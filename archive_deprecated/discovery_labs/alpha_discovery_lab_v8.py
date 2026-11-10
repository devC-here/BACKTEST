#!/usr/bin/env python3
"""
================================================================================
  ALPHA DISCOVERY LAB V8: ASIAN RETESTS, FIX REVERSALS, INSIDE BARS & 200-EMA
================================================================================
Evaluates 4 New Systematic Quantitative Alpha Models across 70% Discovery Partition:
  1. DISC54_ASIAN_BOX_RETEST: Asian Range (00:00 - 06:00 UTC) breakout + pullback retest
     at European Open (07:00 - 11:00 UTC) with 2.0R target.
  2. DISC55_LONDON_FIX_REVERSAL: London 4 PM Fix (15:30 - 16:30 UTC) order flow exhaustion
     mean-reversion back to rolling H1 mean.
  3. DISC56_H1_INSIDE_BAR_EXPANSION: H1 inside bar volatility compression breakout
     aligned with H4 20-EMA trend direction.
  4. DISC57_M15_200EMA_DYNAMIC_BOUNCE: M15 200-EMA institutional multi-day value defense
     aligned with H4 macro trend.

Tested across EURUSD, EURGBP, and USDJPY on 70% Discovery Partition ONLY.
Strict 70/30 deterministic holdout isolation is maintained.
================================================================================
"""

from __future__ import annotations
import sys
import os
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Any, Tuple

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tick_engine.core.engine import BacktestEngine
from tick_engine.core.dataloader import get_week_dataset, get_discovery_weeks
from tick_engine.strategies.base import BaseStrategy, SIGNAL_LONG, SIGNAL_SHORT
from data_engineering_suite.bars import build_time_bars

RESULTS_DIR = ROOT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# Model 1: Asian Box Breakout & Boundary Retest (DISC54)
# ==============================================================================
class AsianBoxRetest(BaseStrategy):
    """
    Measures 00:00 - 06:00 UTC Asian High & Low.
    During 07:00 - 11:00 UTC, if price breaks outside the Asian range,
    waits for a retest of the broken Asian boundary level within 2.5 pips
    and enters on confirmation with a 2.0R target.
    """
    def __init__(self, asset: str, sl_pips: float, tp_pips: float, pip_size: float, max_spread: float, **kwargs):
        super().__init__(**kwargs)
        self.asset = asset
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.pip_size = float(pip_size)
        self.max_spread = float(max_spread)
        self.name = f"{asset}_Asian_Box_Retest"

    def generate_signals(self, ticks: np.ndarray):
        n = len(ticks)
        signals = np.zeros(n, dtype=np.int8)
        sl_arr = np.zeros(n, dtype=np.float64)
        tp_arr = np.zeros(n, dtype=np.float64)
        if n < 500:
            return signals, sl_arr, tp_arr

        m15 = build_time_bars(ticks, timeframe_min=15, pip_size=self.pip_size)
        if len(m15) < 30:
            return signals, sl_arr, tp_arr

        c = m15['close'].values
        h = m15['high'].values
        l = m15['low'].values
        o = m15['open'].values
        spread = m15['spread_pips'].values
        hours = m15.index.hour.values
        dates = m15.index.date
        time_ms = m15.index.astype('datetime64[ms]').view('int64')
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        # Precompute Asian ranges for each date
        asian_ranges = {}
        for d in np.unique(dates):
            mask = (dates == d) & (hours >= 0) & (hours < 6)
            if np.any(mask):
                ah = np.max(h[mask])
                al = np.min(l[mask])
                rng_p = (ah - al) / self.pip_size
                asian_ranges[d] = (ah, al, rng_p)

        traded_days = set()
        tol = 2.5 * self.pip_size
        max_asian_pips = 45.0 if self.asset == "USDJPY" else 30.0

        for i in range(len(m15)):
            if not (7 <= hours[i] <= 11):
                continue
            d = dates[i]
            if d not in asian_ranges or d in traded_days:
                continue
            if spread[i] > self.max_spread:
                continue

            ah, al, arng = asian_ranges[d]
            if arng > max_asian_pips or arng < 5.0:
                continue

            sig = 0

            # Bullish: Previous bar was above Asian High, current bar pulls back to test AH and holds
            if l[i] <= ah + tol and c[i] > ah and c[i] > o[i]:
                # Rejection wick or solid close above AH
                sig = SIGNAL_LONG

            # Bearish: Previous bar was below Asian Low, current bar tests AL and holds
            elif h[i] >= al - tol and c[i] < al and c[i] < o[i]:
                sig = SIGNAL_SHORT

            if sig != 0:
                exec_t = time_ms[i] + dur_ms
                idx = int(np.searchsorted(tick_ts, exec_t))
                if idx < n and signals[idx] == 0:
                    signals[idx] = sig
                    sl_arr[idx] = self.sl_pips
                    tp_arr[idx] = self.tp_pips
                    traded_days.add(d)

        return signals, sl_arr, tp_arr


# ==============================================================================
# Model 2: London 4 PM Fix Order Flow Reversal (DISC55)
# ==============================================================================
class LondonFixReversal(BaseStrategy):
    """
    Between 15:30 and 16:30 UTC, institutional month-end/daily fix flows cause
    extreme one-sided extensions.
    When price extends > 2.2 SD beyond the rolling 20-period H1 SMA,
    enters a mean-reverting counter-trend trade targeting the mean.
    """
    def __init__(self, asset: str, sl_pips: float, tp_pips: float, pip_size: float, max_spread: float, **kwargs):
        super().__init__(**kwargs)
        self.asset = asset
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.pip_size = float(pip_size)
        self.max_spread = float(max_spread)
        self.name = f"{asset}_London_Fix_Reversal"

    def generate_signals(self, ticks: np.ndarray):
        n = len(ticks)
        signals = np.zeros(n, dtype=np.int8)
        sl_arr = np.zeros(n, dtype=np.float64)
        tp_arr = np.zeros(n, dtype=np.float64)
        if n < 500:
            return signals, sl_arr, tp_arr

        m15 = build_time_bars(ticks, timeframe_min=15, pip_size=self.pip_size)
        if len(m15) < 30:
            return signals, sl_arr, tp_arr

        ts = pd.to_datetime(ticks['timestamp'], unit='ms')
        mid = 0.5 * (ticks['bid'] + ticks['ask'])
        base = pd.DataFrame({'close': mid, 'high': mid, 'low': mid}, index=ts)

        h1 = base.resample('1h').agg({'close': 'last'}).dropna()
        if len(h1) < 25:
            return signals, sl_arr, tp_arr

        h1['sma20'] = h1['close'].rolling(20).mean()
        h1['std20'] = h1['close'].rolling(20).std()
        h1['upper22'] = h1['sma20'] + 2.2 * h1['std20']
        h1['lower22'] = h1['sma20'] - 2.2 * h1['std20']

        up_m15 = h1['upper22'].reindex(m15.index, method='ffill').values
        dn_m15 = h1['lower22'].reindex(m15.index, method='ffill').values

        c = m15['close'].values
        h = m15['high'].values
        l = m15['low'].values
        o = m15['open'].values
        spread = m15['spread_pips'].values
        hours = m15.index.hour.values
        mins = m15.index.minute.values
        dates = m15.index.date
        time_ms = m15.index.astype('datetime64[ms]').view('int64')
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        traded_days = set()

        for i in range(25, len(m15)):
            # Fix window: 15:30 to 16:30 UTC
            h_val = hours[i]
            m_val = mins[i]
            in_fix = (h_val == 15 and m_val >= 30) or (h_val == 16 and m_val <= 30)
            if not in_fix:
                continue

            d = dates[i]
            if d in traded_days:
                continue
            if spread[i] > self.max_spread:
                continue

            bar_rng = (h[i] - l[i]) / self.pip_size
            if bar_rng < 3.0:
                continue

            u = up_m15[i]
            dn = dn_m15[i]
            if np.isnan(u) or np.isnan(dn):
                continue

            sig = 0
            # Pierced upper band and closed back inside with bearish candle
            if h[i] >= u and c[i] < u and c[i] < o[i]:
                sig = SIGNAL_SHORT
            # Pierced lower band and closed back inside with bullish candle
            elif l[i] <= dn and c[i] > dn and c[i] > o[i]:
                sig = SIGNAL_LONG

            if sig != 0:
                exec_t = time_ms[i] + dur_ms
                idx = int(np.searchsorted(tick_ts, exec_t))
                if idx < n and signals[idx] == 0:
                    signals[idx] = sig
                    sl_arr[idx] = self.sl_pips
                    tp_arr[idx] = self.tp_pips
                    traded_days.add(d)

        return signals, sl_arr, tp_arr


# ==============================================================================
# Model 3: H1 Inside Bar Volatility Compression Breakout (DISC56)
# ==============================================================================
class H1InsideBarExpansion(BaseStrategy):
    """
    Identifies H1 inside bars (High < prev High and Low > prev Low).
    During European/US hours (07:00 - 16:00 UTC), when M15 breaks decisively beyond
    the mother H1 bar range in alignment with H4 20-EMA trend, enter breakout with 2.0R target.
    """
    def __init__(self, asset: str, sl_pips: float, tp_pips: float, pip_size: float, max_spread: float, **kwargs):
        super().__init__(**kwargs)
        self.asset = asset
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.pip_size = float(pip_size)
        self.max_spread = float(max_spread)
        self.name = f"{asset}_H1_Inside_Bar_Expansion"

    def generate_signals(self, ticks: np.ndarray):
        n = len(ticks)
        signals = np.zeros(n, dtype=np.int8)
        sl_arr = np.zeros(n, dtype=np.float64)
        tp_arr = np.zeros(n, dtype=np.float64)
        if n < 500:
            return signals, sl_arr, tp_arr

        m15 = build_time_bars(ticks, timeframe_min=15, pip_size=self.pip_size)
        if len(m15) < 30:
            return signals, sl_arr, tp_arr

        ts = pd.to_datetime(ticks['timestamp'], unit='ms')
        mid = 0.5 * (ticks['bid'] + ticks['ask'])
        base = pd.DataFrame({'close': mid, 'high': mid, 'low': mid}, index=ts)

        h1 = base.resample('1h').agg({'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
        h4 = base.resample('4h').agg({'close': 'last'}).dropna()
        if len(h1) < 25 or len(h4) < 15:
            return signals, sl_arr, tp_arr

        h4['ema20'] = h4['close'].ewm(span=20, adjust=False).mean()
        h4_trend = np.where(h4['close'] > h4['ema20'], 1, -1)
        h4_bias = pd.Series(h4_trend, index=h4.index).shift(1).reindex(m15.index, method='ffill').fillna(0).values

        # Detect inside bar on H1: bar i has High <= High[i-1] and Low >= Low[i-1]
        h1['mother_h'] = h1['high'].shift(1)
        h1['mother_l'] = h1['low'].shift(1)
        h1['is_inside'] = (h1['high'] <= h1['mother_h']) & (h1['low'] >= h1['mother_l'])

        # Forward fill mother high/low when inside bar occurs
        inside_mother_h = np.where(h1['is_inside'], h1['mother_h'], np.nan)
        inside_mother_l = np.where(h1['is_inside'], h1['mother_l'], np.nan)

        h1['active_mother_h'] = pd.Series(inside_mother_h, index=h1.index).ffill()
        h1['active_mother_l'] = pd.Series(inside_mother_l, index=h1.index).ffill()

        mother_h_m15 = h1['active_mother_h'].reindex(m15.index, method='ffill').values
        mother_l_m15 = h1['active_mother_l'].reindex(m15.index, method='ffill').values

        c = m15['close'].values
        h = m15['high'].values
        l = m15['low'].values
        o = m15['open'].values
        spread = m15['spread_pips'].values
        hours = m15.index.hour.values
        dates = m15.index.date
        time_ms = m15.index.astype('datetime64[ms]').view('int64')
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        traded_days = set()

        for i in range(25, len(m15)):
            if not (7 <= hours[i] <= 16):
                continue
            d = dates[i]
            if d in traded_days:
                continue
            if spread[i] > self.max_spread:
                continue

            mh = mother_h_m15[i]
            ml = mother_l_m15[i]
            if np.isnan(mh) or np.isnan(ml):
                continue

            bias = h4_bias[i]
            bar_rng = (h[i] - l[i]) / self.pip_size
            body = abs(c[i] - o[i]) / self.pip_size
            if bar_rng < 3.0 or (body / bar_rng) < 0.50:
                continue

            sig = 0
            # Bullish: H4 trend up, M15 breaks above mother high with strong close
            if bias > 0 and c[i] > mh and o[i] <= mh and c[i] > o[i]:
                sig = SIGNAL_LONG
            # Bearish: H4 trend down, M15 breaks below mother low with strong close
            elif bias < 0 and c[i] < ml and o[i] >= ml and c[i] < o[i]:
                sig = SIGNAL_SHORT

            if sig != 0:
                exec_t = time_ms[i] + dur_ms
                idx = int(np.searchsorted(tick_ts, exec_t))
                if idx < n and signals[idx] == 0:
                    signals[idx] = sig
                    sl_arr[idx] = self.sl_pips
                    tp_arr[idx] = self.tp_pips
                    traded_days.add(d)

        return signals, sl_arr, tp_arr


# ==============================================================================
# Model 4: M15 200-EMA Institutional Multi-Day Value Bounce (DISC57)
# ==============================================================================
class M15TwoHundredEMABounce(BaseStrategy):
    """
    The 200-period EMA on M15 represents ~50 hours of market price history.
    When a strong H4 trend is active, pullbacks to the M15 200-EMA offer high-conviction
    institutional re-entry with tight risk and 2.0R target.
    """
    def __init__(self, asset: str, sl_pips: float, tp_pips: float, pip_size: float, max_spread: float, **kwargs):
        super().__init__(**kwargs)
        self.asset = asset
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.pip_size = float(pip_size)
        self.max_spread = float(max_spread)
        self.name = f"{asset}_M15_200EMA_Bounce"

    def generate_signals(self, ticks: np.ndarray):
        n = len(ticks)
        signals = np.zeros(n, dtype=np.int8)
        sl_arr = np.zeros(n, dtype=np.float64)
        tp_arr = np.zeros(n, dtype=np.float64)
        if n < 500:
            return signals, sl_arr, tp_arr

        m15 = build_time_bars(ticks, timeframe_min=15, pip_size=self.pip_size)
        if len(m15) < 220:
            return signals, sl_arr, tp_arr

        ts = pd.to_datetime(ticks['timestamp'], unit='ms')
        mid = 0.5 * (ticks['bid'] + ticks['ask'])
        base = pd.DataFrame({'close': mid, 'high': mid, 'low': mid}, index=ts)

        h4 = base.resample('4h').agg({'close': 'last'}).dropna()
        if len(h4) < 15:
            return signals, sl_arr, tp_arr

        h4['ema20'] = h4['close'].ewm(span=20, adjust=False).mean()
        h4_trend = np.where(h4['close'] > h4['ema20'], 1, -1)
        h4_bias = pd.Series(h4_trend, index=h4.index).shift(1).reindex(m15.index, method='ffill').fillna(0).values

        m15['ema200'] = m15['close'].ewm(span=200, adjust=False).mean()
        ema200 = m15['ema200'].values

        c = m15['close'].values
        h = m15['high'].values
        l = m15['low'].values
        o = m15['open'].values
        spread = m15['spread_pips'].values
        hours = m15.index.hour.values
        dates = m15.index.date
        time_ms = m15.index.astype('datetime64[ms]').view('int64')
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        traded_days = set()
        tol = 3.0 * self.pip_size

        for i in range(200, len(m15)):
            if not (7 <= hours[i] <= 17):
                continue
            d = dates[i]
            if d in traded_days:
                continue
            if spread[i] > self.max_spread:
                continue

            e200 = ema200[i]
            bias = h4_bias[i]
            bar_rng = (h[i] - l[i]) / self.pip_size
            if bar_rng < 3.5:
                continue

            sig = 0
            # Bullish trend: Low touches or pierces within 3 pips of EMA200 and closes above
            if bias > 0 and l[i] <= e200 + tol and c[i] > e200 and c[i] > o[i]:
                lower_wick = (min(o[i], c[i]) - l[i]) / self.pip_size
                if (lower_wick / bar_rng) >= 0.40:
                    sig = SIGNAL_LONG

            # Bearish trend: High touches or pierces within 3 pips of EMA200 and closes below
            elif bias < 0 and h[i] >= e200 - tol and c[i] < e200 and c[i] < o[i]:
                upper_wick = (h[i] - max(o[i], c[i])) / self.pip_size
                if (upper_wick / bar_rng) >= 0.40:
                    sig = SIGNAL_SHORT

            if sig != 0:
                exec_t = time_ms[i] + dur_ms
                idx = int(np.searchsorted(tick_ts, exec_t))
                if idx < n and signals[idx] == 0:
                    signals[idx] = sig
                    sl_arr[idx] = self.sl_pips
                    tp_arr[idx] = self.tp_pips
                    traded_days.add(d)

        return signals, sl_arr, tp_arr


# ==============================================================================
# Discovery V8 Multi-Asset Parallel Worker
# ==============================================================================
def evaluate_v8_discovery_week(args: Tuple[str, int]) -> Dict[str, Any]:
    asset, week_num = args
    pip_size = 0.01 if asset == "USDJPY" else 0.0001
    engine = BacktestEngine(pip_size=pip_size, use_slippage=True)

    ticks, meta = get_week_dataset(week_num, raw_dir=f"data/{asset}")
    if len(ticks) == 0:
        return {"asset": asset, "week": week_num, "trades": {}}

    models = []
    if asset == "USDJPY":
        models.append(("ASIAN_BOX_RETEST", AsianBoxRetest(asset=asset, sl_pips=12.0, tp_pips=24.0, pip_size=0.01, max_spread=0.8)))
        models.append(("LONDON_FIX_REV", LondonFixReversal(asset=asset, sl_pips=12.0, tp_pips=20.0, pip_size=0.01, max_spread=0.8)))
        models.append(("H1_INSIDE_EXP", H1InsideBarExpansion(asset=asset, sl_pips=12.0, tp_pips=24.0, pip_size=0.01, max_spread=0.8)))
        models.append(("M15_200EMA_BOUNCE", M15TwoHundredEMABounce(asset=asset, sl_pips=12.0, tp_pips=24.0, pip_size=0.01, max_spread=0.8)))
    elif asset == "EURGBP":
        models.append(("ASIAN_BOX_RETEST", AsianBoxRetest(asset=asset, sl_pips=7.0, tp_pips=14.0, pip_size=0.0001, max_spread=2.0)))
        models.append(("LONDON_FIX_REV", LondonFixReversal(asset=asset, sl_pips=7.0, tp_pips=12.0, pip_size=0.0001, max_spread=2.0)))
        models.append(("H1_INSIDE_EXP", H1InsideBarExpansion(asset=asset, sl_pips=7.0, tp_pips=14.0, pip_size=0.0001, max_spread=2.0)))
        models.append(("M15_200EMA_BOUNCE", M15TwoHundredEMABounce(asset=asset, sl_pips=7.0, tp_pips=14.0, pip_size=0.0001, max_spread=2.0)))
    elif asset == "EURUSD":
        models.append(("ASIAN_BOX_RETEST", AsianBoxRetest(asset=asset, sl_pips=9.0, tp_pips=18.0, pip_size=0.0001, max_spread=0.6)))
        models.append(("LONDON_FIX_REV", LondonFixReversal(asset=asset, sl_pips=9.0, tp_pips=15.0, pip_size=0.0001, max_spread=0.6)))
        models.append(("H1_INSIDE_EXP", H1InsideBarExpansion(asset=asset, sl_pips=9.0, tp_pips=18.0, pip_size=0.0001, max_spread=0.6)))
        models.append(("M15_200EMA_BOUNCE", M15TwoHundredEMABounce(asset=asset, sl_pips=9.0, tp_pips=18.0, pip_size=0.0001, max_spread=0.6)))

    week_trades = {}
    for m_name, model in models:
        try:
            raw_trades = engine.run(ticks, model)
            t_list = []
            for i in range(len(raw_trades)):
                row = raw_trades[i]
                t_list.append({
                    "strategy": m_name,
                    "asset": asset,
                    "week": week_num,
                    "pips": float(row["pips"]),
                    "direction": int(row["direction"])
                })
            week_trades[m_name] = t_list
        except Exception:
            week_trades[m_name] = []

    return {"asset": asset, "week": week_num, "trades": week_trades}


def run_discovery_v8():
    t_start = time.time()
    print("=" * 80, flush=True)
    print("  LAUNCHING ALPHA DISCOVERY V8 EXPEDITION (70% DISCOVERY PARTITIONS)", flush=True)
    print("=" * 80, flush=True)

    assets = [os.environ["DISCOVERY_TARGET_SYMBOL"]] if "DISCOVERY_TARGET_SYMBOL" in os.environ else ["USDJPY", "EURGBP", "EURUSD"]
    all_tasks = []
    for asset in assets:
        disc_weeks = get_discovery_weeks(f"data/{asset}")
        print(f"[*] Queued {len(disc_weeks)} discovery weeks for {asset}...", flush=True)
        for w in disc_weeks:
            all_tasks.append((asset, w))

    print(f"\n[*] Total discovery tasks across all assets: {len(all_tasks)} weeks", flush=True)

    results_by_key: Dict[Tuple[str, str], List[float]] = {}
    completed = 0

    with ProcessPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(evaluate_v8_discovery_week, t): t for t in all_tasks}
        for future in as_completed(futures):
            res = future.result()
            completed += 1
            asset = res["asset"]
            for m_name, t_list in res["trades"].items():
                key = (asset, m_name)
                if key not in results_by_key:
                    results_by_key[key] = []
                results_by_key[key].extend([t["pips"] for t in t_list])

            if completed % 100 == 0 or completed == len(all_tasks):
                print(f"  [+] Processed {completed:3d}/{len(all_tasks)} discovery weeks in {time.time() - t_start:.1f}s...", flush=True)

    # Compute Scorecard
    rows = []
    for (asset, m_name), pips in results_by_key.items():
        n = len(pips)
        if n == 0:
            continue
        p_arr = np.array(pips)
        net_pips = float(np.sum(p_arr))
        wins = p_arr[p_arr > 0]
        losses = -p_arr[p_arr < 0]
        wr = float(len(wins) / n * 100.0)
        sum_w = float(np.sum(wins))
        sum_l = float(np.sum(losses))
        pf = round(sum_w / sum_l, 2) if sum_l > 0 else (99.0 if sum_w > 0 else 0.0)

        equity = np.cumsum(p_arr)
        peak = np.maximum.accumulate(equity)
        dd = peak - equity
        max_dd = float(np.max(dd)) if len(dd) > 0 else 0.0
        ret_dd = round(net_pips / max_dd, 2) if max_dd > 0 else 0.0
        expectancy = round(net_pips / n, 2)

        rows.append({
            "asset": asset,
            "strategy": m_name,
            "trades": n,
            "win_rate": wr,
            "net_pips": net_pips,
            "profit_factor": pf,
            "max_dd_pips": max_dd,
            "ret_dd": ret_dd,
            "expectancy": expectancy
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(by=["asset", "net_pips"], ascending=[True, False])
        out_csv = RESULTS_DIR / "alpha_discovery_lab_v8_scorecard.csv"
        df.to_csv(out_csv, index=False)

        print("\n" + "=" * 90, flush=True)
        print("  ALPHA DISCOVERY LAB V8 SCORECARD (70% DISCOVERY PARTITIONS ONLY)", flush=True)
        print("=" * 90, flush=True)
        print(f"{'Asset':<8} {'Strategy':<20} {'Trades':<8} {'Win Rate':<10} {'Net Pips':<12} {'PF':<8} {'Max DD':<10} {'Ret/DD':<8} {'Exp (p/t)':<10}")
        print("-" * 90, flush=True)
        for _, r in df.iterrows():
            print(f"{r['asset']:<8} {r['strategy']:<20} {int(r['trades']):<8} {r['win_rate']:>6.1f}%   {r['net_pips']:>10.1f}p  {r['profit_factor']:>6.2f}  {r['max_dd_pips']:>8.1f}p  {r['ret_dd']:>6.2f}x  {r['expectancy']:>8.2f}p")
        print("=" * 90, flush=True)

    print(f"\n[*] Alpha Discovery Lab V8 Complete in {time.time() - t_start:.2f}s.", flush=True)


if __name__ == "__main__":
    run_discovery_v8()
