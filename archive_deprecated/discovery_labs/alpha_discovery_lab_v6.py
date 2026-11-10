#!/usr/bin/env python3
"""
================================================================================
  ALPHA DISCOVERY LAB V6: FIBONACCI PIN DEFENSE, TOKYO PIVOTS & NR7 EXPANSIONS
================================================================================
Evaluates 4 New Quantitative Alpha Models across the 70% Discovery Partition:
  1. FIB_PIN_DEFENSE: Calibrated 61.8%/78.6% Golden Pocket with 50% Pin Wick filter
  2. TOKYO_PIVOT_REVERSAL: USDJPY Tokyo Asian Open (00:00-03:00 UTC) Daily Pivot defense
  3. EURGBP_H1_BB_FADE: EURGBP H1 2.0-SD Bollinger Band exhaustion fade
  4. DAILY_NR7_EXPANSION: Post-NR7 volatility expansion breakout at European Open (07:15 UTC)

Tested across EURUSD, EURGBP, and USDJPY on 70% Discovery Partition ONLY.
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
# Model 1: Calibrated Fibonacci Golden Pocket Pin Defense
# ==============================================================================
class FibGoldenPocketPinDefense(BaseStrategy):
    """
    High-conviction Golden Pocket (61.8% - 78.6%) retracement defense.
    Requires M15 rejection wick >= 50% and H4 20-EMA trend agreement.
    Target: 2.0R.
    """
    def __init__(self, asset: str, sl_pips: float, tp_pips: float, pip_size: float, max_spread: float, **kwargs):
        super().__init__(**kwargs)
        self.asset = asset
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.pip_size = float(pip_size)
        self.max_spread = float(max_spread)
        self.name = f"{asset}_Fib_Pin_Defense"

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
        if len(h1) < 25:
            return signals, sl_arr, tp_arr

        h1['swing_h'] = h1['high'].rolling(24).max().shift(1)
        h1['swing_l'] = h1['low'].rolling(24).min().shift(1)
        h1['swing_rng'] = h1['swing_h'] - h1['swing_l']

        h1['fib618_up'] = h1['swing_h'] - 0.618 * h1['swing_rng']
        h1['fib786_up'] = h1['swing_h'] - 0.786 * h1['swing_rng']

        h1['fib618_dn'] = h1['swing_l'] + 0.618 * h1['swing_rng']
        h1['fib786_dn'] = h1['swing_l'] + 0.786 * h1['swing_rng']

        h4 = base.resample('4h').agg({'close': 'last'}).dropna()
        h4['ema20'] = h4['close'].ewm(span=20, adjust=False).mean()
        h4_trend = np.where(h4['close'] > h4['ema20'], 1, -1)
        h4_bias = pd.Series(h4_trend, index=h4.index).shift(1).reindex(m15.index, method='ffill').fillna(0).values

        fib618_up_m15 = h1['fib618_up'].reindex(m15.index, method='ffill').values
        fib786_up_m15 = h1['fib786_up'].reindex(m15.index, method='ffill').values
        fib618_dn_m15 = h1['fib618_dn'].reindex(m15.index, method='ffill').values
        fib786_dn_m15 = h1['fib786_dn'].reindex(m15.index, method='ffill').values

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

            bar_rng = (h[i] - l[i]) / self.pip_size
            if bar_rng < 3.0:
                continue

            bias = h4_bias[i]
            sig = 0

            # Uptrend pullback into 61.8% - 78.6% zone with >= 50% lower wick
            if bias > 0:
                f618 = fib618_up_m15[i]
                f786 = fib786_up_m15[i]
                if f786 <= l[i] <= f618 and c[i] > o[i]:
                    lower_wick = (min(o[i], c[i]) - l[i]) / self.pip_size
                    if (lower_wick / bar_rng) >= 0.50:
                        sig = SIGNAL_LONG

            # Downtrend rally into 61.8% - 78.6% zone with >= 50% upper wick
            elif bias < 0:
                f618 = fib618_dn_m15[i]
                f786 = fib786_dn_m15[i]
                if f618 <= h[i] <= f786 and c[i] < o[i]:
                    upper_wick = (h[i] - max(o[i], c[i])) / self.pip_size
                    if (upper_wick / bar_rng) >= 0.50:
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
# Model 2: USDJPY Tokyo Open Daily Pivot Defense (00:00 - 03:00 UTC)
# ==============================================================================
class TokyoPivotReversal(BaseStrategy):
    """
    Fades early Asian session tests of the Daily Pivot on USDJPY.
    """
    def __init__(self, asset: str, sl_pips: float = 12.0, tp_pips: float = 20.0, pip_size: float = 0.01, max_spread: float = 0.8, **kwargs):
        super().__init__(**kwargs)
        self.asset = asset
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.pip_size = float(pip_size)
        self.max_spread = float(max_spread)
        self.name = f"{asset}_Tokyo_Pivot_Reversal"

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
        daily = base.resample('1D').agg({'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
        if len(daily) < 2:
            return signals, sl_arr, tp_arr

        prev_h = daily['high'].shift(1)
        prev_l = daily['low'].shift(1)
        prev_c = daily['close'].shift(1)
        pivot = (prev_h + prev_l + prev_c) / 3.0
        pivot_m15 = pivot.reindex(m15.index, method='ffill').values

        hours = m15.index.hour.values
        dates = m15.index.date
        c = m15['close'].values
        h = m15['high'].values
        l = m15['low'].values
        o = m15['open'].values
        spread = m15['spread_pips'].values
        time_ms = m15.index.astype('datetime64[ms]').view('int64')
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        traded_days = set()

        for i in range(15, len(m15)):
            if hours[i] not in {0, 1, 2, 3}:
                continue
            d = dates[i]
            if d in traded_days:
                continue
            if spread[i] > self.max_spread:
                continue

            pv = pivot_m15[i]
            if np.isnan(pv):
                continue

            bar_rng = (h[i] - l[i]) / self.pip_size
            if bar_rng < 3.5:
                continue

            sig = 0
            # Test Pivot from above as Support
            dist_pv = abs(l[i] - pv) / self.pip_size
            lower_wick = (min(o[i], c[i]) - l[i]) / self.pip_size
            if dist_pv <= 3.5 and (lower_wick / bar_rng) >= 0.45 and c[i] > o[i]:
                sig = SIGNAL_LONG

            # Test Pivot from below as Resistance
            dist_pv_up = abs(pv - h[i]) / self.pip_size
            upper_wick = (h[i] - max(o[i], c[i])) / self.pip_size
            if dist_pv_up <= 3.5 and (upper_wick / bar_rng) >= 0.45 and c[i] < o[i]:
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
# Model 3: EURGBP H1 Bollinger Band Midline Fade
# ==============================================================================
class EURGBPBandMidlineFade(BaseStrategy):
    """
    Fades 2.0-SD H1 Bollinger Band piercings back to the 20-period moving average midline.
    """
    def __init__(self, asset: str, sl_pips: float = 8.0, tp_pips: float = 14.0, pip_size: float = 0.0001, max_spread: float = 2.0, **kwargs):
        super().__init__(**kwargs)
        self.asset = asset
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.pip_size = float(pip_size)
        self.max_spread = float(max_spread)
        self.name = f"{asset}_Band_Midline_Fade"

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
        if len(h1) < 25:
            return signals, sl_arr, tp_arr

        h1['sma20'] = h1['close'].rolling(20).mean()
        h1['std20'] = h1['close'].rolling(20).std()
        h1['bb_upper'] = h1['sma20'] + 2.0 * h1['std20']
        h1['bb_lower'] = h1['sma20'] - 2.0 * h1['std20']

        bb_up_m15 = h1['bb_upper'].reindex(m15.index, method='ffill').values
        bb_dn_m15 = h1['bb_lower'].reindex(m15.index, method='ffill').values

        hours = m15.index.hour.values
        dates = m15.index.date
        c = m15['close'].values
        h = m15['high'].values
        l = m15['low'].values
        o = m15['open'].values
        spread = m15['spread_pips'].values
        time_ms = m15.index.astype('datetime64[ms]').view('int64')
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        traded_days = set()

        for i in range(15, len(m15)):
            if not (8 <= hours[i] <= 15):
                continue
            d = dates[i]
            if d in traded_days:
                continue
            if spread[i] > self.max_spread:
                continue

            bar_rng = (h[i] - l[i]) / self.pip_size
            if bar_rng < 2.5:
                continue

            sig = 0
            # Fade Upper Band: Pierced above upper band, closed red with upper wick >= 45%
            if h[i] >= bb_up_m15[i] and c[i] < o[i]:
                upper_wick = (h[i] - max(o[i], c[i])) / self.pip_size
                if (upper_wick / bar_rng) >= 0.45:
                    sig = SIGNAL_SHORT

            # Fade Lower Band: Pierced below lower band, closed green with lower wick >= 45%
            elif l[i] <= bb_dn_m15[i] and c[i] > o[i]:
                lower_wick = (min(o[i], c[i]) - l[i]) / self.pip_size
                if (lower_wick / bar_rng) >= 0.45:
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
# Model 4: Daily NR7 Volatility Expansion Breakout
# ==============================================================================
class DailyNR7Expansion(BaseStrategy):
    """
    Toby Crabel NR7 Expansion:
    When yesterday's range was the narrowest of the last 7 calendar days,
    triggers on the London Open (07:00-08:00 UTC) breakout of the 06:00-07:00 pre-market range.
    Target: 2.0R.
    """
    def __init__(self, asset: str, sl_pips: float, tp_pips: float, pip_size: float, max_spread: float, **kwargs):
        super().__init__(**kwargs)
        self.asset = asset
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.pip_size = float(pip_size)
        self.max_spread = float(max_spread)
        self.name = f"{asset}_Daily_NR7_Expansion"

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
        daily = base.resample('1D').agg({'high': 'max', 'low': 'min'}).dropna()
        if len(daily) < 8:
            return signals, sl_arr, tp_arr

        daily['rng'] = daily['high'] - daily['low']
        daily['min_rng7'] = daily['rng'].rolling(7).min()
        daily['is_nr7'] = (daily['rng'] == daily['min_rng7']).shift(1).fillna(False)

        nr7_m15 = daily['is_nr7'].reindex(m15.index, method='ffill').fillna(False).values

        hours = m15.index.hour.values
        dates = m15.index.date
        c = m15['close'].values
        h = m15['high'].values
        l = m15['low'].values
        o = m15['open'].values
        spread = m15['spread_pips'].values
        time_ms = m15.index.astype('datetime64[ms]').view('int64')
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        # Pre-market 06:00-07:00 high/low
        pre_ranges = {}
        for d in np.unique(dates):
            mask = (dates == d) & (hours == 6)
            if np.any(mask):
                pre_ranges[d] = (np.max(h[mask]), np.min(l[mask]))

        traded_days = set()

        for i in range(len(m15)):
            if hours[i] not in {7, 8}:
                continue
            d = dates[i]
            if d not in pre_ranges or d in traded_days:
                continue
            if not nr7_m15[i]:
                continue
            if spread[i] > self.max_spread:
                continue

            ph, pl = pre_ranges[d]
            bar_rng = (h[i] - l[i]) / self.pip_size
            body = abs(c[i] - o[i]) / self.pip_size
            if bar_rng < 3.5 or (body / bar_rng) < 0.55:
                continue

            sig = 0
            if c[i] > ph and c[i] > o[i]:
                sig = SIGNAL_LONG
            elif c[i] < pl and c[i] < o[i]:
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
# Discovery V6 Multi-Asset Harness
# ==============================================================================
def evaluate_v6_discovery_week(args: Tuple[str, int]) -> Dict[str, Any]:
    asset, week_num = args
    pip_size = 0.01 if asset == "USDJPY" else 0.0001
    engine = BacktestEngine(pip_size=pip_size, use_slippage=True)

    ticks, meta = get_week_dataset(week_num, raw_dir=f"data/{asset}")
    if len(ticks) == 0:
        return {"asset": asset, "week": week_num, "trades": {}}

    models = []
    if asset == "USDJPY":
        models.append(("FIB_PIN_DEFENSE", FibGoldenPocketPinDefense(asset=asset, sl_pips=12.0, tp_pips=24.0, pip_size=0.01, max_spread=0.8)))
        models.append(("TOKYO_PIVOT_REV", TokyoPivotReversal(asset=asset, sl_pips=12.0, tp_pips=20.0, pip_size=0.01, max_spread=0.8)))
        models.append(("NR7_EXPANSION", DailyNR7Expansion(asset=asset, sl_pips=14.0, tp_pips=28.0, pip_size=0.01, max_spread=0.8)))
    elif asset == "EURGBP":
        models.append(("FIB_PIN_DEFENSE", FibGoldenPocketPinDefense(asset=asset, sl_pips=8.0, tp_pips=16.0, pip_size=0.0001, max_spread=2.0)))
        models.append(("BB_MIDLINE_FADE", EURGBPBandMidlineFade(asset=asset, sl_pips=8.0, tp_pips=14.0, pip_size=0.0001, max_spread=2.0)))
        models.append(("NR7_EXPANSION", DailyNR7Expansion(asset=asset, sl_pips=8.0, tp_pips=16.0, pip_size=0.0001, max_spread=2.0)))
    elif asset == "EURUSD":
        models.append(("FIB_PIN_DEFENSE", FibGoldenPocketPinDefense(asset=asset, sl_pips=9.0, tp_pips=18.0, pip_size=0.0001, max_spread=0.6)))
        models.append(("NR7_EXPANSION", DailyNR7Expansion(asset=asset, sl_pips=9.0, tp_pips=18.0, pip_size=0.0001, max_spread=0.6)))

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


def run_discovery_v6():
    t_start = time.time()
    print("=" * 80, flush=True)
    print("  LAUNCHING ALPHA DISCOVERY V6 EXPEDITION (70% DISCOVERY PARTITIONS)", flush=True)
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
        futures = {executor.submit(evaluate_v6_discovery_week, t): t for t in all_tasks}
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
            "Asset": asset,
            "Strategy": m_name,
            "Discovery_Trades": n,
            "Win_Rate_Pct": round(wr, 1),
            "Profit_Factor": pf,
            "Net_Pips": round(net_pips, 1),
            "Max_DD_Pips": round(max_dd, 1),
            "Ret_DD_Ratio": ret_dd,
            "Expectancy_pips": expectancy,
            "Discovery_Status": "🌟 VIABLE CANDIDATE" if net_pips > 100.0 and pf >= 1.35 and expectancy >= 1.5 else ("✅ POSITIVE" if net_pips > 0 and pf >= 1.15 else "❌ REJECT")
        })

    df = pd.DataFrame(rows).sort_values(by="Net_Pips", ascending=False)
    csv_out = RESULTS_DIR / "alpha_discovery_v6_scorecard.csv"
    df.to_csv(csv_out, index=False)
    print(f"\n[+] Saved Discovery V6 Scorecard to: {csv_out.resolve()}", flush=True)
    print("\n" + "=" * 80, flush=True)
    print(f"DISCOVERY V6 EXPEDITION COMPLETED IN {time.time() - t_start:.1f}s!", flush=True)
    print("=" * 80, flush=True)
    return df


if __name__ == "__main__":
    run_discovery_v6()
