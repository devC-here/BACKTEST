#!/usr/bin/env python3
"""
================================================================================
  ALPHA DISCOVERY LAB V3: STRUCTURAL LEVEL MINING & TOKYO CALIBRATION
================================================================================
Evaluates 3 Structural High-Expectancy Models across the 70% Discovery Partition:
  1. TOKYO_BOX_CALIBRATED: Quality-filtered Tokyo Fix Breakout on USDJPY.
  2. MACRO_RETEST_EURUSD: Transfer of H1 Donchian + M15 Rejection to EURUSD.
  3. MACRO_RETEST_EURGBP: Transfer of H1 Donchian + M15 Rejection to EURGBP.
  4. DAILY_HIGH_LOW_DEFENSE: Previous Day High/Low retest defense with H4 trend alignment.

All evaluated exclusively on the 70% Discovery Partition.
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
# Model 1: Calibrated Tokyo Fix Expansion (USDJPY)
# ==============================================================================
class USDJPYTokyoFixCalibrated(BaseStrategy):
    """
    Quality-filtered version of Tokyo Fix Breakout.
    - Measures 00:00-01:00 UTC Tokyo opening range.
    - Requires range between 12.0 and 32.0 pips (skips low volatility & news spikes).
    - Requires H4 20-EMA > H4 50-EMA for longs, and vice-versa for shorts.
    - Max 1 trade per day.
    - Trigger window: 01:15 to 03:45 UTC.
    """
    def __init__(self, sl_pips: float = 12.0, tp_pips: float = 20.0, pip_size: float = 0.01, max_spread: float = 0.8, **kwargs):
        super().__init__(**kwargs)
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.pip_size = float(pip_size)
        self.max_spread = float(max_spread)
        self.name = "USDJPY_Tokyo_Fix_Calibrated"

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
        base = pd.DataFrame({'close': mid}, index=ts)
        h4 = base.resample('4h').agg({'close': 'last'}).dropna()
        if len(h4) < 15:
            return signals, sl_arr, tp_arr

        h4['ema20'] = h4['close'].ewm(span=20, adjust=False).mean()
        h4['ema50'] = h4['close'].ewm(span=50, adjust=False).mean()
        h4_trend = np.where(h4['ema20'] > h4['ema50'], 1, -1)
        h4_bias = pd.Series(h4_trend, index=h4.index).shift(1).reindex(m15.index, method='ffill').fillna(0).values

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

        # Measure 00:00-01:00 range
        day_boxes = {}
        for d in np.unique(dates):
            mask_box = (dates == d) & (hours == 0)
            if np.any(mask_box):
                bh = np.max(h[mask_box])
                bl = np.min(l[mask_box])
                br = (bh - bl) / self.pip_size
                if 12.0 <= br <= 32.0:
                    day_boxes[d] = (bh, bl)

        traded_days = set()

        for i in range(len(m15)):
            d = dates[i]
            hr = hours[i]
            if hr not in {1, 2, 3}:
                continue
            if d not in day_boxes or d in traded_days:
                continue
            if spread[i] > self.max_spread:
                continue

            bh, bl = day_boxes[d]
            bias = h4_bias[i]
            bar_rng = (h[i] - l[i]) / self.pip_size
            body = abs(c[i] - o[i]) / self.pip_size
            if bar_rng < 4.0 or (body / bar_rng) < 0.50:
                continue

            sig = 0
            # Long: Bullish H4, Close breaks above box high, strong green body
            if bias > 0 and c[i] > bh and c[i] > o[i]:
                sig = SIGNAL_LONG
            # Short: Bearish H4, Close breaks below box low, strong red body
            elif bias < 0 and c[i] < bl and c[i] < o[i]:
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
# Model 2 & 3: Multi-Asset Intraday Macro Retest (H1 Donchian + M15 Rejection)
# ==============================================================================
class IntradayMacroRetestMultiAsset(BaseStrategy):
    """
    Intraday Macro Retest model evaluated on EURUSD and EURGBP.
    H1 20-bar Donchian level test with M15 rejection wick (>= 45%).
    Active hours: 07:00 to 16:30 UTC.
    """
    def __init__(self, sl_pips: float = 9.0, tp_pips: float = 15.0, pip_size: float = 0.0001, max_spread: float = 0.6, **kwargs):
        super().__init__(**kwargs)
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.pip_size = float(pip_size)
        self.max_spread = float(max_spread)
        self.name = "Intraday_Macro_Retest_MultiAsset"

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

        h1['ema20'] = h1['close'].ewm(span=20, adjust=False).mean()
        h1['bias'] = np.where(h1['close'] > h1['ema20'], 1, -1)
        h1['roll_h'] = h1['high'].rolling(20).max().shift(1).bfill()
        h1['roll_l'] = h1['low'].rolling(20).min().shift(1).bfill()

        bias_m15 = h1['bias'].reindex(m15.index, method='ffill').fillna(0).values
        supp_m15 = h1['roll_l'].reindex(m15.index, method='ffill').values
        res_m15 = h1['roll_h'].reindex(m15.index, method='ffill').values

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

        for i in range(15, len(m15)):
            hr = hours[i]
            if not (7 <= hr <= 16):
                continue
            d = dates[i]
            if d in traded_days:
                continue
            if spread[i] > self.max_spread:
                continue

            bar_rng = (h[i] - l[i]) / self.pip_size
            if bar_rng < 3.0:
                continue

            b = bias_m15[i]
            sig = 0

            # Bullish Retest of H1 Support
            if b > 0:
                dist_supp = abs(l[i] - supp_m15[i]) / self.pip_size
                lower_wick = (min(o[i], c[i]) - l[i]) / self.pip_size
                if dist_supp <= 3.5 and (lower_wick / bar_rng) >= 0.45 and c[i] > o[i]:
                    sig = SIGNAL_LONG

            # Bearish Retest of H1 Resistance
            elif b < 0:
                dist_res = abs(res_m15[i] - h[i]) / self.pip_size
                upper_wick = (h[i] - max(o[i], c[i])) / self.pip_size
                if dist_res <= 3.5 and (upper_wick / bar_rng) >= 0.45 and c[i] < o[i]:
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
# Model 4: Daily High/Low Defense with H4 Trend Alignment
# ==============================================================================
class DailyHighLowDefense(BaseStrategy):
    """
    Tests Previous Day High / Low as key institutional liquidity walls.
    When price retests PDH/PDL with H4 20-EMA trend agreement and forms a rejection wick, enter continuation.
    """
    def __init__(self, sl_pips: float = 10.0, tp_pips: float = 16.0, pip_size: float = 0.0001, max_spread: float = 0.8, **kwargs):
        super().__init__(**kwargs)
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.pip_size = float(pip_size)
        self.max_spread = float(max_spread)
        self.name = "Daily_High_Low_Defense"

    def generate_signals(self, ticks: np.ndarray):
        n = len(ticks)
        signals = np.zeros(n, dtype=np.int8)
        sl_arr = np.zeros(n, dtype=np.float64)
        tp_arr = np.zeros(n, dtype=np.float64)
        if n < 500:
            return signals, sl_arr, tp_arr

        m30 = build_time_bars(ticks, timeframe_min=30, pip_size=self.pip_size)
        if len(m30) < 30:
            return signals, sl_arr, tp_arr

        ts = pd.to_datetime(ticks['timestamp'], unit='ms')
        mid = 0.5 * (ticks['bid'] + ticks['ask'])
        base = pd.DataFrame({'close': mid, 'high': mid, 'low': mid}, index=ts)

        # H4 Trend
        h4 = base.resample('4h').agg({'close': 'last'}).dropna()
        if len(h4) < 15:
            return signals, sl_arr, tp_arr
        h4['ema20'] = h4['close'].ewm(span=20, adjust=False).mean()
        h4_trend = np.where(h4['close'] > h4['ema20'], 1, -1)
        h4_bias = pd.Series(h4_trend, index=h4.index).shift(1).reindex(m30.index, method='ffill').fillna(0).values

        # Daily High / Low
        daily = base.resample('1D').agg({'high': 'max', 'low': 'min'}).dropna()
        if len(daily) < 2:
            return signals, sl_arr, tp_arr
        pdh_series = daily['high'].shift(1)
        pdl_series = daily['low'].shift(1)
        pdh_m30 = pdh_series.reindex(m30.index, method='ffill').values
        pdl_m30 = pdl_series.reindex(m30.index, method='ffill').values

        hours = m30.index.hour.values
        dates = m30.index.date
        c = m30['close'].values
        h = m30['high'].values
        l = m30['low'].values
        o = m30['open'].values
        spread = m30['spread_pips'].values
        time_ms = m30.index.astype('datetime64[ms]').view('int64')
        dur_ms = 30 * 60 * 1000
        tick_ts = ticks['timestamp']

        traded_days = set()

        for i in range(10, len(m30)):
            hr = hours[i]
            if not (7 <= hr <= 16):
                continue
            d = dates[i]
            if d in traded_days:
                continue
            if spread[i] > self.max_spread:
                continue

            pdh = pdh_m30[i]
            pdl = pdl_m30[i]
            if np.isnan(pdh) or np.isnan(pdl):
                continue

            bias = h4_bias[i]
            bar_rng = (h[i] - l[i]) / self.pip_size
            if bar_rng < 4.0:
                continue

            sig = 0
            # Bullish: Macro Up, Price retests PDH from above (PDH acting as support), lower wick >= 45%
            if bias > 0:
                dist_pdh = abs(l[i] - pdh) / self.pip_size
                lower_wick = (min(o[i], c[i]) - l[i]) / self.pip_size
                if dist_pdh <= 4.0 and (lower_wick / bar_rng) >= 0.45 and c[i] > o[i]:
                    sig = SIGNAL_LONG

            # Bearish: Macro Down, Price retests PDL from below (PDL acting as resistance), upper wick >= 45%
            elif bias < 0:
                dist_pdl = abs(pdl - h[i]) / self.pip_size
                upper_wick = (h[i] - max(o[i], c[i])) / self.pip_size
                if dist_pdl <= 4.0 and (upper_wick / bar_rng) >= 0.45 and c[i] < o[i]:
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
# Discovery V3 Execution Harness
# ==============================================================================
def evaluate_v3_discovery_week(args: Tuple[str, int]) -> Dict[str, Any]:
    asset, week_num = args
    pip_size = 0.01 if asset == "USDJPY" else 0.0001
    engine = BacktestEngine(pip_size=pip_size, use_slippage=True)

    ticks, meta = get_week_dataset(week_num, raw_dir=f"data/{asset}")
    if len(ticks) == 0:
        return {"asset": asset, "week": week_num, "trades": {}}

    models = []
    if asset == "USDJPY":
        models.append(("TOKYO_CALIBRATED", USDJPYTokyoFixCalibrated(pip_size=0.01, max_spread=0.8, sl_pips=12.0, tp_pips=20.0)))
        models.append(("PDH_PDL_DEFENSE", DailyHighLowDefense(pip_size=0.01, max_spread=0.8, sl_pips=14.0, tp_pips=22.0)))
    elif asset == "EURGBP":
        models.append(("MACRO_RETEST_EG", IntradayMacroRetestMultiAsset(pip_size=0.0001, max_spread=2.0, sl_pips=8.0, tp_pips=14.0)))
        models.append(("PDH_PDL_DEFENSE", DailyHighLowDefense(pip_size=0.0001, max_spread=2.0, sl_pips=8.0, tp_pips=14.0)))
    elif asset == "EURUSD":
        models.append(("MACRO_RETEST_EU", IntradayMacroRetestMultiAsset(pip_size=0.0001, max_spread=0.6, sl_pips=9.0, tp_pips=15.0)))
        models.append(("PDH_PDL_DEFENSE", DailyHighLowDefense(pip_size=0.0001, max_spread=0.6, sl_pips=10.0, tp_pips=16.0)))

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


def run_discovery_v3():
    t_start = time.time()
    print("=" * 80, flush=True)
    print("  LAUNCHING ALPHA DISCOVERY V3 EXPEDITION (70% DISCOVERY PARTITIONS)", flush=True)
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
        futures = {executor.submit(evaluate_v3_discovery_week, t): t for t in all_tasks}
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
    csv_out = RESULTS_DIR / "alpha_discovery_v3_scorecard.csv"
    df.to_csv(csv_out, index=False)
    print(f"\n[+] Saved Discovery V3 Scorecard to: {csv_out.resolve()}", flush=True)
    print("\n" + "=" * 80, flush=True)
    print(f"DISCOVERY V3 EXPEDITION COMPLETED IN {time.time() - t_start:.1f}s!", flush=True)
    print("=" * 80, flush=True)
    return df


if __name__ == "__main__":
    run_discovery_v3()
