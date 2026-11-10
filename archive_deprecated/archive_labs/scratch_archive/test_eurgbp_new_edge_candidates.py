#!/usr/bin/env python3
"""
================================================================================
  EURGBP SPECIALIZED ALPHA DISCOVERY: 3 HIGH-EDGE CANDIDATE ENGINES
================================================================================
Testing 3 novel architectures specifically designed around EURGBP's proven
structural price action edge (inspired by DISC42's 64.9% WR):

1. EURGBP_H1_M15_Anchor:
   H1 50-EMA Macro Trend + H1 12-bar Donchian Support/Resistance test on M15
   with a >= 50% rejection wick and 1-trade-per-day discipline.
   SL: 8.0p | TP: 13.0p (1.625 RR)

2. EURGBP_Daily_Pivot_Wall:
   Previous Day HLC Pivot Points (S1/R1). Price sweeps S1 (Bullish) or R1 (Bearish)
   aligned with H4 20-EMA trend, printing an M15 rejection candle.
   SL: 9.0p | TP: 14.5p (1.61 RR)

3. EURGBP_London_Lunch_Snapback:
   Between 10:30 and 13:30 UTC, price extended >= 12 pips from daily VWAP
   prints an M15 rejection candle fading back to VWAP median.
   SL: 7.5p | TP: 12.0p (1.60 RR)
================================================================================
"""

from __future__ import annotations
import sys
import os
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Any

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tick_engine.core.engine import BacktestEngine
from tick_engine.core.dataloader import get_week_dataset, load_manifest
from tick_engine.strategies.base import BaseStrategy, SIGNAL_LONG, SIGNAL_SHORT
from data_engineering_suite.bars import build_time_bars

PIP_SIZE = 0.0001
RAW_DIR = ROOT_DIR / "weekly_data" / "EURGBP"


# ==============================================================================
# Model 1: EURGBP H1/M15 Structural Anchor
# ==============================================================================
class EURGBPH1M15Anchor(BaseStrategy):
    """
    H1 50-EMA trend + H1 12-bar S/R level tested on M15.
    Requires >= 50% wick rejection + closes in direction of trend.
    Max 1 trade per day.
    """
    def __init__(
        self,
        sl_pips: float = 8.0,
        tp_pips: float = 13.0,
        lookback_h1: int = 12,
        min_wick: float = 0.50,
        max_dist_pips: float = 4.0,
        max_spread: float = 2.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.lookback_h1 = int(lookback_h1)
        self.min_wick = float(min_wick)
        self.max_dist_pips = float(max_dist_pips)
        self.max_spread = float(max_spread)
        self.name = f"H1_M15_Anchor_{self.sl_pips}SL_{self.tp_pips}TP"

    def generate_signals(self, ticks: np.ndarray):
        n = len(ticks)
        signals = np.zeros(n, dtype=np.int8)
        sl_arr = np.zeros(n, dtype=np.float64)
        tp_arr = np.zeros(n, dtype=np.float64)

        if n < 500:
            return signals, sl_arr, tp_arr

        m15 = build_time_bars(ticks, timeframe_min=15)
        if len(m15) < 30:
            return signals, sl_arr, tp_arr

        # Resample H1 bars
        ts = pd.to_datetime(ticks['timestamp'], unit='ms')
        mid = 0.5 * (ticks['bid'] + ticks['ask'])
        base = pd.DataFrame({'close': mid, 'high': mid, 'low': mid, 'open': mid}, index=ts)
        h1 = base.resample('1h').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
        if len(h1) < self.lookback_h1 + 5:
            return signals, sl_arr, tp_arr

        h1['ema50'] = h1['close'].ewm(span=50, adjust=False).mean()
        h1['macro'] = np.where(h1['close'] > h1['ema50'], 1, -1)
        h1['roll_h'] = h1['high'].rolling(self.lookback_h1).max().shift(1).bfill()
        h1['roll_l'] = h1['low'].rolling(self.lookback_h1).min().shift(1).bfill()

        h1_macro = h1['macro'].reindex(m15.index, method='ffill').fillna(0).values
        h1_supp = h1['roll_l'].reindex(m15.index, method='ffill').values
        h1_res = h1['roll_h'].reindex(m15.index, method='ffill').values

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

        last_trade_date = None

        for i in range(15, len(m15)):
            hr = hours[i]
            if not (7 <= hr <= 16):
                continue
            if spread[i] > self.max_spread:
                continue

            dt = dates[i]
            if dt == last_trade_date:
                continue

            m = h1_macro[i]
            bar_rng = (h[i] - l[i]) / PIP_SIZE
            if bar_rng < 3.0:
                continue

            sig = 0
            if m > 0:
                dist_supp = abs(l[i] - h1_supp[i]) / PIP_SIZE
                lower_wick = (min(o[i], c[i]) - l[i]) / PIP_SIZE
                if dist_supp <= self.max_dist_pips and (lower_wick / bar_rng) >= self.min_wick and c[i] > o[i]:
                    sig = SIGNAL_LONG

            elif m < 0:
                dist_res = abs(h[i] - h1_res[i]) / PIP_SIZE
                upper_wick = (h[i] - max(o[i], c[i])) / PIP_SIZE
                if dist_res <= self.max_dist_pips and (upper_wick / bar_rng) >= self.min_wick and c[i] < o[i]:
                    sig = SIGNAL_SHORT

            if sig != 0:
                exec_t = time_ms[i] + dur_ms
                idx = int(np.searchsorted(tick_ts, exec_t))
                if idx < n and signals[idx] == 0:
                    signals[idx] = sig
                    sl_arr[idx] = self.sl_pips
                    tp_arr[idx] = self.tp_pips
                    last_trade_date = dt

        return signals, sl_arr, tp_arr


# ==============================================================================
# Model 2: EURGBP Daily Pivot Wall Rejection
# ==============================================================================
class EURGBPDailyPivotWall(BaseStrategy):
    """
    Computes classic Daily Pivot Points: P, S1, R1 from prior trading day.
    H4 EMA trend determines direction.
    When price reaches S1 (in uptrend) or R1 (in downtrend) and prints M15 rejection wick >= 45%.
    """
    def __init__(
        self,
        sl_pips: float = 9.0,
        tp_pips: float = 14.5,
        min_wick: float = 0.45,
        max_dist_pips: float = 4.0,
        max_spread: float = 2.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.min_wick = float(min_wick)
        self.max_dist_pips = float(max_dist_pips)
        self.max_spread = float(max_spread)
        self.name = f"Daily_Pivot_Wall_{self.sl_pips}SL_{self.tp_pips}TP"

    def generate_signals(self, ticks: np.ndarray):
        n = len(ticks)
        signals = np.zeros(n, dtype=np.int8)
        sl_arr = np.zeros(n, dtype=np.float64)
        tp_arr = np.zeros(n, dtype=np.float64)

        if n < 500:
            return signals, sl_arr, tp_arr

        m15 = build_time_bars(ticks, timeframe_min=15)
        if len(m15) < 30:
            return signals, sl_arr, tp_arr

        ts = pd.to_datetime(ticks['timestamp'], unit='ms')
        mid = 0.5 * (ticks['bid'] + ticks['ask'])
        base = pd.DataFrame({'close': mid, 'high': mid, 'low': mid, 'open': mid}, index=ts)

        # Daily bars for Pivots
        daily = base.resample('1D').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
        if len(daily) < 2:
            return signals, sl_arr, tp_arr

        # Pivot formulas
        prev_h = daily['high'].shift(1)
        prev_l = daily['low'].shift(1)
        prev_c = daily['close'].shift(1)
        p = (prev_h + prev_l + prev_c) / 3.0
        r1 = (2 * p) - prev_l
        s1 = (2 * p) - prev_h

        daily['s1'] = s1
        daily['r1'] = r1
        daily['pivot'] = p

        # H4 for macro trend
        h4 = base.resample('4h').agg({'close': 'last'}).dropna()
        h4['ema20'] = h4['close'].ewm(span=20, adjust=False).mean()
        h4['macro'] = np.where(h4['close'] > h4['ema20'], 1, -1)

        m15_s1 = daily['s1'].reindex(m15.index, method='ffill').values
        m15_r1 = daily['r1'].reindex(m15.index, method='ffill').values
        m15_macro = h4['macro'].reindex(m15.index, method='ffill').fillna(0).values

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

        last_trade_date = None

        for i in range(15, len(m15)):
            hr = hours[i]
            if not (7 <= hr <= 16):
                continue
            if spread[i] > self.max_spread:
                continue

            dt = dates[i]
            if dt == last_trade_date:
                continue

            m = m15_macro[i]
            bar_rng = (h[i] - l[i]) / PIP_SIZE
            if bar_rng < 2.5:
                continue

            sig = 0
            # S1 Rejection in Uptrend
            if m > 0 and not np.isnan(m15_s1[i]):
                dist_s1 = abs(l[i] - m15_s1[i]) / PIP_SIZE
                lower_wick = (min(o[i], c[i]) - l[i]) / PIP_SIZE
                if dist_s1 <= self.max_dist_pips and (lower_wick / bar_rng) >= self.min_wick and c[i] > o[i]:
                    sig = SIGNAL_LONG

            # R1 Rejection in Downtrend
            elif m < 0 and not np.isnan(m15_r1[i]):
                dist_r1 = abs(h[i] - m15_r1[i]) / PIP_SIZE
                upper_wick = (h[i] - max(o[i], c[i])) / PIP_SIZE
                if dist_r1 <= self.max_dist_pips and (upper_wick / bar_rng) >= self.min_wick and c[i] < o[i]:
                    sig = SIGNAL_SHORT

            if sig != 0:
                exec_t = time_ms[i] + dur_ms
                idx = int(np.searchsorted(tick_ts, exec_t))
                if idx < n and signals[idx] == 0:
                    signals[idx] = sig
                    sl_arr[idx] = self.sl_pips
                    tp_arr[idx] = self.tp_pips
                    last_trade_date = dt

        return signals, sl_arr, tp_arr


# ==============================================================================
# Model 3: London Lunch VWAP Mean Reversion Snapback
# ==============================================================================
class EURGBPLondonLunchSnapback(BaseStrategy):
    """
    Between 10:30 and 13:30 UTC:
    When intraday price deviates >= 10 pips from the daily session VWAP and prints
    an exhaustion rejection wick (>= 48%), fades back toward the VWAP baseline.
    """
    def __init__(
        self,
        min_dev_pips: float = 10.0,
        sl_pips: float = 7.5,
        tp_pips: float = 12.0,
        min_wick: float = 0.48,
        max_spread: float = 2.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.min_dev_pips = float(min_dev_pips)
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.min_wick = float(min_wick)
        self.max_spread = float(max_spread)
        self.name = f"London_Lunch_Snapback_{self.sl_pips}SL_{self.tp_pips}TP"

    def generate_signals(self, ticks: np.ndarray):
        n = len(ticks)
        signals = np.zeros(n, dtype=np.int8)
        sl_arr = np.zeros(n, dtype=np.float64)
        tp_arr = np.zeros(n, dtype=np.float64)

        if n < 500:
            return signals, sl_arr, tp_arr

        m15 = build_time_bars(ticks, timeframe_min=15)
        if len(m15) < 30:
            return signals, sl_arr, tp_arr

        idx = m15.index
        days = np.unique(idx.date)

        c = m15['close'].values
        h = m15['high'].values
        l = m15['low'].values
        o = m15['open'].values
        v = m15['vol'].values if 'vol' in m15.columns else np.ones(len(m15))
        spread = m15['spread_pips'].values

        time_ms = idx.astype('datetime64[ms]').view('int64')
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        for d in days:
            day_mask = idx.date == d
            day_indices = np.where(day_mask)[0]
            if len(day_indices) == 0:
                continue

            # Compute running day VWAP
            day_c = c[day_indices]
            day_v = v[day_indices]
            pv = day_c * day_v
            cum_pv = np.cumsum(pv)
            cum_v = np.cumsum(day_v)
            day_vwap = np.where(cum_v > 0, cum_pv / cum_v, day_c)

            traded_today = False

            for k, i in enumerate(day_indices):
                if traded_today:
                    break

                hr = idx[i].hour
                mn = idx[i].minute
                # Window: 10:30 to 13:30 UTC
                in_window = (hr == 10 and mn >= 30) or (11 <= hr <= 12) or (hr == 13 and mn <= 30)
                if not in_window:
                    continue
                if spread[i] > self.max_spread:
                    continue

                vwap_val = day_vwap[k]
                bar_rng = (h[i] - l[i]) / PIP_SIZE
                if bar_rng < 2.5:
                    continue

                dev = (c[i] - vwap_val) / PIP_SIZE

                sig = 0
                # Overextended Above VWAP -> Fade Short
                if dev >= self.min_dev_pips and c[i] < o[i]:
                    upper_wick = (h[i] - max(o[i], c[i])) / PIP_SIZE
                    if (upper_wick / bar_rng) >= self.min_wick:
                        sig = SIGNAL_SHORT

                # Overextended Below VWAP -> Fade Long
                elif dev <= -self.min_dev_pips and c[i] > o[i]:
                    lower_wick = (min(o[i], c[i]) - l[i]) / PIP_SIZE
                    if (lower_wick / bar_rng) >= self.min_wick:
                        sig = SIGNAL_LONG

                if sig != 0:
                    exec_t = time_ms[i] + dur_ms
                    idx_tick = int(np.searchsorted(tick_ts, exec_t))
                    if idx_tick < n and signals[idx_tick] == 0:
                        signals[idx_tick] = sig
                        sl_arr[idx_tick] = self.sl_pips
                        tp_arr[idx_tick] = self.tp_pips
                        traded_today = True

        return signals, sl_arr, tp_arr


# Evaluation runner across all weeks
def evaluate_week(week_num: int) -> Dict[str, Any]:
    engine = BacktestEngine(pip_size=PIP_SIZE, use_slippage=True)
    ticks, meta = get_week_dataset(week_num, raw_dir=RAW_DIR)

    models = [
        ("H1_M15_Anchor", EURGBPH1M15Anchor(sl_pips=8.0, tp_pips=13.0)),
        ("Daily_Pivot_Wall", EURGBPDailyPivotWall(sl_pips=9.0, tp_pips=14.5)),
        ("London_Lunch_Snapback", EURGBPLondonLunchSnapback(sl_pips=7.5, tp_pips=12.0)),
    ]

    week_res = {}
    for name, m in models:
        raw_trades = engine.run(ticks, m)
        t_list = []
        for i in range(len(raw_trades)):
            row = raw_trades[i]
            t_list.append({
                "strategy": name,
                "week": week_num,
                "entry_time": int(row["entry_time"]),
                "exit_time": int(row["exit_time"]),
                "direction": "BUY" if row["direction"] == 1 else "SELL",
                "pips": float(row["pips"]),
            })
        week_res[name] = t_list

    return {"week": week_num, "trades": week_res}


def compute_stats(trades: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(trades)
    if n == 0:
        return {"trades": 0, "wr": 0.0, "pips": 0.0, "pf": 0.0, "max_dd": 0.0, "ret_dd": 0.0}
    pips = np.array([t["pips"] for t in trades])
    wins = np.sum(pips > 0)
    wr = wins / n * 100
    gp = float(np.sum(pips[pips > 0])) if wins > 0 else 0.0
    gl = float(abs(np.sum(pips[pips < 0]))) if (n - wins) > 0 else 0.0
    pf = gp / gl if gl > 0 else (99.0 if gp > 0 else 0.0)

    cum = np.cumsum(pips)
    dd = np.maximum.accumulate(cum) - cum
    max_dd = float(np.max(dd)) if len(dd) > 0 else 0.0
    net_pips = float(np.sum(pips))
    ret_dd = net_pips / max_dd if max_dd > 0 else 0.0

    return {
        "trades": n,
        "wr": round(wr, 1),
        "pips": round(net_pips, 1),
        "pf": round(pf, 2),
        "max_dd": round(max_dd, 1),
        "ret_dd": round(ret_dd, 2)
    }


def main():
    t0 = time.time()
    manifest = load_manifest(RAW_DIR)
    weeks = sorted(list(manifest.keys()))
    total = len(weeks)

    print("=" * 90)
    print(f"  EURGBP ALPHA DISCOVERY: 3 STRUCTURAL ENGINES ({total} WEEKS)  ".center(90))
    print("=" * 90)

    all_trades = {
        "H1_M15_Anchor": [],
        "Daily_Pivot_Wall": [],
        "London_Lunch_Snapback": []
    }

    workers = min(os.cpu_count() or 4, 10)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(evaluate_week, w): w for w in weeks}
        done = 0
        for f in futures:
            res = f.result()
            done += 1
            for name, t_list in res["trades"].items():
                all_trades[name].extend(t_list)
            if done % 25 == 0 or done == total:
                print(f"  -> Evaluated {done}/{total} weeks ({(done/total)*100:5.1f}%) in {time.time()-t0:.1f}s...")

    print("\n" + "=" * 90)
    print("   EURGBP DISCOVERY RESULTS (FULL 157 WEEKS / 3 YEARS)   ".center(90))
    print("=" * 90)
    print(f"{'Strategy Architecture':<30} | {'Trades':>6} | {'Win Rate':>8} | {'Profit Factor':>13} | {'Net Pips':>9} | {'Max DD':>8} | {'Ret/DD':>7}")
    print("-" * 90)

    combined_curated = []

    for name in all_trades:
        t_list = all_trades[name]
        t_list.sort(key=lambda x: x["entry_time"])
        s = compute_stats(t_list)
        print(f"{name:<30} | {s['trades']:>6} | {s['wr']:>7.1f}% | {s['pf']:>13.2f} | {s['pips']:>+8.1f}p | {s['max_dd']:>7.1f}p | {s['ret_dd']:>6.2f}x")
        if s["pips"] > 0:
            combined_curated.extend(t_list)

    # Save Daily_Pivot_Wall trades
    dpw_trades = all_trades["Daily_Pivot_Wall"]
    if dpw_trades:
        df_dpw = pd.DataFrame(dpw_trades)
        df_dpw["entry_time_utc"] = pd.to_datetime(df_dpw["entry_time"], unit="ms", utc=True).dt.strftime("%Y-%m-%d %H:%M:%S")
        df_dpw["exit_time_utc"] = pd.to_datetime(df_dpw["exit_time"], unit="ms", utc=True).dt.strftime("%Y-%m-%d %H:%M:%S")
        df_dpw.to_csv(ROOT_DIR / "results" / "eurgbp_daily_pivot_wall_trades.csv", index=False)
        print(f"[+] Saved {len(df_dpw)} Daily_Pivot_Wall trades to results/eurgbp_daily_pivot_wall_trades.csv.")

    if combined_curated:
        combined_curated.sort(key=lambda x: x["entry_time"])
        comb_s = compute_stats(combined_curated)
        print("-" * 90)
        print(f"{'COMBINED WINNING PORTFOLIO':<30} | {comb_s['trades']:>6} | {comb_s['wr']:>7.1f}% | {comb_s['pf']:>13.2f} | {comb_s['pips']:>+8.1f}p | {comb_s['max_dd']:>7.1f}p | {comb_s['ret_dd']:>6.2f}x")
    print("=" * 90)
    print(f"Completed research in {time.time()-t0:.1f}s.")


if __name__ == "__main__":
    main()
