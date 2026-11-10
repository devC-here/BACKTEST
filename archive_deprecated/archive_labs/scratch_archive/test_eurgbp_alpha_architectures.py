#!/usr/bin/env python3
"""
================================================================================
  EURGBP ALPHA DISCOVERY LAB: TESTING 4 TAILORED QUANTITATIVE ARCHITECTURES
================================================================================
Specifically exploits EURGBP's unique microstructure:
  - Bounded, mean-reverting cross dynamics (tight European economic coupling)
  - Asian session coiling into London session false breakout / liquidity hunt
  - Support & resistance wick absorption

Architectures Tested:
  1. EURGBP_Asian_Trap_Fade:
     Sweeps Asian range high/low between 07:00-11:00 UTC and rejects with wick.
  2. EURGBP_DISC42_Calibrated:
     H4 Macro Level + M30 Rejection with EURGBP-specific targets (11p SL / 18p TP).
  3. EURGBP_Bollinger_24SD_Exhaustion:
     M15 2.4 SD Band touch + rejection wick fading back to 20 SMA.
  4. EURGBP_DISC45_Calibrated:
     Continuous M5 structural rejection calibrated for EURGBP pip volatility.
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
# Architecture 1: Asian Session Liquidity Trap Fade
# ==============================================================================
class EURGBPAsianTrapFade(BaseStrategy):
    """
    Measures the Asian Range (00:00 to 06:30 UTC).
    During London Open (07:00 to 11:00 UTC), if price sweeps outside Asian High/Low
    by 2 to 16 pips and prints a rejection wick (>= 45%), enters a fade trade
    back toward the Asian median.
    """
    def __init__(
        self,
        min_sweep_pips: float = 2.0,
        max_sweep_pips: float = 16.0,
        sl_pips: float = 10.0,
        tp_pips: float = 16.0,
        min_wick: float = 0.45,
        max_spread: float = 2.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.min_sweep_pips = float(min_sweep_pips)
        self.max_sweep_pips = float(max_sweep_pips)
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.min_wick = float(min_wick)
        self.max_spread = float(max_spread)
        self.name = f"Asian_Trap_Fade_{self.sl_pips}SL_{self.tp_pips}TP"

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

        idx_dt = m15.index
        days = np.unique(idx_dt.date)

        bar_signals = np.zeros(len(m15), dtype=np.int8)

        for d in days:
            day_mask = idx_dt.date == d
            day_indices = np.where(day_mask)[0]
            if len(day_indices) == 0:
                continue

            # Asian Session: 00:00 <= hour < 06:30
            asian_indices = [i for i in day_indices if idx_dt[i].hour < 6 or (idx_dt[i].hour == 6 and idx_dt[i].minute < 30)]
            if len(asian_indices) < 8:
                continue

            asian_high = m15.iloc[asian_indices]['high'].max()
            asian_low = m15.iloc[asian_indices]['low'].min()
            asian_range = (asian_high - asian_low) / PIP_SIZE

            # Skip if Asian range is abnormally wild (> 35 pips)
            if asian_range < 5.0 or asian_range > 35.0:
                continue

            # Trade Window: 07:00 to 11:00 UTC
            london_indices = [i for i in day_indices if 7 <= idx_dt[i].hour <= 11]
            traded_today = False

            for i in london_indices:
                if traded_today:
                    break

                o = m15.iloc[i]['open']
                h = m15.iloc[i]['high']
                l = m15.iloc[i]['low']
                c = m15.iloc[i]['close']
                bar_range = h - l
                if bar_range < 0.0003:
                    continue

                # Bearish Trap (Fakeout above Asian High)
                sweep_high = (h - asian_high) / PIP_SIZE
                if self.min_sweep_pips <= sweep_high <= self.max_sweep_pips and c < asian_high:
                    upper_wick = (h - max(o, c)) / bar_range
                    if upper_wick >= self.min_wick:
                        bar_signals[i] = SIGNAL_SHORT
                        traded_today = True
                        continue

                # Bullish Trap (Fakeout below Asian Low)
                sweep_low = (asian_low - l) / PIP_SIZE
                if self.min_sweep_pips <= sweep_low <= self.max_sweep_pips and c > asian_low:
                    lower_wick = (min(o, c) - l) / bar_range
                    if lower_wick >= self.min_wick:
                        bar_signals[i] = SIGNAL_LONG
                        traded_today = True
                        continue

        # Map M15 signals back to tick array
        sig_indices = np.where(bar_signals != 0)[0]
        if len(sig_indices) == 0:
            return signals, sl_arr, tp_arr

        tick_times = ticks['timestamp']
        spread = (ticks['ask'] - ticks['bid']) / PIP_SIZE

        for bar_idx in sig_indices:
            sig = bar_signals[bar_idx]
            bar_start_ms = int(idx_dt[bar_idx].timestamp() * 1000)
            bar_end_ms = bar_start_ms + 15 * 60 * 1000

            tick_idx = int(np.searchsorted(tick_times, bar_end_ms, side='left'))
            if tick_idx < n:
                if spread[tick_idx] <= self.max_spread:
                    signals[tick_idx] = sig
                    sl_arr[tick_idx] = self.sl_pips
                    tp_arr[tick_idx] = self.tp_pips

        return signals, sl_arr, tp_arr


# ==============================================================================
# Architecture 2: Calibrated H4/M30 Macro Structure (Tailored DISC42)
# ==============================================================================
class EURGBPDISC42Calibrated(BaseStrategy):
    """
    DISC42 adapted for EURGBP:
    H4 Swing High/Low levels tested on M30 with wick rejection.
    Calibrated with tighter SL (11p) and TP (18p) reflecting EURGBP's smaller ATR.
    """
    def __init__(
        self,
        sl_pips: float = 11.0,
        tp_pips: float = 18.0,
        lookback_h4: int = 15,
        min_wick: float = 0.40,
        max_dist_pips: float = 3.5,
        max_spread: float = 2.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.lookback_h4 = int(lookback_h4)
        self.min_wick = float(min_wick)
        self.max_dist_pips = float(max_dist_pips)
        self.max_spread = float(max_spread)
        self.name = f"EURGBP_DISC42_Calibrated_{self.sl_pips}SL_{self.tp_pips}TP"

    def generate_signals(self, ticks: np.ndarray):
        n = len(ticks)
        signals = np.zeros(n, dtype=np.int8)
        sl_arr = np.zeros(n, dtype=np.float64)
        tp_arr = np.zeros(n, dtype=np.float64)

        if n < 1000:
            return signals, sl_arr, tp_arr

        m30 = build_time_bars(ticks, timeframe_min=30)
        h4 = build_time_bars(ticks, timeframe_min=240)

        if len(m30) < 20 or len(h4) < self.lookback_h4:
            return signals, sl_arr, tp_arr

        m30_idx = m30.index
        h4_idx = h4.index
        bar_signals = np.zeros(len(m30), dtype=np.int8)

        # Precompute H4 support & resistance rolling levels
        h4_highs = h4['high'].rolling(self.lookback_h4, min_periods=5).max()
        h4_lows = h4['low'].rolling(self.lookback_h4, min_periods=5).min()

        last_sig_time = -999999

        for i in range(1, len(m30)):
            t_curr = m30_idx[i]
            if t_curr.hour < 7 or t_curr.hour > 17:
                continue

            h4_pos = h4_idx.searchsorted(t_curr, side='right') - 1
            if h4_pos < 5:
                continue

            res_level = h4_highs.iloc[h4_pos]
            sup_level = h4_lows.iloc[h4_pos]

            o = m30.iloc[i]['open']
            h = m30.iloc[i]['high']
            l = m30.iloc[i]['low']
            c = m30.iloc[i]['close']
            bar_range = h - l
            if bar_range < 0.0004:
                continue

            # Bullish Level Rejection (Testing Support)
            dist_to_sup = abs(l - sup_level) / PIP_SIZE
            if dist_to_sup <= self.max_dist_pips and c > o:
                lower_wick = (min(o, c) - l) / bar_range
                if lower_wick >= self.min_wick:
                    curr_ms = int(t_curr.timestamp() * 1000)
                    if curr_ms - last_sig_time > 4 * 3600 * 1000:
                        bar_signals[i] = SIGNAL_LONG
                        last_sig_time = curr_ms
                        continue

            # Bearish Level Rejection (Testing Resistance)
            dist_to_res = abs(h - res_level) / PIP_SIZE
            if dist_to_res <= self.max_dist_pips and c < o:
                upper_wick = (h - max(o, c)) / bar_range
                if upper_wick >= self.min_wick:
                    curr_ms = int(t_curr.timestamp() * 1000)
                    if curr_ms - last_sig_time > 4 * 3600 * 1000:
                        bar_signals[i] = SIGNAL_SHORT
                        last_sig_time = curr_ms
                        continue

        sig_indices = np.where(bar_signals != 0)[0]
        if len(sig_indices) == 0:
            return signals, sl_arr, tp_arr

        tick_times = ticks['timestamp']
        spread = (ticks['ask'] - ticks['bid']) / PIP_SIZE

        for bar_idx in sig_indices:
            sig = bar_signals[bar_idx]
            bar_end_ms = int(m30_idx[bar_idx].timestamp() * 1000) + 30 * 60 * 1000
            tick_idx = int(np.searchsorted(tick_times, bar_end_ms, side='left'))
            if tick_idx < n:
                if spread[tick_idx] <= self.max_spread:
                    signals[tick_idx] = sig
                    sl_arr[tick_idx] = self.sl_pips
                    tp_arr[tick_idx] = self.tp_pips

        return signals, sl_arr, tp_arr


# ==============================================================================
# Architecture 3: London Midday Bollinger 2.4 SD Mean-Reversion Exhaustion
# ==============================================================================
class EURGBPBollingerExhaustion(BaseStrategy):
    """
    During European hours (08:00 to 16:00 UTC):
    When M15 price pierces the 2.4 SD Bollinger Band and produces an exhaustion wick (>= 40%),
    fades back toward the 20-SMA baseline.
    """
    def __init__(
        self,
        bb_period: int = 20,
        bb_std: float = 2.4,
        sl_pips: float = 10.0,
        tp_pips: float = 15.0,
        min_wick: float = 0.40,
        max_spread: float = 2.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.bb_period = int(bb_period)
        self.bb_std = float(bb_std)
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.min_wick = float(min_wick)
        self.max_spread = float(max_spread)
        self.name = f"Bollinger_Exhaustion_2.4SD_{self.sl_pips}SL_{self.tp_pips}TP"

    def generate_signals(self, ticks: np.ndarray):
        n = len(ticks)
        signals = np.zeros(n, dtype=np.int8)
        sl_arr = np.zeros(n, dtype=np.float64)
        tp_arr = np.zeros(n, dtype=np.float64)

        if n < 500:
            return signals, sl_arr, tp_arr

        m15 = build_time_bars(ticks, timeframe_min=15)
        if len(m15) < self.bb_period + 10:
            return signals, sl_arr, tp_arr

        closes = m15['close']
        sma = closes.rolling(self.bb_period).mean()
        rstd = closes.rolling(self.bb_period).std()
        upper_bb = sma + self.bb_std * rstd
        lower_bb = sma - self.bb_std * rstd

        idx = m15.index
        bar_signals = np.zeros(len(m15), dtype=np.int8)
        last_sig_time = -999999

        for i in range(self.bb_period, len(m15)):
            t_curr = idx[i]
            if t_curr.hour < 8 or t_curr.hour > 16:
                continue

            o = m15.iloc[i]['open']
            h = m15.iloc[i]['high']
            l = m15.iloc[i]['low']
            c = m15.iloc[i]['close']
            bar_range = h - l
            if bar_range < 0.0003:
                continue

            # Upper Band Piercing & Rejection (Overbought Exhaustion)
            if h >= upper_bb.iloc[i] and c < upper_bb.iloc[i]:
                upper_wick = (h - max(o, c)) / bar_range
                if upper_wick >= self.min_wick:
                    curr_ms = int(t_curr.timestamp() * 1000)
                    if curr_ms - last_sig_time > 2 * 3600 * 1000:
                        bar_signals[i] = SIGNAL_SHORT
                        last_sig_time = curr_ms
                        continue

            # Lower Band Piercing & Rejection (Oversold Exhaustion)
            if l <= lower_bb.iloc[i] and c > lower_bb.iloc[i]:
                lower_wick = (min(o, c) - l) / bar_range
                if lower_wick >= self.min_wick:
                    curr_ms = int(t_curr.timestamp() * 1000)
                    if curr_ms - last_sig_time > 2 * 3600 * 1000:
                        bar_signals[i] = SIGNAL_LONG
                        last_sig_time = curr_ms
                        continue

        sig_indices = np.where(bar_signals != 0)[0]
        if len(sig_indices) == 0:
            return signals, sl_arr, tp_arr

        tick_times = ticks['timestamp']
        spread = (ticks['ask'] - ticks['bid']) / PIP_SIZE

        for bar_idx in sig_indices:
            sig = bar_signals[bar_idx]
            bar_end_ms = int(idx[bar_idx].timestamp() * 1000) + 15 * 60 * 1000
            tick_idx = int(np.searchsorted(tick_times, bar_end_ms, side='left'))
            if tick_idx < n:
                if spread[tick_idx] <= self.max_spread:
                    signals[tick_idx] = sig
                    sl_arr[tick_idx] = self.sl_pips
                    tp_arr[tick_idx] = self.tp_pips

        return signals, sl_arr, tp_arr


# ==============================================================================
# Architecture 4: Calibrated M5 Structural Rejection (Tailored DISC45)
# ==============================================================================
class EURGBPDISC45Calibrated(BaseStrategy):
    """
    DISC45 calibrated with EURGBP parameters:
    SL 5.5p / TP 8.0p, Wick 48%, Cooldown 6 bars (30 min).
    """
    def __init__(
        self,
        sl_pips: float = 5.5,
        tp_pips: float = 8.0,
        min_wick: float = 0.48,
        max_dist_pips: float = 4.0,
        cooldown_bars: int = 6,
        max_spread: float = 2.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.min_wick = float(min_wick)
        self.max_dist_pips = float(max_dist_pips)
        self.cooldown_bars = int(cooldown_bars)
        self.max_spread = float(max_spread)
        self.name = f"EURGBP_DISC45_Calibrated_{self.sl_pips}SL_{self.tp_pips}TP"

    def generate_signals(self, ticks: np.ndarray):
        n = len(ticks)
        signals = np.zeros(n, dtype=np.int8)
        sl_arr = np.zeros(n, dtype=np.float64)
        tp_arr = np.zeros(n, dtype=np.float64)

        if n < 500:
            return signals, sl_arr, tp_arr

        m5 = build_time_bars(ticks, timeframe_min=5)
        h1 = build_time_bars(ticks, timeframe_min=60)
        if len(m5) < 30 or len(h1) < 10:
            return signals, sl_arr, tp_arr

        m5_idx = m5.index
        h1_idx = h1.index
        bar_signals = np.zeros(len(m5), dtype=np.int8)

        # Rolling 10-bar H1 high and low
        h1_highs = h1['high'].rolling(10, min_periods=3).max()
        h1_lows = h1['low'].rolling(10, min_periods=3).min()

        last_trade_bar = -999

        for i in range(1, len(m5)):
            if i - last_trade_bar < self.cooldown_bars:
                continue

            t_curr = m5_idx[i]
            if t_curr.hour < 7 or t_curr.hour > 16:
                continue

            h1_pos = h1_idx.searchsorted(t_curr, side='right') - 1
            if h1_pos < 3:
                continue

            res_level = h1_highs.iloc[h1_pos]
            sup_level = h1_lows.iloc[h1_pos]

            o = m5.iloc[i]['open']
            h = m5.iloc[i]['high']
            l = m5.iloc[i]['low']
            c = m5.iloc[i]['close']
            bar_range = h - l
            if bar_range < 0.00025:
                continue

            # Bullish Level Rejection
            dist_sup = abs(l - sup_level) / PIP_SIZE
            if dist_sup <= self.max_dist_pips and c > o:
                lower_wick = (min(o, c) - l) / bar_range
                if lower_wick >= self.min_wick:
                    bar_signals[i] = SIGNAL_LONG
                    last_trade_bar = i
                    continue

            # Bearish Level Rejection
            dist_res = abs(h - res_level) / PIP_SIZE
            if dist_res <= self.max_dist_pips and c < o:
                upper_wick = (h - max(o, c)) / bar_range
                if upper_wick >= self.min_wick:
                    bar_signals[i] = SIGNAL_SHORT
                    last_trade_bar = i
                    continue

        sig_indices = np.where(bar_signals != 0)[0]
        if len(sig_indices) == 0:
            return signals, sl_arr, tp_arr

        tick_times = ticks['timestamp']
        spread = (ticks['ask'] - ticks['bid']) / PIP_SIZE

        for bar_idx in sig_indices:
            sig = bar_signals[bar_idx]
            bar_end_ms = int(m5_idx[bar_idx].timestamp() * 1000) + 5 * 60 * 1000
            tick_idx = int(np.searchsorted(tick_times, bar_end_ms, side='left'))
            if tick_idx < n:
                if spread[tick_idx] <= self.max_spread:
                    signals[tick_idx] = sig
                    sl_arr[tick_idx] = self.sl_pips
                    tp_arr[tick_idx] = self.tp_pips

        return signals, sl_arr, tp_arr


# Evaluation runner across all weeks
def evaluate_week(week_num: int) -> Dict[str, Any]:
    engine = BacktestEngine(pip_size=PIP_SIZE, use_slippage=True)
    ticks, meta = get_week_dataset(week_num, raw_dir=RAW_DIR)

    models = [
        ("Asian_Trap_Fade", EURGBPAsianTrapFade(sl_pips=10.0, tp_pips=16.0)),
        ("DISC42_Calibrated", EURGBPDISC42Calibrated(sl_pips=11.0, tp_pips=18.0)),
        ("Bollinger_Exhaustion", EURGBPBollingerExhaustion(sl_pips=10.0, tp_pips=15.0)),
        ("DISC45_Calibrated", EURGBPDISC45Calibrated(sl_pips=5.5, tp_pips=8.0)),
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
    print(f"  EURGBP ALPHA DISCOVERY LAB: TESTING 4 BESPOKE ARCHITECTURES ({total} WEEKS)  ".center(90))
    print("=" * 90)

    all_trades = {
        "Asian_Trap_Fade": [],
        "DISC42_Calibrated": [],
        "Bollinger_Exhaustion": [],
        "DISC45_Calibrated": []
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
    print("   EURGBP CUSTOM ARCHITECTURES SCORECARD (FULL 157 WEEKS / 3 YEARS)   ".center(90))
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

    combined_curated.sort(key=lambda x: x["entry_time"])
    comb_s = compute_stats(combined_curated)
    print("-" * 90)
    print(f"{'COMBINED WINNING PORTFOLIO':<30} | {comb_s['trades']:>6} | {comb_s['wr']:>7.1f}% | {comb_s['pf']:>13.2f} | {comb_s['pips']:>+8.1f}p | {comb_s['max_dd']:>7.1f}p | {comb_s['ret_dd']:>6.2f}x")
    print("=" * 90)
    print(f"Completed research in {time.time()-t0:.1f}s.")


if __name__ == "__main__":
    main()
