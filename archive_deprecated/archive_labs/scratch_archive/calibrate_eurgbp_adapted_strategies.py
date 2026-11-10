#!/usr/bin/env python3
"""
================================================================================
  EURGBP STRATEGY ADAPTATION LAB: MODIFYING EURUSD MODELS FOR EURGBP MICROSTRUCTURE
================================================================================
Modifications Applied:
  1. Spread Filter Alignment: max_spread = 2.0 pips (accommodating EURGBP 1.0-1.8p liquid baseline).
  2. ATR & Pip Target Scaling:
     EURUSD daily ATR is ~75p; EURGBP daily ATR is ~45p (~0.60x scaling).
     Targets reduced proportionally:
       - DISC38 (MACD): SL 14p/TP 24p  --> SL 8.0p / TP 13.5p
       - DISC41 (Thursday): SL 15p/TP 25p --> SL 8.0p / TP 13.5p
       - DISC40 (RSI): Re-oriented from trend shift to Mean-Reverting Extreme Reversal (RSI < 35 / > 65)
         with SL 7.5p / TP 12.0p
       - S14_IB (Inside Bar): Target capped at 1.4x mother bar / 12p TP
       - S66_44MA (H1 44 MA): SL reduced from 20p to 10p, TP 16p
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
# 1. Adapted DISC38 (London Midday MACD Acceleration for EURGBP)
# ==============================================================================
class DISC38EURGBPAdapted(BaseStrategy):
    """
    DISC38 modified for EURGBP:
    - SL 8.0p, TP 13.5p (scaled to EURGBP ATR)
    - max_spread = 2.0 pips
    """
    def __init__(
        self,
        sl_pips: float = 8.0,
        tp_pips: float = 13.5,
        max_spread_pips: float = 2.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.max_spread_pips = float(max_spread_pips)
        self.name = f"DISC38_EURGBP_{self.sl_pips}SL_{self.tp_pips}TP"

    def generate_signals(self, ticks: np.ndarray):
        n = len(ticks)
        signals = np.zeros(n, dtype=np.int8)
        sl_arr = np.zeros(n, dtype=np.float64)
        tp_arr = np.zeros(n, dtype=np.float64)

        if n < 500:
            return signals, sl_arr, tp_arr

        m15 = build_time_bars(ticks, timeframe_min=15)
        if len(m15) < 35:
            return signals, sl_arr, tp_arr

        c = m15['close']
        ema12 = c.ewm(span=12, adjust=False).mean()
        ema26 = c.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        hist = macd - signal

        m15_idx = m15.index
        hours = m15_idx.hour
        dates = m15_idx.date
        bar_signals = np.zeros(len(m15), dtype=np.int8)

        last_trade_date = None

        # London Midday Acceleration: 09:00 to 14:00 UTC
        for i in range(26, len(m15)):
            hr = hours[i]
            if not (9 <= hr <= 14):
                continue

            dt = dates[i]
            if dt == last_trade_date:
                continue

            # Bullish MACD expansion
            if hist.iloc[i] > 0 and hist.iloc[i] > hist.iloc[i-1] and hist.iloc[i-1] > hist.iloc[i-2]:
                if macd.iloc[i] > 0 and c.iloc[i] > c.iloc[i-1]:
                    bar_signals[i] = SIGNAL_LONG
                    last_trade_date = dt
                    continue

            # Bearish MACD expansion
            if hist.iloc[i] < 0 and hist.iloc[i] < hist.iloc[i-1] and hist.iloc[i-1] < hist.iloc[i-2]:
                if macd.iloc[i] < 0 and c.iloc[i] < c.iloc[i-1]:
                    bar_signals[i] = SIGNAL_SHORT
                    last_trade_date = dt
                    continue

        sig_indices = np.where(bar_signals != 0)[0]
        if len(sig_indices) == 0:
            return signals, sl_arr, tp_arr

        tick_times = ticks['timestamp']
        spread = (ticks['ask'] - ticks['bid']) / PIP_SIZE

        for b_idx in sig_indices:
            sig = bar_signals[b_idx]
            bar_end_ms = int(m15_idx[b_idx].timestamp() * 1000) + 15 * 60 * 1000
            t_idx = int(np.searchsorted(tick_times, bar_end_ms, side='left'))
            if t_idx < n and signals[t_idx] == 0:
                if spread[t_idx] <= self.max_spread_pips:
                    signals[t_idx] = sig
                    sl_arr[t_idx] = self.sl_pips
                    tp_arr[t_idx] = self.tp_pips

        return signals, sl_arr, tp_arr


# ==============================================================================
# 2. Adapted DISC40 (RSI Mean-Reversion Exhaustion for EURGBP)
# ==============================================================================
class DISC40EURGBPAdapted(BaseStrategy):
    """
    DISC40 modified from trend breakout to Mean-Reverting Range Reversal:
    - On EURGBP, when RSI reaches extreme oversold (< 35) or overbought (> 65)
      and prints a reversal candle during London/NY hours (08:00 to 16:00 UTC).
    - SL 7.5p / TP 12.0p (1.60 RR).
    """
    def __init__(
        self,
        sl_pips: float = 7.5,
        tp_pips: float = 12.0,
        rsi_period: int = 14,
        max_spread: float = 2.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.rsi_period = int(rsi_period)
        self.max_spread = float(max_spread)
        self.name = f"DISC40_EURGBP_RSI_Rev_{self.sl_pips}SL_{self.tp_pips}TP"

    def generate_signals(self, ticks: np.ndarray):
        n = len(ticks)
        signals = np.zeros(n, dtype=np.int8)
        sl_arr = np.zeros(n, dtype=np.float64)
        tp_arr = np.zeros(n, dtype=np.float64)

        if n < 500:
            return signals, sl_arr, tp_arr

        m15 = build_time_bars(ticks, timeframe_min=15)
        if len(m15) < self.rsi_period + 15:
            return signals, sl_arr, tp_arr

        # Compute RSI
        c = m15['close']
        delta = c.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(self.rsi_period, min_periods=self.rsi_period).mean()
        avg_loss = loss.rolling(self.rsi_period, min_periods=self.rsi_period).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))

        idx = m15.index
        hours = idx.hour
        dates = idx.date
        bar_signals = np.zeros(len(m15), dtype=np.int8)

        last_trade_date = None

        for i in range(self.rsi_period + 1, len(m15)):
            hr = hours[i]
            if not (8 <= hr <= 16):
                continue

            dt = dates[i]
            if dt == last_trade_date:
                continue

            o_val = m15.iloc[i]['open']
            c_val = m15.iloc[i]['close']
            h_val = m15.iloc[i]['high']
            l_val = m15.iloc[i]['low']
            rng = h_val - l_val
            if rng < 0.0003:
                continue

            # Oversold Bounce: Previous RSI < 35, now closing green with lower wick >= 35%
            if rsi.iloc[i-1] < 35.0 and c_val > o_val:
                lower_wick = (min(o_val, c_val) - l_val) / rng
                if lower_wick >= 0.35:
                    bar_signals[i] = SIGNAL_LONG
                    last_trade_date = dt
                    continue

            # Overbought Fade: Previous RSI > 65, now closing red with upper wick >= 35%
            if rsi.iloc[i-1] > 65.0 and c_val < o_val:
                upper_wick = (h_val - max(o_val, c_val)) / rng
                if upper_wick >= 0.35:
                    bar_signals[i] = SIGNAL_SHORT
                    last_trade_date = dt
                    continue

        sig_indices = np.where(bar_signals != 0)[0]
        if len(sig_indices) == 0:
            return signals, sl_arr, tp_arr

        tick_times = ticks['timestamp']
        spread = (ticks['ask'] - ticks['bid']) / PIP_SIZE

        for b_idx in sig_indices:
            sig = bar_signals[b_idx]
            bar_end_ms = int(idx[b_idx].timestamp() * 1000) + 15 * 60 * 1000
            t_idx = int(np.searchsorted(tick_times, bar_end_ms, side='left'))
            if t_idx < n and signals[t_idx] == 0:
                if spread[t_idx] <= self.max_spread:
                    signals[t_idx] = sig
                    sl_arr[t_idx] = self.sl_pips
                    tp_arr[t_idx] = self.tp_pips

        return signals, sl_arr, tp_arr


# ==============================================================================
# 3. Adapted DISC41 (Thursday European Session Ignition for EURGBP)
# ==============================================================================
class DISC41EURGBPAdapted(BaseStrategy):
    """
    DISC41 adapted for EURGBP:
    - Thursday European macro window (08:00 to 14:00 UTC)
    - Calibrated SL 8.0p / TP 13.5p (1.69 RR)
    - max_spread = 2.0 pips
    """
    def __init__(
        self,
        sl_pips: float = 8.0,
        tp_pips: float = 13.5,
        max_spread: float = 2.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.max_spread = float(max_spread)
        self.name = f"DISC41_EURGBP_{self.sl_pips}SL_{self.tp_pips}TP"

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
        weekdays = idx.weekday
        hours = idx.hour
        dates = idx.date

        c = m15['close']
        ema20 = c.ewm(span=20, adjust=False).mean()
        ema50 = c.ewm(span=50, adjust=False).mean()

        bar_signals = np.zeros(len(m15), dtype=np.int8)
        last_trade_date = None

        for i in range(25, len(m15)):
            # Thursday strictly (weekday == 3)
            if weekdays[i] != 3:
                continue

            hr = hours[i]
            if not (8 <= hr <= 14):
                continue

            dt = dates[i]
            if dt == last_trade_date:
                continue

            o_val = m15.iloc[i]['open']
            c_val = m15.iloc[i]['close']
            h_val = m15.iloc[i]['high']
            l_val = m15.iloc[i]['low']
            rng = h_val - l_val
            if rng < 0.0003:
                continue

            # Bullish ignition: Fast EMA > Slow EMA, closes green above EMA20
            if ema20.iloc[i] > ema50.iloc[i] and c_val > ema20.iloc[i] and c_val > o_val:
                bar_signals[i] = SIGNAL_LONG
                last_trade_date = dt
                continue

            # Bearish ignition: Fast EMA < Slow EMA, closes red below EMA20
            if ema20.iloc[i] < ema50.iloc[i] and c_val < ema20.iloc[i] and c_val < o_val:
                bar_signals[i] = SIGNAL_SHORT
                last_trade_date = dt
                continue

        sig_indices = np.where(bar_signals != 0)[0]
        if len(sig_indices) == 0:
            return signals, sl_arr, tp_arr

        tick_times = ticks['timestamp']
        spread = (ticks['ask'] - ticks['bid']) / PIP_SIZE

        for b_idx in sig_indices:
            sig = bar_signals[b_idx]
            bar_end_ms = int(idx[b_idx].timestamp() * 1000) + 15 * 60 * 1000
            t_idx = int(np.searchsorted(tick_times, bar_end_ms, side='left'))
            if t_idx < n and signals[t_idx] == 0:
                if spread[t_idx] <= self.max_spread:
                    signals[t_idx] = sig
                    sl_arr[t_idx] = self.sl_pips
                    tp_arr[t_idx] = self.tp_pips

        return signals, sl_arr, tp_arr


# ==============================================================================
# 4. Adapted S14_IB (Inside Bar Mean-Reversion Target for EURGBP)
# ==============================================================================
class S14IBEURGBPAdapted(BaseStrategy):
    """
    S14 Inside Bar modified for EURGBP:
    - SL 8.0p, TP 12.0p (1.50 RR target) instead of open trend runner.
    - Max 1 trade per day.
    """
    def __init__(
        self,
        sl_pips: float = 8.0,
        tp_pips: float = 12.0,
        max_spread: float = 2.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.max_spread = float(max_spread)
        self.name = f"S14_IB_EURGBP_{self.sl_pips}SL_{self.tp_pips}TP"

    def generate_signals(self, ticks: np.ndarray):
        n = len(ticks)
        signals = np.zeros(n, dtype=np.int8)
        sl_arr = np.zeros(n, dtype=np.float64)
        tp_arr = np.zeros(n, dtype=np.float64)

        if n < 500:
            return signals, sl_arr, tp_arr

        m30 = build_time_bars(ticks, timeframe_min=30)
        if len(m30) < 20:
            return signals, sl_arr, tp_arr

        idx = m30.index
        hours = idx.hour
        dates = idx.date

        bar_signals = np.zeros(len(m30), dtype=np.int8)
        last_trade_date = None

        for i in range(1, len(m30)):
            hr = hours[i]
            if not (7 <= hr <= 16):
                continue

            dt = dates[i]
            if dt == last_trade_date:
                continue

            h_curr = m30.iloc[i]['high']
            l_curr = m30.iloc[i]['low']
            h_prev = m30.iloc[i-1]['high']
            l_prev = m30.iloc[i-1]['low']

            # Check if current bar is inside previous bar (Inside Bar)
            if h_curr < h_prev and l_curr > l_prev:
                c_curr = m30.iloc[i]['close']
                o_curr = m30.iloc[i]['open']
                if c_curr > o_curr:
                    bar_signals[i] = SIGNAL_LONG
                    last_trade_date = dt
                elif c_curr < o_curr:
                    bar_signals[i] = SIGNAL_SHORT
                    last_trade_date = dt

        sig_indices = np.where(bar_signals != 0)[0]
        if len(sig_indices) == 0:
            return signals, sl_arr, tp_arr

        tick_times = ticks['timestamp']
        spread = (ticks['ask'] - ticks['bid']) / PIP_SIZE

        for b_idx in sig_indices:
            sig = bar_signals[b_idx]
            bar_end_ms = int(idx[b_idx].timestamp() * 1000) + 30 * 60 * 1000
            t_idx = int(np.searchsorted(tick_times, bar_end_ms, side='left'))
            if t_idx < n and signals[t_idx] == 0:
                if spread[t_idx] <= self.max_spread:
                    signals[t_idx] = sig
                    sl_arr[t_idx] = self.sl_pips
                    tp_arr[t_idx] = self.tp_pips

        return signals, sl_arr, tp_arr


# Evaluation runner across all weeks
def evaluate_week(week_num: int) -> Dict[str, Any]:
    engine = BacktestEngine(pip_size=PIP_SIZE, use_slippage=True)
    ticks, meta = get_week_dataset(week_num, raw_dir=RAW_DIR)

    models = [
        ("DISC38_Adapted", DISC38EURGBPAdapted(sl_pips=8.0, tp_pips=13.5)),
        ("DISC40_Adapted", DISC40EURGBPAdapted(sl_pips=7.5, tp_pips=12.0)),
        ("DISC41_Adapted", DISC41EURGBPAdapted(sl_pips=8.0, tp_pips=13.5)),
        ("S14_IB_Adapted", S14IBEURGBPAdapted(sl_pips=8.0, tp_pips=12.0)),
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

    print("=" * 95)
    print(f"  EURGBP ADAPTATION LAB: EVALUATING MODIFIED STRATEGIES ({total} WEEKS)  ".center(95))
    print("=" * 95)

    all_trades = {
        "DISC38_Adapted": [],
        "DISC40_Adapted": [],
        "DISC41_Adapted": [],
        "S14_IB_Adapted": []
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

    print("\n" + "=" * 95)
    print("   ADAPTED EURGBP MODELS SCORECARD (FULL 157 WEEKS / 3 YEARS)   ".center(95))
    print("=" * 95)
    print(f"{'Adapted Strategy':<25} | {'Trades':>6} | {'Win Rate':>8} | {'Profit Factor':>13} | {'Net Pips':>9} | {'Max DD':>8} | {'Ret/DD':>7}")
    print("-" * 95)

    for name in all_trades:
        t_list = all_trades[name]
        t_list.sort(key=lambda x: x["entry_time"])
        s = compute_stats(t_list)
        print(f"{name:<25} | {s['trades']:>6} | {s['wr']:>7.1f}% | {s['pf']:>13.2f} | {s['pips']:>+8.1f}p | {s['max_dd']:>7.1f}p | {s['ret_dd']:>6.2f}x")

    print("=" * 95)
    print(f"Completed adaptation lab in {time.time()-t0:.1f}s.")


if __name__ == "__main__":
    main()
