#!/usr/bin/env python3
"""
================================================================================
  M5 FVG REFINEMENT LAB: BEATING SPREAD FRICTION
================================================================================
Investigates how to transform 1,000+ M5 FVG setups from PF 0.89 into positive edge:
  Test 1: FVG Size Threshold (1.5p vs 2.5p vs 3.5p vs 4.5p)
  Test 2: Stop Loss Buffer Sizing (Tight 5p vs Dynamic ATR 8-15p)
  Test 3: Macro Confluence (H1 50-EMA vs H4 20-EMA vs No filter)
  Test 4: Target R:R Spectrum (1.8R, 2.2R, 2.5R, 3.0R)
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

from tick_engine.core.engine import BacktestEngine, TRADE_DTYPE
from tick_engine.core.dataloader import get_week_dataset
from tick_engine.core.metrics import compute_metrics
from tick_engine.strategies.base import BaseStrategy, SIGNAL_LONG, SIGNAL_SHORT

PIP_SIZE = 0.0001


class RefinedM5FVGScalper(BaseStrategy):
    def __init__(
        self,
        min_fvg_pips: float = 2.5,
        target_rr: float = 2.2,
        min_sl: float = 7.0,
        max_sl: float = 16.0,
        macro_htf: str = "h1",   # "h1", "h4", "none"
        **kwargs
    ):
        super().__init__(**kwargs)
        self.min_fvg_pips = min_fvg_pips
        self.target_rr = target_rr
        self.min_sl = min_sl
        self.max_sl = max_sl
        self.macro_htf = macro_htf
        self.name = f"M5_FVG_{min_fvg_pips}p_RR{target_rr}_{macro_htf}"
        self.allowed_hours = {7, 8, 9, 10, 12, 13, 14, 15, 16}

    def generate_signals(self, ticks: np.ndarray):
        n_ticks = len(ticks)
        signals = np.zeros(n_ticks, dtype=np.int8)
        sl_pips_arr = np.zeros(n_ticks, dtype=np.float64)
        tp_pips_arr = np.zeros(n_ticks, dtype=np.float64)

        if n_ticks < 1000:
            return signals, sl_pips_arr, tp_pips_arr

        ts = pd.to_datetime(ticks['timestamp'], unit='ms')
        mid = 0.5 * (ticks['bid'] + ticks['ask'])
        base_df = pd.DataFrame({'open': mid, 'high': mid, 'low': mid, 'close': mid}, index=ts)

        m5 = base_df.resample('5min').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
        if len(m5) < 60:
            return signals, sl_pips_arr, tp_pips_arr

        # Macro Trend Calculation
        if self.macro_htf == "h1":
            h1 = base_df.resample('1h').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
            h1['ema50'] = h1['close'].ewm(span=50, adjust=False).mean()
            h1['trend'] = np.where(h1['close'] > h1['ema50'], 1, -1)
            m5['macro'] = h1['trend'].shift(1).reindex(m5.index, method='ffill').fillna(0)
        elif self.macro_htf == "h4":
            h4 = base_df.resample('4h').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
            h4['ema20'] = h4['close'].ewm(span=20, adjust=False).mean()
            h4['trend'] = np.where(h4['close'] > h4['ema20'], 1, -1)
            m5['macro'] = h4['trend'].shift(1).reindex(m5.index, method='ffill').fillna(0)
        else:
            m5['macro'] = 1

        tr = np.maximum(m5['high'] - m5['low'], np.maximum(abs(m5['high'] - m5['close'].shift(1)), abs(m5['low'] - m5['close'].shift(1))))
        m5['atr'] = tr.rolling(14).mean().bfill()

        c = m5['close'].values
        h = m5['high'].values
        l = m5['low'].values
        o = m5['open'].values
        macro = m5['macro'].values
        atr_pips = (m5['atr'] / PIP_SIZE).values
        hours = m5.index.hour.values
        weekdays = m5.index.dayofweek.values
        time_ms = m5.index.astype('datetime64[ms]').view('int64')
        dur_ms = 5 * 60 * 1000
        tick_ts = ticks['timestamp']
        n_bars = len(m5)

        for i in range(25, n_bars - 1):
            hr = hours[i]
            wd = weekdays[i]
            if hr not in self.allowed_hours:
                continue
            if wd == 4 and hr >= 16:
                continue

            sig = 0
            risk_p = 0.0
            m_dir = macro[i]

            dyn_min = max(self.min_sl, 0.9 * atr_pips[i])
            dyn_max = max(self.max_sl, 1.8 * atr_pips[i])

            # Bullish FVG
            gap_up = (l[i] - h[i-2]) / PIP_SIZE
            if gap_up >= self.min_fvg_pips and c[i-1] > o[i-1]:
                if m_dir >= 0 and c[i] > o[i]:
                    sl_p = h[i-2] - 1.2 * PIP_SIZE
                    r = max((c[i] - sl_p) / PIP_SIZE, dyn_min)
                    if r <= dyn_max:
                        sig = SIGNAL_LONG
                        risk_p = r

            # Bearish FVG
            gap_dn = (l[i-2] - h[i]) / PIP_SIZE
            if gap_dn >= self.min_fvg_pips and c[i-1] < o[i-1]:
                if m_dir <= 0 and c[i] < o[i]:
                    sl_p = l[i-2] + 1.2 * PIP_SIZE
                    r = max((sl_p - c[i]) / PIP_SIZE, dyn_min)
                    if r <= dyn_max:
                        sig = SIGNAL_SHORT
                        risk_p = r

            if sig != 0:
                exec_t = time_ms[i] + dur_ms
                idx = int(np.searchsorted(tick_ts, exec_t))
                if idx < n_ticks and signals[idx] == 0:
                    signals[idx] = sig
                    sl_pips_arr[idx] = risk_p
                    tp_pips_arr[idx] = risk_p * self.target_rr

        return signals, sl_pips_arr, tp_pips_arr


def evaluate_week(args):
    w, min_fvg, rr, macro = args
    ticks, _ = get_week_dataset(w)
    engine = BacktestEngine(pip_size=PIP_SIZE, use_slippage=True)
    strat = RefinedM5FVGScalper(min_fvg_pips=min_fvg, target_rr=rr, macro_htf=macro)
    trades = engine.run(ticks, strat)
    return trades


def run_test(min_fvg: float, rr: float, macro: str, weeks: List[int]):
    args = [(w, min_fvg, rr, macro) for w in weeks]
    num_workers = min(8, os.cpu_count() or 4)
    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        week_trades = list(ex.map(evaluate_week, args))

    all_t = [t for wt in week_trades if len(wt) > 0 for t in wt]
    if not all_t:
        return {'fvg': min_fvg, 'rr': rr, 'macro': macro, 'trades': 0, 'wr': 0, 'pf': 0, 'pips': 0, 'dd': 0}

    tr_arr = np.concatenate(week_trades) if week_trades else np.empty(0, dtype=TRADE_DTYPE)
    m = compute_metrics(tr_arr)
    return {
        'fvg': min_fvg,
        'rr': rr,
        'macro': macro,
        'trades': m['total_trades'],
        'trades_wk': m['total_trades'] / len(weeks),
        'wr': m['win_rate_pct'],
        'pf': m['profit_factor'],
        'pips': m['net_pips'],
        'dd': m['max_drawdown_pips']
    }


def main():
    print("=" * 115)
    print("   M5 FVG REFINEMENT LAB: FILTERING NOISE & BEATING SPREAD (52 WEEKS)   ".center(115))
    print("=" * 115)

    weeks = list(range(1, 53))
    tests = [
        # min_fvg, RR, macro
        (2.0, 2.0, "h1"),
        (2.5, 2.0, "h1"),
        (3.0, 2.0, "h1"),
        (3.5, 2.0, "h1"),
        (3.5, 2.2, "h1"),
        (3.5, 2.5, "h1"),
        (3.5, 2.2, "h4"),
        (4.0, 2.2, "h1"),
        (4.0, 2.5, "h1"),
    ]

    results = []
    for fvg_sz, rr, macro in tests:
        t0 = time.time()
        r = run_test(fvg_sz, rr, macro, weeks)
        results.append(r)
        dt = time.time() - t0
        print(f"  -> [FVG >={fvg_sz:3.1f}p | RR={rr:3.1f} | Macro={macro:<2}] Trades={r['trades']:<4} ({r['trades_wk']:<4.1f}/wk) | WR={r['wr']:<5.1f}% | PF={r['pf']:<5.2f} | Pips={r['pips']:<+8.1f} | DD={r['dd']:<6.1f} ({dt:.1f}s)")

    print("-" * 115)
    print(f"{'FVG Threshold':<15} | {'Target RR':<10} | {'Macro':<6} | {'Trades':<8} | {'Trades/Wk':<10} | {'Win Rate':<10} | {'Profit Factor':<15} | {'Net Pips':<12} | {'Max DD'}")
    print("-" * 115)
    for r in results:
        print(f"FVG >= {r['fvg']:<3.1f} p   | {r['rr']:<10.1f} | {r['macro']:<6} | {r['trades']:<8} | {r['trades_wk']:<10.1f} | {r['wr']:<9.1f}% | {r['pf']:<15.2f} | {r['pips']:<+11.1f}p | {r['dd']:<7.1f}p")
    print("-" * 115)


if __name__ == '__main__':
    main()
