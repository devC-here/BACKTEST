#!/usr/bin/env python3
"""
================================================================================
  M5 PURE HIGH-FREQUENCY SCALPER RESEARCH LAB (52 WEEKS TICK AUDIT)
================================================================================
Deeply analyzes the 3 microstructure setups from the 2nd analysis:
  Setup A: M5 Fair Value Gap (FVG) Mitigation
  Setup B: M5 Displacement & 50% Equilibrium Retest
  Setup C: M5 NR4 Compression Box Breakout
  Setup D: Hybrid M5 Multi-Setup Scalper

Simulates tick-by-tick with real Dukascopy floating spreads and slippage.
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


class M5MicroScalperEngine(BaseStrategy):
    def __init__(
        self,
        mode: str = "hybrid",          # "fvg", "displacement", "nr4", "hybrid"
        target_rr: float = 1.8,
        min_sl: float = 5.0,
        max_sl: float = 12.0,
        use_macro_filter: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.mode = mode
        self.target_rr = target_rr
        self.min_sl = min_sl
        self.max_sl = max_sl
        self.use_macro_filter = use_macro_filter
        self.name = f"M5_Scalper_{mode.upper()}_RR{target_rr}"
        # Peak institutional liquidity hours only (London 07-10, NY 12-16 UTC)
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

        # M5 Resampling
        m5 = base_df.resample('5min').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
        if len(m5) < 60:
            return signals, sl_pips_arr, tp_pips_arr

        # Strict Zero-Lookahead H1 Macro Trend (shift 1)
        h1 = base_df.resample('1h').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
        h1['ema50'] = h1['close'].ewm(span=50, adjust=False).mean()
        h1['trend'] = np.where(h1['close'] > h1['ema50'], 1, -1)
        m5['macro'] = h1['trend'].shift(1).reindex(m5.index, method='ffill').fillna(0)

        # ATR for dynamic sizing
        tr = np.maximum(m5['high'] - m5['low'], np.maximum(abs(m5['high'] - m5['close'].shift(1)), abs(m5['low'] - m5['close'].shift(1))))
        m5['atr'] = tr.rolling(14).mean().bfill()
        m5['ranges'] = (m5['high'] - m5['low']) / PIP_SIZE

        c = m5['close'].values
        h = m5['high'].values
        l = m5['low'].values
        o = m5['open'].values
        macro = m5['macro'].values
        ranges = m5['ranges'].values
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
            if wd == 4 and hr >= 16:  # Skip Friday late hours
                continue

            sig = 0
            risk_p = 0.0
            m_dir = macro[i] if self.use_macro_filter else 1

            dyn_min = max(self.min_sl, 0.7 * atr_pips[i])
            dyn_max = max(self.max_sl, 1.6 * atr_pips[i])

            # -------------------------------------------------------------
            # SETUP 1: M5 FAIR VALUE GAP (FVG) MITIGATION
            # -------------------------------------------------------------
            if self.mode in ("fvg", "hybrid") and sig == 0:
                # Bullish FVG formed at i-2, i-1, i: High[i-2] < Low[i] with gap >= 1.5p
                if (l[i] - h[i-2]) / PIP_SIZE >= 1.5 and c[i-1] > o[i-1]:
                    fvg_top = l[i]
                    fvg_bot = h[i-2]
                    # Check if bar i closed strong and is mitigating
                    if m_dir >= 0:
                        sl_p = fvg_bot - 0.8 * PIP_SIZE
                        r = max((c[i] - sl_p) / PIP_SIZE, dyn_min)
                        if r <= dyn_max:
                            sig = SIGNAL_LONG
                            risk_p = r

                # Bearish FVG: Low[i-2] > High[i] with gap >= 1.5p
                elif (l[i-2] - h[i]) / PIP_SIZE >= 1.5 and c[i-1] < o[i-1]:
                    fvg_top = l[i-2]
                    fvg_bot = h[i]
                    if m_dir <= 0:
                        sl_p = fvg_top + 0.8 * PIP_SIZE
                        r = max((sl_p - c[i]) / PIP_SIZE, dyn_min)
                        if r <= dyn_max:
                            sig = SIGNAL_SHORT
                            risk_p = r

            # -------------------------------------------------------------
            # SETUP 2: M5 DISPLACEMENT & 50% MIDPOINT RETEST
            # -------------------------------------------------------------
            if self.mode in ("displacement", "hybrid") and sig == 0:
                # Check prior bar for strong displacement: Range >= 6.5p, Body >= 65%
                prev_rng = ranges[i-1]
                prev_body = abs(c[i-1] - o[i-1]) / PIP_SIZE
                if prev_rng >= 6.5 and (prev_body / max(prev_rng, 0.1)) >= 0.65:
                    midpoint = 0.5 * (o[i-1] + c[i-1])

                    # Bullish impulse, retested 50% and rejected upwards
                    if c[i-1] > o[i-1] and m_dir >= 0:
                        if l[i] <= midpoint and c[i] > midpoint and c[i] > o[i]:
                            sl_p = l[i-1] - 0.8 * PIP_SIZE
                            r = max((c[i] - sl_p) / PIP_SIZE, dyn_min)
                            if r <= dyn_max:
                                sig = SIGNAL_LONG
                                risk_p = r

                    # Bearish impulse, retested 50% and rejected downwards
                    elif c[i-1] < o[i-1] and m_dir <= 0:
                        if h[i] >= midpoint and c[i] < midpoint and c[i] < o[i]:
                            sl_p = h[i-1] + 0.8 * PIP_SIZE
                            r = max((sl_p - c[i]) / PIP_SIZE, dyn_min)
                            if r <= dyn_max:
                                sig = SIGNAL_SHORT
                                risk_p = r

            # -------------------------------------------------------------
            # SETUP 3: M5 NR4 COMPRESSION BOX BREAKOUT
            # -------------------------------------------------------------
            if self.mode in ("nr4", "hybrid") and sig == 0:
                is_nr4 = (ranges[i-1] <= 3.2) and (ranges[i-2] <= 3.2) and (ranges[i-3] <= 3.2) and (ranges[i-4] <= 3.2)
                if is_nr4:
                    box_h = np.max(h[i-4:i])
                    box_l = np.min(l[i-4:i])

                    if c[i] > box_h and c[i] > o[i] and m_dir >= 0:
                        sl_p = box_l - 0.8 * PIP_SIZE
                        r = max((c[i] - sl_p) / PIP_SIZE, dyn_min)
                        if r <= dyn_max:
                            sig = SIGNAL_LONG
                            risk_p = r

                    elif c[i] < box_l and c[i] < o[i] and m_dir <= 0:
                        sl_p = box_h + 0.8 * PIP_SIZE
                        r = max((sl_p - c[i]) / PIP_SIZE, dyn_min)
                        if r <= dyn_max:
                            sig = SIGNAL_SHORT
                            risk_p = r

            # Signal Registration
            if sig != 0:
                exec_t = time_ms[i] + dur_ms
                idx = int(np.searchsorted(tick_ts, exec_t))
                if idx < n_ticks and signals[idx] == 0:
                    signals[idx] = sig
                    sl_pips_arr[idx] = risk_p
                    tp_pips_arr[idx] = risk_p * self.target_rr

        return signals, sl_pips_arr, tp_pips_arr


def evaluate_week(args):
    w, mode, rr = args
    ticks, _ = get_week_dataset(w)
    engine = BacktestEngine(pip_size=PIP_SIZE, use_slippage=True)
    strat = M5MicroScalperEngine(mode=mode, target_rr=rr)
    trades = engine.run(ticks, strat)
    return trades


def run_experiment(mode: str, rr: float, weeks: List[int]):
    args = [(w, mode, rr) for w in weeks]
    num_workers = min(8, os.cpu_count() or 4)
    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        week_trades = list(ex.map(evaluate_week, args))

    all_t = [t for wt in week_trades if len(wt) > 0 for t in wt]
    if not all_t:
        return {'mode': mode, 'rr': rr, 'trades': 0, 'win_rate': 0, 'pf': 0, 'pips': 0, 'dd': 0}

    tr_arr = np.concatenate(week_trades) if week_trades else np.empty(0, dtype=TRADE_DTYPE)
    m = compute_metrics(tr_arr)
    return {
        'mode': mode,
        'rr': rr,
        'trades': m['total_trades'],
        'trades_per_wk': m['total_trades'] / len(weeks),
        'win_rate': m['win_rate_pct'],
        'pf': m['profit_factor'],
        'pips': m['net_pips'],
        'dd': m['max_drawdown_pips'],
        'ret_dd': m['net_pips'] / max(m['max_drawdown_pips'], 1e-6)
    }


def main():
    print("=" * 115)
    print("   RESEARCH LAB: M5 PURE MICRO-SCALPER 52-WEEK BENCHMARK (REAL TICK ENGINE)   ".center(115))
    print("=" * 115)

    weeks = list(range(1, 53))  # 52 weeks (1 year)
    t0 = time.time()

    configurations = [
        # Mode, Target R:R
        ("fvg", 1.8),
        ("fvg", 2.0),
        ("displacement", 1.8),
        ("displacement", 2.0),
        ("nr4", 1.8),
        ("nr4", 2.0),
        ("hybrid", 1.8),
        ("hybrid", 2.0),
        ("hybrid", 2.2),
    ]

    results = []
    print(f"Testing {len(configurations)} configuration sets across 52 weeks (~26M ticks)...")
    for mode, rr in configurations:
        t_sub = time.time()
        res = run_experiment(mode, rr, weeks)
        results.append(res)
        dt_sub = time.time() - t_sub
        print(f"  -> [{mode.upper():<12} RR={rr:<4}] Trades={res['trades']:<5} ({res['trades_per_wk']:<4.1f}/wk) | WR={res['win_rate']:<5.1f}% | PF={res['pf']:<5.2f} | Pips={res['pips']:<+8.1f} | DD={res['dd']:<6.1f} ({dt_sub:.1f}s)")

    dt = time.time() - t0
    print(f"\n[+] Research lab completed in {dt:.1f}s!\n")

    print("-" * 115)
    print(f"{'M5 Scalper Configuration':<28} | {'52W Trades':<11} | {'Trades/Wk':<10} | {'Win Rate':<10} | {'Profit Factor':<15} | {'Net Pips':<12} | {'Max DD'}")
    print("-" * 115)
    for r in results:
        label = f"{r['mode'].upper()} (RR {r['rr']})"
        print(f"{label:<28} | {r['trades']:<11} | {r['trades_per_wk']:<10.1f} | {r['win_rate']:<9.1f}% | {r['pf']:<15.2f} | {r['pips']:<+11.1f}p | {r['dd']:<7.1f}p")
    print("-" * 115)


if __name__ == "__main__":
    main()
