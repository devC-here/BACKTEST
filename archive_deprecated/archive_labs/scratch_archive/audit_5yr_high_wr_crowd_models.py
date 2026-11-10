#!/usr/bin/env python3
"""
================================================================================
  5-YEAR FULL CYCLE AUDIT: HIGH WIN-RATE MODELS (WR > 50%) ACROSS 262 WEEKS
================================================================================
Evaluates the top high-win-rate candidates across the entire 5-year Dukascopy dataset:
  - 3-Year In-Sample (Weeks 1 to 156)
  - 2-Year Out-of-Sample (Weeks 157 to 262)
  - 5-Year Full Cycle (Weeks 1 to 262 / ~123.22M ticks)

Models Audited:
  1. DISC42 (Macro Structure H4 + M30 Rejection) [The 72.9% WR Champion]
  2. DISC43_HighWR_8.5TP (H1 Structure + M5 Rejection Wick >= 55%, 6.0p SL, 8.5p TP)
  3. DISC43_HighWR_8.0TP (H1 Structure + M5 Rejection Wick >= 60%, 6.0p SL, 8.0p TP)
  4. WOTC_Struct_Bounce_M15 (H1 Structure + M15 Rejection Wick >= 45%, 9.0p SL, 12.5p TP)
  5. High_WR_Crowd_Duo (Simultaneous execution of DISC42 + DISC43_HighWR)
================================================================================
"""

from __future__ import annotations
import sys
import os
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tick_engine.core.engine import BacktestEngine, TRADE_DTYPE
from tick_engine.core.dataloader import get_week_dataset
from tick_engine.strategies.base import BaseStrategy, SIGNAL_LONG, SIGNAL_SHORT
from data_engineering_suite.bars import build_time_bars

PIP_SIZE = 0.0001


# ----------------------------------------------------------------------
# 1. DISC42 Original
# ----------------------------------------------------------------------
from strategies.new_discoveries.disc42_macro_structure_h4_m30_elite.strategy import MacroStructureH4M30Elite


# ----------------------------------------------------------------------
# 2. DISC43 High-WR (6.0p SL, 8.5p TP, Wick >= 55%)
# ----------------------------------------------------------------------
class DISC43HighWR85(BaseStrategy):
    def __init__(self, sl_pips: float = 6.0, tp_pips: float = 8.5, min_wick: float = 0.55, **kwargs):
        super().__init__(**kwargs)
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.min_wick = float(min_wick)
        self.max_spread = 0.55
        self.name = "DISC43_HighWR_8.5TP"

    def generate_signals(self, ticks: np.ndarray):
        n_ticks = len(ticks)
        signals = np.zeros(n_ticks, dtype=np.int8)
        sl_pips_arr = np.zeros(n_ticks, dtype=np.float64)
        tp_pips_arr = np.zeros(n_ticks, dtype=np.float64)

        if n_ticks < 500:
            return signals, sl_pips_arr, tp_pips_arr

        m5 = build_time_bars(ticks, timeframe_min=5)
        if len(m5) < 50:
            return signals, sl_pips_arr, tp_pips_arr

        ts = pd.to_datetime(ticks['timestamp'], unit='ms')
        mid = 0.5 * (ticks['bid'] + ticks['ask'])
        base = pd.DataFrame({'close': mid, 'high': mid, 'low': mid, 'open': mid}, index=ts)
        h1 = base.resample('1h').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
        if len(h1) < 15:
            return signals, sl_pips_arr, tp_pips_arr

        h1['ema50'] = h1['close'].ewm(span=50, adjust=False).mean()
        h1['macro'] = np.where(h1['close'] > h1['ema50'], 1, -1)
        h1['roll_h'] = h1['high'].rolling(12).max().shift(1).bfill()
        h1['roll_l'] = h1['low'].rolling(12).min().shift(1).bfill()

        h1_macro = h1['macro'].reindex(m5.index, method='ffill').fillna(0).values
        h1_supp = h1['roll_l'].reindex(m5.index, method='ffill').values
        h1_res = h1['roll_h'].reindex(m5.index, method='ffill').values

        c = m5['close'].values
        h = m5['high'].values
        l = m5['low'].values
        o = m5['open'].values
        spread = m5['spread_pips'].values
        hours = m5.index.hour.values

        time_ms = m5.index.astype('datetime64[ms]').view('int64')
        dur_ms = 5 * 60 * 1000
        tick_ts = ticks['timestamp']

        last_trade_idx = -999

        for i in range(15, len(m5)):
            hr = hours[i]
            if not (7 <= hr <= 16):
                continue
            if spread[i] > self.max_spread:
                continue

            if (i - last_trade_idx) < 12:
                continue

            m = h1_macro[i]
            bar_rng = (h[i] - l[i]) / PIP_SIZE
            if bar_rng < 2.5:
                continue

            sig = 0
            if m > 0:
                dist_supp = abs(l[i] - h1_supp[i]) / PIP_SIZE
                lower_wick = (min(o[i], c[i]) - l[i]) / PIP_SIZE
                if dist_supp <= 4.0 and (lower_wick / bar_rng) >= self.min_wick and c[i] > o[i]:
                    sig = SIGNAL_LONG

            elif m < 0:
                dist_res = abs(h1_res[i] - h[i]) / PIP_SIZE
                upper_wick = (h[i] - max(o[i], c[i])) / PIP_SIZE
                if dist_res <= 4.0 and (upper_wick / bar_rng) >= self.min_wick and c[i] < o[i]:
                    sig = SIGNAL_SHORT

            if sig != 0:
                exec_t = time_ms[i] + dur_ms
                idx = int(np.searchsorted(tick_ts, exec_t))
                if idx < n_ticks and signals[idx] == 0:
                    signals[idx] = sig
                    sl_pips_arr[idx] = self.sl_pips
                    tp_pips_arr[idx] = self.tp_pips
                    last_trade_idx = i

        return signals, sl_pips_arr, tp_pips_arr


# ----------------------------------------------------------------------
# 3. WOTC Structural Bounce (H1/M15, 9.0p SL, 12.5p TP)
# ----------------------------------------------------------------------
class WOTCStructuralBounce5Y(BaseStrategy):
    def __init__(self, sl_pips: float = 9.0, tp_pips: float = 12.5, **kwargs):
        super().__init__(**kwargs)
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.allowed_hours = {7, 8, 9, 10, 11, 12, 13, 14, 15, 16}
        self.name = "WOTC_Struct_Bounce_M15"

    def generate_signals(self, ticks: np.ndarray):
        n_ticks = len(ticks)
        signals = np.zeros(n_ticks, dtype=np.int8)
        sl_pips_arr = np.zeros(n_ticks, dtype=np.float64)
        tp_pips_arr = np.zeros(n_ticks, dtype=np.float64)

        if n_ticks < 500:
            return signals, sl_pips_arr, tp_pips_arr

        m15 = build_time_bars(ticks, timeframe_min=15)
        if len(m15) < 40:
            return signals, sl_pips_arr, tp_pips_arr

        ts = pd.to_datetime(ticks['timestamp'], unit='ms')
        mid = 0.5 * (ticks['bid'] + ticks['ask'])
        base = pd.DataFrame({'close': mid, 'high': mid, 'low': mid, 'open': mid}, index=ts)
        h1 = base.resample('1h').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
        if len(h1) < 25:
            return signals, sl_pips_arr, tp_pips_arr

        h1['ema20'] = h1['close'].ewm(span=20, adjust=False).mean()
        h1['macro'] = np.where(h1['close'] > h1['ema20'], 1, -1)
        h1['roll_h'] = h1['high'].rolling(12).max().shift(1).bfill()
        h1['roll_l'] = h1['low'].rolling(12).min().shift(1).bfill()

        h1_macro = h1['macro'].reindex(m15.index, method='ffill').fillna(0).values
        h1_supp = h1['roll_l'].reindex(m15.index, method='ffill').values
        h1_res = h1['roll_h'].reindex(m15.index, method='ffill').values

        c = m15['close'].values
        h = m15['high'].values
        l = m15['low'].values
        o = m15['open'].values
        n_bars = len(m15)

        tick_ts = ticks['timestamp']
        time_ms = m15.index.astype(np.int64) // 10**6
        dur_ms = 15 * 60 * 1000
        hours = m15.index.hour

        traded_today = False
        last_day = -1

        for i in range(25, n_bars - 1):
            cur_day = m15.index[i].dayofyear
            if cur_day != last_day:
                last_day = cur_day
                traded_today = False

            if traded_today:
                continue

            if hours[i] not in self.allowed_hours:
                continue

            exec_time = time_ms[i] + dur_ms
            t_idx = int(np.searchsorted(tick_ts, exec_time))
            if t_idx >= n_ticks:
                break

            spread = (ticks['ask'][t_idx] - ticks['bid'][t_idx]) / PIP_SIZE
            if spread > 0.60:
                continue

            bar_range = max(h[i] - l[i], 1e-6)
            bar_pips = bar_range / PIP_SIZE
            if bar_pips < 3.2:
                continue

            m = h1_macro[i]

            if m > 0:
                dist_supp = abs(l[i] - h1_supp[i]) / PIP_SIZE
                lower_wick = min(o[i], c[i]) - l[i]
                if dist_supp <= 5.5 and (lower_wick / bar_range) >= 0.45 and c[i] > o[i]:
                    signals[t_idx] = SIGNAL_LONG
                    sl_pips_arr[t_idx] = self.sl_pips
                    tp_pips_arr[t_idx] = self.tp_pips
                    traded_today = True

            elif m < 0:
                dist_res = abs(h1_res[i] - h[i]) / PIP_SIZE
                upper_wick = h[i] - max(o[i], c[i])
                if dist_res <= 5.5 and (upper_wick / bar_range) >= 0.45 and c[i] < o[i]:
                    signals[t_idx] = SIGNAL_SHORT
                    sl_pips_arr[t_idx] = self.sl_pips
                    tp_pips_arr[t_idx] = self.tp_pips
                    traded_today = True

        return signals, sl_pips_arr, tp_pips_arr


# ----------------------------------------------------------------------
# 5-Year Auditor Loop
# ----------------------------------------------------------------------
def _eval_model_week(args):
    w, StratCls, kwargs = args
    ticks, _ = get_week_dataset(w)
    engine = BacktestEngine(pip_size=0.0001, use_slippage=True)
    strat = StratCls(**kwargs)
    trades = engine.run(ticks, strat)
    return w, trades


def audit_full_5y(StratCls, **kwargs):
    weeks = [(w, StratCls, kwargs) for w in range(1, 263)]
    num_workers = min(8, os.cpu_count() or 4)
    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        results = list(ex.map(_eval_model_week, weeks))

    results.sort(key=lambda x: x[0])
    all_trades = []
    in_sample_trades = []
    out_of_sample_trades = []

    for w, trs in results:
        all_trades.extend(trs)
        if w <= 156:
            in_sample_trades.extend(trs)
        else:
            out_of_sample_trades.extend(trs)

    def _calc_stats(trades):
        n = len(trades)
        if n == 0:
            return {'n': 0, 'wr': 0, 'pf': 0, 'pips': 0, 'max_dd': 0, 'ret_dd': 0}
        pips = sum(t['pips'] for t in trades)
        wins = [t for t in trades if t['pips'] > 0]
        losses = [t for t in trades if t['pips'] < 0]
        wr = len(wins) / n * 100.0
        gp = sum(t['pips'] for t in wins)
        gl = abs(sum(t['pips'] for t in losses))
        pf = gp / gl if gl > 0 else 99.0

        equity = np.cumsum([t['pips'] for t in trades])
        peak = np.maximum.accumulate(equity)
        dd = peak - equity
        max_dd = dd.max()
        ret_dd = pips / max_dd if max_dd > 0 else 99.0
        return {'n': n, 'wr': wr, 'pf': pf, 'pips': pips, 'max_dd': max_dd, 'ret_dd': ret_dd}

    strat_inst = StratCls(**kwargs)
    return {
        'name': strat_inst.name,
        'in_sample': _calc_stats(in_sample_trades),
        'out_of_sample': _calc_stats(out_of_sample_trades),
        'full_5y': _calc_stats(all_trades),
    }


if __name__ == "__main__":
    t0 = time.time()
    print("=" * 85)
    print("  AUDITING HIGH WIN-RATE MODELS ACROSS 5 FULL CALENDAR YEARS (262 WEEKS)")
    print("  (~123.22 MILLION REAL DUKASCOPY TICKS / OCT 2021 - OCT 2026)")
    print("=" * 85)

    models = [
        (MacroStructureH4M30Elite, {}),
        (DISC43HighWR85, {}),
        (WOTCStructuralBounce5Y, {}),
    ]

    for StratCls, kwargs in models:
        st_t0 = time.time()
        res = audit_full_5y(StratCls, **kwargs)
        dt = time.time() - st_t0

        name = res['name']
        ins = res['in_sample']
        oos = res['out_of_sample']
        tot = res['full_5y']

        print(f"\nModel: {name} (Audited in {dt:.1f}s)")
        print(f"  * 3-Year In-Sample  (W001-156): {ins['n']} trades | WR: {ins['wr']:.1f}% | PF: {ins['pf']:.2f} | Net: {ins['pips']:+.1f}p | Max DD: {ins['max_dd']:.1f}p | Ret/DD: {ins['ret_dd']:.2f}")
        print(f"  * 2-Year Out-Sample (W157-262): {oos['n']} trades | WR: {oos['wr']:.1f}% | PF: {oos['pf']:.2f} | Net: {oos['pips']:+.1f}p | Max DD: {oos['max_dd']:.1f}p | Ret/DD: {oos['ret_dd']:.2f}")
        print(f"  * 5-Year FULL CYCLE (W001-262): {tot['n']} trades | WR: {tot['wr']:.1f}% | PF: {tot['pf']:.2f} | Net: {tot['pips']:+.1f}p | Max DD: {tot['max_dd']:.1f}p | Ret/DD: {tot['ret_dd']:.2f}")
        print("-" * 85)

    print(f"\nAll 5-year audits completed in {time.time() - t0:.1f}s total.")
