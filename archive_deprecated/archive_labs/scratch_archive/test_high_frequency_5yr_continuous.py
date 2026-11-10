#!/usr/bin/env python3
"""
================================================================================
  5-YEAR FULL CYCLE AUDIT: HIGH-FREQUENCY CONTINUOUS REJECTION ENGINE (262 WEEKS)
================================================================================
Tests the high-frequency M5 structural rejection engine across all 262 weeks:
  - 3-Year In-Sample (Weeks 1 to 156)
  - 2-Year Out-of-Sample (Weeks 157 to 262)
  - 5-Year Full Cycle Total (~123.22M Dukascopy ticks)

Parameter Variations:
  Var A: Cooldown 8 bars (40 min) | Dist 4.5p | Wick 50% | SL 6.0p | TP 8.5p
  Var B: Cooldown 6 bars (30 min) | Dist 4.5p | Wick 50% | SL 6.0p | TP 8.5p
  Var C: Cooldown 6 bars (30 min) | Dist 5.0p | Wick 48% | SL 6.0p | TP 8.5p
  Var D: Cooldown 4 bars (20 min) | Dist 5.0p | Wick 48% | SL 6.0p | TP 8.0p
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


class HighFrequencyContinuousRejection(BaseStrategy):
    def __init__(
        self,
        sl_pips: float = 6.0,
        tp_pips: float = 8.5,
        min_wick: float = 0.50,
        max_dist: float = 4.5,
        cooldown_bars: int = 6,
        lookback_h1: int = 10,
        name_suffix: str = "",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.min_wick = float(min_wick)
        self.max_dist = float(max_dist)
        self.cooldown_bars = int(cooldown_bars)
        self.lookback_h1 = int(lookback_h1)
        self.allowed_hours = {7, 8, 9, 10, 11, 12, 13, 14, 15, 16}
        self.name = f"HF_Rejection_CD{self.cooldown_bars}_Wick{int(self.min_wick*100)}_{self.sl_pips}SL_{self.tp_pips}TP{name_suffix}"

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
        h1['roll_h'] = h1['high'].rolling(self.lookback_h1).max().shift(1).bfill()
        h1['roll_l'] = h1['low'].rolling(self.lookback_h1).min().shift(1).bfill()

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
            if hours[i] not in self.allowed_hours:
                continue
            if spread[i] > 0.55:
                continue

            if (i - last_trade_idx) < self.cooldown_bars:
                continue

            m = h1_macro[i]
            bar_rng = (h[i] - l[i]) / PIP_SIZE
            if bar_rng < 2.5:
                continue

            sig = 0
            if m > 0:
                dist_supp = abs(l[i] - h1_supp[i]) / PIP_SIZE
                lower_wick = (min(o[i], c[i]) - l[i]) / PIP_SIZE
                if dist_supp <= self.max_dist and (lower_wick / bar_rng) >= self.min_wick and c[i] > o[i]:
                    sig = SIGNAL_LONG

            elif m < 0:
                dist_res = abs(h1_res[i] - h[i]) / PIP_SIZE
                upper_wick = (h[i] - max(o[i], c[i])) / PIP_SIZE
                if dist_res <= self.max_dist and (upper_wick / bar_rng) >= self.min_wick and c[i] < o[i]:
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
# 5-Year Parallel Auditor Loop
# ----------------------------------------------------------------------
def _eval_model_week(args):
    w, kwargs = args
    ticks, _ = get_week_dataset(w)
    engine = BacktestEngine(pip_size=0.0001, use_slippage=True)
    strat = HighFrequencyContinuousRejection(**kwargs)
    trades = engine.run(ticks, strat)
    return w, trades


def audit_full_5y(kwargs):
    weeks = [(w, kwargs) for w in range(1, 263)]
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

    strat_inst = HighFrequencyContinuousRejection(**kwargs)
    return {
        'name': strat_inst.name,
        'in_sample': _calc_stats(in_sample_trades),
        'out_of_sample': _calc_stats(out_of_sample_trades),
        'full_5y': _calc_stats(all_trades),
    }


if __name__ == "__main__":
    t0 = time.time()
    print("=" * 85)
    print("  5-YEAR TICK AUDIT: HIGH-FREQUENCY CONTINUOUS REJECTION (262 WEEKS / ~123.22M TICKS)")
    print("  OBJECTIVE: HIGH TRADE VOLUME (400-800+ TRADES) + WIN RATE > 50% + SMOOTH EQUITY")
    print("=" * 85)

    variations = [
        {'sl_pips': 6.0, 'tp_pips': 8.5, 'min_wick': 0.50, 'max_dist': 4.5, 'cooldown_bars': 8, 'lookback_h1': 10, 'name_suffix': '_Standard'},
        {'sl_pips': 6.0, 'tp_pips': 8.5, 'min_wick': 0.50, 'max_dist': 4.5, 'cooldown_bars': 6, 'lookback_h1': 10, 'name_suffix': '_FastCD30m'},
        {'sl_pips': 6.0, 'tp_pips': 8.0, 'min_wick': 0.48, 'max_dist': 5.0, 'cooldown_bars': 6, 'lookback_h1': 10, 'name_suffix': '_UltraHighFreq'},
    ]

    for kwargs in variations:
        st_t0 = time.time()
        res = audit_full_5y(kwargs)
        dt = time.time() - st_t0

        name = res['name']
        ins = res['in_sample']
        oos = res['out_of_sample']
        tot = res['full_5y']

        print(f"\nModel: {name} (Audited in {dt:.1f}s)")
        print(f"  * 3-Year In-Sample  (W001-156): {ins['n']} trades ({ins['n']/156:.1f}/wk) | WR: {ins['wr']:.1f}% | PF: {ins['pf']:.2f} | Net: {ins['pips']:+.1f}p | Max DD: {ins['max_dd']:.1f}p | Ret/DD: {ins['ret_dd']:.2f}")
        print(f"  * 2-Year Out-Sample (W157-262): {oos['n']} trades ({oos['n']/106:.1f}/wk) | WR: {oos['wr']:.1f}% | PF: {oos['pf']:.2f} | Net: {oos['pips']:+.1f}p | Max DD: {oos['max_dd']:.1f}p | Ret/DD: {oos['ret_dd']:.2f}")
        print(f"  * 5-Year FULL CYCLE (W001-262): {tot['n']} trades ({tot['n']/262:.1f}/wk | ~{tot['n']/5:.1f}/yr) | WR: {tot['wr']:.1f}% | PF: {tot['pf']:.2f} | Net: {tot['pips']:+.1f}p | Max DD: {tot['max_dd']:.1f}p | Ret/DD: {tot['ret_dd']:.2f}")
        print("-" * 85)

    print(f"\nAll 5-year audits completed in {time.time() - t0:.1f}s total.")
