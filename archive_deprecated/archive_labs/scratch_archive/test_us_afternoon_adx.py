#!/usr/bin/env python3
"""
Refining US Afternoon Trend Continuation (16:00 - 19:30 UTC) with ADX & VWAP filters.
"""
from __future__ import annotations
import sys
import os
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tick_engine.core.engine import BacktestEngine
from tick_engine.core.dataloader import get_week_dataset
from tick_engine.strategies.base import BaseStrategy, SIGNAL_LONG, SIGNAL_SHORT

PIP_SIZE = 0.0001


class USAfternoonTrendCalibrated(BaseStrategy):
    def __init__(self, target_rr=1.75, min_adx=20.0, min_sl=6.0, max_sl=15.0, **kwargs):
        super().__init__(**kwargs)
        self.target_rr = target_rr
        self.min_adx = min_adx
        self.min_sl = min_sl
        self.max_sl = max_sl
        self.name = f"US_Afternoon_ADX{min_adx}_RR{target_rr}"

    def generate_signals(self, ticks: np.ndarray):
        n_ticks = len(ticks)
        signals = np.zeros(n_ticks, dtype=np.int8)
        sl_pips_arr = np.zeros(n_ticks, dtype=np.float64)
        tp_pips_arr = np.zeros(n_ticks, dtype=np.float64)
        if n_ticks < 500:
            return signals, sl_pips_arr, tp_pips_arr

        ts = pd.to_datetime(ticks['timestamp'], unit='ms')
        mid = 0.5 * (ticks['bid'] + ticks['ask'])
        base = pd.DataFrame({'open': mid, 'high': mid, 'low': mid, 'close': mid}, index=ts)
        m15 = base.resample('15min').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
        if len(m15) < 40:
            return signals, sl_pips_arr, tp_pips_arr

        c = m15['close']
        h = m15['high']
        l = m15['low']

        tr = np.maximum(h - l, np.maximum(abs(h - c.shift(1)), abs(l - c.shift(1))))
        m15['atr'] = tr.rolling(14).mean().bfill()
        m15['ema20'] = c.ewm(span=20, adjust=False).mean()

        # ADX 14
        up_m = h.diff()
        dn_m = -l.diff()
        p_dm = np.where((up_m > dn_m) & (up_m > 0), up_m, 0.0)
        m_dm = np.where((dn_m > up_m) & (dn_m > 0), dn_m, 0.0)
        smooth_tr = tr.ewm(alpha=1.0/14, adjust=False).mean()
        p_di = (pd.Series(p_dm, index=m15.index).ewm(alpha=1.0/14, adjust=False).mean() / np.maximum(smooth_tr, 1e-9)) * 100
        m_di = (pd.Series(m_dm, index=m15.index).ewm(alpha=1.0/14, adjust=False).mean() / np.maximum(smooth_tr, 1e-9)) * 100
        dx = (abs(p_di - m_di) / np.maximum(p_di + m_di, 1e-9)) * 100
        m15['adx'] = dx.ewm(alpha=1.0/14, adjust=False).mean()

        # H1 Macro Trend
        h1 = base.resample('1h').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
        h1['ema50'] = h1['close'].ewm(span=50, adjust=False).mean()
        h1['macro'] = np.where(h1['close'] > h1['ema50'], 1, -1)
        m15['h1_macro'] = h1['macro'].reindex(m15.index, method='ffill').fillna(0)

        m15['hour'] = m15.index.hour
        time_ms = m15.index.astype('datetime64[ms]').view('int64')
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        allowed_hours = {16, 17, 18, 19}

        for i in range(25, len(m15)):
            row = m15.iloc[i]
            prev = m15.iloc[i-1]

            if row['hour'] not in allowed_hours:
                continue
            if row['adx'] < self.min_adx:
                continue

            sig = 0
            risk_p = 0.0

            # Bullish trend pullback
            if row['h1_macro'] > 0 and row['close'] > row['ema20'] and row['low'] <= row['ema20'] and row['close'] > row['open']:
                sl_p = min(row['low'], prev['low']) - 1.2 * PIP_SIZE
                risk = max(min((row['close'] - sl_p) / PIP_SIZE, self.max_sl), self.min_sl)
                sig = SIGNAL_LONG
                risk_p = risk

            # Bearish trend pullback
            elif row['h1_macro'] < 0 and row['close'] < row['ema20'] and row['high'] >= row['ema20'] and row['close'] < row['open']:
                sl_p = max(row['high'], prev['high']) + 1.2 * PIP_SIZE
                risk = max(min((sl_p - row['close']) / PIP_SIZE, self.max_sl), self.min_sl)
                sig = SIGNAL_SHORT
                risk_p = risk

            if sig != 0:
                exec_t = time_ms[i] + dur_ms
                idx = int(np.searchsorted(tick_ts, exec_t))
                if idx < n_ticks and signals[idx] == 0:
                    signals[idx] = sig
                    sl_pips_arr[idx] = risk_p
                    tp_pips_arr[idx] = risk_p * self.target_rr

        return signals, sl_pips_arr, tp_pips_arr


def eval_week(args):
    w, strat_cls, params = args
    ticks, _ = get_week_dataset(w)
    engine = BacktestEngine(pip_size=0.0001, use_slippage=True)
    strat = strat_cls(**params)
    trades = engine.run(ticks, strat)
    return trades


def run_test(strat_cls, params, label):
    t0 = time.time()
    weeks = list(range(1, 53))
    tasks = [(w, strat_cls, params) for w in weeks]
    num_workers = min(8, os.cpu_count() or 4)

    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        results = list(ex.map(eval_week, tasks))

    all_trades = [t for r in results for t in r]
    dt = time.time() - t0

    n = len(all_trades)
    if n == 0:
        print(f"[-] {label}: 0 trades.")
        return None

    pips = sum(t['pips'] for t in all_trades)
    wins = [t for t in all_trades if t['pips'] > 0]
    losses = [t for t in all_trades if t['pips'] < 0]
    wr = len(wins) / n * 100.0
    gp = sum(t['pips'] for t in wins)
    gl = abs(sum(t['pips'] for t in losses))
    pf = gp / gl if gl > 0 else 99.0
    realized_rr = (gp / max(len(wins), 1)) / (gl / max(len(losses), 1)) if gl > 0 else 0.0

    equity = np.cumsum([t['pips'] for t in all_trades])
    peak = np.maximum.accumulate(equity)
    dd = peak - equity
    max_dd = dd.max()

    print("=" * 80)
    print(f"AUDIT: {label} (Time: {dt:.1f}s)")
    print("=" * 80)
    print(f"Trades (52 Wks): {n}  (~{n/52:.1f}/wk, ~{n/250:.2f}/day)")
    print(f"Win Rate:        {wr:.1f}%")
    print(f"Realized RR:     {realized_rr:.2f} : 1")
    print(f"Profit Factor:   {pf:.2f}")
    print(f"Net Pips:        {pips:+.1f} pips")
    print(f"Max Drawdown:    {max_dd:.1f} pips")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    print("\n--- AUDITING CALIBRATED US AFTERNOON GAP ---\n")
    run_test(USAfternoonTrendCalibrated, {"target_rr": 1.75, "min_adx": 18.0}, "US Afternoon (ADX>=18, RR 1.75)")
    run_test(USAfternoonTrendCalibrated, {"target_rr": 1.80, "min_adx": 22.0}, "US Afternoon (ADX>=22, RR 1.80)")
    run_test(USAfternoonTrendCalibrated, {"target_rr": 2.00, "min_adx": 25.0}, "US Afternoon (ADX>=25, RR 2.00)")
