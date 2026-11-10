#!/usr/bin/env python3
"""
Test Hybrid Session Liquidity + Indicator Strategy on 52 Weeks of EUR/USD Ticks.
Combines:
- Asian Session Range (00:00 to 06:00 UTC)
- London Open Sweep Fade (07:00 to 10:00 UTC)
- RSI / Extreme confirmation
- Target RR: 1.8 to 2.2
"""
import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import os
import time

from tick_engine.core.engine import BacktestEngine
from tick_engine.core.dataloader import get_week_dataset
from tick_engine.strategies.base import BaseStrategy, SIGNAL_LONG, SIGNAL_SHORT

PIP_SIZE = 0.0001


class LondonAsianSweepStrategy(BaseStrategy):
    def __init__(self, target_rr=2.0, min_sweep=1.0, max_sweep=12.0, **kwargs):
        super().__init__(**kwargs)
        self.target_rr = target_rr
        self.min_sweep = min_sweep
        self.max_sweep = max_sweep
        self.name = 'London_Asian_Sweep'

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

        # H1 Macro EMA 50
        h1 = base.resample('1h').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
        h1['ema50'] = h1['close'].ewm(span=50, adjust=False).mean()
        h1['macro'] = np.where(h1['close'] > h1['ema50'], 1, -1)
        m15['h1_macro'] = h1['macro'].reindex(m15.index, method='ffill').fillna(0)

        # RSI (14)
        delta = m15['close'].diff()
        gain = delta.clip(lower=0.0).ewm(alpha=1.0/14, adjust=False).mean()
        loss = (-delta.clip(upper=0.0)).ewm(alpha=1.0/14, adjust=False).mean()
        rs = gain / np.maximum(loss, 1e-9)
        m15['rsi'] = 100.0 - (100.0 / (1.0 + rs))

        m15['hour'] = m15.index.hour
        m15['date'] = m15.index.date
        m15['time_ms'] = m15.index.astype('datetime64[ms]').view('int64')

        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        dates = m15['date'].unique()
        for d in dates:
            day_mask = m15['date'] == d
            asian_mask = day_mask & (m15['hour'] < 6)
            asian_bars = m15[asian_mask]
            if len(asian_bars) < 8:
                continue
            asian_h = asian_bars['high'].max()
            asian_l = asian_bars['low'].min()
            asian_range_pips = (asian_h - asian_l) / PIP_SIZE
            if asian_range_pips < 8.0 or asian_range_pips > 45.0:
                continue

            london_indices = np.where(day_mask & (m15['hour'] >= 7) & (m15['hour'] <= 10))[0]
            traded_today = False

            for idx in london_indices:
                if traded_today:
                    break
                row = m15.iloc[idx]
                macro = row['h1_macro']

                sig = 0
                risk_p = 0.0

                # Bullish: Swept Asian Low and closed back inside range
                sweep_low_pips = (asian_l - row['low']) / PIP_SIZE
                if self.min_sweep <= sweep_low_pips <= self.max_sweep and row['close'] > asian_l and row['close'] > row['open']:
                    if macro >= 0 or row['rsi'] <= 42.0:
                        sl_dist = (row['close'] - row['low']) / PIP_SIZE + 1.2
                        if 5.0 <= sl_dist <= 16.0:
                            sig = SIGNAL_LONG
                            risk_p = sl_dist
                            traded_today = True

                # Bearish: Swept Asian High and closed back inside range
                sweep_high_pips = (row['high'] - asian_h) / PIP_SIZE
                if self.min_sweep <= sweep_high_pips <= self.max_sweep and row['close'] < asian_h and row['close'] < row['open']:
                    if macro <= 0 or row['rsi'] >= 58.0:
                        sl_dist = (row['high'] - row['close']) / PIP_SIZE + 1.2
                        if 5.0 <= sl_dist <= 16.0:
                            sig = SIGNAL_SHORT
                            risk_p = sl_dist
                            traded_today = True

                if sig != 0:
                    exec_t = m15['time_ms'].iloc[idx] + dur_ms
                    t_idx = int(np.searchsorted(tick_ts, exec_t))
                    if t_idx < n_ticks and signals[t_idx] == 0:
                        signals[t_idx] = sig
                        sl_pips_arr[t_idx] = risk_p
                        tp_pips_arr[t_idx] = risk_p * self.target_rr

        return signals, sl_pips_arr, tp_pips_arr


def run_week(w):
    ticks, _ = get_week_dataset(w)
    engine = BacktestEngine(pip_size=0.0001, use_slippage=True)
    strat = LondonAsianSweepStrategy(target_rr=2.2)
    trades = engine.run(ticks, strat)
    return trades


if __name__ == "__main__":
    t0 = time.time()
    weeks = list(range(1, 53))
    num_workers = min(8, os.cpu_count() or 4)

    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        results = list(ex.map(run_week, weeks))

    all_trades = []
    for r in results:
        all_trades.extend(r)

    dt = time.time() - t0
    n = len(all_trades)
    pips = sum(t['pips'] for t in all_trades)
    wins = sum(1 for t in all_trades if t['pips'] > 0)
    wr = (wins / n * 100) if n > 0 else 0
    gp = sum(t['pips'] for t in all_trades if t['pips'] > 0)
    gl = abs(sum(t['pips'] for t in all_trades if t['pips'] < 0))
    pf = gp / gl if gl > 0 else 0

    print("=" * 80)
    print(f"LONDON ASIAN RANGE SWEEP + RSI/MACRO HYBRID (52 WEEKS TICK AUDIT in {dt:.1f}s)")
    print("=" * 80)
    print(f"Total Trades:        {n} trades (~{n/52:.1f} trades/week / ~{n/250:.2f} trades/day)")
    print(f"Win Rate:            {wr:.1f}%")
    print(f"Profit Factor:       {pf:.2f}")
    print(f"Net Pips:            {pips:+.1f} pips")
    print("=" * 80)
