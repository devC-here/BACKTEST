#!/usr/bin/env python3
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
from tick_engine.core.dataloader import get_week_dataset
from tick_engine.strategies.base import BaseStrategy, SIGNAL_LONG, SIGNAL_SHORT

PIP_SIZE = 0.0001

class ElderTripleScreenStrict(BaseStrategy):
    """
    Alexander Elder's Triple Screen with STRICT zero-lookahead:
    H1 MACD Histogram tide is shifted by 1 hour (so at 10:15, only the closed 09:00 H1 bar is used).
    """
    def __init__(self, target_rr: float = 2.2, min_sl: float = 6.0, max_sl: float = 16.0, **kwargs):
        super().__init__(**kwargs)
        self.target_rr = target_rr
        self.min_sl = min_sl
        self.max_sl = max_sl
        self.pip_size = PIP_SIZE
        self.name = "Elder_Triple_Screen_Strict"

    def generate_signals(self, ticks: np.ndarray):
        n_ticks = len(ticks)
        signals = np.zeros(n_ticks, dtype=np.int8)
        sl_pips_arr = np.zeros(n_ticks, dtype=np.float64)
        tp_pips_arr = np.zeros(n_ticks, dtype=np.float64)
        if n_ticks < 500:
            return signals, sl_pips_arr, tp_pips_arr

        ts = pd.to_datetime(ticks['timestamp'], unit='ms')
        mid = 0.5 * (ticks['bid'] + ticks['ask'])
        base_df = pd.DataFrame({'open': mid, 'high': mid, 'low': mid, 'close': mid}, index=ts)
        m15 = base_df.resample('15min').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
        if len(m15) < 40:
            return signals, sl_pips_arr, tp_pips_arr

        # M15 RSI 2
        delta = m15['close'].diff()
        gain2 = delta.clip(lower=0.0).ewm(alpha=1.0/2, adjust=False).mean()
        loss2 = (-delta.clip(upper=0.0)).ewm(alpha=1.0/2, adjust=False).mean()
        rs2 = gain2 / np.maximum(loss2, 1e-9)
        m15['rsi_2'] = 100.0 - (100.0 / (1.0 + rs2))

        # H1 MACD Histogram Tide with STRICT SHIFT(1)
        h1 = base_df.resample('1h').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
        h1_ema12 = h1['close'].ewm(span=12, adjust=False).mean()
        h1_ema26 = h1['close'].ewm(span=26, adjust=False).mean()
        h1_macd = h1_ema12 - h1_ema26
        h1_sig = h1_macd.ewm(span=9, adjust=False).mean()
        h1_hist = h1_macd - h1_sig
        h1['tide'] = np.where(h1_hist > h1_hist.shift(1), 1, -1)
        # Shift H1 by 1 bar to strictly guarantee no lookahead
        m15['tide'] = h1['tide'].shift(1).reindex(m15.index, method='ffill').fillna(0)

        m15['hour'] = m15.index.hour
        m15['time_ms'] = m15.index.astype('datetime64[ms]').view('int64')

        time_ms = m15['time_ms'].values
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        for i in range(25, len(m15)):
            row = m15.iloc[i]
            prev = m15.iloc[i-1]

            if row['hour'] < 7 or row['hour'] > 15:
                continue

            sig = 0
            risk_p = 0.0

            # Long: H1 Tide Bullish, Prev RSI(2) < 20, Current breaks above Prev High and closes green
            if row['tide'] > 0 and prev['rsi_2'] < 20.0 and row['close'] > prev['high'] and row['close'] > row['open']:
                sl_p = min(prev['low'], row['low']) - 1.2 * self.pip_size
                risk = max((row['close'] - sl_p) / self.pip_size, self.min_sl)
                if risk <= self.max_sl:
                    sig = SIGNAL_LONG
                    risk_p = risk

            # Short: H1 Tide Bearish, Prev RSI(2) > 80, Current breaks below Prev Low and closes red
            elif row['tide'] < 0 and prev['rsi_2'] > 80.0 and row['close'] < prev['low'] and row['close'] < row['open']:
                sl_p = max(prev['high'], row['high']) + 1.2 * self.pip_size
                risk = max((sl_p - row['close']) / self.pip_size, self.min_sl)
                if risk <= self.max_sl:
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


def run_week(w: int):
    ticks, meta = get_week_dataset(w)
    engine = BacktestEngine(pip_size=PIP_SIZE, use_slippage=True)
    strat = ElderTripleScreenStrict()
    trades = engine.run(ticks, strat)
    return [float(t["pips"]) for t in trades]

def main():
    print("Testing Elder Triple Screen with STRICT H1 shift(1) across 52 weeks...")
    with ProcessPoolExecutor() as executor:
        week_res = list(executor.map(run_week, range(1, 53)))
    
    all_pips = []
    for r in week_res:
        all_pips.extend(r)
    
    pips = np.array(all_pips)
    wins = pips[pips > 0]
    losses = pips[pips < 0]
    wr = len(wins) / len(pips) * 100 if len(pips) > 0 else 0
    pf = np.sum(wins) / abs(np.sum(losses)) if len(losses) > 0 else 0
    rr = np.mean(wins) / abs(np.mean(losses)) if len(losses) > 0 else 0
    print(f"Results: Trades={len(pips)}, WR={wr:.1f}%, Realized RR={rr:.2f}, PF={pf:.2f}, Net Pips={np.sum(pips):+.1f}")

if __name__ == "__main__":
    main()
