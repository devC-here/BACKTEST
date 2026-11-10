#!/usr/bin/env python3
"""
Squeeze Breakout Parameter & Frequency Sweep:
Objective: Scale trade frequency of INST04 (48% WR, PF 1.66) to 100-200 trades/year
while maintaining Win Rate >= 45% and Profit Factor >= 1.40.
"""

from __future__ import annotations
import sys
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Any
import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tick_engine.core.engine import BacktestEngine
from tick_engine.core.dataloader import get_week_dataset
from tick_engine.strategies.base import BaseStrategy, SIGNAL_LONG, SIGNAL_SHORT

PIP_SIZE = 0.0001


class ParameterizedSqueezeStrategy(BaseStrategy):
    def __init__(
        self,
        lookback: int = 30,
        squeeze_ratio: float = 1.35,
        target_rr: float = 1.60,
        body_ratio: float = 0.70,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.lookback = lookback
        self.squeeze_ratio = squeeze_ratio
        self.target_rr = target_rr
        self.body_ratio = body_ratio
        self.pip_size = PIP_SIZE
        self.min_sl = 6.0
        self.max_sl = 16.0

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
        if len(m15) < self.lookback + 5:
            return signals, sl_pips_arr, tp_pips_arr

        c = m15['close']
        h = m15['high']
        l = m15['low']

        tr = np.maximum(h - l, np.maximum(abs(h - c.shift(1)), abs(l - c.shift(1))))
        m15['atr_14'] = tr.rolling(14).mean().bfill()
        m15['atr_pips'] = m15['atr_14'] / self.pip_size

        m15['sma_20'] = c.rolling(20).mean().bfill()
        std20 = c.rolling(20).std().bfill()
        m15['bb_up'] = m15['sma_20'] + (2.0 * std20)
        m15['bb_low'] = m15['sma_20'] - (2.0 * std20)
        m15['bb_width'] = (m15['bb_up'] - m15['bb_low']) / self.pip_size
        m15['bb_pct_b'] = (c - m15['bb_low']) / np.maximum(m15['bb_up'] - m15['bb_low'], 1e-9)
        m15['bb_w_min'] = m15['bb_width'].rolling(self.lookback).min().shift(1).bfill()

        m15['hour'] = m15.index.hour
        time_ms = m15.index.astype('datetime64[ms]').view('int64')
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        for i in range(self.lookback, len(m15)):
            row = m15.iloc[i]
            prev = m15.iloc[i-1]

            if row['hour'] < 7 or row['hour'] > 15:
                continue

            sig = 0
            risk_p = 0.0
            body_pips = abs(row['close'] - row['open']) / self.pip_size

            was_tight = prev['bb_width'] <= self.squeeze_ratio * row['bb_w_min']
            if not was_tight:
                continue

            # Bullish Breakout
            if prev['bb_pct_b'] <= 1.0 and row['bb_pct_b'] > 1.0 and row['close'] > row['open']:
                if body_pips >= self.body_ratio * row['atr_pips']:
                    sl_p = row['sma_20']
                    risk = max((row['close'] - sl_p) / self.pip_size, self.min_sl)
                    if risk <= self.max_sl:
                        sig = SIGNAL_LONG
                        risk_p = risk

            # Bearish Breakout
            elif prev['bb_pct_b'] >= 0.0 and row['bb_pct_b'] < 0.0 and row['close'] < row['open']:
                if body_pips >= self.body_ratio * row['atr_pips']:
                    sl_p = row['sma_20']
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


CONFIGS = [
    {"name": "Lookback=40, RR=1.6, Squeeze=1.35", "lb": 40, "sq": 1.35, "rr": 1.6, "br": 0.8},
    {"name": "Lookback=30, RR=1.6, Squeeze=1.35", "lb": 30, "sq": 1.35, "rr": 1.6, "br": 0.7},
    {"name": "Lookback=25, RR=1.6, Squeeze=1.35", "lb": 25, "sq": 1.35, "rr": 1.6, "br": 0.7},
    {"name": "Lookback=30, RR=1.5, Squeeze=1.40", "lb": 30, "sq": 1.40, "rr": 1.5, "br": 0.7},
    {"name": "Lookback=25, RR=1.5, Squeeze=1.40", "lb": 25, "sq": 1.40, "rr": 1.5, "br": 0.65},
    {"name": "Lookback=20, RR=1.6, Squeeze=1.35", "lb": 20, "sq": 1.35, "rr": 1.6, "br": 0.7},
]


def run_week(week_num: int):
    ticks, meta = get_week_dataset(week_num)
    engine = BacktestEngine(pip_size=PIP_SIZE, use_slippage=True)
    res = {}
    for cfg in CONFIGS:
        strat = ParameterizedSqueezeStrategy(
            lookback=cfg["lb"], squeeze_ratio=cfg["sq"], target_rr=cfg["rr"], body_ratio=cfg["br"]
        )
        trades = engine.run(ticks, strat)
        res[cfg["name"]] = [float(t["pips"]) for t in trades]
    return res


def main():
    print("Testing Parameterized Squeeze Configurations across 52 weeks...")
    with ProcessPoolExecutor() as executor:
        week_res = list(executor.map(run_week, range(1, 53)))

    all_res = {cfg["name"]: [] for cfg in CONFIGS}
    for r in week_res:
        for k, v in r.items():
            all_res[k].extend(v)

    print("=" * 105)
    print(f"{'Configuration':<42} | {'Trades':<6} | {'Win %':<6} | {'Real RR':<7} | {'PF':<5} | {'Net Pips':<9}")
    print("-" * 105)
    for cfg in CONFIGS:
        name = cfg["name"]
        pips = np.array(all_res[name])
        wins = pips[pips > 0]
        losses = pips[pips < 0]
        n_t = len(pips)
        wr = len(wins) / n_t * 100.0 if n_t > 0 else 0.0
        rr = (np.mean(wins) / abs(np.mean(losses))) if len(losses) > 0 else 0.0
        pf = np.sum(wins) / abs(np.sum(losses)) if len(losses) > 0 else 0.0
        net = np.sum(pips)
        print(f"{name:<42} | {n_t:<6} | {wr:<4.1f} % | {rr:<7.2f} | {pf:<5.2f} | {net:<+9.1f}")
    print("=" * 105)


if __name__ == "__main__":
    main()
