#!/usr/bin/env python3
"""
================================================================================
  PROTOTYPE BENCHMARK: LONDON & MIDDAY MACD ACCELERATION ELITE (DISC38 CANDIDATE)
================================================================================
Exploits the empirically discovered 1.81:1 to 1.86:1 MFE/MAE edge of MACD Histogram
acceleration in London & European Midday sessions:
  - Allowed Session Windows:
      Window 1: London Morning Momentum (08:00 - 09:30 UTC)
      Window 2: European Midday Expansion (11:00 - 12:30 UTC)
  - Setup:
      1. Trend: H1 50-EMA Macro Trend alignment (Close > H1 EMA50 for Long, < for Short).
      2. M15 MACD (12, 26, 9):
         Bullish: MACD line > 0, MACD Histogram (diff) ticks up decisively from a pullback (diff[i-1] < 0 and diff[i] > 0 or diff[i] > diff[i-1] * 1.5).
         Bearish: MACD line < 0, MACD Histogram ticks down from a pullback.
      3. Floating Spread: <= 0.60 pips (liquid European execution).
      4. Risk: SL based on recent 3-bar swing (7 - 14 pips).
      5. Target: 1.65R - 1.80R.
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
import ta

sys.dont_write_bytecode = True

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tick_engine.core.engine import BacktestEngine, TRADE_DTYPE
from tick_engine.core.dataloader import get_week_dataset
from tick_engine.core.metrics import compute_metrics
from tick_engine.strategies.base import BaseStrategy, SIGNAL_LONG, SIGNAL_SHORT
from data_engineering_suite.bars import build_time_bars

PIP_SIZE = 0.0001


class LondonMACDAccelerationStrategy(BaseStrategy):
    def __init__(
        self,
        target_rr: float = 1.70,
        min_risk_pips: float = 7.0,
        max_risk_pips: float = 14.0,
        allowed_hours: tuple = (8, 11),
        **kwargs
    ):
        super().__init__(**kwargs)
        self.target_rr = float(target_rr)
        self.min_risk_pips = float(min_risk_pips)
        self.max_risk_pips = float(max_risk_pips)
        self.allowed_hours = set(allowed_hours)
        self.name = "London_MACD_Acceleration"

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

        c = m15['close']
        h = m15['high']
        l = m15['low']
        o = m15['open']
        spread = m15['spread_pips'].values
        hours = m15.index.hour.values

        # Macro Trend (H1 50-EMA)
        ts = pd.to_datetime(ticks['timestamp'], unit='ms')
        mid = 0.5 * (ticks['bid'] + ticks['ask'])
        base = pd.DataFrame({'close': mid}, index=ts)
        h1 = base.resample('1h').last().dropna()
        h1['ema50'] = h1['close'].ewm(span=50, adjust=False).mean()
        h1['macro'] = np.where(h1['close'] > h1['ema50'], 1, -1)
        macro = h1['macro'].reindex(m15.index, method='ffill').fillna(0).values

        # M15 MACD (12, 26, 9)
        macd = ta.trend.MACD(close=c, window_slow=26, window_fast=12, window_sign=9)
        macd_line = macd.macd().values
        macd_diff = macd.macd_diff().values

        time_ms = m15.index.astype('datetime64[ms]').view('int64')
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        c_vals = c.values
        h_vals = h.values
        l_vals = l.values
        o_vals = o.values

        last_trade_day = -1

        for i in range(35, len(m15)):
            hr = hours[i]
            if hr not in self.allowed_hours:
                continue

            if spread[i] > 0.65:
                continue

            day_val = m15.index[i].day
            if day_val == last_trade_day:
                continue

            m = macro[i]
            sig = 0
            risk = 0.0

            # Bullish MACD Acceleration
            if m > 0 and macd_line[i] > 0 and macd_diff[i-1] <= 0 and macd_diff[i] > 0 and c_vals[i] > o_vals[i]:
                sl_ref = np.min(l_vals[max(0, i-3):i+1])
                risk = max(min((c_vals[i] - sl_ref) / PIP_SIZE, self.max_risk_pips), self.min_risk_pips)
                sig = SIGNAL_LONG

            # Bearish MACD Acceleration
            elif m < 0 and macd_line[i] < 0 and macd_diff[i-1] >= 0 and macd_diff[i] < 0 and c_vals[i] < o_vals[i]:
                sl_ref = np.max(h_vals[max(0, i-3):i+1])
                risk = max(min((sl_ref - c_vals[i]) / PIP_SIZE, self.max_risk_pips), self.min_risk_pips)
                sig = SIGNAL_SHORT

            if sig != 0:
                exec_t = time_ms[i] + dur_ms
                idx = int(np.searchsorted(tick_ts, exec_t))
                if idx < n_ticks and signals[idx] == 0:
                    signals[idx] = sig
                    sl_pips_arr[idx] = risk
                    tp_pips_arr[idx] = risk * self.target_rr
                    last_trade_day = day_val

        return signals, sl_pips_arr, tp_pips_arr


def _test_week(w: int):
    ticks, _ = get_week_dataset(w)
    strat = LondonMACDAccelerationStrategy(
        target_rr=1.70,
        min_risk_pips=7.0,
        max_risk_pips=14.0,
        allowed_hours=(8, 11)
    )
    engine = BacktestEngine(pip_size=PIP_SIZE, use_slippage=True)
    trades = engine.run(ticks, strat)
    return trades


def run():
    print("=" * 80)
    print("  TESTING LONDON & MIDDAY MACD ACCELERATION ACROSS 52 WEEKS (~26M TICKS)")
    print("=" * 80)

    weeks = list(range(1, 53))
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=min(8, os.cpu_count() or 4)) as ex:
        res = list(ex.map(_test_week, weeks))

    dt = time.time() - t0
    all_trades = [r for r in res if len(r) > 0]
    tr_arr = np.concatenate(all_trades) if all_trades else np.empty(0, dtype=TRADE_DTYPE)

    m = compute_metrics(tr_arr)
    print(f"Total Weeks:   {len(weeks)}")
    print(f"Total Trades:  {m['total_trades']} ({m['total_trades']/52:.2f}/wk)")
    print(f"Win Rate:      {m['win_rate_pct']:.1f}%")
    print(f"Profit Factor: {m['profit_factor']:.2f}")
    print(f"Net Pips:      {m['net_pips']:+.1f} pips")
    print(f"Max Drawdown:  {m['max_drawdown_pips']:.1f} pips")
    print(f"Execution:     {dt:.1f}s")
    print("=" * 80)


if __name__ == "__main__":
    run()
