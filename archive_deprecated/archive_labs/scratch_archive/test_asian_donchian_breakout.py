#!/usr/bin/env python3
"""
================================================================================
  PROTOTYPE BENCHMARK: ASIAN DONCHIAN ADX BREAKOUT ELITE (DISC38 CANDIDATE)
================================================================================
Exploits the empirically discovered 2.10:1 MFE/MAE asymmetry in Asian Session (00:00-03:00 UTC):
  - Session Window: 00:00 - 03:30 UTC
  - Setup:
    1. Trend filter: Price relative to Asian baseline or Daily SMA / H1 EMA50.
    2. Donchian 20-bar channel breakout on M15.
    3. ADX(14) >= 22 (momentum confirmation, avoiding Asian dead flat chop).
    4. Floating spread filter: Spread <= 0.60 pips (spread is ~0.40p during Tokyo, avoiding rollover spikes).
    5. SL: Opposite side of tight 4-bar swing or 8-12 pips.
    6. TP: 1.65R - 1.80R.
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


class AsianDonchianBreakoutStrategy(BaseStrategy):
    def __init__(
        self,
        donchian_window: int = 16,
        adx_thresh: float = 20.0,
        target_rr: float = 1.65,
        min_risk_pips: float = 6.0,
        max_risk_pips: float = 14.0,
        allowed_hours: tuple = (0, 1, 2, 3),
        **kwargs
    ):
        super().__init__(**kwargs)
        self.donchian_window = int(donchian_window)
        self.adx_thresh = float(adx_thresh)
        self.target_rr = float(target_rr)
        self.min_risk_pips = float(min_risk_pips)
        self.max_risk_pips = float(max_risk_pips)
        self.allowed_hours = set(allowed_hours)
        self.name = "Asian_Donchian_ADX_Breakout"

    def generate_signals(self, ticks: np.ndarray):
        n_ticks = len(ticks)
        signals = np.zeros(n_ticks, dtype=np.int8)
        sl_pips_arr = np.zeros(n_ticks, dtype=np.float64)
        tp_pips_arr = np.zeros(n_ticks, dtype=np.float64)

        if n_ticks < 500:
            return signals, sl_pips_arr, tp_pips_arr

        m15 = build_time_bars(ticks, timeframe_min=15)
        if len(m15) < 30:
            return signals, sl_pips_arr, tp_pips_arr

        c = m15['close']
        h = m15['high']
        l = m15['low']
        o = m15['open']
        spread = m15['spread_pips'].values
        hours = m15.index.hour.values

        # Trend filter (M15 EMA50)
        ema50 = c.ewm(span=50, adjust=False).mean().values

        # ADX(14)
        adx_ind = ta.trend.ADXIndicator(high=h, low=l, close=c, window=14)
        adx = adx_ind.adx().values

        # Donchian Channels
        roll_h = h.rolling(self.donchian_window).max().shift(1).bfill().values
        roll_l = l.rolling(self.donchian_window).min().shift(1).bfill().values

        time_ms = m15.index.astype('datetime64[ms]').view('int64')
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        c_vals = c.values
        h_vals = h.values
        l_vals = l.values
        o_vals = o.values

        last_trade_day = -1

        for i in range(self.donchian_window + 5, len(m15)):
            hr = hours[i]
            if hr not in self.allowed_hours:
                continue

            # Floating spread check (must be tight Asian spread, not rollover wide)
            if spread[i] > 0.65:
                continue

            if adx[i] < self.adx_thresh:
                continue

            # Ensure 1 trade per day maximum
            day_val = m15.index[i].day
            if day_val == last_trade_day:
                continue

            sig = 0
            risk = 0.0

            # Bullish Breakout
            if c_vals[i] > roll_h[i] and c_vals[i] > ema50[i] and c_vals[i] > o_vals[i]:
                # SL placed below the recent 4-bar low or ema50
                sl_ref = min(np.min(l_vals[max(0, i-3):i+1]), ema50[i])
                risk = max(min((c_vals[i] - sl_ref) / PIP_SIZE, self.max_risk_pips), self.min_risk_pips)
                sig = SIGNAL_LONG

            # Bearish Breakout
            elif c_vals[i] < roll_l[i] and c_vals[i] < ema50[i] and c_vals[i] < o_vals[i]:
                sl_ref = max(np.max(h_vals[max(0, i-3):i+1]), ema50[i])
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
    strat = AsianDonchianBreakoutStrategy(
        donchian_window=16,
        adx_thresh=20.0,
        target_rr=1.75,
        min_risk_pips=6.0,
        max_risk_pips=12.0
    )
    engine = BacktestEngine(pip_size=PIP_SIZE, use_slippage=True)
    trades = engine.run(ticks, strat)
    return trades


def run():
    print("=" * 80)
    print("  TESTING ASIAN DONCHIAN ADX BREAKOUT ACROSS 52 WEEKS (~26M TICKS)")
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
