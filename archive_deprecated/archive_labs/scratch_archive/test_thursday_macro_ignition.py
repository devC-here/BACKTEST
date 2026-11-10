#!/usr/bin/env python3
"""
================================================================================
  CANDIDATE DISCOVERY: THURSDAY MACRO TREND IGNITION ELITE (DISC41 CANDIDATE)
================================================================================
Exploits the empirically discovered 1.98:1 MFE/MAE asymmetry and 97.8-pip average
range of Thursdays during the New York Overlap (13:00 - 15:30 UTC):
  - Day: Thursday Only (Day of Week = 3)
  - Time Window: 13:00 - 15:30 UTC (Post-Macro Release Institutional Flow)
  - Setup:
      1. Macro Trend: Price aligned with H1 50-EMA trend.
      2. M15 Momentum: Donchian 20-bar channel breakout.
      3. Trend Strength Filter: ADX(14) >= 22.0.
      4. Spread Filter: Floating spread <= 0.60 pips.
      5. Dynamic Stop: Recent 3-bar swing low/high bounded between 7.0 and 15.0 pips.
      6. Target: 1.70R.
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


class ThursdayMacroTrendIgnitionElite(BaseStrategy):
    def __init__(
        self,
        donchian_window: int = 20,
        adx_thresh: float = 22.0,
        target_rr: float = 1.70,
        min_risk_pips: float = 7.0,
        max_risk_pips: float = 15.0,
        allowed_hours: tuple = (13, 14, 15),
        max_spread_pips: float = 0.60,
        pip_size: float = PIP_SIZE,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.donchian_window = int(donchian_window)
        self.adx_thresh = float(adx_thresh)
        self.target_rr = float(target_rr)
        self.min_risk_pips = float(min_risk_pips)
        self.max_risk_pips = float(max_risk_pips)
        self.allowed_hours = set(allowed_hours)
        self.max_spread_pips = float(max_spread_pips)
        self.pip_size = float(pip_size)
        self.name = "Thursday_Macro_Trend_Ignition_Elite"

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
        days = m15.index.dayofweek.values

        # Macro Trend (H1 50-EMA)
        ts = pd.to_datetime(ticks['timestamp'], unit='ms')
        mid = 0.5 * (ticks['bid'] + ticks['ask'])
        base = pd.DataFrame({'close': mid}, index=ts)
        h1 = base.resample('1h').last().dropna()
        h1['ema50'] = h1['close'].ewm(span=50, adjust=False).mean()
        h1['macro'] = np.where(h1['close'] > h1['ema50'], 1, -1)
        macro = h1['macro'].reindex(m15.index, method='ffill').fillna(0).values

        # Donchian & ADX
        roll_h = h.rolling(self.donchian_window).max().shift(1).bfill().values
        roll_l = l.rolling(self.donchian_window).min().shift(1).bfill().values

        adx_ind = ta.trend.ADXIndicator(high=h, low=l, close=c, window=14)
        adx = adx_ind.adx().values

        time_ms = m15.index.astype('datetime64[ms]').view('int64')
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        c_vals = c.values
        h_vals = h.values
        l_vals = l.values
        o_vals = o.values

        last_trade_day = -1

        for i in range(35, len(m15)):
            # Strictly Thursday (dayofweek == 3)
            if days[i] != 3:
                continue

            hr = hours[i]
            if hr not in self.allowed_hours:
                continue

            if spread[i] > self.max_spread_pips:
                continue

            day_val = m15.index[i].day
            if day_val == last_trade_day:
                continue

            if adx[i] < self.adx_thresh:
                continue

            m = macro[i]
            sig = 0
            risk = 0.0

            # Bullish Thursday Macro Ignition
            if c_vals[i] > roll_h[i] and m > 0 and c_vals[i] > o_vals[i]:
                sl_ref = np.min(l_vals[max(0, i-3):i+1])
                risk = max(min((c_vals[i] - sl_ref) / self.pip_size, self.max_risk_pips), self.min_risk_pips)
                sig = SIGNAL_LONG

            # Bearish Thursday Macro Ignition
            elif c_vals[i] < roll_l[i] and m < 0 and c_vals[i] < o_vals[i]:
                sl_ref = np.max(h_vals[max(0, i-3):i+1])
                risk = max(min((sl_ref - c_vals[i]) / self.pip_size, self.max_risk_pips), self.min_risk_pips)
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
    strat = ThursdayMacroTrendIgnitionElite()
    engine = BacktestEngine(pip_size=PIP_SIZE, use_slippage=True)
    trades = engine.run(ticks, strat)
    return w, trades


def run():
    print("=" * 95)
    print("  5-YEAR TICK AUDIT: THURSDAY MACRO TREND IGNITION ELITE (262 WEEKS / ~123.22M TICKS)  ".center(95))
    print("=" * 95)

    weeks = list(range(1, 263))
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=min(8, os.cpu_count() or 4)) as ex:
        results = list(ex.map(_test_week, weeks))

    dt = time.time() - t0
    print(f"[+] All 262 weeks simulated in {dt:.1f}s!\n")

    results.sort(key=lambda x: x[0])

    train_trades = [r[1] for r in results if r[0] <= 156 and len(r[1]) > 0]
    test_trades = [r[1] for r in results if r[0] > 156 and len(r[1]) > 0]
    all_trades = [r[1] for r in results if len(r[1]) > 0]

    tr_train = np.concatenate(train_trades) if train_trades else np.empty(0, dtype=TRADE_DTYPE)
    tr_test = np.concatenate(test_trades) if test_trades else np.empty(0, dtype=TRADE_DTYPE)
    tr_all = np.concatenate(all_trades) if all_trades else np.empty(0, dtype=TRADE_DTYPE)

    m_train = compute_metrics(tr_train)
    m_test = compute_metrics(tr_test)
    m_all = compute_metrics(tr_all)

    print("-" * 95)
    print(f"{'Metric':<25} | {'3Y Train (W1-156)':<20} | {'2Y Test (W157-262)':<20} | {'5Y Total (W1-262)':<20}")
    print("-" * 95)
    print(f"{'Total Trades':<25} | {m_train['total_trades']:<20} | {m_test['total_trades']:<20} | {m_all['total_trades']:<20}")
    print(f"{'Trades per Week':<25} | {m_train['total_trades']/156:<20.2f} | {m_test['total_trades']/106:<20.2f} | {m_all['total_trades']/262:<20.2f}")
    print(f"{'Win Rate (%)':<25} | {m_train['win_rate_pct']:<19.1f}% | {m_test['win_rate_pct']:<19.1f}% | {m_all['win_rate_pct']:<19.1f}%")
    print(f"{'Profit Factor':<25} | {m_train['profit_factor']:<20.2f} | {m_test['profit_factor']:<20.2f} | {m_all['profit_factor']:<20.2f}")
    print(f"{'Net Cumulative Pips':<25} | {m_train['net_pips']:<+19.1f}p | {m_test['net_pips']:<+19.1f}p | {m_all['net_pips']:<+19.1f}p")
    print(f"{'Max Drawdown (p)':<25} | {m_train['max_drawdown_pips']:<19.1f}p | {m_test['max_drawdown_pips']:<19.1f}p | {m_all['max_drawdown_pips']:<19.1f}p")
    print(f"{'Return / DD Ratio':<25} | {m_train['net_pips']/max(m_train['max_drawdown_pips'], 1e-6):<20.2f} | {m_test['net_pips']/max(m_test['max_drawdown_pips'], 1e-6):<20.2f} | {m_all['net_pips']/max(m_all['max_drawdown_pips'], 1e-6):<20.2f}")
    print("-" * 95)


if __name__ == "__main__":
    run()
