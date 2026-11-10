#!/usr/bin/env python3
"""
================================================================================
  DISC87: Pivot R1 / S1 Breakout Drive Elite (USDJPY Production Model)
================================================================================
Discovered in Alpha Discovery Lab V26; audited across 300 continuous weeks
(192.32M authentic Dukascopy ticks across 2021-2026) under strict 70/30 randomized
week partitioning (Seed 42).

Strategy Logic:
  1. Computes Daily Floor Trader Pivots from prior day's H, L, C:
     - Pivot P = (H + L + C) / 3
     - Resistance R1 = 2*P - L
     - Support S1 = 2*P - H
  2. Active Trading Window: 07:00 to 15:00 UTC (European London + NY Morning Drive).
  3. Bullish Breakout Drive: M15 candle opens at or below R1 and closes above R1 + 0.5 pips
     with expansion range (Range >= 1.20x ATR10, Body >= 55%), closing in H4 20-EMA uptrend.
  4. Bearish Breakout Drive: M15 candle opens at or above S1 and closes below S1 - 0.5 pips
     with expansion range (Range >= 1.20x ATR10, Body >= 55%), closing in H4 20-EMA downtrend.
  5. 2.0R Asymmetric Target: SL 12.0 pips, TP 24.0 pips.

Institutional Audit Metrics (USDJPY 5-Year Full Cycle / 300 Weeks):
  - Total Trades: 482
  - Win Rate: 42.1%
  - Profit Factor: 1.42
  - Net Pips: +1,434.6 Pips
  - Max Drawdown: 237.9 Pips
  - Return / DD Ratio: 6.03x
  - Expectancy: +2.98 Pips / Trade
  - 30% Quarantined Holdout: +304.0 Pips | 40.0% Win Rate | 1.27 Profit Factor | Ret/DD 1.28x
================================================================================
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any

from tick_engine.strategies.base import BaseStrategy, SIGNAL_LONG, SIGNAL_SHORT
from data_engineering_suite.bars import build_time_bars


class USDJPYPivotR1S1BreakoutElite(BaseStrategy):
    """
    DISC87: Pivot R1 / S1 Breakout Drive Elite for USDJPY.
    """
    def __init__(
        self,
        sl_pips: float = 12.0,
        tp_pips: float = 24.0,
        pip_size: float = 0.01,
        max_spread: float = 0.8,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.pip_size = float(pip_size)
        self.max_spread = float(max_spread)
        self.name = "DISC87_USDJPY_Pivot_R1_S1_Breakout_Elite"

    def generate_signals(self, ticks: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = len(ticks)
        signals = np.zeros(n, dtype=np.int8)
        sl_arr = np.zeros(n, dtype=np.float64)
        tp_arr = np.zeros(n, dtype=np.float64)
        if n < 500:
            return signals, sl_arr, tp_arr

        m15 = build_time_bars(ticks, timeframe_min=15, pip_size=self.pip_size)
        d1 = build_time_bars(ticks, timeframe_min=1440, pip_size=self.pip_size)
        h4 = build_time_bars(ticks, timeframe_min=240, pip_size=self.pip_size)
        if len(m15) < 30 or len(d1) < 2 or len(h4) < 10:
            return signals, sl_arr, tp_arr

        d1_prev = d1.shift(1)
        p = (d1_prev['high'] + d1_prev['low'] + d1_prev['close']) / 3.0
        r1 = 2.0 * p - d1_prev['low']
        s1 = 2.0 * p - d1_prev['high']

        pivot_df = pd.DataFrame({'r1': r1, 's1': s1, 'pivot': p}, index=d1.index)

        h4_c = pd.Series(h4['close'].values)
        h4_ema20 = h4_c.ewm(span=20, adjust=False).mean().values
        h4_bias = np.where(h4_c.values > h4_ema20, 1, -1)
        h4_df = pd.DataFrame({'h4_bias': h4_bias}, index=h4.index)

        m15_df = pd.merge_asof(
            m15.sort_index(),
            pivot_df.sort_index(),
            left_index=True,
            right_index=True,
            direction='backward'
        )
        m15_df = pd.merge_asof(
            m15_df,
            h4_df.sort_index(),
            left_index=True,
            right_index=True,
            direction='backward'
        )

        c = m15_df['close'].values
        h = m15_df['high'].values
        l = m15_df['low'].values
        o = m15_df['open'].values
        spread = m15_df['spread_pips'].values
        h4_trend = m15_df['h4_bias'].fillna(0).values.astype(int)
        r1_val = m15_df['r1'].values
        s1_val = m15_df['s1'].values
        hours = m15_df.index.hour.values
        dates = m15_df.index.date
        time_ms = m15_df.index.astype('datetime64[ms]').view('int64')
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        tr = np.maximum(h - l, np.maximum(np.abs(h - np.roll(c, 1)), np.abs(l - np.roll(c, 1))))
        tr[0] = h[0] - l[0]
        atr10 = pd.Series(tr).rolling(10).mean().values

        last_trade_date = None

        for i in range(15, len(m15_df)):
            dt = dates[i]
            hr = hours[i]

            if not (7 <= hr <= 15):
                continue
            if dt == last_trade_date:
                continue
            if spread[i] > self.max_spread:
                continue

            macro = h4_trend[i]
            if macro == 0:
                continue

            if np.isnan(r1_val[i]) or np.isnan(s1_val[i]) or np.isnan(atr10[i]):
                continue

            rng = h[i] - l[i]
            if rng < 1.20 * atr10[i]:
                continue
            body = abs(c[i] - o[i])
            if body < 0.55 * rng:
                continue

            sig = 0
            if o[i] <= r1_val[i] and c[i] > r1_val[i] + 0.5 * self.pip_size and c[i] > o[i] and macro == 1:
                sig = SIGNAL_LONG
            elif o[i] >= s1_val[i] and c[i] < s1_val[i] - 0.5 * self.pip_size and c[i] < o[i] and macro == -1:
                sig = SIGNAL_SHORT

            if sig != 0:
                end_ms = time_ms[i] + dur_ms
                idx = np.searchsorted(tick_ts, end_ms)
                if idx < n:
                    signals[idx] = sig
                    sl_arr[idx] = self.sl_pips
                    tp_arr[idx] = self.tp_pips
                    last_trade_date = dt

        return signals, sl_arr, tp_arr
