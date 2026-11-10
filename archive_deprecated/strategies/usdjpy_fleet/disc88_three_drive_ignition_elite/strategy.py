#!/usr/bin/env python3
"""
================================================================================
  DISC88: Consecutive Three-Drive Ignition Elite (USDJPY Production Model)
================================================================================
Discovered in Alpha Discovery Lab V26; audited across 300 continuous weeks
(192.32M authentic Dukascopy ticks across 2021-2026) under strict 70/30 randomized
week partitioning (Seed 42).

Strategy Logic:
  1. Identifies 3 consecutive M15 candles in the same direction:
     - Bullish Three-Drive: c[i] > c[i-1] > c[i-2] with all 3 bars closing green.
     - Bearish Three-Drive: c[i] < c[i-1] < c[i-2] with all 3 bars closing red.
  2. Ignition Bar Condition: 3rd bar is an expansion impulse candle:
     Range >= 1.25x ATR10, Body >= 55%.
  3. Higher Timeframe Alignment: Strictly follows H4 20-EMA trend.
  4. Active Hours: 07:00 - 17:00 UTC (European London + NY Active Sessions).
  5. 2.0R Asymmetric Target: SL 12.0 pips, TP 24.0 pips.

Institutional Audit Metrics (USDJPY 5-Year Full Cycle / 300 Weeks):
  - Total Trades: 858
  - Win Rate: 38.3%
  - Profit Factor: 1.22
  - Net Pips: +1,394.2 Pips
  - Max Drawdown: 213.7 Pips
  - Return / DD Ratio: 6.52x
  - Expectancy: +1.62 Pips / Trade
  - 30% Quarantined Holdout: +459.7 Pips | 39.0% Win Rate | 1.23 Profit Factor | Ret/DD 3.02x
================================================================================
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any

from tick_engine.strategies.base import BaseStrategy, SIGNAL_LONG, SIGNAL_SHORT
from data_engineering_suite.bars import build_time_bars


class USDJPYThreeDriveIgnitionElite(BaseStrategy):
    """
    DISC88: Consecutive Three-Drive Ignition Elite for USDJPY.
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
        self.name = "DISC88_USDJPY_Three_Drive_Ignition_Elite"

    def generate_signals(self, ticks: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = len(ticks)
        signals = np.zeros(n, dtype=np.int8)
        sl_arr = np.zeros(n, dtype=np.float64)
        tp_arr = np.zeros(n, dtype=np.float64)
        if n < 500:
            return signals, sl_arr, tp_arr

        m15 = build_time_bars(ticks, timeframe_min=15, pip_size=self.pip_size)
        h4 = build_time_bars(ticks, timeframe_min=240, pip_size=self.pip_size)
        if len(m15) < 30 or len(h4) < 10:
            return signals, sl_arr, tp_arr

        h4_c = pd.Series(h4['close'].values)
        h4_ema20 = h4_c.ewm(span=20, adjust=False).mean().values
        h4_bias = np.where(h4_c.values > h4_ema20, 1, -1)
        h4_df = pd.DataFrame({'h4_bias': h4_bias}, index=h4.index)

        m15_df = pd.merge_asof(
            m15.sort_index(),
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

            if not (7 <= hr <= 17):
                continue
            if dt == last_trade_date:
                continue
            if spread[i] > self.max_spread:
                continue

            macro = h4_trend[i]
            if macro == 0:
                continue

            if np.isnan(atr10[i]) or atr10[i] <= 0:
                continue

            rng3 = h[i] - l[i]
            if rng3 < 1.25 * atr10[i]:
                continue
            body3 = abs(c[i] - o[i])
            if body3 < 0.55 * rng3:
                continue

            sig = 0
            if (c[i] > c[i-1] > c[i-2]) and (c[i] > o[i] and c[i-1] > o[i-1] and c[i-2] > o[i-2]) and macro == 1:
                sig = SIGNAL_LONG
            elif (c[i] < c[i-1] < c[i-2]) and (c[i] < o[i] and c[i-1] < o[i-1] and c[i-2] < o[i-2]) and macro == -1:
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
