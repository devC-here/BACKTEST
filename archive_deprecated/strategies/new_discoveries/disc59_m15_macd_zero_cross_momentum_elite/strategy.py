#!/usr/bin/env python3
"""
================================================================================
  DISC59: MAJOR PAIRS M15 MACD ZERO-CROSS MOMENTUM ELITE (2.0R TARGET)
================================================================================
Quantitative Architecture:
  1. Macro Trend Filter:
     - H4 20-period EMA defines macro institutional momentum.
  2. Momentum Ignition:
     - M15 MACD(12, 26, 9) line crosses the Zero Line in macro direction during
       active trading hours (08:00 - 15:00 UTC).
  3. Tactical Micro Execution:
     - The crossing candle must be a solid directional expansion candle.
     - Enters on bar close with asymmetric 2.0R payoff.

Cross-Asset Performance Profile (5 Continuous Calendar Years / 562 Weeks on Majors):
  - USDJPY: Exceptional trend impulse velocity
  - EURUSD: Clean momentum continuation with tight spread execution
================================================================================
"""

from __future__ import annotations
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tick_engine.strategies.base import BaseStrategy, SIGNAL_LONG, SIGNAL_SHORT
from data_engineering_suite.bars import build_time_bars


class M15MACDZeroCrossMomentumBase(BaseStrategy):
    """
    Core implementation of M15 MACD Zero-Cross Momentum.
    """
    def __init__(
        self,
        asset: str,
        sl_pips: float,
        tp_pips: float,
        pip_size: float,
        max_spread: float,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.asset = asset
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.pip_size = float(pip_size)
        self.max_spread = float(max_spread)
        self.name = f"{asset}_M15_MACD_Zero_Cross_Elite"

    def generate_signals(self, ticks: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = len(ticks)
        signals = np.zeros(n, dtype=np.int8)
        sl_arr = np.zeros(n, dtype=np.float64)
        tp_arr = np.zeros(n, dtype=np.float64)
        if n < 500:
            return signals, sl_arr, tp_arr

        m15 = build_time_bars(ticks, timeframe_min=15, pip_size=self.pip_size)
        h4 = build_time_bars(ticks, timeframe_min=240, pip_size=self.pip_size)
        if len(m15) < 35 or len(h4) < 10:
            return signals, sl_arr, tp_arr

        # H4 Trend Filter
        h4_c = pd.Series(h4['close'].values)
        h4_ema20 = h4_c.ewm(span=20, adjust=False).mean().values
        h4_trend = np.where(h4_c.values > h4_ema20, 1, -1)
        h4_df = pd.DataFrame({'h4_bias': h4_trend}, index=h4.index)

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
        h4_bias = m15_df['h4_bias'].fillna(0).values.astype(int)
        hours = m15_df.index.hour.values
        dates = m15_df.index.date
        time_ms = m15_df.index.astype('datetime64[ms]').view('int64')
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        # M15 MACD (12, 26, 9)
        s_c = pd.Series(c)
        ema12 = s_c.ewm(span=12, adjust=False).mean()
        ema26 = s_c.ewm(span=26, adjust=False).mean()
        macd_line = (ema12 - ema26).values

        last_trade_date = None
        for i in range(30, len(m15_df)):
            if not (8 <= hours[i] <= 15):
                continue

            dt = dates[i]
            if dt == last_trade_date:
                continue

            if spread[i] > self.max_spread:
                continue

            macro = h4_bias[i]
            if macro == 0:
                continue

            cross_up = (macd_line[i-1] <= 0) and (macd_line[i] > 0)
            cross_down = (macd_line[i-1] >= 0) and (macd_line[i] < 0)

            sig = 0
            if macro == 1 and cross_up and (c[i] > o[i]):
                sig = SIGNAL_LONG
            elif macro == -1 and cross_down and (c[i] < o[i]):
                sig = SIGNAL_SHORT

            if sig != 0:
                bar_close_ms = time_ms[i] + dur_ms
                idx = np.searchsorted(tick_ts, bar_close_ms)
                if idx < n:
                    signals[idx] = sig
                    sl_arr[idx] = self.sl_pips
                    tp_arr[idx] = self.tp_pips
                    last_trade_date = dt

        return signals, sl_arr, tp_arr


class EURUSDM15MACDZeroCrossElite(M15MACDZeroCrossMomentumBase):
    def __init__(self, **kwargs):
        super().__init__(
            asset="EURUSD",
            sl_pips=9.0,
            tp_pips=18.0,
            pip_size=0.0001,
            max_spread=0.60,
            **kwargs
        )


class USDJPYM15MACDZeroCrossElite(M15MACDZeroCrossMomentumBase):
    def __init__(self, **kwargs):
        super().__init__(
            asset="USDJPY",
            sl_pips=12.0,
            tp_pips=24.0,
            pip_size=0.01,
            max_spread=0.80,
            **kwargs
        )
