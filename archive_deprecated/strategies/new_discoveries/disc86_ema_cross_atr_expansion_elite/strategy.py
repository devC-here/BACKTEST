#!/usr/bin/env python3
"""
================================================================================
  DISC86: EMA Cross ATR Band Expansion Elite (Multi-Asset Incubator)
================================================================================
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any

from tick_engine.strategies.base import BaseStrategy, SIGNAL_LONG, SIGNAL_SHORT
from data_engineering_suite.bars import build_time_bars


class DISC86EMACrossATRExpansionElite(BaseStrategy):
    """
    DISC86: EMA Cross ATR Band Expansion Elite for Multi-Asset Incubator.
    """
    def __init__(
        self,
        asset: str = "EURUSD",
        sl_pips: float | None = None,
        tp_pips: float | None = None,
        pip_size: float | None = None,
        max_spread: float | None = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.asset = asset.upper()
        if self.asset == "USDJPY":
            self.sl_pips = float(sl_pips if sl_pips is not None else 12.0)
            self.tp_pips = float(tp_pips if tp_pips is not None else 24.0)
            self.pip_size = float(pip_size if pip_size is not None else 0.01)
            self.max_spread = float(max_spread if max_spread is not None else 0.8)
        else:
            self.sl_pips = float(sl_pips if sl_pips is not None else 8.5)
            self.tp_pips = float(tp_pips if tp_pips is not None else 17.0)
            self.pip_size = float(pip_size if pip_size is not None else 0.0001)
            self.max_spread = float(max_spread if max_spread is not None else 0.6)
        self.name = f"DISC86_{self.asset}_EMA_Cross_ATR_Expansion_Elite"

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

        c_ser = pd.Series(c)
        ema9 = c_ser.ewm(span=9, adjust=False).mean().values
        ema21 = c_ser.ewm(span=21, adjust=False).mean().values

        tr = np.maximum(h - l, np.maximum(np.abs(h - np.roll(c, 1)), np.abs(l - np.roll(c, 1))))
        tr[0] = h[0] - l[0]
        atr10 = pd.Series(tr).rolling(10).mean().values

        last_trade_date = None

        for i in range(25, len(m15_df)):
            dt = dates[i]
            hr = hours[i]

            if not (7 <= hr <= 16):
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

            bull_cross = (ema9[i] > ema21[i]) and (ema9[i-1] <= ema21[i-1] or ema9[i-2] <= ema21[i-2])
            bear_cross = (ema9[i] < ema21[i]) and (ema9[i-1] >= ema21[i-1] or ema9[i-2] >= ema21[i-2])

            sig = 0
            if bull_cross and c[i] > ema21[i] + 0.8 * atr10[i] and c[i] > o[i] and macro == 1:
                sig = SIGNAL_LONG
            elif bear_cross and c[i] < ema21[i] - 0.8 * atr10[i] and c[i] < o[i] and macro == -1:
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
