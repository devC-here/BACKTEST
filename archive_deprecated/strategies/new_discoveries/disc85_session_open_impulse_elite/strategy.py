#!/usr/bin/env python3
"""
================================================================================
  DISC85: London Session Open Benchmark Impulse Elite (Multi-Asset Incubator)
================================================================================
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any

from tick_engine.strategies.base import BaseStrategy, SIGNAL_LONG, SIGNAL_SHORT
from data_engineering_suite.bars import build_time_bars


class DISC85SessionOpenImpulseElite(BaseStrategy):
    """
    DISC85: London Session Open Benchmark Impulse Elite for Multi-Asset Incubator.
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
        self.name = f"DISC85_{self.asset}_Session_Open_Impulse_Elite"

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
        minutes = m15_df.index.minute.values
        dates = m15_df.index.date
        time_ms = m15_df.index.astype('datetime64[ms]').view('int64')
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        tr = np.maximum(h - l, np.maximum(np.abs(h - np.roll(c, 1)), np.abs(l - np.roll(c, 1))))
        tr[0] = h[0] - l[0]
        atr10 = pd.Series(tr).rolling(10).mean().values

        session_open_price: Dict[Any, float] = {}
        for idx_bar in range(len(m15_df)):
            dt = dates[idx_bar]
            hr = hours[idx_bar]
            mn = minutes[idx_bar]
            if hr == 7 and mn == 0 and dt not in session_open_price:
                session_open_price[dt] = o[idx_bar]

        last_trade_date = None

        for i in range(10, len(m15_df)):
            dt = dates[i]
            hr = hours[i]

            if not ((hr == 7 and minutes[i] > 0) or (8 <= hr <= 11)):
                continue
            if dt == last_trade_date:
                continue
            if spread[i] > self.max_spread:
                continue

            open_p = session_open_price.get(dt)
            if open_p is None:
                continue

            macro = h4_trend[i]
            if macro == 0:
                continue

            rng = h[i] - l[i]
            if np.isnan(atr10[i]) or atr10[i] <= 0:
                continue
            if rng < 1.20 * atr10[i]:
                continue

            body = abs(c[i] - o[i])
            if body < 0.50 * rng:
                continue

            sig = 0
            if c[i] > o[i] and l[i] >= open_p and c[i] > np.max(h[i-3:i]) and macro == 1:
                sig = SIGNAL_LONG
            elif c[i] < o[i] and h[i] <= open_p and c[i] < np.min(l[i-3:i]) and macro == -1:
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
