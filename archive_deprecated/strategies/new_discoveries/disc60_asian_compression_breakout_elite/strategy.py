#!/usr/bin/env python3
"""
================================================================================
  DISC60: MULTI-ASSET ASIAN COMPRESSION BREAKOUT ELITE (2.0R TARGET)
================================================================================
Quantitative Architecture:
  1. Asian Box Compression Measurement:
     - Measures Tokyo session range (00:00 - 07:00 UTC).
     - Identifies compressed consolidation regimes where total range <= max_asia_pips:
       * EURUSD: <= 18.0 pips
       * USDJPY: <= 22.0 pips
       * EURGBP: <= 14.0 pips
  2. Macro Trend Alignment:
     - H4 20-period EMA dictates expansion direction.
  3. Tactical Micro Execution:
     - During London Opening (07:00 - 10:00 UTC), first M15 candle closing decisively
       beyond the Asian box in H4 trend direction triggers breakout.
     - Asymmetric 2.0R payoff.
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


class AsianCompressionBreakoutBase(BaseStrategy):
    """
    Core implementation of Asian Compression Breakout.
    """
    def __init__(
        self,
        asset: str,
        sl_pips: float,
        tp_pips: float,
        pip_size: float,
        max_spread: float,
        max_asia_pips: float,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.asset = asset
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.pip_size = float(pip_size)
        self.max_spread = float(max_spread)
        self.max_asia_pips = float(max_asia_pips)
        self.name = f"{asset}_Asian_Compression_Breakout_Elite"

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

        # H4 trend
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

        # Calculate Asian ranges (00:00 - 07:00 UTC)
        unique_dates = np.unique(dates)
        asia_ranges = {}
        for ud in unique_dates:
            mask = (dates == ud) & (hours < 7)
            if np.any(mask):
                a_hi = np.max(h[mask])
                a_lo = np.min(l[mask])
                rng_pips = (a_hi - a_lo) / self.pip_size
                if rng_pips <= self.max_asia_pips and rng_pips >= 5.0:
                    asia_ranges[ud] = (a_hi, a_lo)

        last_trade_date = None
        for i in range(20, len(m15_df)):
            if not (7 <= hours[i] <= 10):
                continue

            dt = dates[i]
            if dt == last_trade_date:
                continue

            if dt not in asia_ranges:
                continue

            if spread[i] > self.max_spread:
                continue

            macro = h4_bias[i]
            a_hi, a_lo = asia_ranges[dt]

            sig = 0
            if macro == 1 and (c[i] > a_hi) and (c[i] > o[i]):
                sig = SIGNAL_LONG
            elif macro == -1 and (c[i] < a_lo) and (c[i] < o[i]):
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


class EURUSDAsianCompressionBreakoutElite(AsianCompressionBreakoutBase):
    def __init__(self, **kwargs):
        super().__init__(
            asset="EURUSD",
            sl_pips=9.0,
            tp_pips=18.0,
            pip_size=0.0001,
            max_spread=0.60,
            max_asia_pips=18.0,
            **kwargs
        )


class USDJPYAsianCompressionBreakoutElite(AsianCompressionBreakoutBase):
    def __init__(self, **kwargs):
        super().__init__(
            asset="USDJPY",
            sl_pips=12.0,
            tp_pips=24.0,
            pip_size=0.01,
            max_spread=0.80,
            max_asia_pips=22.0,
            **kwargs
        )


class EURGBPAsianCompressionBreakoutElite(AsianCompressionBreakoutBase):
    def __init__(self, **kwargs):
        super().__init__(
            asset="EURGBP",
            sl_pips=8.0,
            tp_pips=16.0,
            pip_size=0.0001,
            max_spread=0.80,
            max_asia_pips=14.0,
            **kwargs
        )
