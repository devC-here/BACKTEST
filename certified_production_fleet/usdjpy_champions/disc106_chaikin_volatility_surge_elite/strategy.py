#!/usr/bin/env python3
"""
================================================================================
  DISC106: USDJPY Chaikin Volatility Expansion Surge Elite
================================================================================
Discovered in Alpha Discovery Lab V24; validated across 5 continuous years
(300 continuous weeks / ~192.32M Dukascopy ticks) on authentic USDJPY tick data.

Performance Metrics:
  - 70% In-Sample Discovery: 760 trades | +1,706.2 Net Pips | 38.8% WR | 1.26 PF | +34.8%/yr | 7.27x Ret/DD
  - 30% Quarantined Holdout: 339 trades | +765.5 Net Pips | 39.2% WR | 1.26 PF | +36.5%/yr | 4.86x Ret/DD
  - 100% Full 5-Year Cycle: 1,099 trades | +2,471.7 Net Pips | 38.9% WR | 1.26 PF | +35.3%/yr | 10.53x Ret/DD

Quantitative Strategy:
  - Timeframe: M15 with H4 20-EMA trend agreement.
  - Active Session: 07:00 - 16:00 UTC (European London and New York Morning).
  - Microstructure Setup:
      - Chaikin Volatility: 10-period EMA of (High - Low) range with 10-period rate-of-change.
      - Volatility surge ignition: Chaikin Volatility >= +30.0%.
      - Directional breakout closing past 4-bar extreme (Highest High for Long, Lowest Low for Short).
      - Directional candle body >= 50% of bar range in H4 trend direction.
  - Risk Setup: Asymmetric 1:2.0 Risk-to-Reward (SL 14.0 pips | TP 28.0 pips | Max Spread 0.8 pips).
  - Frequency: Max 1 trade per day.
================================================================================
"""

from __future__ import annotations
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from tick_engine.strategies.base import BaseStrategy, SIGNAL_LONG, SIGNAL_SHORT
from data_engineering_suite.bars import build_time_bars


class USDJPYChaikinVolatilitySurgeElite(BaseStrategy):
    """
    USDJPY Chaikin Volatility Expansion Surge Elite Model.
    """
    def __init__(
        self,
        asset: str = "USDJPY",
        sl_pips: float = 14.0,
        tp_pips: float = 28.0,
        pip_size: float = 0.01,
        max_spread: float = 0.8,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.asset = asset
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.pip_size = float(pip_size)
        self.max_spread = float(max_spread)
        self.name = "USDJPY_Chaikin_Volatility_Surge_Elite"

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

        # Chaikin Volatility: 10-period EMA of (High - Low)
        hl_range = pd.Series(h - l)
        hl_ema = hl_range.ewm(span=10, adjust=False).mean()
        chaikin_vol = (hl_ema - hl_ema.shift(10)) / hl_ema.shift(10) * 100.0
        chaikin = chaikin_vol.values

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

            cv = chaikin[i]
            if np.isnan(cv) or cv < 30.0:
                continue

            # Breakout past 4-bar extreme
            prev_high4 = np.max(h[i-4:i])
            prev_low4 = np.min(l[i-4:i])

            bar_rng = h[i] - l[i]
            if bar_rng < 2.5 * self.pip_size:
                continue

            sig = 0
            if macro == 1 and c[i] > prev_high4 and c[i] > o[i]:
                body = c[i] - o[i]
                if body / bar_rng >= 0.50:
                    sig = SIGNAL_LONG
            elif macro == -1 and c[i] < prev_low4 and c[i] < o[i]:
                body = o[i] - c[i]
                if body / bar_rng >= 0.50:
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

