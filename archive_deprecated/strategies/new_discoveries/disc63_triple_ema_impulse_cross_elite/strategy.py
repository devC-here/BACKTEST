#!/usr/bin/env python3
"""
================================================================================
  DISC63: M15 Triple EMA Trend Impulse Cross Elite (2.0R Target)
================================================================================
Discovered in Alpha Discovery Lab V15; validated across 5 continuous years
(862 total weeks / 419M ticks) across EURUSD, USDJPY, and EURGBP on authentic
Dukascopy tick data with floating spread and slippage.

Quantitative Microstructure Edge:
  - Architecture: Triple Exponential Moving Average Trend Impulse (EMA 9, 21, 55).
  - Volatility Expansion Trigger: Candle range (High - Low) >= 0.8x ATR(14) to eliminate
    low-momentum chop and false crosses.
  - Impulse Cross Conditions:
      * Bullish Impulse: EMA 9 crosses above EMA 21 while both are above EMA 55.
      * Bearish Impulse: EMA 9 crosses below EMA 21 while both are below EMA 55.
  - Active Hours: 07:00 - 16:00 UTC (European open into US afternoon session).
  - Risk Structure: Asymmetric 1:2.0 Risk-to-Reward Ratio:
      * EURUSD: SL 9.0 pips | TP 18.0 pips | Max Spread 0.6 pips.
      * USDJPY: SL 12.0 pips | TP 24.0 pips | Max Spread 0.8 pips.
      * EURGBP: SL 8.0 pips | TP 16.0 pips | Max Spread 0.8 pips.
================================================================================
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from tick_engine.strategies.base import BaseStrategy, SIGNAL_LONG, SIGNAL_SHORT
from data_engineering_suite.bars import build_time_bars


class TripleEMATrendImpulseBase(BaseStrategy):
    """
    Base quantitative engine for M15 Triple EMA Trend Impulse Cross Elite.
    """
    def __init__(
        self,
        asset: str,
        sl_pips: float,
        tp_pips: float,
        pip_size: float,
        max_spread: float = 0.8,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.asset = asset
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.pip_size = float(pip_size)
        self.max_spread = float(max_spread)
        self.name = f"{asset}_Triple_EMA_Cross_Elite"

    def generate_signals(self, ticks: np.ndarray):
        n = len(ticks)
        signals = np.zeros(n, dtype=np.int8)
        sl_arr = np.zeros(n, dtype=np.float64)
        tp_arr = np.zeros(n, dtype=np.float64)
        if n < 500:
            return signals, sl_arr, tp_arr

        m15 = build_time_bars(ticks, timeframe_min=15, pip_size=self.pip_size)
        if len(m15) < 60:
            return signals, sl_arr, tp_arr

        c = m15['close'].values
        h = m15['high'].values
        l = m15['low'].values
        spread = m15['spread_pips'].values
        hours = m15.index.hour.values
        dates = m15.index.date
        time_ms = m15.index.astype('datetime64[ms]').view('int64')
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        c_s = pd.Series(c)
        ema9 = c_s.ewm(span=9, adjust=False).mean().values
        ema21 = c_s.ewm(span=21, adjust=False).mean().values
        ema55 = c_s.ewm(span=55, adjust=False).mean().values

        # ATR 14 calculation
        tr = np.maximum(h[1:] - l[1:], np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))
        tr = np.insert(tr, 0, h[0] - l[0])
        atr = pd.Series(tr).rolling(14).mean().values

        last_trade_date = None
        for i in range(56, len(m15)):
            if not (7 <= hours[i] <= 16):
                continue
            dt = dates[i]
            if dt == last_trade_date:
                continue
            if spread[i] > self.max_spread:
                continue

            bar_range = h[i] - l[i]
            if np.isnan(atr[i]) or bar_range < 0.8 * atr[i]:
                continue

            sig = 0
            # Bullish cross: EMA 9 crosses above EMA 21, both above EMA 55
            if ema9[i-1] <= ema21[i-1] and ema9[i] > ema21[i] and ema21[i] > ema55[i]:
                sig = SIGNAL_LONG
            # Bearish cross: EMA 9 crosses below EMA 21, both below EMA 55
            elif ema9[i-1] >= ema21[i-1] and ema9[i] < ema21[i] and ema21[i] < ema55[i]:
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


class EURUSD_TripleEMAImpulseCrossElite(TripleEMATrendImpulseBase):
    """
    EURUSD Triple EMA Trend Impulse Cross Elite (2.0R Target).
    """
    def __init__(self, **kwargs):
        super().__init__(
            asset="EURUSD",
            sl_pips=9.0,
            tp_pips=18.0,
            pip_size=0.0001,
            max_spread=0.6,
            **kwargs
        )


class USDJPY_TripleEMAImpulseCrossElite(TripleEMATrendImpulseBase):
    """
    USDJPY Triple EMA Trend Impulse Cross Elite (2.0R Target).
    """
    def __init__(self, **kwargs):
        super().__init__(
            asset="USDJPY",
            sl_pips=12.0,
            tp_pips=24.0,
            pip_size=0.01,
            max_spread=0.8,
            **kwargs
        )


class EURGBP_TripleEMAImpulseCrossElite(TripleEMATrendImpulseBase):
    """
    EURGBP Triple EMA Trend Impulse Cross Elite (2.0R Target).
    """
    def __init__(self, **kwargs):
        super().__init__(
            asset="EURGBP",
            sl_pips=8.0,
            tp_pips=16.0,
            pip_size=0.0001,
            max_spread=0.8,
            **kwargs
        )
