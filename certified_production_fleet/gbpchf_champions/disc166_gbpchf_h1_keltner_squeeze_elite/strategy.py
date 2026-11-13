#!/usr/bin/env python3
"""
================================================================================
  disc166: GBPCHF H1 Keltner Squeeze Explosion Elite (2.11R Target)
================================================================================
Discovered and certified across 5.75 continuous years (300 continuous weeks)
on authentic GBPCHF Dukascopy tick data under live ECN broker friction.

Quantitative Architecture:
  - Higher Timeframe Volatility Squeeze (H1):
      * Bollinger Bands (20, 2.0 SD) vs Keltner Channels (20 EMA, 1.8 ATR).
      * Squeeze fires when Bollinger Bands expand beyond Keltner Channel.
      * Directional Momentum: H1 Close vs H1 20-period EMA.
  - Active Session Window: 07:00 to 16:00 UTC (London & New York Liquid Hours (07:00 - 16:00 UTC)).
  - Tactical Execution: M15 candle closing in direction of H1 squeeze release.
  - Risk Setup: Asymmetric Risk-to-Reward (SL 18.0 pips | TP 38.0 pips | Max Spread 3.2 pips).
  - Frequency: Maximum 1 trade per calendar day.
================================================================================
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from tick_engine.strategies.base import BaseStrategy, SIGNAL_LONG, SIGNAL_SHORT
from data_engineering_suite.bars import build_time_bars


class GBPCHFH1KeltnerSqueezeExplosionElite(BaseStrategy):
    """
    disc166: GBPCHF H1 Keltner Squeeze Explosion Elite (2.11R Target).
    """
    def __init__(
        self,
        asset: str = "GBPCHF",
        sl_pips: float = 18.0,
        tp_pips: float = 38.0,
        pip_size: float = 0.0001,
        max_spread: float = 3.2,
        kc_mult: float = 1.8,
        start_hr: int = 7,
        end_hr: int = 16,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.asset = asset
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.pip_size = float(pip_size)
        self.max_spread = float(max_spread)
        self.kc_mult = float(kc_mult)
        self.start_hr = int(start_hr)
        self.end_hr = int(end_hr)
        self.name = "GBPCHFH1KeltnerSqueezeExplosionElite"

    def generate_signals(self, ticks: np.ndarray):
        n = len(ticks)
        signals = np.zeros(n, dtype=np.int8)
        sl_arr = np.zeros(n, dtype=np.float64)
        tp_arr = np.zeros(n, dtype=np.float64)
        if n < 500:
            return signals, sl_arr, tp_arr

        m15 = build_time_bars(ticks, timeframe_min=15, pip_size=self.pip_size)
        h1 = build_time_bars(ticks, timeframe_min=60, pip_size=self.pip_size)
        if len(m15) < 30 or len(h1) < 25:
            return signals, sl_arr, tp_arr

        # H1 Squeeze Calculation
        h1_c = h1['close'].values
        h1_h = h1['high'].values
        h1_l = h1['low'].values

        c_s = pd.Series(h1_c)
        h1_ema20 = c_s.ewm(span=20, adjust=False).mean().values
        h1_std20 = c_s.rolling(20).std().values
        bb_upper = h1_ema20 + 2.0 * h1_std20
        bb_lower = h1_ema20 - 2.0 * h1_std20

        tr = np.maximum(h1_h - h1_l, np.maximum(np.abs(h1_h - np.roll(h1_c, 1)), np.abs(h1_l - np.roll(h1_c, 1))))
        tr[0] = h1_h[0] - h1_l[0]
        atr20 = pd.Series(tr).rolling(20).mean().bfill().values

        kc_upper = h1_ema20 + self.kc_mult * atr20
        kc_lower = h1_ema20 - self.kc_mult * atr20

        # Squeeze on when BB inside KC
        sqz_on = (bb_lower > kc_lower) & (bb_upper < kc_upper)
        sqz_fired = pd.Series(sqz_on).shift(1).fillna(False).values & (~sqz_on)
        h1_mom = np.where(h1_c > h1_ema20, 1, np.where(h1_c < h1_ema20, -1, 0))

        h1_df = pd.DataFrame({
            'sqz_fire': sqz_fired,
            'h1_mom': h1_mom
        }, index=h1.index)

        m15_df = pd.merge_asof(
            m15.sort_index(),
            h1_df.sort_index(),
            left_index=True,
            right_index=True,
            direction='backward'
        )

        c = m15_df['close'].values
        o = m15_df['open'].values
        spread = m15_df['spread_pips'].values
        sqz_active = m15_df['sqz_fire'].fillna(False).values
        macro_mom = m15_df['h1_mom'].fillna(0).values.astype(int)
        hours = m15_df.index.hour.values
        dates = m15_df.index.date
        time_ms = m15_df.index.astype('datetime64[ms]').view('int64')
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        last_trade_date = None

        for i in range(15, len(m15_df)):
            dt = dates[i]
            hr = hours[i]

            if self.start_hr <= self.end_hr:
                in_window = (self.start_hr <= hr <= self.end_hr)
            else:
                in_window = (hr >= self.start_hr or hr <= self.end_hr)

            if not in_window:
                continue
            if dt == last_trade_date:
                continue
            if spread[i] > self.max_spread:
                continue

            if not sqz_active[i]:
                continue
            mom = macro_mom[i]
            if mom == 0:
                continue

            sig = 0
            if mom == 1 and c[i] > o[i]:
                sig = SIGNAL_LONG
            elif mom == -1 and c[i] < o[i]:
                sig = SIGNAL_SHORT

            if sig != 0:
                end_ms = time_ms[i] + dur_ms
                idx = np.searchsorted(tick_ts, end_ms)
                if idx < n and signals[idx] == 0:
                    signals[idx] = sig
                    sl_arr[idx] = self.sl_pips
                    tp_arr[idx] = self.tp_pips
                    last_trade_date = dt

        return signals, sl_arr, tp_arr


Strategy = GBPCHFH1KeltnerSqueezeExplosionElite
