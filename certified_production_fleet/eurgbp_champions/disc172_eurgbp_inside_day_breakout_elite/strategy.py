#!/usr/bin/env python3
"""
================================================================================
  disc172: EURGBP Inside Day Compression Breakout Elite (2.0R Target)
================================================================================
Discovered in Alpha Discovery Lab V23; validated across 5.75 continuous years
(300 continuous weeks) on authentic EURGBP Dukascopy tick data under live ECN friction.

Quantitative Architecture:
  - Multi-Day Compression Anchor: Detects prior-day Inside Bar (H_d-1 < H_d-2 and L_d-1 > L_d-2),
    identifying extreme volatility compression across consecutive calendar days.
  - Active Session Window: 07:00 to 16:00 UTC (European Frankfurt/London/NY cash session).
  - Microstructure Setup:
      - Breakout Buffer: M15 candle closing >= 0.5 pips beyond previous day's high or low.
      - Candle Body Ratio: M15 breakout bar must have body >= 45% of total bar range.
      - Bullish: Green expansion bar closing above prior-day high + 0.5p.
      - Bearish: Red expansion bar closing below prior-day low - 0.5p.
  - Risk Setup: Asymmetric 1:2.0 Risk-to-Reward (SL 10.0 pips | TP 20.0 pips | Max Spread 0.8 pips).
  - Frequency: Maximum 1 trade per calendar day.
================================================================================
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from tick_engine.strategies.base import BaseStrategy, SIGNAL_LONG, SIGNAL_SHORT
from data_engineering_suite.bars import build_time_bars


class EURGBPInsideDayBreakoutElite(BaseStrategy):
    """
    disc172: EURGBP Inside Day Compression Breakout Elite Model.
    """
    def __init__(
        self,
        asset: str = "EURGBP",
        sl_pips: float = 10.0,
        tp_pips: float = 20.0,
        buffer_pips: float = 0.5,
        body_ratio: float = 0.45,
        pip_size: float = 0.0001,
        max_spread: float = 0.8,
        start_hr: int = 7,
        end_hr: int = 16,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.asset = asset
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.buffer_pips = float(buffer_pips)
        self.body_ratio = float(body_ratio)
        self.pip_size = float(pip_size)
        self.max_spread = float(max_spread)
        self.start_hr = int(start_hr)
        self.end_hr = int(end_hr)
        self.name = "EURGBPInsideDayBreakoutElite"

    def generate_signals(self, ticks: np.ndarray):
        n = len(ticks)
        signals = np.zeros(n, dtype=np.int8)
        sl_arr = np.zeros(n, dtype=np.float64)
        tp_arr = np.zeros(n, dtype=np.float64)
        if n < 500:
            return signals, sl_arr, tp_arr

        m15 = build_time_bars(ticks, timeframe_min=15, pip_size=self.pip_size)
        if len(m15) < 35:
            return signals, sl_arr, tp_arr

        c = m15['close'].values
        h = m15['high'].values
        l = m15['low'].values
        o = m15['open'].values
        spread = m15['spread_pips'].values
        hours = m15.index.hour.values
        dates = m15.index.date
        time_ms = m15.index.astype('datetime64[ms]').view('int64')
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        # Daily bar statistics from completed daily M15 bars
        unique_dates = np.unique(dates)
        daily_stats = {}
        for d in unique_dates:
            mask = (dates == d)
            if np.sum(mask) >= 15:
                dh = float(np.max(h[mask]))
                dl = float(np.min(l[mask]))
                daily_stats[d] = (dh, dl)

        # Track Inside Day status
        is_inside_day = {}
        prev_dh = {}
        prev_dl = {}
        for idx_d in range(2, len(unique_dates)):
            d_cur = unique_dates[idx_d]
            d_p1 = unique_dates[idx_d - 1]
            d_p2 = unique_dates[idx_d - 2]
            if d_p1 in daily_stats and d_p2 in daily_stats:
                h1, l1 = daily_stats[d_p1]
                h2, l2 = daily_stats[d_p2]
                if h1 < h2 and l1 > l2:
                    is_inside_day[d_cur] = True
                    prev_dh[d_cur] = h1
                    prev_dl[d_cur] = l1

        buf = self.buffer_pips * self.pip_size
        last_trade_date = None

        for i in range(15, len(m15)):
            dt = dates[i]
            hr = hours[i]

            if not (self.start_hr <= hr <= self.end_hr):
                continue
            if dt == last_trade_date:
                continue
            if dt not in is_inside_day:
                continue
            if spread[i] > self.max_spread:
                continue

            bar_rng = h[i] - l[i]
            if bar_rng < 1.5 * self.pip_size:
                continue

            body = abs(c[i] - o[i])
            if (body / bar_rng) < self.body_ratio:
                continue

            pd_h = prev_dh[dt]
            pd_l = prev_dl[dt]

            sig = 0
            if c[i] > pd_h + buf and c[i] > o[i]:
                sig = SIGNAL_LONG
            elif c[i] < pd_l - buf and c[i] < o[i]:
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
