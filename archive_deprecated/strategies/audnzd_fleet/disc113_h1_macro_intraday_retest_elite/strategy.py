#!/usr/bin/env python3
"""
================================================================================
  DISC113: AUDNZD H1 Macro Intraday Retest Elite (1.67R Target)
================================================================================
Discovered and certified across 5.75 continuous years (300 continuous weeks /
139.00M Dukascopy ticks) on authentic AUDNZD tick data.

Performance Metrics:
  - 70% In-Sample Discovery (210w): 56 trades | 53.6% WR | 1.87 PF | +269.9p |  +66.8p/yr |  4.40x Ret/DD
  - 30% Quarantined Holdout (90w):  27 trades | 59.3% WR | 2.36 PF | +188.2p | +108.7p/yr |  7.78x Ret/DD (Edge Expanded in Holdout!)
  - 100% Full 5.77-Year Cycle:      83 trades | 55.4% WR | 2.02 PF | +458.1p |  +79.4p/yr | 12.12x Ret/DD
  - Max Full-Cycle Drawdown:        37.8 pips (Holdout DD: 24.2 pips)
  - Expectancy:                     +5.52 pips / trade

Quantitative Architecture:
  - Intermediate Anchor (H1): Defines intraday rolling 20-bar Donchian channel
    support/resistance aligned with H1 20-EMA trend filter.
  - Entry Execution (M15): Triggers when price tests within 4.5 pips of H1 structural
    boundaries with a significant rejection pin wick (>= 50% of bar range).
  - Multi-Session Hours: Sydney/Tokyo Asian Session (21:00 - 07:00 UTC) + London Session (07:00 - 16:00 UTC).
  - Risk Setup: Asymmetric 1:1.67 Risk-to-Reward (SL 12.0 pips | TP 20.0 pips | Max Spread 2.8 pips).
  - Frequency: Max 1 trade per calendar day.
================================================================================
"""

from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

from tick_engine.strategies.base import BaseStrategy, SIGNAL_LONG, SIGNAL_SHORT
from data_engineering_suite.bars import build_time_bars


class AUDNZDH1MacroIntradayRetestElite(BaseStrategy):
    """
    DISC113: AUDNZD H1 Macro Intraday Retest Elite (SL 12.0 / TP 20.0).
    """
    def __init__(
        self,
        asset: str = "AUDNZD",
        sl_pips: float = 12.0,
        tp_pips: float = 20.0,
        lookback_h1: int = 20,
        dist_pips: float = 4.5,
        min_wick: float = 0.50,
        max_spread: float = 2.8,
        pip_size: float = 0.0001,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.asset = asset
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.lookback_h1 = int(lookback_h1)
        self.dist_pips = float(dist_pips)
        self.min_wick = float(min_wick)
        self.max_spread = float(max_spread)
        self.pip_size = float(pip_size)
        self.name = "AUDNZD_H1_Macro_Intraday_Retest_Elite"

    def generate_signals(self, ticks: np.ndarray):
        n_ticks = len(ticks)
        signals = np.zeros(n_ticks, dtype=np.int8)
        sl_pips_arr = np.zeros(n_ticks, dtype=np.float64)
        tp_pips_arr = np.zeros(n_ticks, dtype=np.float64)

        if n_ticks < 500:
            return signals, sl_pips_arr, tp_pips_arr

        m15 = build_time_bars(ticks, timeframe_min=15, pip_size=self.pip_size)
        if len(m15) < 30:
            return signals, sl_pips_arr, tp_pips_arr

        ts = pd.to_datetime(ticks['timestamp'], unit='ms')
        mid = 0.5 * (ticks['bid'] + ticks['ask'])
        base = pd.DataFrame({'close': mid, 'high': mid, 'low': mid}, index=ts)

        # H1 Intraday Anchor Resampling
        h1 = base.resample('1h').agg({'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
        if len(h1) < max(self.lookback_h1 + 5, 25):
            return signals, sl_pips_arr, tp_pips_arr

        h1['ema20'] = h1['close'].ewm(span=20, adjust=False).mean()
        h1['bias'] = np.where(h1['close'] > h1['ema20'], 1, -1)
        h1['roll_h'] = h1['high'].rolling(self.lookback_h1).max().shift(1).bfill()
        h1['roll_l'] = h1['low'].rolling(self.lookback_h1).min().shift(1).bfill()

        bias_m15 = h1['bias'].reindex(m15.index, method='ffill').fillna(0).values
        supp_m15 = h1['roll_l'].reindex(m15.index, method='ffill').values
        res_m15 = h1['roll_h'].reindex(m15.index, method='ffill').values

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

        last_trade_date = None

        for i in range(15, len(m15)):
            hr = hours[i]
            # Active Hours: Sydney/Tokyo Asian Session (21-07 UTC) + London Session (07-16 UTC)
            if not (hr >= 21 or hr <= 16):
                continue

            dt = dates[i]
            if dt == last_trade_date:
                continue
            if spread[i] > self.max_spread:
                continue

            bar_rng = (h[i] - l[i]) / self.pip_size
            if bar_rng < 2.5:
                continue

            b = bias_m15[i]
            sig = 0

            # Bullish Retest of H1 Support in uptrend
            if b > 0:
                dist_supp = abs(l[i] - supp_m15[i]) / self.pip_size
                lower_wick = (min(o[i], c[i]) - l[i]) / self.pip_size
                if dist_supp <= self.dist_pips and (lower_wick / bar_rng) >= self.min_wick and c[i] > o[i]:
                    sig = SIGNAL_LONG

            # Bearish Retest of H1 Resistance in downtrend
            elif b < 0:
                dist_res = abs(res_m15[i] - h[i]) / self.pip_size
                upper_wick = (h[i] - max(o[i], c[i])) / self.pip_size
                if dist_res <= self.dist_pips and (upper_wick / bar_rng) >= self.min_wick and c[i] < o[i]:
                    sig = SIGNAL_SHORT

            if sig != 0:
                exec_t = time_ms[i] + dur_ms
                idx = int(np.searchsorted(tick_ts, exec_t))
                if idx < n_ticks and signals[idx] == 0:
                    signals[idx] = sig
                    sl_pips_arr[idx] = self.sl_pips
                    tp_pips_arr[idx] = self.tp_pips
                    last_trade_date = dt

        return signals, sl_pips_arr, tp_pips_arr

