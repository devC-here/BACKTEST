#!/usr/bin/env python3
"""
================================================================================
  DISC84: London 30M ORB Continuation Elite (Multi-Asset Incubator)
================================================================================
Discovered in Alpha Discovery Lab V24; forward-audited across 300 continuous weeks
of USDJPY (192.32M authentic Dukascopy ticks across 2021-2026) under strict 70/30
randomized week partitioning (Seed 42).

Strategy Logic:
  1. Computes the London Cash Open 30-Minute Range (07:00 - 07:30 UTC High and Low).
  2. Active Trading Window: 07:30 to 11:30 UTC.
  3. When M15 candle breaks past the 30M ORB High (> High + 1.5 pips) in an H4 20-EMA
     uptrend with candle closing in trend direction, triggers Long.
  4. When M15 candle breaks past the 30M ORB Low (< Low - 1.5 pips) in an H4 20-EMA
     downtrend with candle closing in trend direction, triggers Short.
  5. 2.0R Asymmetric Target.

Institutional Audit Metrics (USDJPY 5-Year Full Cycle / 300 Weeks):
  - Total Trades: 1,020
  - Win Rate: 38.8%
  - Profit Factor: 1.25
  - Net Pips: +1,925.2 Pips
  - Max Drawdown: 240.2 Pips
  - Return / DD Ratio: 8.01x
  - Expectancy: +1.89 Pips / Trade
  - 30% Quarantined Holdout: +724.4 Pips | 40.0% Win Rate | 1.31 Profit Factor | Ret/DD 3.02x
================================================================================
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any

from tick_engine.strategies.base import BaseStrategy, SIGNAL_LONG, SIGNAL_SHORT
from data_engineering_suite.bars import build_time_bars


class London30MORBContinuationElite(BaseStrategy):
    """
    DISC84: London 30M ORB Continuation Elite (Multi-Asset).
    """
    def __init__(
        self,
        asset: str = "USDJPY",
        sl_pips: float = 12.0,
        tp_pips: float = 24.0,
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
        self.name = f"DISC84_{asset}_London_30M_ORB_Continuation_Elite"

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
        minutes = m15_df.index.minute.values
        dates = m15_df.index.date
        time_ms = m15_df.index.astype('datetime64[ms]').view('int64')
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        # Determine daily London 30M ORB High and Low (07:00 - 07:30 UTC)
        orb_high: Dict[Any, float] = {}
        orb_low: Dict[Any, float] = {}

        for i in range(len(m15_df)):
            dt = dates[i]
            hr = hours[i]
            mn = minutes[i]
            if hr == 7 and (mn == 0 or mn == 15):
                if dt not in orb_high:
                    orb_high[dt] = h[i]
                    orb_low[dt] = l[i]
                else:
                    orb_high[dt] = max(orb_high[dt], h[i])
                    orb_low[dt] = min(orb_low[dt], l[i])

        last_trade_date = None

        for i in range(10, len(m15_df)):
            dt = dates[i]
            hr = hours[i]
            mn = minutes[i]

            # Trading window: 07:30 to 11:30 UTC
            time_val = hr * 60 + mn
            if not (7 * 60 + 30 <= time_val <= 11 * 60 + 30):
                continue
            if dt == last_trade_date:
                continue
            if spread[i] > self.max_spread:
                continue

            if dt not in orb_high or dt not in orb_low:
                continue

            oh = orb_high[dt]
            ol = orb_low[dt]
            orb_rng = oh - ol
            if orb_rng < 4.0 * self.pip_size:
                continue

            macro = h4_trend[i]
            if macro == 0:
                continue

            sig = 0
            if macro == 1 and c[i] > oh + 1.5 * self.pip_size and c[i] > o[i]:
                sig = SIGNAL_LONG
            elif macro == -1 and c[i] < ol - 1.5 * self.pip_size and c[i] < o[i]:
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
