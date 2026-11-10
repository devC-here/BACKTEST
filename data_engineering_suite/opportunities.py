#!/usr/bin/env python3
"""
================================================================================
  DATA ENGINEERING SUITE: MICROSTRUCTURE ALPHA OPPORTUNITY DETECTORS
================================================================================
Exploits three distinct quantitative market microstructure phenomena:
  1. OFI_Absorption_Fade: Order Flow Imbalance (CVD) absorption at extremes
  2. Tick_Velocity_Ignition: Sudden Poisson arrival burst momentum continuation
  3. Range_Bar_Channel_Expansion: Time-invariant 10-pip range bar breakout
================================================================================
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import numba

from tick_engine.strategies.base import BaseStrategy, SIGNAL_LONG, SIGNAL_SHORT
from data_engineering_suite.bars import build_time_bars, build_range_bars

PIP_SIZE = 0.0001


# ------------------------------------------------------------------------------
# 1. ORDER FLOW IMBALANCE (CVD) ABSORPTION FADE STRATEGY (M15)
# ------------------------------------------------------------------------------
class OFIAbsorptionFadeStrategy(BaseStrategy):
    """
    OFI Absorption Fade:
    - Monitors 20-bar rolling highs and lows on M15 bars.
    - Setup:
      Bullish Absorption: Price makes a new 20-bar Low, BUT Cumulative Volume Delta (CVD)
      is POSITIVE over the last 3 bars (aggressive selling was absorbed by institutional limit buyers).
      The candle closes with a rejection wick > 35% of total bar range.
      -> Fade Long with 1.65R target.
      
      Bearish Absorption: Price makes a new 20-bar High, BUT Cumulative Volume Delta (CVD)
      is NEGATIVE over the last 3 bars (aggressive buying was absorbed by institutional limit sellers).
      The candle closes with an upper rejection wick > 35% of total bar range.
      -> Fade Short with 1.65R target.
    - Active strictly during peak European & US liquid hours (07:00 - 16:00 UTC).
    """
    def __init__(self, target_rr: float = 1.65, **kwargs):
        super().__init__(**kwargs)
        self.target_rr = float(target_rr)
        self.allowed_hours = set(range(7, 17))
        self.name = "OFI_Absorption_Fade_Elite"

    def generate_signals(self, ticks: np.ndarray):
        n_ticks = len(ticks)
        signals = np.zeros(n_ticks, dtype=np.int8)
        sl_pips_arr = np.zeros(n_ticks, dtype=np.float64)
        tp_pips_arr = np.zeros(n_ticks, dtype=np.float64)
        if n_ticks < 500:
            return signals, sl_pips_arr, tp_pips_arr

        m15 = build_time_bars(ticks, timeframe_min=15)
        if len(m15) < 30:
            return signals, sl_pips_arr, tp_pips_arr

        c = m15['close'].values
        h = m15['high'].values
        l = m15['low'].values
        o = m15['open'].values
        deltas = m15['delta'].values
        hours = m15.index.hour.values

        time_ms = m15.index.astype('datetime64[ms]').view('int64')
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        # 20-bar rolling highs and lows
        s_h = pd.Series(h)
        s_l = pd.Series(l)
        roll_h = s_h.rolling(20).max().shift(1).values
        roll_l = s_l.rolling(20).min().shift(1).values

        for i in range(25, len(m15)):
            if hours[i] not in self.allowed_hours:
                continue

            bar_range = h[i] - l[i]
            if bar_range < 6.0 * PIP_SIZE:
                continue

            # 3-bar net tick delta
            three_bar_delta = deltas[i] + deltas[i-1] + deltas[i-2]

            upper_wick = h[i] - max(o[i], c[i])
            lower_wick = min(o[i], c[i]) - l[i]

            sig = 0
            risk_p = 0.0

            # Bearish Absorption: Price tested above 20-bar high, but sellers absorbed (delta negative)
            if h[i] >= roll_h[i] and three_bar_delta < 0 and upper_wick >= 0.35 * bar_range and c[i] < o[i]:
                sl_p = h[i] + 1.2 * PIP_SIZE
                risk = max(min((sl_p - c[i]) / PIP_SIZE, 15.0), 5.5)
                sig = SIGNAL_SHORT
                risk_p = risk

            # Bullish Absorption: Price tested below 20-bar low, but buyers absorbed (delta positive)
            elif l[i] <= roll_l[i] and three_bar_delta > 0 and lower_wick >= 0.35 * bar_range and c[i] > o[i]:
                sl_p = l[i] - 1.2 * PIP_SIZE
                risk = max(min((c[i] - sl_p) / PIP_SIZE, 15.0), 5.5)
                sig = SIGNAL_LONG
                risk_p = risk

            if sig != 0:
                exec_t = time_ms[i] + dur_ms
                idx = int(np.searchsorted(tick_ts, exec_t))
                if idx < n_ticks and signals[idx] == 0:
                    signals[idx] = sig
                    sl_pips_arr[idx] = risk_p
                    tp_pips_arr[idx] = risk_p * self.target_rr

        return signals, sl_pips_arr, tp_pips_arr


# ------------------------------------------------------------------------------
# 2. TICK VELOCITY IGNITION BURST STRATEGY (M15)
# ------------------------------------------------------------------------------
class TickVelocityIgnitionStrategy(BaseStrategy):
    """
    Tick Velocity Ignition Burst:
    - Measures tick arrival intensity.
    - Setup: When an M15 bar's tick volume surges >= 2.5x the 20-bar moving average,
      AND the body constitutes >= 70% of the bar's range (decisive directional displacement),
      AND spread is normal/tight (<= 1.5 pips, ensuring it's not a news gap),
      AND aligned with H1 50-EMA trend.
    - Ride the institutional momentum ignition with 1.65R target.
    """
    def __init__(self, target_rr: float = 1.65, **kwargs):
        super().__init__(**kwargs)
        self.target_rr = float(target_rr)
        self.allowed_hours = set(range(7, 16))
        self.name = "Tick_Velocity_Ignition_Elite"

    def generate_signals(self, ticks: np.ndarray):
        n_ticks = len(ticks)
        signals = np.zeros(n_ticks, dtype=np.int8)
        sl_pips_arr = np.zeros(n_ticks, dtype=np.float64)
        tp_pips_arr = np.zeros(n_ticks, dtype=np.float64)
        if n_ticks < 500:
            return signals, sl_pips_arr, tp_pips_arr

        m15 = build_time_bars(ticks, timeframe_min=15)
        if len(m15) < 30:
            return signals, sl_pips_arr, tp_pips_arr

        ts = pd.to_datetime(ticks['timestamp'], unit='ms')
        mid = 0.5 * (ticks['bid'] + ticks['ask'])
        base = pd.DataFrame({'close': mid}, index=ts)
        h1 = base.resample('1h').last().dropna()
        h1['ema50'] = h1['close'].ewm(span=50, adjust=False).mean()
        h1['macro'] = np.where(h1['close'] > h1['ema50'], 1, -1)
        m15['macro'] = h1['macro'].reindex(m15.index, method='ffill').fillna(0)

        c = m15['close'].values
        h = m15['high'].values
        l = m15['low'].values
        o = m15['open'].values
        v = m15['vol'].values
        spreads = m15['spread_pips'].values
        macros = m15['macro'].values
        hours = m15.index.hour.values

        vol_sma = pd.Series(v).rolling(20).mean().shift(1).bfill().values

        time_ms = m15.index.astype('datetime64[ms]').view('int64')
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        for i in range(20, len(m15)):
            if hours[i] not in self.allowed_hours:
                continue

            # Spread must be tight/normal
            if spreads[i] > 1.8:
                continue

            bar_range = h[i] - l[i]
            if bar_range < 7.0 * PIP_SIZE:
                continue

            body = abs(c[i] - o[i])
            if body < 0.70 * bar_range:
                continue

            is_burst = v[i] >= 2.4 * vol_sma[i]
            if not is_burst:
                continue

            sig = 0
            risk_p = 0.0

            # Bullish ignition burst
            if c[i] > o[i] and macros[i] > 0:
                sl_p = l[i] - 1.5 * PIP_SIZE
                risk = max(min((c[i] - sl_p) / PIP_SIZE, 15.0), 6.0)
                sig = SIGNAL_LONG
                risk_p = risk
            # Bearish ignition burst
            elif c[i] < o[i] and macros[i] < 0:
                sl_p = h[i] + 1.5 * PIP_SIZE
                risk = max(min((sl_p - c[i]) / PIP_SIZE, 15.0), 6.0)
                sig = SIGNAL_SHORT
                risk_p = risk

            if sig != 0:
                exec_t = time_ms[i] + dur_ms
                idx = int(np.searchsorted(tick_ts, exec_t))
                if idx < n_ticks and signals[idx] == 0:
                    signals[idx] = sig
                    sl_pips_arr[idx] = risk_p
                    tp_pips_arr[idx] = risk_p * self.target_rr

        return signals, sl_pips_arr, tp_pips_arr
