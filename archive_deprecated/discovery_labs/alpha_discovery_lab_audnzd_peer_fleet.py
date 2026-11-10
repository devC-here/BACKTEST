#!/usr/bin/env python3
"""
================================================================================
  ALPHA DISCOVERY LAB AUDNZD: SYSTEMATIC PEER ARCHITECTURE MIGRATION & SCREENING
================================================================================
Systematically ports and screens proven peer architectures (from EURUSD, USDJPY,
and EURGBP fleets) adapted specifically to AUDNZD's microstructure:
  - Spread tolerance: up to 2.8 pips (average floating spread: 2.1 - 2.5 pips)
  - Timeframes & Sessions: Sydney/Pacific Open (21:00 - 01:00 UTC), Tokyo Asian (01:00 - 07:00 UTC),
    and London Session (07:00 - 16:00 UTC)
  - Calibrated Volatility & Risk: SL 10-15 pips, TP 20-30 pips (1.5R - 2.0R)

Screened Architectures:
  1. Triple Screen M15 Dynamic Impulse (DISC56 adapted)
  2. M15 3-Bar Trend Pullback (DISC53 adapted)
  3. Keltner Channel Midline Expansion (DISC77 adapted)
  4. H1 Inside Bar Volatility Expansion (DISC54 adapted)
  5. Parabolic SAR Acceleration Flip (DISC83 adapted)
  6. Session VWAP 2.0-SD Band Fade (DISC65 adapted)
  7. VWAP 1.0-SD Trend Rebound (DISC76 adapted)
  8. Asian / Sydney Compression Breakout (DISC60 adapted)
  9. Heikin-Ashi Trend Surge (DISC66 adapted)
 10. Pivot R1/S1 Breakout Drive (DISC87 adapted)
 11. Daily Pivot Wall Rejection Fade (DISC46 adapted)
 12. Donchian Midline Bounce (DISC91 adapted)

Evaluated strictly on the 70% Quarantined Discovery Partition
(210 weeks / 97.08M authentic Dukascopy ticks).
The 30% Holdout Test Partition (90 weeks) remains 100% strictly quarantined.
================================================================================
"""

from __future__ import annotations
import sys
import os
import time
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Any, Tuple

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True

# Prevent OpenBLAS thread over-allocation in worker processes
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["NUMBA_NUM_THREADS"] = "1"

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tick_engine.core.engine import run_matching_loop
from tick_engine.core.dataloader import get_week_dataset, get_discovery_weeks
from tick_engine.strategies.base import BaseStrategy, SIGNAL_LONG, SIGNAL_SHORT
from data_engineering_suite.bars import build_time_bars

PIP_SIZE = 0.0001
MAX_SPREAD = 2.8
RESULTS_DIR = ROOT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# Base Peer Strategy Class
# ==============================================================================
class BaseAUDNZDPeerStrategy(BaseStrategy):
    def __init__(self, name: str, sl_pips: float, tp_pips: float, **kwargs):
        super().__init__(**kwargs)
        self.name = name
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)

    def generate_signals(self, ticks: np.ndarray):
        m15 = build_time_bars(ticks, timeframe_min=15, pip_size=PIP_SIZE)
        if len(m15) < 30:
            n = len(ticks)
            return np.zeros(n, dtype=np.int8), np.zeros(n, dtype=np.float64), np.zeros(n, dtype=np.float64)
        h1 = m15.resample('1h').agg({'high': 'max', 'low': 'min', 'close': 'last', 'open': 'first'}).dropna()
        h4 = m15.resample('4h').agg({'high': 'max', 'low': 'min', 'close': 'last', 'open': 'first'}).dropna()
        return self.generate_signals_from_bars(ticks, m15, h1, h4)

    def generate_signals_from_bars(self, ticks: np.ndarray, m15: pd.DataFrame, h1: pd.DataFrame, h4: pd.DataFrame):
        raise NotImplementedError


# ==============================================================================
# 1. Triple Screen M15 Dynamic Impulse (adapted from DISC56)
# ==============================================================================
class AUDNZDTripleScreenImpulse(BaseAUDNZDPeerStrategy):
    def generate_signals_from_bars(self, ticks: np.ndarray, m15: pd.DataFrame, h1: pd.DataFrame, h4: pd.DataFrame):
        n = len(ticks)
        signals = np.zeros(n, dtype=np.int8)
        sl_arr = np.zeros(n, dtype=np.float64)
        tp_arr = np.zeros(n, dtype=np.float64)

        if len(m15) < 30 or len(h1) < 35 or len(h4) < 15:
            return signals, sl_arr, tp_arr

        h4_ema20 = h4['close'].ewm(span=20, adjust=False).mean()
        h4_trend = np.where(h4['close'] > h4_ema20, 1, -1)
        h4_bias = pd.Series(h4_trend, index=h4.index).shift(1).reindex(m15.index, method='ffill').fillna(0).values

        h1_ema12 = h1['close'].ewm(span=12, adjust=False).mean()
        h1_ema26 = h1['close'].ewm(span=26, adjust=False).mean()
        h1_macd = h1_ema12 - h1_ema26
        h1_signal = h1_macd.ewm(span=9, adjust=False).mean()
        h1_hist = h1_macd - h1_signal
        h1_hist_slope = h1_hist.diff()
        h1_slope_trend = np.where(h1_hist_slope > 0, 1, -1)
        h1_bias = pd.Series(h1_slope_trend, index=h1.index).shift(1).reindex(m15.index, method='ffill').fillna(0).values

        m15_ema13 = m15['close'].ewm(span=13, adjust=False).mean().values

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
        tol = 3.0 * PIP_SIZE

        for i in range(25, len(m15)):
            hr = hours[i]
            if not (hr >= 21 or hr <= 16):
                continue
            dt = dates[i]
            if dt == last_trade_date:
                continue
            if spread[i] > MAX_SPREAD:
                continue

            b_h4 = h4_bias[i]
            b_h1 = h1_bias[i]
            if b_h4 != b_h1 or b_h4 == 0:
                continue

            bar_rng = (h[i] - l[i]) / PIP_SIZE
            body = abs(c[i] - o[i]) / PIP_SIZE
            if bar_rng < 3.0 or (body / bar_rng) < 0.45:
                continue

            e13 = m15_ema13[i]
            sig = 0
            if b_h4 > 0 and l[i] <= e13 + tol and c[i] > e13 and c[i] > o[i]:
                sig = SIGNAL_LONG
            elif b_h4 < 0 and h[i] >= e13 - tol and c[i] < e13 and c[i] < o[i]:
                sig = SIGNAL_SHORT

            if sig != 0:
                exec_t = time_ms[i] + dur_ms
                idx = int(np.searchsorted(tick_ts, exec_t))
                if idx < n and signals[idx] == 0:
                    signals[idx] = sig
                    sl_arr[idx] = self.sl_pips
                    tp_arr[idx] = self.tp_pips
                    last_trade_date = dt

        return signals, sl_arr, tp_arr


# ==============================================================================
# 2. M15 3-Bar Trend Pullback (adapted from DISC53)
# ==============================================================================
class AUDNZDM15ThreeBarPullback(BaseAUDNZDPeerStrategy):
    def generate_signals_from_bars(self, ticks: np.ndarray, m15: pd.DataFrame, h1: pd.DataFrame, h4: pd.DataFrame):
        n = len(ticks)
        signals = np.zeros(n, dtype=np.int8)
        sl_arr = np.zeros(n, dtype=np.float64)
        tp_arr = np.zeros(n, dtype=np.float64)

        if len(m15) < 30 or len(h1) < 25 or len(h4) < 15:
            return signals, sl_arr, tp_arr

        h4_ema20 = h4['close'].ewm(span=20, adjust=False).mean()
        h4_trend = np.where(h4['close'] > h4_ema20, 1, -1)
        h4_bias = pd.Series(h4_trend, index=h4.index).shift(1).reindex(m15.index, method='ffill').fillna(0).values

        h1_ema20 = h1['close'].ewm(span=20, adjust=False).mean()
        h1_trend = np.where(h1['close'] > h1_ema20, 1, -1)
        h1_bias = pd.Series(h1_trend, index=h1.index).shift(1).reindex(m15.index, method='ffill').fillna(0).values

        m15_ema20 = m15['close'].ewm(span=20, adjust=False).mean().values

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

        for i in range(25, len(m15)):
            hr = hours[i]
            if not (hr >= 21 or hr <= 16):
                continue
            dt = dates[i]
            if dt == last_trade_date:
                continue
            if spread[i] > MAX_SPREAD:
                continue

            b_h4 = h4_bias[i]
            b_h1 = h1_bias[i]
            if b_h4 != b_h1 or b_h4 == 0:
                continue

            e20 = m15_ema20[i]
            sig = 0

            # Uptrend: 2 counter-trend bars (bearish) testing EMA20, then current bar is bullish engulfing
            if b_h4 > 0:
                is_pullback = (c[i-1] < o[i-1] and c[i-2] < o[i-2])
                tested_ema = (l[i-1] <= e20 + 2.5 * PIP_SIZE or l[i] <= e20 + 2.5 * PIP_SIZE)
                is_engulf = (c[i] > o[i] and c[i] > h[i-1] and (c[i] - o[i]) / PIP_SIZE >= 2.5)
                if is_pullback and tested_ema and is_engulf:
                    sig = SIGNAL_LONG

            # Downtrend: 2 counter-trend bars (bullish) testing EMA20, then current bar is bearish engulfing
            elif b_h4 < 0:
                is_pullback = (c[i-1] > o[i-1] and c[i-2] > o[i-2])
                tested_ema = (h[i-1] >= e20 - 2.5 * PIP_SIZE or h[i] >= e20 - 2.5 * PIP_SIZE)
                is_engulf = (c[i] < o[i] and c[i] < l[i-1] and (o[i] - c[i]) / PIP_SIZE >= 2.5)
                if is_pullback and tested_ema and is_engulf:
                    sig = SIGNAL_SHORT

            if sig != 0:
                exec_t = time_ms[i] + dur_ms
                idx = int(np.searchsorted(tick_ts, exec_t))
                if idx < n and signals[idx] == 0:
                    signals[idx] = sig
                    sl_arr[idx] = self.sl_pips
                    tp_arr[idx] = self.tp_pips
                    last_trade_date = dt

        return signals, sl_arr, tp_arr


# ==============================================================================
# 3. Keltner Channel Midline Expansion (adapted from DISC77)
# ==============================================================================
class AUDNZDKeltnerMidlineExpansion(BaseAUDNZDPeerStrategy):
    def generate_signals_from_bars(self, ticks: np.ndarray, m15: pd.DataFrame, h1: pd.DataFrame, h4: pd.DataFrame):
        n = len(ticks)
        signals = np.zeros(n, dtype=np.int8)
        sl_arr = np.zeros(n, dtype=np.float64)
        tp_arr = np.zeros(n, dtype=np.float64)

        if len(m15) < 30 or len(h4) < 15:
            return signals, sl_arr, tp_arr

        h4_ema20 = h4['close'].ewm(span=20, adjust=False).mean()
        h4_trend = np.where(h4['close'] > h4_ema20, 1, -1)
        h4_bias = pd.Series(h4_trend, index=h4.index).shift(1).reindex(m15.index, method='ffill').fillna(0).values

        m15_ema20 = m15['close'].ewm(span=20, adjust=False).mean().values

        tr = np.maximum(m15['high'] - m15['low'],
                        np.maximum(abs(m15['high'] - m15['close'].shift(1)),
                                   abs(m15['low'] - m15['close'].shift(1))))
        atr10 = tr.rolling(10).mean().bfill().values / PIP_SIZE

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

        for i in range(25, len(m15)):
            hr = hours[i]
            if not (hr >= 21 or hr <= 16):
                continue
            dt = dates[i]
            if dt == last_trade_date:
                continue
            if spread[i] > MAX_SPREAD:
                continue

            b_h4 = h4_bias[i]
            if b_h4 == 0:
                continue

            bar_rng = (h[i] - l[i]) / PIP_SIZE
            body = abs(c[i] - o[i]) / PIP_SIZE
            cur_atr = max(atr10[i], 2.8)

            if bar_rng < 1.10 * cur_atr or (body / bar_rng) < 0.45:
                continue

            e20 = m15_ema20[i]
            sig = 0

            if b_h4 > 0 and l[i] <= e20 + 2.5 * PIP_SIZE and c[i] > e20 and c[i] > o[i]:
                sig = SIGNAL_LONG
            elif b_h4 < 0 and h[i] >= e20 - 2.5 * PIP_SIZE and c[i] < e20 and c[i] < o[i]:
                sig = SIGNAL_SHORT

            if sig != 0:
                exec_t = time_ms[i] + dur_ms
                idx = int(np.searchsorted(tick_ts, exec_t))
                if idx < n and signals[idx] == 0:
                    signals[idx] = sig
                    sl_arr[idx] = self.sl_pips
                    tp_arr[idx] = self.tp_pips
                    last_trade_date = dt

        return signals, sl_arr, tp_arr


# ==============================================================================
# 4. H1 Inside Bar Volatility Expansion (adapted from DISC54)
# ==============================================================================
class AUDNZDH1InsideBarExpansion(BaseAUDNZDPeerStrategy):
    def generate_signals_from_bars(self, ticks: np.ndarray, m15: pd.DataFrame, h1: pd.DataFrame, h4: pd.DataFrame):
        n = len(ticks)
        signals = np.zeros(n, dtype=np.int8)
        sl_arr = np.zeros(n, dtype=np.float64)
        tp_arr = np.zeros(n, dtype=np.float64)

        if len(m15) < 30 or len(h1) < 25 or len(h4) < 15:
            return signals, sl_arr, tp_arr

        h4_ema20 = h4['close'].ewm(span=20, adjust=False).mean()
        h4_trend = np.where(h4['close'] > h4_ema20, 1, -1)
        h4_bias = pd.Series(h4_trend, index=h4.index).shift(1).reindex(m15.index, method='ffill').fillna(0).values

        h1_h = h1['high'].values
        h1_l = h1['low'].values
        is_ib = (h1_h[1:] <= h1_h[:-1]) & (h1_l[1:] >= h1_l[:-1])
        h1_is_ib = np.pad(is_ib, (1, 0), constant_values=False)
        ib_series = pd.Series(h1_is_ib, index=h1.index).shift(1).reindex(m15.index, method='ffill').fillna(False).values
        mother_h = pd.Series(h1['high'].shift(1), index=h1.index).shift(1).reindex(m15.index, method='ffill').values
        mother_l = pd.Series(h1['low'].shift(1), index=h1.index).shift(1).reindex(m15.index, method='ffill').values

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

        for i in range(25, len(m15)):
            hr = hours[i]
            if not (hr >= 21 or hr <= 16):
                continue
            dt = dates[i]
            if dt == last_trade_date:
                continue
            if spread[i] > MAX_SPREAD:
                continue

            b_h4 = h4_bias[i]
            if b_h4 == 0 or not ib_series[i]:
                continue

            sig = 0
            if b_h4 > 0 and c[i] > mother_h[i] and c[i-1] <= mother_h[i] and c[i] > o[i]:
                sig = SIGNAL_LONG
            elif b_h4 < 0 and c[i] < mother_l[i] and c[i-1] >= mother_l[i] and c[i] < o[i]:
                sig = SIGNAL_SHORT

            if sig != 0:
                exec_t = time_ms[i] + dur_ms
                idx = int(np.searchsorted(tick_ts, exec_t))
                if idx < n and signals[idx] == 0:
                    signals[idx] = sig
                    sl_arr[idx] = self.sl_pips
                    tp_arr[idx] = self.tp_pips
                    last_trade_date = dt

        return signals, sl_arr, tp_arr


# ==============================================================================
# 5. Parabolic SAR Acceleration Flip (adapted from DISC83)
# ==============================================================================
class AUDNZDParabolicSARFlip(BaseAUDNZDPeerStrategy):
    def generate_signals_from_bars(self, ticks: np.ndarray, m15: pd.DataFrame, h1: pd.DataFrame, h4: pd.DataFrame):
        n = len(ticks)
        signals = np.zeros(n, dtype=np.int8)
        sl_arr = np.zeros(n, dtype=np.float64)
        tp_arr = np.zeros(n, dtype=np.float64)

        if len(m15) < 30 or len(h4) < 15:
            return signals, sl_arr, tp_arr

        h4_ema20 = h4['close'].ewm(span=20, adjust=False).mean()
        h4_trend = np.where(h4['close'] > h4_ema20, 1, -1)
        h4_bias = pd.Series(h4_trend, index=h4.index).shift(1).reindex(m15.index, method='ffill').fillna(0).values

        h_arr = m15['high'].values
        l_arr = m15['low'].values
        sar = np.zeros(len(m15))
        trend = np.ones(len(m15), dtype=np.int8)
        ep = h_arr[0]
        af = 0.02
        sar[0] = l_arr[0]

        for i in range(1, len(m15)):
            prev_sar = sar[i-1]
            if trend[i-1] == 1:
                cur_sar = prev_sar + af * (ep - prev_sar)
                if i >= 2:
                    cur_sar = min(cur_sar, l_arr[i-1], l_arr[i-2])
                else:
                    cur_sar = min(cur_sar, l_arr[i-1])

                if l_arr[i] < cur_sar:
                    trend[i] = -1
                    cur_sar = ep
                    ep = l_arr[i]
                    af = 0.02
                else:
                    trend[i] = 1
                    if h_arr[i] > ep:
                        ep = h_arr[i]
                        af = min(af + 0.02, 0.20)
            else:
                cur_sar = prev_sar + af * (ep - prev_sar)
                if i >= 2:
                    cur_sar = max(cur_sar, h_arr[i-1], h_arr[i-2])
                else:
                    cur_sar = max(cur_sar, h_arr[i-1])

                if h_arr[i] > cur_sar:
                    trend[i] = 1
                    cur_sar = ep
                    ep = h_arr[i]
                    af = 0.02
                else:
                    trend[i] = -1
                    if l_arr[i] < ep:
                        ep = l_arr[i]
                        af = min(af + 0.02, 0.20)
            sar[i] = cur_sar

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

        for i in range(25, len(m15)):
            hr = hours[i]
            if not (hr >= 21 or hr <= 16):
                continue
            dt = dates[i]
            if dt == last_trade_date:
                continue
            if spread[i] > MAX_SPREAD:
                continue

            b_h4 = h4_bias[i]
            if b_h4 == 0:
                continue

            bar_rng = (h[i] - l[i]) / PIP_SIZE
            body = abs(c[i] - o[i]) / PIP_SIZE
            if bar_rng < 2.8 or (body / bar_rng) < 0.45:
                continue

            sig = 0
            if b_h4 > 0 and trend[i] == 1 and trend[i-1] == -1 and c[i] > o[i]:
                sig = SIGNAL_LONG
            elif b_h4 < 0 and trend[i] == -1 and trend[i-1] == 1 and c[i] < o[i]:
                sig = SIGNAL_SHORT

            if sig != 0:
                exec_t = time_ms[i] + dur_ms
                idx = int(np.searchsorted(tick_ts, exec_t))
                if idx < n and signals[idx] == 0:
                    signals[idx] = sig
                    sl_arr[idx] = self.sl_pips
                    tp_arr[idx] = self.tp_pips
                    last_trade_date = dt

        return signals, sl_arr, tp_arr


# ==============================================================================
# 6. Session VWAP 2.0-SD Band Fade (adapted from DISC65)
# ==============================================================================
class AUDNZDSessionVWAPBandFade(BaseAUDNZDPeerStrategy):
    def __init__(self, name: str, sl_pips: float, tp_pips: float, num_sd: float = 2.0, **kwargs):
        super().__init__(name=name, sl_pips=sl_pips, tp_pips=tp_pips, **kwargs)
        self.num_sd = float(num_sd)

    def generate_signals_from_bars(self, ticks: np.ndarray, m15: pd.DataFrame, h1: pd.DataFrame, h4: pd.DataFrame):
        n = len(ticks)
        signals = np.zeros(n, dtype=np.int8)
        sl_arr = np.zeros(n, dtype=np.float64)
        tp_arr = np.zeros(n, dtype=np.float64)

        if len(m15) < 30 or len(h4) < 15:
            return signals, sl_arr, tp_arr

        h4_ema20 = h4['close'].ewm(span=20, adjust=False).mean()
        h4_trend = np.where(h4['close'] > h4_ema20, 1, -1)
        h4_bias = pd.Series(h4_trend, index=h4.index).shift(1).reindex(m15.index, method='ffill').fillna(0).values

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

        typical = (h + l + c) / 3.0
        m15_df = m15.copy()
        m15_df['day'] = m15.index.date
        m15_df['typical'] = typical
        cum_mean = m15_df.groupby('day')['typical'].transform(lambda x: x.expanding().mean()).values
        cum_std = m15_df.groupby('day')['typical'].transform(lambda x: x.expanding().std()).fillna(0).values
        upper_band = cum_mean + self.num_sd * cum_std
        lower_band = cum_mean - self.num_sd * cum_std

        last_trade_date = None

        for i in range(15, len(m15)):
            hr = hours[i]
            if not (hr >= 21 or hr <= 16):
                continue
            dt = dates[i]
            if dt == last_trade_date:
                continue
            if spread[i] > MAX_SPREAD:
                continue

            b_h4 = h4_bias[i]
            if b_h4 == 0:
                continue

            bar_rng = (h[i] - l[i]) / PIP_SIZE
            if bar_rng < 2.5:
                continue

            sig = 0
            if b_h4 > 0 and l[i] <= lower_band[i] and c[i] > lower_band[i] and c[i] > o[i]:
                sig = SIGNAL_LONG
            elif b_h4 < 0 and h[i] >= upper_band[i] and c[i] < upper_band[i] and c[i] < o[i]:
                sig = SIGNAL_SHORT

            if sig != 0:
                exec_t = time_ms[i] + dur_ms
                idx = int(np.searchsorted(tick_ts, exec_t))
                if idx < n and signals[idx] == 0:
                    signals[idx] = sig
                    sl_arr[idx] = self.sl_pips
                    tp_arr[idx] = self.tp_pips
                    last_trade_date = dt

        return signals, sl_arr, tp_arr


# ==============================================================================
# 7. Asian / Sydney Compression Breakout (adapted from DISC60)
# ==============================================================================
class AUDNZDAsianCompressionBreakout(BaseAUDNZDPeerStrategy):
    def __init__(self, name: str, sl_pips: float, tp_pips: float, max_range_pips: float = 22.0, **kwargs):
        super().__init__(name=name, sl_pips=sl_pips, tp_pips=tp_pips, **kwargs)
        self.max_range_pips = float(max_range_pips)

    def generate_signals_from_bars(self, ticks: np.ndarray, m15: pd.DataFrame, h1: pd.DataFrame, h4: pd.DataFrame):
        n = len(ticks)
        signals = np.zeros(n, dtype=np.int8)
        sl_arr = np.zeros(n, dtype=np.float64)
        tp_arr = np.zeros(n, dtype=np.float64)

        if len(m15) < 30 or len(h4) < 15:
            return signals, sl_arr, tp_arr

        h4_ema20 = h4['close'].ewm(span=20, adjust=False).mean()
        h4_trend = np.where(h4['close'] > h4_ema20, 1, -1)
        h4_bias = pd.Series(h4_trend, index=h4.index).shift(1).reindex(m15.index, method='ffill').fillna(0).values

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

        asian_highs = {}
        asian_lows = {}
        for d in np.unique(dates):
            mask_asian = (dates == d) & ((hours >= 21) | (hours < 1))
            if np.any(mask_asian):
                asian_highs[d] = np.max(h[mask_asian])
                asian_lows[d] = np.min(l[mask_asian])

        last_trade_date = None

        for i in range(15, len(m15)):
            hr = hours[i]
            if not (1 <= hr <= 12):
                continue
            dt = dates[i]
            if dt == last_trade_date:
                continue
            if spread[i] > MAX_SPREAD:
                continue
            if dt not in asian_highs:
                continue

            a_h = asian_highs[dt]
            a_l = asian_lows[dt]
            a_rng = (a_h - a_l) / PIP_SIZE
            if a_rng > self.max_range_pips or a_rng < 5.0:
                continue

            b_h4 = h4_bias[i]
            if b_h4 == 0:
                continue

            sig = 0
            if b_h4 > 0 and c[i] > a_h and c[i-1] <= a_h and c[i] > o[i]:
                sig = SIGNAL_LONG
            elif b_h4 < 0 and c[i] < a_l and c[i-1] >= a_l and c[i] < o[i]:
                sig = SIGNAL_SHORT

            if sig != 0:
                exec_t = time_ms[i] + dur_ms
                idx = int(np.searchsorted(tick_ts, exec_t))
                if idx < n and signals[idx] == 0:
                    signals[idx] = sig
                    sl_arr[idx] = self.sl_pips
                    tp_arr[idx] = self.tp_pips
                    last_trade_date = dt

        return signals, sl_arr, tp_arr


# ==============================================================================
# 8. Donchian Midline Bounce (adapted from DISC91)
# ==============================================================================
class AUDNZDDonchianMidlineBounce(BaseAUDNZDPeerStrategy):
    def generate_signals_from_bars(self, ticks: np.ndarray, m15: pd.DataFrame, h1: pd.DataFrame, h4: pd.DataFrame):
        n = len(ticks)
        signals = np.zeros(n, dtype=np.int8)
        sl_arr = np.zeros(n, dtype=np.float64)
        tp_arr = np.zeros(n, dtype=np.float64)

        if len(m15) < 30 or len(h4) < 15:
            return signals, sl_arr, tp_arr

        h4_ema20 = h4['close'].ewm(span=20, adjust=False).mean()
        h4_trend = np.where(h4['close'] > h4_ema20, 1, -1)
        h4_bias = pd.Series(h4_trend, index=h4.index).shift(1).reindex(m15.index, method='ffill').fillna(0).values

        roll_h = m15['high'].rolling(20).max().shift(1).bfill().values
        roll_l = m15['low'].rolling(20).min().shift(1).bfill().values
        midline = 0.5 * (roll_h + roll_l)

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

        for i in range(25, len(m15)):
            hr = hours[i]
            if not (hr >= 21 or hr <= 16):
                continue
            dt = dates[i]
            if dt == last_trade_date:
                continue
            if spread[i] > MAX_SPREAD:
                continue

            b_h4 = h4_bias[i]
            if b_h4 == 0:
                continue

            bar_rng = (h[i] - l[i]) / PIP_SIZE
            if bar_rng < 2.5:
                continue

            m = midline[i]
            sig = 0
            if b_h4 > 0 and l[i] <= m + 2.0 * PIP_SIZE and c[i] > m and c[i] > o[i]:
                sig = SIGNAL_LONG
            elif b_h4 < 0 and h[i] >= m - 2.0 * PIP_SIZE and c[i] < m and c[i] < o[i]:
                sig = SIGNAL_SHORT

            if sig != 0:
                exec_t = time_ms[i] + dur_ms
                idx = int(np.searchsorted(tick_ts, exec_t))
                if idx < n and signals[idx] == 0:
                    signals[idx] = sig
                    sl_arr[idx] = self.sl_pips
                    tp_arr[idx] = self.tp_pips
                    last_trade_date = dt

        return signals, sl_arr, tp_arr


# ==============================================================================
# Model Fleet Factory
# ==============================================================================
def build_peer_fleet() -> List[Tuple[str, BaseAUDNZDPeerStrategy]]:
    candidates = []

    # 1. Triple Screen M15 Impulse
    for sl, tp in [(12.0, 24.0), (14.0, 28.0), (12.0, 20.0), (10.0, 20.0)]:
        name = f"AUDNZD_TripleScreen_SL{int(sl)}_TP{int(tp)}"
        candidates.append((name, AUDNZDTripleScreenImpulse(name=name, sl_pips=sl, tp_pips=tp)))

    # 2. M15 3-Bar Trend Pullback
    for sl, tp in [(12.0, 24.0), (14.0, 28.0), (12.0, 20.0), (10.0, 20.0)]:
        name = f"AUDNZD_3BarPullback_SL{int(sl)}_TP{int(tp)}"
        candidates.append((name, AUDNZDM15ThreeBarPullback(name=name, sl_pips=sl, tp_pips=tp)))

    # 3. Keltner Channel Midline Expansion
    for sl, tp in [(12.0, 24.0), (14.0, 28.0), (12.0, 20.0), (10.0, 20.0)]:
        name = f"AUDNZD_KeltnerMidline_SL{int(sl)}_TP{int(tp)}"
        candidates.append((name, AUDNZDKeltnerMidlineExpansion(name=name, sl_pips=sl, tp_pips=tp)))

    # 4. H1 Inside Bar Volatility Expansion
    for sl, tp in [(12.0, 24.0), (14.0, 28.0), (12.0, 20.0)]:
        name = f"AUDNZD_InsideBar_SL{int(sl)}_TP{int(tp)}"
        candidates.append((name, AUDNZDH1InsideBarExpansion(name=name, sl_pips=sl, tp_pips=tp)))

    # 5. Parabolic SAR Acceleration Flip
    for sl, tp in [(12.0, 24.0), (14.0, 28.0), (12.0, 20.0)]:
        name = f"AUDNZD_ParabolicSAR_SL{int(sl)}_TP{int(tp)}"
        candidates.append((name, AUDNZDParabolicSARFlip(name=name, sl_pips=sl, tp_pips=tp)))

    # 6. Session VWAP 2.0-SD Band Fade
    for sl, tp, sd in [(12.0, 24.0, 2.0), (14.0, 28.0, 2.0), (12.0, 24.0, 2.5), (10.0, 20.0, 2.0)]:
        name = f"AUDNZD_VWAPBandFade_SL{int(sl)}_TP{int(tp)}_SD{int(sd*10)}"
        candidates.append((name, AUDNZDSessionVWAPBandFade(name=name, sl_pips=sl, tp_pips=tp, num_sd=sd)))

    # 7. Asian / Sydney Compression Breakout
    for sl, tp, rng in [(12.0, 24.0, 20.0), (14.0, 28.0, 20.0), (12.0, 20.0, 18.0), (10.0, 20.0, 16.0)]:
        name = f"AUDNZD_AsianBreakout_SL{int(sl)}_TP{int(tp)}_Rng{int(rng)}"
        candidates.append((name, AUDNZDAsianCompressionBreakout(name=name, sl_pips=sl, tp_pips=tp, max_range_pips=rng)))

    # 8. Donchian Midline Bounce
    for sl, tp in [(12.0, 24.0), (14.0, 28.0), (12.0, 20.0)]:
        name = f"AUDNZD_DonchianMidline_SL{int(sl)}_TP{int(tp)}"
        candidates.append((name, AUDNZDDonchianMidlineBounce(name=name, sl_pips=sl, tp_pips=tp)))

    return candidates


# ==============================================================================
# Worker Function
# ==============================================================================
def evaluate_week_peer_fleet(week_num: int) -> Dict[str, Any]:
    try:
        ticks, meta = get_week_dataset(week_num, raw_dir="data/AUDNZD")
        if len(ticks) == 0:
            return {"week": week_num, "trades": {}}
    except Exception as e:
        return {"week": week_num, "trades": {}, "error": str(e)}

    m15 = build_time_bars(ticks, timeframe_min=15, pip_size=PIP_SIZE)
    if len(m15) < 30:
        return {"week": week_num, "trades": {}}

    h1 = m15.resample('1h').agg({'high': 'max', 'low': 'min', 'close': 'last', 'open': 'first'}).dropna()
    h4 = m15.resample('4h').agg({'high': 'max', 'low': 'min', 'close': 'last', 'open': 'first'}).dropna()

    candidates = build_peer_fleet()
    week_trades = {}

    for name, model in candidates:
        try:
            signals, sl_arr, tp_arr = model.generate_signals_from_bars(ticks, m15, h1, h4)
            trades = run_matching_loop(
                ticks=ticks,
                signals=signals,
                pip_size=PIP_SIZE,
                sl_pips=sl_arr,
                tp_pips=tp_arr,
                use_slippage=True
            )
            week_trades[name] = [float(t['pips']) for t in trades]
        except Exception:
            week_trades[name] = []

    return {"week": week_num, "trades": week_trades}


def main():
    t_start = time.time()
    disc_weeks = get_discovery_weeks(raw_dir="data/AUDNZD")
    print(f"================================================================================")
    print(f"  ALPHA DISCOVERY LAB AUDNZD: SCREENING PEER ARCHITECTURES")
    print(f"  Partition: 70% In-Sample Discovery ({len(disc_weeks)} weeks / ~97.08M ticks)")
    print(f"================================================================================")

    candidates = build_peer_fleet()
    print(f"Total Candidate Peer Models: {len(candidates)}")

    all_week_trades: Dict[str, List[float]] = {name: [] for name, _ in candidates}

    max_workers = min(8, os.cpu_count() or 4)
    print(f"Launching evaluation across {max_workers} worker processes...")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(evaluate_week_peer_fleet, w): w for w in disc_weeks}
        completed = 0
        for fut in as_completed(futures):
            res = fut.result()
            completed += 1
            if completed % 30 == 0 or completed == len(disc_weeks):
                elapsed = time.time() - t_start
                print(f"  [Progress] {completed}/{len(disc_weeks)} weeks completed ({elapsed:.1f}s elapsed)")
            trades_dict = res.get("trades", {})
            for m_name, pips in trades_dict.items():
                if m_name in all_week_trades:
                    all_week_trades[m_name].extend(pips)

    # Compute metrics
    scorecard = []
    years = len(disc_weeks) / 52.0

    for name, pips in all_week_trades.items():
        if len(pips) == 0:
            continue
        pips_arr = np.array(pips)
        trades = len(pips_arr)
        wins = np.sum(pips_arr > 0)
        losses = np.sum(pips_arr < 0)
        win_rate = (wins / trades * 100.0) if trades > 0 else 0.0

        gross_win = np.sum(pips_arr[pips_arr > 0])
        gross_loss = abs(np.sum(pips_arr[pips_arr < 0]))
        pf = (gross_win / gross_loss) if gross_loss > 0 else (99.0 if gross_win > 0 else 0.0)

        net_pips = float(np.sum(pips_arr))
        ann_pips = net_pips / years if years > 0 else 0.0

        equity = np.cumsum(pips_arr)
        peak = np.maximum.accumulate(equity)
        dd = peak - equity
        max_dd = float(np.max(dd)) if len(dd) > 0 else 0.0
        ret_to_dd = (net_pips / max_dd) if max_dd > 0 else 0.0

        scorecard.append({
            "model": name,
            "trades": int(trades),
            "win_rate": round(float(win_rate), 1),
            "profit_factor": round(float(pf), 2),
            "net_pips": round(net_pips, 1),
            "annual_pips": round(ann_pips, 1),
            "max_dd_pips": round(max_dd, 1),
            "return_to_dd": round(ret_to_dd, 2),
            "expectancy_pips": round(float(np.mean(pips_arr)), 2)
        })

    # Sort by profit factor and net pips
    scorecard.sort(key=lambda x: (x["profit_factor"], x["net_pips"]), reverse=True)

    print("\n" + "=" * 110)
    print(f"{'Model Name':<42} | {'Trades':<6} | {'WR %':<6} | {'PF':<5} | {'Net Pips':<9} | {'Ann Pips':<8} | {'Max DD':<7} | {'Ret/DD':<6} | {'Exp'}")
    print("-" * 110)
    for row in scorecard:
        print(f"{row['model']:<42} | {row['trades']:<6} | {row['win_rate']:<6.1f} | {row['profit_factor']:<5.2f} | {row['net_pips']:<+9.1f} | {row['annual_pips']:<+8.1f} | {row['max_dd_pips']:<7.1f} | {row['return_to_dd']:<6.2f} | {row['expectancy_pips']:+.2f}")
    print("=" * 110)

    out_file = RESULTS_DIR / "audnzd_peer_fleet_screening.json"
    with open(out_file, "w") as f:
        json.dump({
            "asset": "AUDNZD",
            "partition": "discovery_70",
            "total_weeks": len(disc_weeks),
            "years_equivalent": round(years, 2),
            "timestamp_utc": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "scorecard": scorecard
        }, f, indent=2)
    print(f"\n[Saved] Complete screening scorecard saved to: {out_file}")


if __name__ == "__main__":
    main()
