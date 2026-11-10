#!/usr/bin/env python3
"""
================================================================================
  ALPHA DISCOVERY LAB: EURGBP SPECIALIST EXPANSION & SCREENING (PHASE 1)
================================================================================
Screens 40+ diverse quantitative architectures tailored to EURGBP's cointegrated
microstructure across the 70% Quarantined Discovery Partition (210 weeks / ~60M ticks).

Target Qualification Hurdle:
  - Annualized Net Return >= 15.0% / year (Annual Pips / SL_pips >= 15.0 R/year)
  - Profit Factor >= 1.25
  - Trades >= 150 (minimum statistical significance)
  - Return / Drawdown >= 4.0x
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

# OpenBLAS thread control
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["NUMBA_NUM_THREADS"] = "1"

sys.dont_write_bytecode = True

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tick_engine.core.engine import run_matching_loop
from tick_engine.core.dataloader import get_week_dataset, get_discovery_weeks
from tick_engine.strategies.base import BaseStrategy, SIGNAL_LONG, SIGNAL_SHORT
from data_engineering_suite.bars import build_time_bars

PIP_SIZE = 0.0001
MAX_SPREAD = 1.8
RESULTS_DIR = ROOT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# Base Strategy Class
# ==============================================================================
class BaseEURGBPDiscoveryStrategy(BaseStrategy):
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
# 1. Fibonacci Golden Pocket (adapted for EURGBP cointegrated swings)
# ==============================================================================
class EURGBPFibGoldenPocket(BaseEURGBPDiscoveryStrategy):
    def __init__(self, name: str, sl_pips: float, tp_pips: float, lookback_h1: int = 48, min_wick: float = 0.35, **kwargs):
        super().__init__(name=name, sl_pips=sl_pips, tp_pips=tp_pips, **kwargs)
        self.lookback_h1 = int(lookback_h1)
        self.min_wick = float(min_wick)

    def generate_signals_from_bars(self, ticks: np.ndarray, m15: pd.DataFrame, h1: pd.DataFrame, h4: pd.DataFrame):
        n = len(ticks)
        signals = np.zeros(n, dtype=np.int8)
        sl_arr = np.zeros(n, dtype=np.float64)
        tp_arr = np.zeros(n, dtype=np.float64)

        if len(m15) < 30 or len(h1) < self.lookback_h1 or len(h4) < 15:
            return signals, sl_arr, tp_arr

        h4_ema20 = h4['close'].ewm(span=20, adjust=False).mean()
        h4_trend = np.where(h4['close'] > h4_ema20, 1, -1)
        h4_bias = pd.Series(h4_trend, index=h4.index).shift(1).reindex(m15.index, method='ffill').fillna(0).values

        h1_high = h1['high'].rolling(self.lookback_h1).max().shift(1)
        h1_low = h1['low'].rolling(self.lookback_h1).min().shift(1)
        m15_swing_h = h1_high.reindex(m15.index, method='ffill').values
        m15_swing_l = h1_low.reindex(m15.index, method='ffill').values

        c = m15['close'].values
        h = m15['high'].values
        l = m15['low'].values
        o = m15['open'].values
        spread = m15['spread_pips'].values
        hours = m15.index.hour.values
        time_ms = m15.index.astype('datetime64[ms]').view('int64')
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        last_entry_bar = -999

        for i in range(self.lookback_h1, len(m15)):
            if i - last_entry_bar < 6:
                continue
            hr = hours[i]
            if not (6 <= hr <= 17):
                continue
            if spread[i] > MAX_SPREAD:
                continue

            bias = h4_bias[i]
            sw_h = m15_swing_h[i]
            sw_l = m15_swing_l[i]
            if np.isnan(sw_h) or np.isnan(sw_l):
                continue
            rng = sw_h - sw_l
            if rng < 20.0 * PIP_SIZE:
                continue

            bar_rng = h[i] - l[i]
            if bar_rng <= 0:
                continue

            if bias > 0:
                fib_618 = sw_h - 0.618 * rng
                fib_786 = sw_h - 0.786 * rng
                if l[i] <= fib_618 and c[i] >= fib_786:
                    lower_wick = min(o[i], c[i]) - l[i]
                    if (lower_wick / bar_rng) >= self.min_wick and c[i] > o[i]:
                        idx = np.searchsorted(tick_ts, time_ms[i] + dur_ms)
                        if idx < n:
                            signals[idx] = SIGNAL_LONG
                            sl_arr[idx] = self.sl_pips * PIP_SIZE
                            tp_arr[idx] = self.tp_pips * PIP_SIZE
                            last_entry_bar = i

            elif bias < 0:
                fib_618 = sw_l + 0.618 * rng
                fib_786 = sw_l + 0.786 * rng
                if h[i] >= fib_618 and c[i] <= fib_786:
                    upper_wick = h[i] - max(o[i], c[i])
                    if (upper_wick / bar_rng) >= self.min_wick and c[i] < o[i]:
                        idx = np.searchsorted(tick_ts, time_ms[i] + dur_ms)
                        if idx < n:
                            signals[idx] = SIGNAL_SHORT
                            sl_arr[idx] = self.sl_pips * PIP_SIZE
                            tp_arr[idx] = self.tp_pips * PIP_SIZE
                            last_entry_bar = i

        return signals, sl_arr, tp_arr


# ==============================================================================
# 2. Frankfurt / London Liquidity Sweep & Reversal
# ==============================================================================
class EURGBPFrankfurtSweepFade(BaseEURGBPDiscoveryStrategy):
    def __init__(self, name: str, sl_pips: float, tp_pips: float, sweep_min_pips: float = 1.5, **kwargs):
        super().__init__(name=name, sl_pips=sl_pips, tp_pips=tp_pips, **kwargs)
        self.sweep_min = float(sweep_min_pips) * PIP_SIZE

    def generate_signals_from_bars(self, ticks: np.ndarray, m15: pd.DataFrame, h1: pd.DataFrame, h4: pd.DataFrame):
        n = len(ticks)
        signals = np.zeros(n, dtype=np.int8)
        sl_arr = np.zeros(n, dtype=np.float64)
        tp_arr = np.zeros(n, dtype=np.float64)

        if len(m15) < 30:
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

        ff_high = {}
        ff_low = {}
        unique_dates = np.unique(dates)
        for d in unique_dates:
            mask = (dates == d) & (hours == 6)
            if np.any(mask):
                ff_high[d] = np.max(h[mask])
                ff_low[d] = np.min(l[mask])

        last_trade_date = None

        for i in range(15, len(m15)):
            hr = hours[i]
            if not (7 <= hr <= 10):
                continue
            if spread[i] > MAX_SPREAD:
                continue
            dt = dates[i]
            if dt == last_trade_date or dt not in ff_high:
                continue

            fh = ff_high[dt]
            fl = ff_low[dt]
            bar_rng = h[i] - l[i]
            if bar_rng <= 0:
                continue

            # Sweep of Frankfurt High -> Bearish Reversal
            if h[i] >= fh + self.sweep_min and c[i] < fh:
                upper_wick = h[i] - max(o[i], c[i])
                if (upper_wick / bar_rng) >= 0.40 and c[i] < o[i]:
                    idx = np.searchsorted(tick_ts, time_ms[i] + dur_ms)
                    if idx < n:
                        signals[idx] = SIGNAL_SHORT
                        sl_arr[idx] = self.sl_pips * PIP_SIZE
                        tp_arr[idx] = self.tp_pips * PIP_SIZE
                        last_trade_date = dt

            # Sweep of Frankfurt Low -> Bullish Reversal
            elif l[i] <= fl - self.sweep_min and c[i] > fl:
                lower_wick = min(o[i], c[i]) - l[i]
                if (lower_wick / bar_rng) >= 0.40 and c[i] > o[i]:
                    idx = np.searchsorted(tick_ts, time_ms[i] + dur_ms)
                    if idx < n:
                        signals[idx] = SIGNAL_LONG
                        sl_arr[idx] = self.sl_pips * PIP_SIZE
                        tp_arr[idx] = self.tp_pips * PIP_SIZE
                        last_trade_date = dt

        return signals, sl_arr, tp_arr


# ==============================================================================
# 3. Keltner Midline Expansion (DISC77 adapted for EURGBP)
# ==============================================================================
class EURGBPKeltnerMidlineExpansion(BaseEURGBPDiscoveryStrategy):
    def __init__(self, name: str, sl_pips: float, tp_pips: float, **kwargs):
        super().__init__(name=name, sl_pips=sl_pips, tp_pips=tp_pips, **kwargs)

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
        m15_atr10 = (m15['high'] - m15['low']).rolling(10).mean().values

        c = m15['close'].values
        h = m15['high'].values
        l = m15['low'].values
        o = m15['open'].values
        spread = m15['spread_pips'].values
        hours = m15.index.hour.values
        time_ms = m15.index.astype('datetime64[ms]').view('int64')
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        last_entry_bar = -999

        for i in range(20, len(m15)):
            if i - last_entry_bar < 6:
                continue
            hr = hours[i]
            if not (7 <= hr <= 16):
                continue
            if spread[i] > MAX_SPREAD:
                continue

            bias = h4_bias[i]
            mid = m15_ema20[i]
            atr = m15_atr10[i]
            if np.isnan(atr) or atr <= 0:
                continue

            bar_rng = h[i] - l[i]
            body = abs(c[i] - o[i])

            if bias > 0:
                if l[i] <= mid + 1.0 * PIP_SIZE and c[i] > mid:
                    if bar_rng >= 1.15 * atr and (body / bar_rng) >= 0.55 and c[i] > o[i]:
                        idx = np.searchsorted(tick_ts, time_ms[i] + dur_ms)
                        if idx < n:
                            signals[idx] = SIGNAL_LONG
                            sl_arr[idx] = self.sl_pips * PIP_SIZE
                            tp_arr[idx] = self.tp_pips * PIP_SIZE
                            last_entry_bar = i

            elif bias < 0:
                if h[i] >= mid - 1.0 * PIP_SIZE and c[i] < mid:
                    if bar_rng >= 1.15 * atr and (body / bar_rng) >= 0.55 and c[i] < o[i]:
                        idx = np.searchsorted(tick_ts, time_ms[i] + dur_ms)
                        if idx < n:
                            signals[idx] = SIGNAL_SHORT
                            sl_arr[idx] = self.sl_pips * PIP_SIZE
                            tp_arr[idx] = self.tp_pips * PIP_SIZE
                            last_entry_bar = i

        return signals, sl_arr, tp_arr


# ==============================================================================
# 4. Chaikin Volatility Expansion Surge (DISC82 adapted for EURGBP)
# ==============================================================================
class EURGBPChaikinVolSurge(BaseEURGBPDiscoveryStrategy):
    def __init__(self, name: str, sl_pips: float, tp_pips: float, **kwargs):
        super().__init__(name=name, sl_pips=sl_pips, tp_pips=tp_pips, **kwargs)

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

        hl_diff = (m15['high'] - m15['low']).ewm(span=10, adjust=False).mean()
        chaikin_vol = (hl_diff - hl_diff.shift(10)) / (hl_diff.shift(10) + 1e-9)

        c = m15['close'].values
        h = m15['high'].values
        l = m15['low'].values
        o = m15['open'].values
        cvol = chaikin_vol.values
        spread = m15['spread_pips'].values
        hours = m15.index.hour.values
        time_ms = m15.index.astype('datetime64[ms]').view('int64')
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        last_entry_bar = -999

        for i in range(20, len(m15)):
            if i - last_entry_bar < 6:
                continue
            hr = hours[i]
            if not (7 <= hr <= 16):
                continue
            if spread[i] > MAX_SPREAD:
                continue

            bias = h4_bias[i]
            cv = cvol[i]
            if np.isnan(cv) or cv < 0.25:
                continue

            bar_rng = h[i] - l[i]
            body = abs(c[i] - o[i])
            if bar_rng <= 0:
                continue

            prev_4_max = np.max(h[i-4:i])
            prev_4_min = np.min(l[i-4:i])

            if bias > 0 and c[i] > prev_4_max and c[i] > o[i] and (body / bar_rng) >= 0.50:
                idx = np.searchsorted(tick_ts, time_ms[i] + dur_ms)
                if idx < n:
                    signals[idx] = SIGNAL_LONG
                    sl_arr[idx] = self.sl_pips * PIP_SIZE
                    tp_arr[idx] = self.tp_pips * PIP_SIZE
                    last_entry_bar = i

            elif bias < 0 and c[i] < prev_4_min and c[i] < o[i] and (body / bar_rng) >= 0.50:
                idx = np.searchsorted(tick_ts, time_ms[i] + dur_ms)
                if idx < n:
                    signals[idx] = SIGNAL_SHORT
                    sl_arr[idx] = self.sl_pips * PIP_SIZE
                    tp_arr[idx] = self.tp_pips * PIP_SIZE
                    last_entry_bar = i

        return signals, sl_arr, tp_arr


# ==============================================================================
# 5. Dual EMA Pullback Trend Scalper (S10 adapted for EURGBP)
# ==============================================================================
class EURGBPDualEMAPullback(BaseEURGBPDiscoveryStrategy):
    def __init__(self, name: str, sl_pips: float, tp_pips: float, fast_span: int = 9, slow_span: int = 21, **kwargs):
        super().__init__(name=name, sl_pips=sl_pips, tp_pips=tp_pips, **kwargs)
        self.fast_span = int(fast_span)
        self.slow_span = int(slow_span)

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

        m15_fast = m15['close'].ewm(span=self.fast_span, adjust=False).mean().values
        m15_slow = m15['close'].ewm(span=self.slow_span, adjust=False).mean().values

        c = m15['close'].values
        h = m15['high'].values
        l = m15['low'].values
        o = m15['open'].values
        spread = m15['spread_pips'].values
        hours = m15.index.hour.values
        time_ms = m15.index.astype('datetime64[ms]').view('int64')
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        last_entry_bar = -999

        for i in range(25, len(m15)):
            if i - last_entry_bar < 6:
                continue
            hr = hours[i]
            if not (7 <= hr <= 16):
                continue
            if spread[i] > MAX_SPREAD:
                continue

            bias = h4_bias[i]
            fast = m15_fast[i]
            slow = m15_slow[i]
            bar_rng = h[i] - l[i]
            if bar_rng <= 0:
                continue

            if bias > 0 and fast > slow:
                if l[i] <= fast and c[i] > fast and c[i] > o[i]:
                    lower_wick = min(o[i], c[i]) - l[i]
                    if (lower_wick / bar_rng) >= 0.35:
                        idx = np.searchsorted(tick_ts, time_ms[i] + dur_ms)
                        if idx < n:
                            signals[idx] = SIGNAL_LONG
                            sl_arr[idx] = self.sl_pips * PIP_SIZE
                            tp_arr[idx] = self.tp_pips * PIP_SIZE
                            last_entry_bar = i

            elif bias < 0 and fast < slow:
                if h[i] >= fast and c[i] < fast and c[i] < o[i]:
                    upper_wick = h[i] - max(o[i], c[i])
                    if (upper_wick / bar_rng) >= 0.35:
                        idx = np.searchsorted(tick_ts, time_ms[i] + dur_ms)
                        if idx < n:
                            signals[idx] = SIGNAL_SHORT
                            sl_arr[idx] = self.sl_pips * PIP_SIZE
                            tp_arr[idx] = self.tp_pips * PIP_SIZE
                            last_entry_bar = i

        return signals, sl_arr, tp_arr


# ==============================================================================
# 6. Session Anchored VWAP 2.0-SD Band Fade
# ==============================================================================
class EURGBPSessionVWAPBandFade(BaseEURGBPDiscoveryStrategy):
    def __init__(self, name: str, sl_pips: float, tp_pips: float, band_mult: float = 2.0, **kwargs):
        super().__init__(name=name, sl_pips=sl_pips, tp_pips=tp_pips, **kwargs)
        self.band_mult = float(band_mult)

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

        dates = m15.index.date
        c = m15['close'].values
        h = m15['high'].values
        l = m15['low'].values
        o = m15['open'].values
        spread = m15['spread_pips'].values
        hours = m15.index.hour.values
        vol = m15['tick_volume'].values if 'tick_volume' in m15.columns else np.ones(len(m15))
        time_ms = m15.index.astype('datetime64[ms]').view('int64')
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        vwap = np.zeros(len(m15))
        sd = np.zeros(len(m15))

        cur_date = None
        cum_pv = 0.0
        cum_v = 0.0
        price_buffer = []

        for i in range(len(m15)):
            d = dates[i]
            if d != cur_date:
                cur_date = d
                cum_pv = 0.0
                cum_v = 0.0
                price_buffer = []

            tp_val = (h[i] + l[i] + c[i]) / 3.0
            v_val = max(1.0, vol[i])
            cum_pv += tp_val * v_val
            cum_v += v_val
            price_buffer.append(tp_val)

            vwap[i] = cum_pv / cum_v
            sd[i] = np.std(price_buffer) if len(price_buffer) > 2 else 0.0

        last_entry_bar = -999

        for i in range(15, len(m15)):
            if i - last_entry_bar < 6:
                continue
            hr = hours[i]
            if not (7 <= hr <= 16):
                continue
            if spread[i] > MAX_SPREAD:
                continue

            bias = h4_bias[i]
            v = vwap[i]
            s = sd[i]
            if s <= 0:
                continue

            upper = v + self.band_mult * s
            lower = v - self.band_mult * s
            bar_rng = h[i] - l[i]
            if bar_rng <= 0:
                continue

            if bias > 0 and l[i] <= lower and c[i] > lower:
                lower_wick = min(o[i], c[i]) - l[i]
                if (lower_wick / bar_rng) >= 0.35:
                    idx = np.searchsorted(tick_ts, time_ms[i] + dur_ms)
                    if idx < n:
                        signals[idx] = SIGNAL_LONG
                        sl_arr[idx] = self.sl_pips * PIP_SIZE
                        tp_arr[idx] = self.tp_pips * PIP_SIZE
                        last_entry_bar = i

            elif bias < 0 and h[i] >= upper and c[i] < upper:
                upper_wick = h[i] - max(o[i], c[i])
                if (upper_wick / bar_rng) >= 0.35:
                    idx = np.searchsorted(tick_ts, time_ms[i] + dur_ms)
                    if idx < n:
                        signals[idx] = SIGNAL_SHORT
                        sl_arr[idx] = self.sl_pips * PIP_SIZE
                        tp_arr[idx] = self.tp_pips * PIP_SIZE
                        last_entry_bar = i

        return signals, sl_arr, tp_arr


# ==============================================================================
# 7. Daily Pivot Trend Pullback Rejection (DISC74 tuned for EURGBP)
# ==============================================================================
class EURGBPDailyPivotPullback(BaseEURGBPDiscoveryStrategy):
    def __init__(self, name: str, sl_pips: float, tp_pips: float, **kwargs):
        super().__init__(name=name, sl_pips=sl_pips, tp_pips=tp_pips, **kwargs)

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

        # Daily pivots
        daily = m15.resample('1D').agg({'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
        pp = (daily['high'] + daily['low'] + daily['close']) / 3.0
        s1 = 2 * pp - daily['high']
        r1 = 2 * pp - daily['low']

        pp_m15 = pp.shift(1).reindex(m15.index, method='ffill').values
        s1_m15 = s1.shift(1).reindex(m15.index, method='ffill').values
        r1_m15 = r1.shift(1).reindex(m15.index, method='ffill').values

        c = m15['close'].values
        h = m15['high'].values
        l = m15['low'].values
        o = m15['open'].values
        spread = m15['spread_pips'].values
        hours = m15.index.hour.values
        time_ms = m15.index.astype('datetime64[ms]').view('int64')
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        last_entry_bar = -999

        for i in range(15, len(m15)):
            if i - last_entry_bar < 6:
                continue
            hr = hours[i]
            if not (7 <= hr <= 16):
                continue
            if spread[i] > MAX_SPREAD:
                continue

            bias = h4_bias[i]
            p_val = pp_m15[i]
            s1_val = s1_m15[i]
            r1_val = r1_m15[i]
            if np.isnan(p_val):
                continue

            bar_rng = h[i] - l[i]
            if bar_rng <= 0:
                continue

            # Bullish test of Pivot P or S1
            if bias > 0:
                near_p = abs(l[i] - p_val) <= 2.0 * PIP_SIZE
                near_s1 = abs(l[i] - s1_val) <= 2.0 * PIP_SIZE
                if (near_p or near_s1) and c[i] > o[i]:
                    lower_wick = min(o[i], c[i]) - l[i]
                    if (lower_wick / bar_rng) >= 0.35:
                        idx = np.searchsorted(tick_ts, time_ms[i] + dur_ms)
                        if idx < n:
                            signals[idx] = SIGNAL_LONG
                            sl_arr[idx] = self.sl_pips * PIP_SIZE
                            tp_arr[idx] = self.tp_pips * PIP_SIZE
                            last_entry_bar = i

            elif bias < 0:
                near_p = abs(h[i] - p_val) <= 2.0 * PIP_SIZE
                near_r1 = abs(h[i] - r1_val) <= 2.0 * PIP_SIZE
                if (near_p or near_r1) and c[i] < o[i]:
                    upper_wick = h[i] - max(o[i], c[i])
                    if (upper_wick / bar_rng) >= 0.35:
                        idx = np.searchsorted(tick_ts, time_ms[i] + dur_ms)
                        if idx < n:
                            signals[idx] = SIGNAL_SHORT
                            sl_arr[idx] = self.sl_pips * PIP_SIZE
                            tp_arr[idx] = self.tp_pips * PIP_SIZE
                            last_entry_bar = i

        return signals, sl_arr, tp_arr


# ==============================================================================
# Build Fleet of EURGBP Discovery Architectures
# ==============================================================================
def build_eurgbp_expansion_fleet() -> List[Tuple[str, Any]]:
    fleet = []

    # 1. Fibonacci Golden Pocket (various lookbacks, SLs and TPs)
    for lb in [36, 48, 72]:
        for sl, tp in [(6.5, 13.0), (7.0, 14.0), (7.0, 17.5), (8.0, 16.0)]:
            name = f"EG_FibGoldenPocket_LB{lb}_SL{sl}_TP{tp}"
            fleet.append((name, EURGBPFibGoldenPocket(name=name, sl_pips=sl, tp_pips=tp, lookback_h1=lb, min_wick=0.35)))

    # 2. Frankfurt Sweep Fade
    for sweep in [1.5, 2.0, 2.5]:
        for sl, tp in [(6.5, 13.0), (7.0, 14.0), (7.5, 15.0)]:
            name = f"EG_FrankfurtSweep_Sw{sweep}_SL{sl}_TP{tp}"
            fleet.append((name, EURGBPFrankfurtSweepFade(name=name, sl_pips=sl, tp_pips=tp, sweep_min_pips=sweep)))

    # 3. Keltner Midline Expansion
    for sl, tp in [(6.0, 12.0), (6.5, 13.0), (7.0, 14.0), (7.5, 15.0)]:
        name = f"EG_KeltnerMidline_SL{sl}_TP{tp}"
        fleet.append((name, EURGBPKeltnerMidlineExpansion(name=name, sl_pips=sl, tp_pips=tp)))

    # 4. Chaikin Volatility Surge
    for sl, tp in [(6.0, 12.0), (6.5, 13.0), (7.0, 14.0), (7.5, 15.0)]:
        name = f"EG_ChaikinVolSurge_SL{sl}_TP{tp}"
        fleet.append((name, EURGBPChaikinVolSurge(name=name, sl_pips=sl, tp_pips=tp)))

    # 5. Dual EMA Pullback
    for f_span, s_span in [(9, 21), (13, 34)]:
        for sl, tp in [(6.5, 13.0), (7.0, 14.0), (7.5, 15.0)]:
            name = f"EG_DualEMA_{f_span}_{s_span}_SL{sl}_TP{tp}"
            fleet.append((name, EURGBPDualEMAPullback(name=name, sl_pips=sl, tp_pips=tp, fast_span=f_span, slow_span=s_span)))

    # 6. Session VWAP Band Fade
    for b_mult in [2.0, 2.5]:
        for sl, tp in [(6.5, 13.0), (7.0, 14.0)]:
            name = f"EG_SessionVWAP_B{b_mult}_SL{sl}_TP{tp}"
            fleet.append((name, EURGBPSessionVWAPBandFade(name=name, sl_pips=sl, tp_pips=tp, band_mult=b_mult)))

    # 7. Daily Pivot Pullback
    for sl, tp in [(6.5, 13.0), (7.0, 14.0), (7.0, 17.5), (7.5, 15.0)]:
        name = f"EG_DailyPivot_SL{sl}_TP{tp}"
        fleet.append((name, EURGBPDailyPivotPullback(name=name, sl_pips=sl, tp_pips=tp)))

    return fleet


# ==============================================================================
# Worker Function
# ==============================================================================
def evaluate_week_batch(week_num: int) -> Dict[str, List[Dict[str, Any]]]:
    t_res = get_week_dataset(week_num, raw_dir="data/EURGBP")
    if t_res is None:
        return {}
    ticks, meta = t_res if isinstance(t_res, tuple) else (t_res, None)
    if ticks is None or len(ticks) < 1000:
        return {}

    fleet = build_eurgbp_expansion_fleet()
    week_results = {}

    for name, strategy in fleet:
        signals, sl_arr, tp_arr = strategy.generate_signals(ticks)
        trade_log = run_matching_loop(ticks, signals, pip_size=PIP_SIZE, sl_pips=strategy.sl_pips, tp_pips=strategy.tp_pips)

        parsed_trades = []
        for t in trade_log:
            parsed_trades.append({
                "pnl_pips": float(t['pips']),
                "is_win": bool(t['pips'] > 0),
                "sl_pips": strategy.sl_pips,
            })
        week_results[name] = parsed_trades

    return week_results


# ==============================================================================
# Main Discovery Runner
# ==============================================================================
def main():
    start_time = time.time()
    print("=" * 80)
    print("  EURGBP DEEP ALPHA EXPANSION DISCOVERY (PHASE 1 - 210 WEEKS)")
    print("=" * 80)

    discovery_weeks = get_discovery_weeks(raw_dir="data/EURGBP")
    print(f"Loaded {len(discovery_weeks)} discovery weeks for EURGBP.")

    sample_fleet = build_eurgbp_expansion_fleet()
    print(f"Screening {len(sample_fleet)} distinct quantitative architectures across 210 weeks...")

    fleet_trades: Dict[str, List[Dict[str, Any]]] = {name: [] for name, _ in sample_fleet}

    completed = 0
    with ProcessPoolExecutor(max_workers=12) as executor:
        futures = {executor.submit(evaluate_week_batch, w): w for w in discovery_weeks}
        for future in as_completed(futures):
            res = future.result()
            for name, trades in res.items():
                fleet_trades[name].extend(trades)
            completed += 1
            if completed % 35 == 0 or completed == len(discovery_weeks):
                pct = completed / len(discovery_weeks) * 100
                print(f"  Progress: {completed}/{len(discovery_weeks)} weeks ({pct:.1f}%) in {time.time() - start_time:.1f}s")

    print("\nProcessing metrics and ranking candidates by Annualized Net Return...")

    DISCOVERY_YEARS = 4.04  # 210 weeks / 52.0
    scorecard = []

    for name, _ in sample_fleet:
        trades = fleet_trades[name]
        n_trades = len(trades)
        if n_trades < 30:
            continue

        pnls = np.array([t["pnl_pips"] for t in trades])
        sl_val = trades[0]["sl_pips"]
        net_pips = float(np.sum(pnls))
        wins = pnls[pnls > 0]
        losses = pnls[pnls < 0]

        gross_profit = float(np.sum(wins)) if len(wins) > 0 else 0.0
        gross_loss = float(abs(np.sum(losses))) if len(losses) > 0 else 1e-9
        pf = gross_profit / gross_loss
        wr = len(wins) / n_trades * 100.0

        cum = np.cumsum(pnls)
        peaks = np.maximum.accumulate(cum)
        dd_series = peaks - cum
        max_dd = float(np.max(dd_series)) if len(dd_series) > 0 else 1.0
        ret_to_dd = net_pips / max_dd if max_dd > 0 else 0.0

        ann_pips = net_pips / DISCOVERY_YEARS
        ann_ret_pct = (ann_pips / sl_val) * 1.0 * 100.0  # 1% risk per trade

        scorecard.append({
            "name": name,
            "sl": sl_val,
            "trades": n_trades,
            "net_pips": round(net_pips, 1),
            "ann_pips": round(ann_pips, 1),
            "ann_ret_pct": round(ann_ret_pct, 1),
            "profit_factor": round(pf, 2),
            "win_rate": round(wr, 1),
            "max_dd": round(max_dd, 1),
            "ret_to_dd": round(ret_to_dd, 2),
            "exp_pips": round(net_pips / n_trades, 2)
        })

    if len(scorecard) == 0:
        print("No models met the trade count threshold.")
        return

    df = pd.DataFrame(scorecard)
    df = df.sort_values(by="ann_ret_pct", ascending=False)

    out_csv = RESULTS_DIR / "eurgbp_expansion_discovery_scorecard.csv"
    df.to_csv(out_csv, index=False)

    print("\n" + "=" * 120)
    print("  TOP EURGBP CANDIDATES RANKED BY ANNUALIZED RETURN % (DISCOVERY PARTITION)")
    print("=" * 120)
    print(f"{'Rank':<4} | {'Strategy Name':<42} | {'Trades':<6} | {'Net Pips':<9} | {'Ann Ret%':<8} | {'PF':<5} | {'WR%':<5} | {'Max DD':<7} | {'Ret/DD':<7} | {'Exp'}")
    print("-" * 120)

    top_candidates = df.head(30)
    for idx, (_, row) in enumerate(top_candidates.iterrows(), 1):
        flag = "⭐" if row['ann_ret_pct'] >= 15.0 and row['profit_factor'] >= 1.25 else "  "
        print(f"{idx:<4} | {row['name']:<42} | {int(row['trades']):<6} | {row['net_pips']:<+9.1f} | {row['ann_ret_pct']:<+7.1f}% | {row['profit_factor']:<5.2f} | {row['win_rate']:<5.1f} | {row['max_dd']:<7.1f} | {row['ret_to_dd']:<6.2f}x | {row['exp_pips']:<+5.2f} {flag}")

    print("-" * 120)
    qualified = df[(df['ann_ret_pct'] >= 15.0) & (df['profit_factor'] >= 1.25) & (df['ret_to_dd'] >= 4.0)]
    print(f"Total architectures exceeding >= 15.0% annual return and PF >= 1.25: {len(qualified)}")
    print(f"Detailed scorecard saved to: {out_csv}")
    print(f"Total Elapsed Time: {time.time() - start_time:.2f}s")


if __name__ == "__main__":
    main()
