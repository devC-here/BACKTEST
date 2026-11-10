#!/usr/bin/env python3
"""
================================================================================
  ALPHA DISCOVERY LAB USDCAD: COMMODITY-MAJOR SPECIALIST ALPHA MINING
================================================================================
High-performance batch-evaluated discovery lab for USDCAD:
  1. USDCAD Daily Pivot Wall Rejection (DISC46 adapted with shifted H4 macro)
  2. USDCAD Weekly Pivot Wall Rejection (DISC47 adapted with shifted H4 macro)
  3. USDCAD Camarilla L3/H3 Floor/Ceiling Rejection (DISC48 adapted)
  4. USDCAD Fibonacci Golden Pocket Pin Defense (DISC52 adapted)
  5. USDCAD NY Momentum Range Ignition (London-NY overlap breakout)
  6. USDCAD H4 Macro Donchian Structure Retest (DISC42 adapted)

Evaluates:
  - Phase 1: 52-Week 2021 In-Sample Calendar Tick Audit under 0.7p friction
  - Phase 2: Standalone Hurdle Filter (Annual Net Return >= 12.0% / R >= 12.0)
  - Phase 3: 90-Week Quarantined Holdout (Seed 42 / 126.6M Ticks Unseen Data)
  - Phase 4: Dual-Certification (OOS Ann Ret >= 10.0% - 12.0% and Net Pips > 0)
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

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["NUMBA_NUM_THREADS"] = "1"
sys.dont_write_bytecode = True

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tick_engine.core.engine import run_matching_loop
from tick_engine.core.dataloader import get_week_dataset
from tick_engine.strategies.base import BaseStrategy, SIGNAL_LONG, SIGNAL_SHORT
from data_engineering_suite.bars import build_time_bars

PIP_SIZE = 0.0001
MAX_SPREAD = 2.4
COMMISSION_PIPS = 0.5
ENTRY_SLIPPAGE_PIPS = 0.2
RESULTS_DIR = ROOT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


class BaseUSDCADStrategy(BaseStrategy):
    def __init__(self, name: str, sl_pips: float, tp_pips: float, **kwargs):
        super().__init__(**kwargs)
        self.name = name
        self.symbol = "USDCAD"
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
# 1. Daily Pivot Wall Rejection (Zero-Lookahead Causal Alignment)
# ==============================================================================
class USDCADDailyPivotWall(BaseUSDCADStrategy):
    def __init__(self, name: str, sl_pips: float, tp_pips: float, min_wick: float = 0.40, max_dist_pips: float = 4.5, **kwargs):
        super().__init__(name=name, sl_pips=sl_pips, tp_pips=tp_pips, **kwargs)
        self.min_wick = float(min_wick)
        self.max_dist_pips = float(max_dist_pips)

    def generate_signals_from_bars(self, ticks: np.ndarray, m15: pd.DataFrame, h1: pd.DataFrame, h4: pd.DataFrame):
        n = len(ticks)
        signals = np.zeros(n, dtype=np.int8)
        sl_arr = np.zeros(n, dtype=np.float64)
        tp_arr = np.zeros(n, dtype=np.float64)

        if len(m15) < 30 or len(h4) < 15:
            return signals, sl_arr, tp_arr

        daily = m15.resample('1D').agg({'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
        if len(daily) < 2:
            return signals, sl_arr, tp_arr

        prev_h = daily['high'].shift(1)
        prev_l = daily['low'].shift(1)
        prev_c = daily['close'].shift(1)
        p = (prev_h + prev_l + prev_c) / 3.0
        r1 = (2 * p) - prev_l
        s1 = (2 * p) - prev_h

        daily['s1'] = s1
        daily['r1'] = r1

        h4_ema20 = h4['close'].ewm(span=20, adjust=False).mean()
        h4_macro = np.where(h4['close'] > h4_ema20, 1, -1)

        m15_s1 = daily['s1'].reindex(m15.index, method='ffill').values
        m15_r1 = daily['r1'].reindex(m15.index, method='ffill').values
        # Causal Shift(1) on H4 Macro Filter
        m15_macro = pd.Series(h4_macro, index=h4.index).shift(1).reindex(m15.index, method='ffill').fillna(0).values

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
            if not (7 <= hr <= 18):
                continue
            if spread[i] > MAX_SPREAD:
                continue

            dt = dates[i]
            if dt == last_trade_date:
                continue

            m = m15_macro[i]
            bar_rng = (h[i] - l[i]) / PIP_SIZE
            if bar_rng < 3.0:
                continue

            sig = 0
            if m > 0 and not np.isnan(m15_s1[i]):
                dist_s1 = abs(l[i] - m15_s1[i]) / PIP_SIZE
                lower_wick = (min(o[i], c[i]) - l[i]) / PIP_SIZE
                if dist_s1 <= self.max_dist_pips and (lower_wick / bar_rng) >= self.min_wick and c[i] > o[i]:
                    sig = SIGNAL_LONG

            elif m < 0 and not np.isnan(m15_r1[i]):
                dist_r1 = abs(h[i] - m15_r1[i]) / PIP_SIZE
                upper_wick = (h[i] - max(o[i], c[i])) / PIP_SIZE
                if dist_r1 <= self.max_dist_pips and (upper_wick / bar_rng) >= self.min_wick and c[i] < o[i]:
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
# 2. Weekly Pivot Wall Rejection (Zero-Lookahead Causal Alignment)
# ==============================================================================
class USDCADWeeklyPivotWall(BaseUSDCADStrategy):
    def __init__(self, name: str, sl_pips: float, tp_pips: float, min_wick: float = 0.40, max_dist_pips: float = 5.0, **kwargs):
        super().__init__(name=name, sl_pips=sl_pips, tp_pips=tp_pips, **kwargs)
        self.min_wick = float(min_wick)
        self.max_dist_pips = float(max_dist_pips)

    def generate_signals_from_bars(self, ticks: np.ndarray, m15: pd.DataFrame, h1: pd.DataFrame, h4: pd.DataFrame):
        n = len(ticks)
        signals = np.zeros(n, dtype=np.int8)
        sl_arr = np.zeros(n, dtype=np.float64)
        tp_arr = np.zeros(n, dtype=np.float64)

        if len(m15) < 30 or len(h4) < 15:
            return signals, sl_arr, tp_arr

        weekly = m15.resample('W').agg({'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
        if len(weekly) < 2:
            return signals, sl_arr, tp_arr

        prev_h = weekly['high'].shift(1)
        prev_l = weekly['low'].shift(1)
        prev_c = weekly['close'].shift(1)
        p = (prev_h + prev_l + prev_c) / 3.0
        r1 = (2 * p) - prev_l
        s1 = (2 * p) - prev_h

        weekly['s1'] = s1
        weekly['r1'] = r1
        weekly['pivot'] = p

        h4_ema20 = h4['close'].ewm(span=20, adjust=False).mean()
        h4_macro = np.where(h4['close'] > h4_ema20, 1, -1)

        m15_s1 = weekly['s1'].reindex(m15.index, method='ffill').values
        m15_r1 = weekly['r1'].reindex(m15.index, method='ffill').values
        m15_p = weekly['pivot'].reindex(m15.index, method='ffill').values
        # Causal Shift(1) on H4 Macro Filter
        m15_macro = pd.Series(h4_macro, index=h4.index).shift(1).reindex(m15.index, method='ffill').fillna(0).values

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
            if not (7 <= hr <= 18):
                continue
            if spread[i] > MAX_SPREAD:
                continue

            dt = dates[i]
            if dt == last_trade_date:
                continue

            m = m15_macro[i]
            bar_rng = (h[i] - l[i]) / PIP_SIZE
            if bar_rng < 3.0:
                continue

            sig = 0
            if m > 0 and not np.isnan(m15_s1[i]):
                dist_s1 = abs(l[i] - m15_s1[i]) / PIP_SIZE
                dist_p = abs(l[i] - m15_p[i]) / PIP_SIZE
                lower_wick = (min(o[i], c[i]) - l[i]) / PIP_SIZE
                if (dist_s1 <= self.max_dist_pips or dist_p <= self.max_dist_pips) and (lower_wick / bar_rng) >= self.min_wick and c[i] > o[i]:
                    sig = SIGNAL_LONG

            elif m < 0 and not np.isnan(m15_r1[i]):
                dist_r1 = abs(h[i] - m15_r1[i]) / PIP_SIZE
                dist_p = abs(h[i] - m15_p[i]) / PIP_SIZE
                upper_wick = (h[i] - max(o[i], c[i])) / PIP_SIZE
                if (dist_r1 <= self.max_dist_pips or dist_p <= self.max_dist_pips) and (upper_wick / bar_rng) >= self.min_wick and c[i] < o[i]:
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
# 3. Camarilla Floor / Ceiling Rejection (Zero-Lookahead Causal Alignment)
# ==============================================================================
class USDCADCamarillaFloor(BaseUSDCADStrategy):
    def __init__(self, name: str, sl_pips: float, tp_pips: float, min_wick: float = 0.40, max_dist_pips: float = 4.0, **kwargs):
        super().__init__(name=name, sl_pips=sl_pips, tp_pips=tp_pips, **kwargs)
        self.min_wick = float(min_wick)
        self.max_dist_pips = float(max_dist_pips)

    def generate_signals_from_bars(self, ticks: np.ndarray, m15: pd.DataFrame, h1: pd.DataFrame, h4: pd.DataFrame):
        n = len(ticks)
        signals = np.zeros(n, dtype=np.int8)
        sl_arr = np.zeros(n, dtype=np.float64)
        tp_arr = np.zeros(n, dtype=np.float64)

        if len(m15) < 30 or len(h4) < 15:
            return signals, sl_arr, tp_arr

        daily = m15.resample('1D').agg({'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
        if len(daily) < 2:
            return signals, sl_arr, tp_arr

        prev_h = daily['high'].shift(1)
        prev_l = daily['low'].shift(1)
        prev_c = daily['close'].shift(1)
        rng = prev_h - prev_l

        daily['h3'] = prev_c + rng * (1.1 / 4.0)
        daily['l3'] = prev_c - rng * (1.1 / 4.0)

        h4_ema20 = h4['close'].ewm(span=20, adjust=False).mean()
        h4_macro = np.where(h4['close'] > h4_ema20, 1, -1)

        m15_l3 = daily['l3'].reindex(m15.index, method='ffill').values
        m15_h3 = daily['h3'].reindex(m15.index, method='ffill').values
        # Causal Shift(1) on H4 Macro Filter
        m15_macro = pd.Series(h4_macro, index=h4.index).shift(1).reindex(m15.index, method='ffill').fillna(0).values

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
            if not (7 <= hr <= 18):
                continue
            if spread[i] > MAX_SPREAD:
                continue

            dt = dates[i]
            if dt == last_trade_date:
                continue

            m = m15_macro[i]
            bar_rng = (h[i] - l[i]) / PIP_SIZE
            if bar_rng < 3.0:
                continue

            sig = 0
            if m > 0 and not np.isnan(m15_l3[i]):
                dist_l3 = abs(l[i] - m15_l3[i]) / PIP_SIZE
                lower_wick = (min(o[i], c[i]) - l[i]) / PIP_SIZE
                if dist_l3 <= self.max_dist_pips and (lower_wick / bar_rng) >= self.min_wick and c[i] > o[i]:
                    sig = SIGNAL_LONG

            elif m < 0 and not np.isnan(m15_h3[i]):
                dist_h3 = abs(h[i] - m15_h3[i]) / PIP_SIZE
                upper_wick = (h[i] - max(o[i], c[i])) / PIP_SIZE
                if dist_h3 <= self.max_dist_pips and (upper_wick / bar_rng) >= self.min_wick and c[i] < o[i]:
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
# 4. Fibonacci Golden Pocket (Zero-Lookahead Causal Alignment)
# ==============================================================================
class USDCADFibGoldenPocket(BaseUSDCADStrategy):
    def __init__(self, name: str, sl_pips: float, tp_pips: float, lookback_h1: int = 48, **kwargs):
        super().__init__(name=name, sl_pips=sl_pips, tp_pips=tp_pips, **kwargs)
        self.lookback_h1 = int(lookback_h1)

    def generate_signals_from_bars(self, ticks: np.ndarray, m15: pd.DataFrame, h1: pd.DataFrame, h4: pd.DataFrame):
        n = len(ticks)
        signals = np.zeros(n, dtype=np.int8)
        sl_arr = np.zeros(n, dtype=np.float64)
        tp_arr = np.zeros(n, dtype=np.float64)

        if len(m15) < 30 or len(h1) < self.lookback_h1 + 5:
            return signals, sl_arr, tp_arr

        # Causal Shift(1) on H1 Swing High and Low
        h1_high = h1['high'].rolling(self.lookback_h1).max().shift(1)
        h1_low = h1['low'].rolling(self.lookback_h1).min().shift(1)
        h1_ema20 = h1['close'].ewm(span=20, adjust=False).mean()
        h1_trend = pd.Series(np.where(h1['close'] > h1_ema20, 1, -1), index=h1.index).shift(1)

        m15_sw_h = h1_high.reindex(m15.index, method='ffill').values
        m15_sw_l = h1_low.reindex(m15.index, method='ffill').values
        m15_trend = h1_trend.reindex(m15.index, method='ffill').fillna(0).values

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
            if not (7 <= hr <= 18):
                continue
            if spread[i] > MAX_SPREAD:
                continue

            dt = dates[i]
            if dt == last_trade_date:
                continue

            sw_h = m15_sw_h[i]
            sw_l = m15_sw_l[i]
            if np.isnan(sw_h) or np.isnan(sw_l):
                continue

            rng = sw_h - sw_l
            if (rng / PIP_SIZE) < 30.0:
                continue

            tr = m15_trend[i]
            bar_rng = (h[i] - l[i]) / PIP_SIZE
            if bar_rng < 2.5:
                continue

            sig = 0
            if tr > 0:
                fib618 = sw_h - (rng * 0.618)
                fib650 = sw_h - (rng * 0.650)
                if fib650 <= l[i] <= fib618 or abs(l[i] - fib618) / PIP_SIZE <= 3.5:
                    lower_wick = (min(o[i], c[i]) - l[i]) / PIP_SIZE
                    if (lower_wick / bar_rng) >= 0.40 and c[i] > o[i]:
                        sig = SIGNAL_LONG

            elif tr < 0:
                fib618 = sw_l + (rng * 0.618)
                fib650 = sw_l + (rng * 0.650)
                if fib618 <= h[i] <= fib650 or abs(h[i] - fib618) / PIP_SIZE <= 3.5:
                    upper_wick = (h[i] - max(o[i], c[i])) / PIP_SIZE
                    if (upper_wick / bar_rng) >= 0.40 and c[i] < o[i]:
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
# 5. London-NY Overlap Momentum Range Ignition (USDCAD Specialist)
# ==============================================================================
class USDCADNYOverlapIgnition(BaseUSDCADStrategy):
    def __init__(self, name: str, sl_pips: float, tp_pips: float, lookback_bars: int = 16, **kwargs):
        super().__init__(name=name, sl_pips=sl_pips, tp_pips=tp_pips, **kwargs)
        self.lookback_bars = int(lookback_bars)

    def generate_signals_from_bars(self, ticks: np.ndarray, m15: pd.DataFrame, h1: pd.DataFrame, h4: pd.DataFrame):
        n = len(ticks)
        signals = np.zeros(n, dtype=np.int8)
        sl_arr = np.zeros(n, dtype=np.float64)
        tp_arr = np.zeros(n, dtype=np.float64)

        if len(m15) < self.lookback_bars + 15 or len(h4) < 15:
            return signals, sl_arr, tp_arr

        h4_ema20 = h4['close'].ewm(span=20, adjust=False).mean()
        h4_macro = pd.Series(np.where(h4['close'] > h4_ema20, 1, -1), index=h4.index).shift(1)
        m15_macro = h4_macro.reindex(m15.index, method='ffill').fillna(0).values

        m15_high_roll = m15['high'].rolling(self.lookback_bars).max().shift(1).values
        m15_low_roll = m15['low'].rolling(self.lookback_bars).min().shift(1).values

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

        for i in range(self.lookback_bars + 2, len(m15)):
            hr = hours[i]
            if not (12 <= hr <= 16):
                continue
            if spread[i] > MAX_SPREAD:
                continue

            dt = dates[i]
            if dt == last_trade_date:
                continue

            m = m15_macro[i]
            box_h = m15_high_roll[i]
            box_l = m15_low_roll[i]
            if np.isnan(box_h) or np.isnan(box_l):
                continue

            rng = (box_h - box_l) / PIP_SIZE
            if not (15.0 <= rng <= 55.0):
                continue

            bar_body = abs(c[i] - o[i]) / PIP_SIZE
            bar_rng = (h[i] - l[i]) / PIP_SIZE
            if bar_rng < 4.0:
                continue

            sig = 0
            if m > 0 and c[i] > box_h and (bar_body / bar_rng) >= 0.55:
                sig = SIGNAL_LONG
            elif m < 0 and c[i] < box_l and (bar_body / bar_rng) >= 0.55:
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
# 6. H4 Macro Donchian Retest (Zero-Lookahead Causal Alignment)
# ==============================================================================
class USDCADMacroDonchianRetest(BaseUSDCADStrategy):
    def __init__(self, name: str, sl_pips: float, tp_pips: float, lookback_h4: int = 15, **kwargs):
        super().__init__(name=name, sl_pips=sl_pips, tp_pips=tp_pips, **kwargs)
        self.lookback_h4 = int(lookback_h4)

    def generate_signals_from_bars(self, ticks: np.ndarray, m15: pd.DataFrame, h1: pd.DataFrame, h4: pd.DataFrame):
        n = len(ticks)
        signals = np.zeros(n, dtype=np.int8)
        sl_arr = np.zeros(n, dtype=np.float64)
        tp_arr = np.zeros(n, dtype=np.float64)

        if len(m15) < 30 or len(h4) < self.lookback_h4 + 5:
            return signals, sl_arr, tp_arr

        h4_high = h4['high'].rolling(self.lookback_h4).max().shift(1)
        h4_low = h4['low'].rolling(self.lookback_h4).min().shift(1)
        h4_ema20 = h4['close'].ewm(span=20, adjust=False).mean()
        h4_trend = pd.Series(np.where(h4['close'] > h4_ema20, 1, -1), index=h4.index).shift(1)

        m15_don_h = h4_high.reindex(m15.index, method='ffill').values
        m15_don_l = h4_low.reindex(m15.index, method='ffill').values
        m15_trend = h4_trend.reindex(m15.index, method='ffill').fillna(0).values

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
            if not (7 <= hr <= 18):
                continue
            if spread[i] > MAX_SPREAD:
                continue

            dt = dates[i]
            if dt == last_trade_date:
                continue

            tr = m15_trend[i]
            lvl_h = m15_don_h[i]
            lvl_l = m15_don_l[i]
            if np.isnan(lvl_h) or np.isnan(lvl_l):
                continue

            bar_rng = (h[i] - l[i]) / PIP_SIZE
            if bar_rng < 3.0:
                continue

            sig = 0
            if tr > 0:
                dist = abs(l[i] - lvl_h) / PIP_SIZE
                lower_wick = (min(o[i], c[i]) - l[i]) / PIP_SIZE
                if dist <= 4.0 and (lower_wick / bar_rng) >= 0.45 and c[i] > o[i]:
                    sig = SIGNAL_LONG
            elif tr < 0:
                dist = abs(h[i] - lvl_l) / PIP_SIZE
                upper_wick = (h[i] - max(o[i], c[i])) / PIP_SIZE
                if dist <= 4.0 and (upper_wick / bar_rng) >= 0.45 and c[i] < o[i]:
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
# CANDIDATE FACTORY FOR USDCAD
# ==============================================================================
def generate_usdcad_candidate_fleet() -> List[BaseUSDCADStrategy]:
    fleet = []

    # 1. Daily Pivot Wall (SL: 12, 14, 16 | TP: 24, 28, 35, 40)
    for sl, tp in [(12.0, 24.0), (12.0, 30.0), (14.0, 28.0), (14.0, 35.0), (16.0, 32.0), (16.0, 40.0)]:
        fleet.append(USDCADDailyPivotWall(f"USDCAD_DailyPivot_SL{sl:.0f}_TP{tp:.0f}", sl_pips=sl, tp_pips=tp))

    # 2. Weekly Pivot Wall (SL: 14, 16, 18 | TP: 28, 35, 45)
    for sl, tp in [(14.0, 28.0), (14.0, 35.0), (16.0, 32.0), (16.0, 40.0), (18.0, 45.0)]:
        fleet.append(USDCADWeeklyPivotWall(f"USDCAD_WeeklyPivot_SL{sl:.0f}_TP{tp:.0f}", sl_pips=sl, tp_pips=tp))

    # 3. Camarilla Floor / Ceiling (SL: 12, 14, 16 | TP: 24, 28, 35)
    for sl, tp in [(12.0, 24.0), (12.0, 30.0), (14.0, 28.0), (14.0, 35.0), (16.0, 32.0)]:
        fleet.append(USDCADCamarillaFloor(f"USDCAD_Camarilla_SL{sl:.0f}_TP{tp:.0f}", sl_pips=sl, tp_pips=tp))

    # 4. Fibonacci Golden Pocket (LB: 36, 48, 72 | SL: 14, 16, 18 | TP: 28, 35, 45)
    for lb in [36, 48, 72]:
        for sl, tp in [(14.0, 28.0), (14.0, 35.0), (16.0, 32.0), (16.0, 40.0)]:
            fleet.append(USDCADFibGoldenPocket(f"USDCAD_FibPocket_LB{lb}_SL{sl:.0f}_TP{tp:.0f}", sl_pips=sl, tp_pips=tp, lookback_h1=lb))

    # 5. NY Overlap Momentum Range Ignition (LB: 12, 16, 20 | SL: 14, 16, 18 | TP: 28, 35, 45)
    for lb in [12, 16, 20]:
        for sl, tp in [(14.0, 28.0), (14.0, 35.0), (16.0, 32.0), (16.0, 40.0)]:
            fleet.append(USDCADNYOverlapIgnition(f"USDCAD_NYIgnition_LB{lb}_SL{sl:.0f}_TP{tp:.0f}", sl_pips=sl, tp_pips=tp, lookback_bars=lb))

    # 6. H4 Macro Donchian Retest (LB: 15, 20 | SL: 14, 16, 18 | TP: 28, 35, 45)
    for lb in [15, 20]:
        for sl, tp in [(14.0, 28.0), (14.0, 35.0), (16.0, 32.0), (16.0, 40.0)]:
            fleet.append(USDCADMacroDonchianRetest(f"USDCAD_MacroDonchian_LB{lb}_SL{sl:.0f}_TP{tp:.0f}", sl_pips=sl, tp_pips=tp, lookback_h4=lb))

    return fleet


# ==============================================================================
# HIGH PERFORMANCE BATCH EVALUATOR (Loads Each Week ONCE)
# ==============================================================================
def evaluate_week_chunk_for_all(args):
    weeks_chunk, cand_classes_and_params = args
    raw_dir = Path("data/USDCAD")

    # Instantiate strategies inside worker
    strats = []
    for cls_name, params in cand_classes_and_params:
        cls_map = {
            "USDCADDailyPivotWall": USDCADDailyPivotWall,
            "USDCADWeeklyPivotWall": USDCADWeeklyPivotWall,
            "USDCADCamarillaFloor": USDCADCamarillaFloor,
            "USDCADFibGoldenPocket": USDCADFibGoldenPocket,
            "USDCADNYOverlapIgnition": USDCADNYOverlapIgnition,
            "USDCADMacroDonchianRetest": USDCADMacroDonchianRetest
        }
        strat = cls_map[cls_name](**params)
        strats.append(strat)

    stats = {s.name: {"trades": 0, "wins": 0, "losses": 0, "win_pips": 0.0, "loss_pips": 0.0, "net_pips": 0.0} for s in strats}

    for w in weeks_chunk:
        t_res = get_week_dataset(w, raw_dir=raw_dir, auto_convert=True)
        if t_res is None:
            continue
        ticks, meta = t_res
        if len(ticks) < 1000:
            continue

        # Build bars ONCE for this week
        m15 = build_time_bars(ticks, timeframe_min=15, pip_size=PIP_SIZE)
        if len(m15) < 30:
            continue
        h1 = m15.resample('1h').agg({'high': 'max', 'low': 'min', 'close': 'last', 'open': 'first'}).dropna()
        h4 = m15.resample('4h').agg({'high': 'max', 'low': 'min', 'close': 'last', 'open': 'first'}).dropna()

        for s in strats:
            try:
                sigs, sl_arr, tp_arr = s.generate_signals_from_bars(ticks, m15, h1, h4)
            except Exception:
                continue

            if not np.any(sigs != 0):
                continue

            tr = run_matching_loop(
                ticks=ticks,
                signals=sigs,
                pip_size=PIP_SIZE,
                sl_pips=sl_arr,
                tp_pips=tp_arr,
                commission_pips=COMMISSION_PIPS,
                entry_slippage_pips=ENTRY_SLIPPAGE_PIPS
            )
            if len(tr) == 0:
                continue

            for t in tr:
                p = float(t["pips"])
                stats[s.name]["trades"] += 1
                stats[s.name]["net_pips"] += p
                if p > 0:
                    stats[s.name]["wins"] += 1
                    stats[s.name]["win_pips"] += p
                else:
                    stats[s.name]["losses"] += 1
                    stats[s.name]["loss_pips"] += abs(p)

    return stats


def run_parallel_battery(candidates: List[BaseUSDCADStrategy], weeks: List[int], n_workers: int = 6) -> Dict[str, Dict[str, Any]]:
    cand_info = []
    for c in candidates:
        params = {
            "name": c.name,
            "sl_pips": c.sl_pips,
            "tp_pips": c.tp_pips,
        }
        if hasattr(c, "lookback_h1"):
            params["lookback_h1"] = c.lookback_h1
        if hasattr(c, "lookback_bars"):
            params["lookback_bars"] = c.lookback_bars
        if hasattr(c, "lookback_h4"):
            params["lookback_h4"] = c.lookback_h4
        if hasattr(c, "min_wick"):
            params["min_wick"] = c.min_wick
        if hasattr(c, "max_dist_pips"):
            params["max_dist_pips"] = c.max_dist_pips
        cand_info.append((c.__class__.__name__, params))

    # Divide weeks into n_workers chunks
    chunk_size = max(1, len(weeks) // n_workers)
    week_chunks = [weeks[i:i + chunk_size] for i in range(0, len(weeks), chunk_size)]
    tasks = [(chunk, cand_info) for chunk in week_chunks]

    combined = {c.name: {
        "name": c.name,
        "sl_pips": c.sl_pips,
        "tp_pips": c.tp_pips,
        "trades": 0, "wins": 0, "losses": 0, "win_pips": 0.0, "loss_pips": 0.0, "net_pips": 0.0
    } for c in candidates}

    with ProcessPoolExecutor(max_workers=min(n_workers, len(tasks))) as executor:
        futures = [executor.submit(evaluate_week_chunk_for_all, t) for t in tasks]
        for fut in as_completed(futures):
            chunk_res = fut.result()
            for name, data in chunk_res.items():
                combined[name]["trades"] += data["trades"]
                combined[name]["wins"] += data["wins"]
                combined[name]["losses"] += data["losses"]
                combined[name]["win_pips"] += data["win_pips"]
                combined[name]["loss_pips"] += data["loss_pips"]
                combined[name]["net_pips"] += data["net_pips"]

    for name, c in combined.items():
        tr = c["trades"]
        c["net_pips"] = round(c["net_pips"], 1)
        c["win_pips"] = round(c["win_pips"], 1)
        c["loss_pips"] = round(c["loss_pips"], 1)
        c["win_rate"] = round((c["wins"] / tr * 100.0), 1) if tr > 0 else 0.0
        c["profit_factor"] = round((c["win_pips"] / c["loss_pips"]), 2) if c["loss_pips"] > 0 else (99.0 if c["win_pips"] > 0 else 0.0)

    return combined


def main():
    print("=" * 115)
    print("        USDCAD SPECIALIST ALPHA DISCOVERY & DUAL-CERTIFICATION BATTERY        ".center(115))
    print("=" * 115)
    print("Zero-Lookahead Causal Alignment: All HTF Indicators Shifted by 1 Bar (.shift(1))")
    print(f"Microstructure: Pip Size={PIP_SIZE}, Spread Limit={MAX_SPREAD}p, Comm={COMMISSION_PIPS}p, Slip={ENTRY_SLIPPAGE_PIPS}p\n")

    candidates = generate_usdcad_candidate_fleet()
    print(f"Synthesized {len(candidates)} USDCAD bespoke candidate strategies across 6 specialist architectures.\n")

    # Load holdout weeks
    splits_file = ROOT_DIR / "data_splits" / "master_70_30_splits.json"
    with open(splits_file, "r") as f:
        splits = json.load(f)
    holdout_weeks = sorted(splits["assets"]["USDCAD"]["holdout_test_30"]["weeks"])
    in_sample_weeks = list(range(1, 53))

    # Phase 1: In-Sample 52-Week 2021 Screen
    print(f"[PHASE 1] Executing In-Sample 52-Week (Calendar 2021) Tick Audit ({len(in_sample_weeks)} weeks)...")
    t0 = time.time()
    is_results = run_parallel_battery(candidates, in_sample_weeks, n_workers=6)
    print(f"In-Sample Audit Completed in {time.time()-t0:.1f}s.")

    for name, data in is_results.items():
        sl = data["sl_pips"]
        net_p = data["net_pips"]
        r_mult = net_p / sl if sl > 0 else 0.0
        data["r_mult"] = round(r_mult, 1)
        data["ann_return_pct"] = round(r_mult * 1.0, 1)

    # Phase 2: Standalone Hurdle Filter (>= 12.0% Net Return & PF >= 1.20)
    passing_is = [d for d in is_results.values() if d["ann_return_pct"] >= 12.0 and d["profit_factor"] >= 1.20]
    passing_is.sort(key=lambda x: x["ann_return_pct"], reverse=True)

    print(f"\nIn-Sample Hurdle Filter: {len(passing_is)} / {len(is_results)} candidates pass >= 12.0% net profit hurdle!")
    for i, p in enumerate(passing_is[:10]):
        print(f"  {i+1:2d}. {p['name']:40s}: Ret={p['ann_return_pct']:+5.1f}% ({p['net_pips']:+6.1f}p, {p['trades']:3d} tr), WR={p['win_rate']:4.1f}%, PF={p['profit_factor']:4.2f}")

    if not passing_is:
        print("\nNo candidates passed the 12% in-sample hurdle in this parameter grid.")
        all_sorted = sorted(is_results.values(), key=lambda x: x["ann_return_pct"], reverse=True)
        print("Top 5 closest candidates:")
        for it in all_sorted[:5]:
            print(f"  - {it['name']:40s}: Ret={it['ann_return_pct']:+5.1f}% ({it['net_pips']:+6.1f}p), WR={it['win_rate']:4.1f}%, PF={it['profit_factor']:4.2f}")
        return

    # Phase 3: Out-of-Sample 90-Week Quarantined Holdout Audit (Seed 42 / 126.6M Ticks)
    print(f"\n[PHASE 3] Executing 90-Week Quarantined Holdout Audit ({len(holdout_weeks)} weeks / Unseen Data)...")
    t0 = time.time()
    oos_candidates = [c for c in candidates if c.name in {p["name"] for p in passing_is}]
    oos_results = run_parallel_battery(oos_candidates, holdout_weeks, n_workers=6)
    print(f"Holdout Audit Completed in {time.time()-t0:.1f}s.")

    years_oos = len(holdout_weeks) / 52.0
    for name, data in oos_results.items():
        sl = data["sl_pips"]
        net_p = data["net_pips"]
        ann_pips = net_p / years_oos if years_oos > 0 else 0.0
        ann_ret = (ann_pips / sl) if sl > 0 else 0.0
        data["ann_net_pips"] = round(ann_pips, 1)
        data["ann_return_pct"] = round(ann_ret, 1)

    # Phase 4: Dual-Certification
    dual_certified = []
    for is_cand in passing_is:
        name = is_cand["name"]
        oos_cand = oos_results.get(name, {})
        oos_pips = oos_cand.get("net_pips", 0.0)
        oos_ann_ret = oos_cand.get("ann_return_pct", 0.0)

        if oos_pips > 0 and oos_ann_ret >= 10.0:
            dual_certified.append({
                "symbol": "USDCAD",
                "name": name,
                "sl_pips": is_cand["sl_pips"],
                "tp_pips": is_cand["tp_pips"],
                "is_trades": is_cand["trades"],
                "is_net_pips": is_cand["net_pips"],
                "is_return_pct": is_cand["ann_return_pct"],
                "is_win_rate": is_cand["win_rate"],
                "is_profit_factor": is_cand["profit_factor"],
                "oos_trades": oos_cand.get("trades", 0),
                "oos_net_pips": oos_pips,
                "oos_ann_net_pips": oos_cand.get("ann_net_pips", 0.0),
                "oos_ann_return_pct": oos_ann_ret,
                "oos_win_rate": oos_cand.get("win_rate", 0.0),
                "oos_profit_factor": oos_cand.get("profit_factor", 0.0),
                "decay_ratio": round(oos_ann_ret / is_cand["ann_return_pct"], 2) if is_cand["ann_return_pct"] > 0 else 0.0
            })

    dual_certified.sort(key=lambda x: (x["oos_ann_return_pct"] + x["is_return_pct"]), reverse=True)

    print("\n" + "=" * 115)
    print("            FINAL DUAL-CERTIFIED PRODUCTION SCORECARD FOR USDCAD            ".center(115))
    print("=" * 115)
    print(f"Total Initial Candidates:               {len(candidates)}")
    print(f"Passing 52w In-Sample (>=12% Hurdle):   {len(passing_is)}")
    print(f"Dual-Certified (IS >= 12% & OOS >= 10%):{len(dual_certified)}")
    print("-" * 115)
    print(f"{'Strategy Name':<38} | {'SL/TP':<10} | {'52w IS Ret (pips)':<20} | {'90w OOS Ret (pips/yr)':<23} | {'Decay':<6} | {'Status':<12}")
    print("-" * 115)
    for c in dual_certified:
        status = "DUAL PASS" if c["oos_ann_return_pct"] >= 12.0 else "PASS (>=10%)"
        print(f"{c['name']:<38} | {c['sl_pips']:4.1f}/{c['tp_pips']:4.1f} | {c['is_return_pct']:5.1f}% (+{c['is_net_pips']:5.1f}p) | {c['oos_ann_return_pct']:5.1f}% (+{c['oos_ann_net_pips']:5.1f}p/yr) | {c['decay_ratio']:5.2f}x | {status:<12}")
    print("=" * 115)

    out_file = RESULTS_DIR / "usdcad_specialist_dual_certified.json"
    with open(out_file, "w") as f:
        json.dump({
            "symbol": "USDCAD",
            "microstructure_profile": {"pip_size": PIP_SIZE, "spread_limit": MAX_SPREAD, "comm": COMMISSION_PIPS, "slip": ENTRY_SLIPPAGE_PIPS},
            "total_candidates": len(candidates),
            "passing_is_count": len(passing_is),
            "dual_certified_count": len(dual_certified),
            "dual_certified": dual_certified,
            "all_is_results": is_results,
            "all_oos_results": oos_results
        }, f, indent=2)
    print(f"\nSaved dual-certified scorecard to {out_file}")


if __name__ == "__main__":
    main()
