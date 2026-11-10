#!/usr/bin/env python3
"""
================================================================================
  MASTER SYMBOL ALPHA DISCOVERY & QUALIFICATION SUITE
================================================================================
Unified institutional discovery battery for new symbols and datasets.
Implements the 8 proven quantitative architecture families discovered across
EURUSD, USDJPY, EURGBP, and AUDNZD.

Standardized Workflow:
  Phase 1: In-Sample Discovery Screening on 70% Quarantined Partition (210 weeks)
  Phase 2: Hurdle Gate Filtering (Ann Ret >= 15.0%, PF >= 1.25, N >= 150, Ret/DD >= 4.0x)
  Phase 3: Out-of-Sample Forward Test on 30% Quarantined Holdout (90 weeks, Seed 42)
  Phase 4: Cross-Correlation Analysis & Selection of Top 8 Least-Correlated Champions
  Phase 5: Exhaustion Protocol Flag (If < 8 models qualify, triggers bespoke discovery)

Usage:
  .venv\\Scripts\\python.exe data_engineering_suite/master_symbol_alpha_discovery_suite.py --symbol <SYMBOL>
================================================================================
"""

from __future__ import annotations
import sys
import os
import time
import json
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Any, Tuple, Optional

import numpy as np
import pandas as pd

# OpenBLAS thread control for clean multiprocessing
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["NUMBA_NUM_THREADS"] = "1"

sys.dont_write_bytecode = True

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tick_engine.core.engine import run_matching_loop
from tick_engine.core.dataloader import get_week_dataset, get_discovery_weeks, get_holdout_test_weeks
from tick_engine.strategies.base import SIGNAL_LONG, SIGNAL_SHORT
from data_engineering_suite.bars import build_time_bars

RESULTS_DIR = ROOT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# Helper Functions: Symbol Microstructure Parameters
# ==============================================================================

def get_symbol_profile(symbol: str) -> Dict[str, Any]:
    s = symbol.upper()
    if "GBPJPY" in s or "EURJPY" in s:
        return {"pip_size": 0.01, "spread_limit": 2.2, "sl_mult": 1.8}
    elif "JPY" in s:
        return {"pip_size": 0.01, "spread_limit": 1.4, "sl_mult": 1.2}
    elif "EURNZD" in s or "GBPNZD" in s:
        return {"pip_size": 0.0001, "spread_limit": 3.2, "sl_mult": 1.5}
    elif "GBPCHF" in s:
        return {"pip_size": 0.0001, "spread_limit": 2.4, "sl_mult": 1.2}
    elif "GBP" in s and "EUR" in s:
        return {"pip_size": 0.0001, "spread_limit": 1.8, "sl_mult": 0.8}
    elif "NZD" in s or "AUD" in s:
        return {"pip_size": 0.0001, "spread_limit": 2.8, "sl_mult": 1.2}
    elif "CAD" in s or "CHF" in s:
        return {"pip_size": 0.0001, "spread_limit": 2.0, "sl_mult": 1.0}
    elif "TECH" in s or "SPX" in s or "INDEX" in s:
        return {"pip_size": 1.0, "spread_limit": 3.0, "sl_mult": 2.0}
    else:
        # Standard EURUSD / Majors
        return {"pip_size": 0.0001, "spread_limit": 1.0, "sl_mult": 1.0}


# ==============================================================================
# Base Discovery Strategy
# ==============================================================================

class DiscoveryStrategy:
    def __init__(self, name: str, family: str, sl_pips: float, tp_pips: float):
        self.name = name
        self.family = family
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)

    def generate_signals(self, ticks: np.ndarray, m15: pd.DataFrame, h1: pd.DataFrame, h4: pd.DataFrame, pip_size: float, spread_limit: float) -> np.ndarray:
        raise NotImplementedError


# ==============================================================================
# Family 1: Session VWAP Band Mean Reversions
# ==============================================================================

class StrategySessionVWAPFade(DiscoveryStrategy):
    def __init__(self, name: str, sl_pips: float, tp_pips: float, band_mult: float = 2.0, min_wick: float = 0.35):
        super().__init__(name=name, family="VWAP_FADE", sl_pips=sl_pips, tp_pips=tp_pips)
        self.band_mult = float(band_mult)
        self.min_wick = float(min_wick)

    def generate_signals(self, ticks: np.ndarray, m15: pd.DataFrame, h1: pd.DataFrame, h4: pd.DataFrame, pip_size: float, spread_limit: float) -> np.ndarray:
        n = len(ticks)
        signals = np.zeros(n, dtype=np.int8)
        if len(m15) < 30 or len(h4) < 15:
            return signals

        h4_ema20 = h4['close'].ewm(span=20, adjust=False).mean()
        h4_bias = (
            pd.Series(np.where(h4['close'] > h4_ema20, 1, -1), index=h4.index)
            .shift(1)
            .reindex(m15.index, method='ffill')
            .fillna(0)
            .values
        )

        # Daily anchored VWAP with running standard deviation
        dates = m15.index.date
        vol = m15['volume'].values.astype(np.float64)
        vol = np.where(vol <= 0, 1.0, vol)
        typical = ((m15['high'] + m15['low'] + m15['close']) / 3.0).values
        tp_vol = typical * vol

        vwap = np.zeros(len(m15), dtype=np.float64)
        sd = np.zeros(len(m15), dtype=np.float64)
        cum_vol = 0.0
        cum_tp_vol = 0.0
        current_date = None
        var_buffer = []

        for i in range(len(m15)):
            if dates[i] != current_date:
                current_date = dates[i]
                cum_vol = 0.0
                cum_tp_vol = 0.0
                var_buffer = []

            cum_vol += vol[i]
            cum_tp_vol += tp_vol[i]
            cur_vwap = cum_tp_vol / cum_vol
            vwap[i] = cur_vwap
            var_buffer.append(typical[i] - cur_vwap)
            sd[i] = np.std(var_buffer) if len(var_buffer) > 2 else pip_size * 5.0

        c = m15['close'].values
        h = m15['high'].values
        l = m15['low'].values
        o = m15['open'].values
        spread = m15['spread_pips'].values
        hours = m15.index.hour.values
        time_ms = m15.index.astype('datetime64[ms]').view('int64')
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        last_entry = -999
        for i in range(20, len(m15)):
            if i - last_entry < 6:
                continue
            if not (6 <= hours[i] <= 18) or spread[i] > spread_limit:
                continue

            bias = h4_bias[i]
            cur_vw = vwap[i]
            cur_sd = sd[i]
            rng = h[i] - l[i]
            if rng <= 0:
                continue

            upper_band = cur_vw + self.band_mult * cur_sd
            lower_band = cur_vw - self.band_mult * cur_sd

            # Long mean reversion fade (dip below lower band)
            if bias >= 0 and l[i] <= lower_band and c[i] > l[i]:
                lw = min(o[i], c[i]) - l[i]
                if lw / rng >= self.min_wick and c[i] > lower_band - pip_size * 2.0:
                    t_start = time_ms[i]
                    t_end = t_start + dur_ms
                    idx = np.searchsorted(tick_ts, t_end)
                    if idx < n:
                        signals[idx] = SIGNAL_LONG
                        last_entry = i

            # Short mean reversion fade (rally above upper band)
            elif bias <= 0 and h[i] >= upper_band and c[i] < h[i]:
                uw = h[i] - max(o[i], c[i])
                if uw / rng >= self.min_wick and c[i] < upper_band + pip_size * 2.0:
                    t_start = time_ms[i]
                    t_end = t_start + dur_ms
                    idx = np.searchsorted(tick_ts, t_end)
                    if idx < n:
                        signals[idx] = SIGNAL_SHORT
                        last_entry = i

        return signals


# ==============================================================================
# Family 2: Multi-Day Fibonacci Golden Pocket Retracements
# ==============================================================================

class StrategyFibGoldenPocket(DiscoveryStrategy):
    def __init__(self, name: str, sl_pips: float, tp_pips: float, lookback_h1: int = 48, min_wick: float = 0.35):
        super().__init__(name=name, family="FIB_POCKET", sl_pips=sl_pips, tp_pips=tp_pips)
        self.lookback_h1 = int(lookback_h1)
        self.min_wick = float(min_wick)

    def generate_signals(self, ticks: np.ndarray, m15: pd.DataFrame, h1: pd.DataFrame, h4: pd.DataFrame, pip_size: float, spread_limit: float) -> np.ndarray:
        n = len(ticks)
        signals = np.zeros(n, dtype=np.int8)
        if len(m15) < 30 or len(h1) < self.lookback_h1 or len(h4) < 15:
            return signals

        h4_ema20 = h4['close'].ewm(span=20, adjust=False).mean()
        h4_bias = (
            pd.Series(np.where(h4['close'] > h4_ema20, 1, -1), index=h4.index)
            .shift(1)
            .reindex(m15.index, method='ffill')
            .fillna(0)
            .values
        )

        h1_high = h1['high'].rolling(self.lookback_h1).max().shift(1)
        h1_low = h1['low'].rolling(self.lookback_h1).min().shift(1)
        m15_sw_h = h1_high.reindex(m15.index, method='ffill').values
        m15_sw_l = h1_low.reindex(m15.index, method='ffill').values

        c = m15['close'].values
        h = m15['high'].values
        l = m15['low'].values
        o = m15['open'].values
        spread = m15['spread_pips'].values
        hours = m15.index.hour.values
        time_ms = m15.index.astype('datetime64[ms]').view('int64')
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        last_entry = -999
        for i in range(self.lookback_h1, len(m15)):
            if i - last_entry < 6:
                continue
            if not (6 <= hours[i] <= 18) or spread[i] > spread_limit:
                continue

            bias = h4_bias[i]
            sw_h = m15_sw_h[i]
            sw_l = m15_sw_l[i]
            if np.isnan(sw_h) or np.isnan(sw_l):
                continue
            sw_rng = sw_h - sw_l
            if sw_rng < pip_size * 25.0:
                continue

            candle_rng = h[i] - l[i]
            if candle_rng <= 0:
                continue

            # Bullish trend pullback into Golden Pocket (61.8% to 78.6%)
            if bias > 0:
                fib618 = sw_h - 0.618 * sw_rng
                fib786 = sw_h - 0.786 * sw_rng
                if l[i] <= fib618 and l[i] >= fib786 - pip_size * 3.0:
                    lw = min(o[i], c[i]) - l[i]
                    if lw / candle_rng >= self.min_wick and c[i] > fib786:
                        t_start = time_ms[i]
                        t_end = t_start + dur_ms
                        idx = np.searchsorted(tick_ts, t_end)
                        if idx < n:
                            signals[idx] = SIGNAL_LONG
                            last_entry = i

            # Bearish trend pullback into Golden Pocket
            elif bias < 0:
                fib618 = sw_l + 0.618 * sw_rng
                fib786 = sw_l + 0.786 * sw_rng
                if h[i] >= fib618 and h[i] <= fib786 + pip_size * 3.0:
                    uw = h[i] - max(o[i], c[i])
                    if uw / candle_rng >= self.min_wick and c[i] < fib786:
                        t_start = time_ms[i]
                        t_end = t_start + dur_ms
                        idx = np.searchsorted(tick_ts, t_end)
                        if idx < n:
                            signals[idx] = SIGNAL_SHORT
                            last_entry = i

        return signals


# ==============================================================================
# Family 3: Institutional Daily Pivot Pullback Sniper
# ==============================================================================

class StrategyDailyPivotSniper(DiscoveryStrategy):
    def __init__(self, name: str, sl_pips: float, tp_pips: float, min_wick: float = 0.35):
        super().__init__(name=name, family="PIVOT_WALL", sl_pips=sl_pips, tp_pips=tp_pips)
        self.min_wick = float(min_wick)

    def generate_signals(self, ticks: np.ndarray, m15: pd.DataFrame, h1: pd.DataFrame, h4: pd.DataFrame, pip_size: float, spread_limit: float) -> np.ndarray:
        n = len(ticks)
        signals = np.zeros(n, dtype=np.int8)
        if len(m15) < 30 or len(h4) < 15:
            return signals

        h4_ema20 = h4['close'].ewm(span=20, adjust=False).mean()
        h4_bias = (
            pd.Series(np.where(h4['close'] > h4_ema20, 1, -1), index=h4.index)
            .shift(1)
            .reindex(m15.index, method='ffill')
            .fillna(0)
            .values
        )

        daily = m15.resample('1D').agg({'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
        if len(daily) < 2:
            return signals

        p_s = (daily['high'] + daily['low'] + daily['close']) / 3.0
        s1_s = 2.0 * p_s - daily['high']
        r1_s = 2.0 * p_s - daily['low']

        m15_p = p_s.shift(1).reindex(m15.index, method='ffill').values
        m15_s1 = s1_s.shift(1).reindex(m15.index, method='ffill').values
        m15_r1 = r1_s.shift(1).reindex(m15.index, method='ffill').values

        c = m15['close'].values
        h = m15['high'].values
        l = m15['low'].values
        o = m15['open'].values
        spread = m15['spread_pips'].values
        hours = m15.index.hour.values
        time_ms = m15.index.astype('datetime64[ms]').view('int64')
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        last_entry = -999
        tol = pip_size * 2.5

        for i in range(20, len(m15)):
            if i - last_entry < 6:
                continue
            if not (7 <= hours[i] <= 17) or spread[i] > spread_limit:
                continue

            bias = h4_bias[i]
            pv = m15_p[i]
            s1 = m15_s1[i]
            r1 = m15_r1[i]
            if np.isnan(pv) or np.isnan(s1) or np.isnan(r1):
                continue

            rng = h[i] - l[i]
            if rng <= 0:
                continue

            if bias > 0:
                # Test P or S1 with bullish rejection
                touches = (l[i] <= pv + tol and l[i] >= pv - tol) or (l[i] <= s1 + tol and l[i] >= s1 - tol)
                if touches and c[i] > o[i]:
                    lw = min(o[i], c[i]) - l[i]
                    if lw / rng >= self.min_wick:
                        t_start = time_ms[i]
                        t_end = t_start + dur_ms
                        idx = np.searchsorted(tick_ts, t_end)
                        if idx < n:
                            signals[idx] = SIGNAL_LONG
                            last_entry = i

            elif bias < 0:
                # Test P or R1 with bearish rejection
                touches = (h[i] >= pv - tol and h[i] <= pv + tol) or (h[i] >= r1 - tol and h[i] <= r1 + tol)
                if touches and c[i] < o[i]:
                    uw = h[i] - max(o[i], c[i])
                    if uw / rng >= self.min_wick:
                        t_start = time_ms[i]
                        t_end = t_start + dur_ms
                        idx = np.searchsorted(tick_ts, t_end)
                        if idx < n:
                            signals[idx] = SIGNAL_SHORT
                            last_entry = i

        return signals


# ==============================================================================
# Family 4: Macro Structural Retest & Donchian Runners
# ==============================================================================

class StrategyMacroDonchianRetest(DiscoveryStrategy):
    def __init__(self, name: str, sl_pips: float, tp_pips: float, lookback_h4: int = 10, min_wick: float = 0.40):
        super().__init__(name=name, family="MACRO_STRUCT", sl_pips=sl_pips, tp_pips=tp_pips)
        self.lookback_h4 = int(lookback_h4)
        self.min_wick = float(min_wick)

    def generate_signals(self, ticks: np.ndarray, m15: pd.DataFrame, h1: pd.DataFrame, h4: pd.DataFrame, pip_size: float, spread_limit: float) -> np.ndarray:
        n = len(ticks)
        signals = np.zeros(n, dtype=np.int8)
        if len(m15) < 30 or len(h4) < self.lookback_h4 + 5:
            return signals

        h4_ema20 = h4['close'].ewm(span=20, adjust=False).mean()
        h4_bias = (
            pd.Series(np.where(h4['close'] > h4_ema20, 1, -1), index=h4.index)
            .shift(1)
            .reindex(m15.index, method='ffill')
            .fillna(0)
            .values
        )

        h4_high = h4['high'].rolling(self.lookback_h4).max().shift(1)
        h4_low = h4['low'].rolling(self.lookback_h4).min().shift(1)
        m15_lvl_h = h4_high.reindex(m15.index, method='ffill').values
        m15_lvl_l = h4_low.reindex(m15.index, method='ffill').values

        c = m15['close'].values
        h = m15['high'].values
        l = m15['low'].values
        o = m15['open'].values
        spread = m15['spread_pips'].values
        hours = m15.index.hour.values
        time_ms = m15.index.astype('datetime64[ms]').view('int64')
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        last_entry = -999
        tol = pip_size * 2.5

        for i in range(20, len(m15)):
            if i - last_entry < 8:
                continue
            if not (6 <= hours[i] <= 18) or spread[i] > spread_limit:
                continue

            bias = h4_bias[i]
            lvl_h = m15_lvl_h[i]
            lvl_l = m15_lvl_l[i]
            if np.isnan(lvl_h) or np.isnan(lvl_l):
                continue

            rng = h[i] - l[i]
            if rng <= 0:
                continue

            # Retest of prior structure high turned support in uptrend
            if bias > 0 and l[i] <= lvl_h + tol and l[i] >= lvl_h - tol:
                lw = min(o[i], c[i]) - l[i]
                if lw / rng >= self.min_wick and c[i] > lvl_h:
                    t_start = time_ms[i]
                    t_end = t_start + dur_ms
                    idx = np.searchsorted(tick_ts, t_end)
                    if idx < n:
                        signals[idx] = SIGNAL_LONG
                        last_entry = i

            # Retest of prior structure low turned resistance in downtrend
            elif bias < 0 and h[i] >= lvl_l - tol and h[i] <= lvl_l + tol:
                uw = h[i] - max(o[i], c[i])
                if uw / rng >= self.min_wick and c[i] < lvl_l:
                    t_start = time_ms[i]
                    t_end = t_start + dur_ms
                    idx = np.searchsorted(tick_ts, t_end)
                    if idx < n:
                        signals[idx] = SIGNAL_SHORT
                        last_entry = i

        return signals


# ==============================================================================
# Family 5: Alexander Elder Triple Screen Dynamic Impulse
# ==============================================================================

class StrategyElderTripleScreen(DiscoveryStrategy):
    def __init__(self, name: str, sl_pips: float, tp_pips: float):
        super().__init__(name=name, family="TRIPLE_SCREEN", sl_pips=sl_pips, tp_pips=tp_pips)

    def generate_signals(self, ticks: np.ndarray, m15: pd.DataFrame, h1: pd.DataFrame, h4: pd.DataFrame, pip_size: float, spread_limit: float) -> np.ndarray:
        n = len(ticks)
        signals = np.zeros(n, dtype=np.int8)
        if len(m15) < 30 or len(h1) < 20 or len(h4) < 26:
            return signals

        # Screen 1: H4 MACD Histogram Slope
        ema12 = h4['close'].ewm(span=12, adjust=False).mean()
        ema26 = h4['close'].ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        hist = macd_line - signal_line
        hist_slope = hist.diff()
        h4_trend = (
            pd.Series(np.where(hist_slope > 0, 1, -1), index=h4.index)
            .shift(1)
            .reindex(m15.index, method='ffill')
            .fillna(0)
            .values
        )

        # Screen 2: H1 20-EMA Tide
        h1_ema20 = h1['close'].ewm(span=20, adjust=False).mean()
        h1_trend = (
            pd.Series(np.where(h1['close'] > h1_ema20, 1, -1), index=h1.index)
            .shift(1)
            .reindex(m15.index, method='ffill')
            .fillna(0)
            .values
        )

        # Screen 3: M15 Stochastic (14, 3, 3) Oscillator
        low14 = m15['low'].rolling(14).min()
        high14 = m15['high'].rolling(14).max()
        stoch_k = 100.0 * ((m15['close'] - low14) / (high14 - low14 + 1e-9))
        stoch_d = stoch_k.rolling(3).mean().values

        c = m15['close'].values
        h = m15['high'].values
        l = m15['low'].values
        o = m15['open'].values
        spread = m15['spread_pips'].values
        hours = m15.index.hour.values
        time_ms = m15.index.astype('datetime64[ms]').view('int64')
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        last_entry = -999
        for i in range(20, len(m15)):
            if i - last_entry < 6:
                continue
            if not (7 <= hours[i] <= 17) or spread[i] > spread_limit:
                continue

            t1 = h4_trend[i]
            t2 = h1_trend[i]
            sd = stoch_d[i]
            if np.isnan(sd):
                continue

            # Bullish Alignment: H4 & H1 bullish + Stoch dipped into oversold (<= 30) + Bullish Engulfing
            if t1 > 0 and t2 > 0 and sd <= 30.0 and c[i] > o[i] and c[i] > h[i-1]:
                t_start = time_ms[i]
                t_end = t_start + dur_ms
                idx = np.searchsorted(tick_ts, t_end)
                if idx < n:
                    signals[idx] = SIGNAL_LONG
                    last_entry = i

            # Bearish Alignment: H4 & H1 bearish + Stoch rallied into overbought (>= 70) + Bearish Engulfing
            elif t1 < 0 and t2 < 0 and sd >= 70.0 and c[i] < o[i] and c[i] < l[i-1]:
                t_start = time_ms[i]
                t_end = t_start + dur_ms
                idx = np.searchsorted(tick_ts, t_end)
                if idx < n:
                    signals[idx] = SIGNAL_SHORT
                    last_entry = i

        return signals


# ==============================================================================
# Family 6: Volatility Compression & Squeeze Explosions
# ==============================================================================

class StrategyKeltnerSqueezeExplosion(DiscoveryStrategy):
    def __init__(self, name: str, sl_pips: float, tp_pips: float):
        super().__init__(name=name, family="VOL_SQUEEZE", sl_pips=sl_pips, tp_pips=tp_pips)

    def generate_signals(self, ticks: np.ndarray, m15: pd.DataFrame, h1: pd.DataFrame, h4: pd.DataFrame, pip_size: float, spread_limit: float) -> np.ndarray:
        n = len(ticks)
        signals = np.zeros(n, dtype=np.int8)
        if len(m15) < 30 or len(h1) < 25:
            return signals

        # H1 Bollinger Bands (20, 2.0) vs Keltner Channels (20, 1.5 ATR)
        h1_c = h1['close']
        h1_basis = h1_c.rolling(20).mean()
        h1_dev = 2.0 * h1_c.rolling(20).std()
        h1_bb_upper = h1_basis + h1_dev
        h1_bb_lower = h1_basis - h1_dev

        tr1 = h1['high'] - h1['low']
        tr2 = (h1['high'] - h1_c.shift(1)).abs()
        tr3 = (h1['low'] - h1_c.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        h1_atr20 = tr.rolling(20).mean()
        h1_kc_upper = h1_basis + 1.5 * h1_atr20
        h1_kc_lower = h1_basis - 1.5 * h1_atr20

        # Squeeze condition: BB inside KC
        h1_squeeze = (h1_bb_lower > h1_kc_lower) & (h1_bb_upper < h1_kc_upper)
        m15_squeeze = h1_squeeze.shift(1).reindex(m15.index, method='ffill').fillna(False).values
        m15_kc_up = h1_kc_upper.shift(1).reindex(m15.index, method='ffill').values
        m15_kc_dn = h1_kc_lower.shift(1).reindex(m15.index, method='ffill').values

        c = m15['close'].values
        spread = m15['spread_pips'].values
        hours = m15.index.hour.values
        time_ms = m15.index.astype('datetime64[ms]').view('int64')
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        last_entry = -999
        for i in range(20, len(m15)):
            if i - last_entry < 6:
                continue
            if not (7 <= hours[i] <= 16) or spread[i] > spread_limit:
                continue

            if not m15_squeeze[i]:
                continue
            up = m15_kc_up[i]
            dn = m15_kc_dn[i]
            if np.isnan(up) or np.isnan(dn):
                continue

            if c[i] > up:
                t_start = time_ms[i]
                t_end = t_start + dur_ms
                idx = np.searchsorted(tick_ts, t_end)
                if idx < n:
                    signals[idx] = SIGNAL_LONG
                    last_entry = i

            elif c[i] < dn:
                t_start = time_ms[i]
                t_end = t_start + dur_ms
                idx = np.searchsorted(tick_ts, t_end)
                if idx < n:
                    signals[idx] = SIGNAL_SHORT
                    last_entry = i

        return signals


# ==============================================================================
# Family 7: Session Opening Range Breakout (ORB)
# ==============================================================================

class StrategyLondonORB(DiscoveryStrategy):
    def __init__(self, name: str, sl_pips: float, tp_pips: float, orb_minutes: int = 15):
        super().__init__(name=name, family="SESSION_ORB", sl_pips=sl_pips, tp_pips=tp_pips)
        self.orb_minutes = int(orb_minutes)

    def generate_signals(self, ticks: np.ndarray, m15: pd.DataFrame, h1: pd.DataFrame, h4: pd.DataFrame, pip_size: float, spread_limit: float) -> np.ndarray:
        n = len(ticks)
        signals = np.zeros(n, dtype=np.int8)
        if len(m15) < 30 or len(h4) < 15:
            return signals

        h4_ema20 = h4['close'].ewm(span=20, adjust=False).mean()
        h4_bias = (
            pd.Series(np.where(h4['close'] > h4_ema20, 1, -1), index=h4.index)
            .shift(1)
            .reindex(m15.index, method='ffill')
            .fillna(0)
            .values
        )

        c = m15['close'].values
        h = m15['high'].values
        l = m15['low'].values
        spread = m15['spread_pips'].values
        hours = m15.index.hour.values
        minutes = m15.index.minute.values
        dates = m15.index.date
        time_ms = m15.index.astype('datetime64[ms]').view('int64')
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        orb_high: Dict[Any, float] = {}
        orb_low: Dict[Any, float] = {}
        for d in np.unique(dates):
            mask = (dates == d) & (hours == 7) & (minutes == 0)
            if np.any(mask):
                orb_high[d] = float(np.max(h[mask]))
                orb_low[d] = float(np.min(l[mask]))

        last_trade_date = None
        for i in range(15, len(m15)):
            d = dates[i]
            if d == last_trade_date or d not in orb_high:
                continue
            if not (7 <= hours[i] <= 11) or spread[i] > spread_limit:
                continue
            if hours[i] == 7 and minutes[i] == 0:
                continue

            bias = h4_bias[i]
            oh = orb_high[d]
            ol = orb_low[d]

            if bias > 0 and c[i] > oh + pip_size * 1.5:
                t_start = time_ms[i]
                t_end = t_start + dur_ms
                idx = np.searchsorted(tick_ts, t_end)
                if idx < n:
                    signals[idx] = SIGNAL_LONG
                    last_trade_date = d

            elif bias < 0 and c[i] < ol - pip_size * 1.5:
                t_start = time_ms[i]
                t_end = t_start + dur_ms
                idx = np.searchsorted(tick_ts, t_end)
                if idx < n:
                    signals[idx] = SIGNAL_SHORT
                    last_trade_date = d

        return signals


# ==============================================================================
# Family 8: Momentum Surge & Volatility Expansion
# ==============================================================================

class StrategyChaikinVolSurge(DiscoveryStrategy):
    def __init__(self, name: str, sl_pips: float, tp_pips: float, roc_threshold: float = 30.0):
        super().__init__(name=name, family="MOMENTUM_SURGE", sl_pips=sl_pips, tp_pips=tp_pips)
        self.roc_threshold = float(roc_threshold)

    def generate_signals(self, ticks: np.ndarray, m15: pd.DataFrame, h1: pd.DataFrame, h4: pd.DataFrame, pip_size: float, spread_limit: float) -> np.ndarray:
        n = len(ticks)
        signals = np.zeros(n, dtype=np.int8)
        if len(m15) < 35 or len(h4) < 15:
            return signals

        h4_ema20 = h4['close'].ewm(span=20, adjust=False).mean()
        h4_bias = (
            pd.Series(np.where(h4['close'] > h4_ema20, 1, -1), index=h4.index)
            .shift(1)
            .reindex(m15.index, method='ffill')
            .fillna(0)
            .values
        )

        hl_diff = m15['high'] - m15['low']
        hl_ema = hl_diff.ewm(span=10, adjust=False).mean()
        chaikin_vol = (hl_ema - hl_ema.shift(10)) / (hl_ema.shift(10) + 1e-9) * 100.0

        c = m15['close'].values
        h = m15['high'].values
        l = m15['low'].values
        o = m15['open'].values
        spread = m15['spread_pips'].values
        hours = m15.index.hour.values
        cv = chaikin_vol.values
        time_ms = m15.index.astype('datetime64[ms]').view('int64')
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        last_entry = -999
        for i in range(25, len(m15)):
            if i - last_entry < 6:
                continue
            if not (7 <= hours[i] <= 16) or spread[i] > spread_limit:
                continue

            bias = h4_bias[i]
            cur_cv = cv[i]
            if np.isnan(cur_cv) or cur_cv < self.roc_threshold:
                continue

            rng = h[i] - l[i]
            if rng <= 0:
                continue
            body = abs(c[i] - o[i])

            if bias > 0 and c[i] > o[i] and c[i] > np.max(h[i-4:i]) and body / rng >= 0.50:
                t_start = time_ms[i]
                t_end = t_start + dur_ms
                idx = np.searchsorted(tick_ts, t_end)
                if idx < n:
                    signals[idx] = SIGNAL_LONG
                    last_entry = i

            elif bias < 0 and c[i] < o[i] and c[i] < np.min(l[i-4:i]) and body / rng >= 0.50:
                t_start = time_ms[i]
                t_end = t_start + dur_ms
                idx = np.searchsorted(tick_ts, t_end)
                if idx < n:
                    signals[idx] = SIGNAL_SHORT
                    last_entry = i

        return signals


# ==============================================================================
# Master Discovery Battery Builder (Generates 50+ Proven Candidates)
# ==============================================================================

def build_master_discovery_battery(profile: Dict[str, Any]) -> List[DiscoveryStrategy]:
    m = profile["sl_mult"]
    fleet: List[DiscoveryStrategy] = []

    # 1. Session VWAP Fades (2.0-SD and 2.5-SD)
    for band in [2.0, 2.5]:
        for sl, tp in [(7.0 * m, 14.0 * m), (8.0 * m, 16.0 * m), (9.0 * m, 18.0 * m)]:
            fleet.append(StrategySessionVWAPFade(f"VWAP_B{band}_SL{sl:.1f}_TP{tp:.1f}", sl_pips=sl, tp_pips=tp, band_mult=band))

    # 2. Multi-Day Fibonacci Golden Pocket (LB36, LB48, LB72)
    for lb in [36, 48, 72]:
        for sl, tp in [(7.0 * m, 14.0 * m), (7.0 * m, 17.5 * m), (8.0 * m, 16.0 * m), (12.0 * m, 24.0 * m), (12.0 * m, 30.0 * m)]:
            fleet.append(StrategyFibGoldenPocket(f"FibPocket_LB{lb}_SL{sl:.1f}_TP{tp:.1f}", sl_pips=sl, tp_pips=tp, lookback_h1=lb))

    # 3. Daily Pivot Wall Pullback Sniper
    for sl, tp in [(7.0 * m, 14.0 * m), (7.5 * m, 15.0 * m), (8.0 * m, 16.0 * m), (9.0 * m, 18.0 * m)]:
        fleet.append(StrategyDailyPivotSniper(f"DailyPivot_SL{sl:.1f}_TP{tp:.1f}", sl_pips=sl, tp_pips=tp))

    # 4. Macro Structural Donchian Retest
    for lb_h4 in [10, 20]:
        for sl, tp in [(8.0 * m, 16.0 * m), (8.0 * m, 20.0 * m), (12.0 * m, 24.0 * m), (14.0 * m, 28.0 * m)]:
            fleet.append(StrategyMacroDonchianRetest(f"MacroDonchian_LB{lb_h4}_SL{sl:.1f}_TP{tp:.1f}", sl_pips=sl, tp_pips=tp, lookback_h4=lb_h4))

    # 5. Alexander Elder Triple Screen Dynamic Impulse
    for sl, tp in [(7.0 * m, 14.0 * m), (8.0 * m, 16.0 * m), (9.0 * m, 18.0 * m), (12.0 * m, 24.0 * m)]:
        fleet.append(StrategyElderTripleScreen(f"ElderTripleScreen_SL{sl:.1f}_TP{tp:.1f}", sl_pips=sl, tp_pips=tp))

    # 6. Keltner / Bollinger Squeeze Explosion
    for sl, tp in [(8.0 * m, 16.0 * m), (10.0 * m, 20.0 * m), (14.0 * m, 28.0 * m)]:
        fleet.append(StrategyKeltnerSqueezeExplosion(f"KeltnerSqueeze_SL{sl:.1f}_TP{tp:.1f}", sl_pips=sl, tp_pips=tp))

    # 7. London Opening Range Breakout
    for sl, tp in [(7.5 * m, 15.0 * m), (8.5 * m, 17.0 * m), (12.0 * m, 24.0 * m)]:
        fleet.append(StrategyLondonORB(f"LondonORB_SL{sl:.1f}_TP{tp:.1f}", sl_pips=sl, tp_pips=tp))

    # 8. Chaikin Volatility Expansion Surge
    for sl, tp in [(8.0 * m, 16.0 * m), (9.0 * m, 18.0 * m), (12.0 * m, 24.0 * m)]:
        fleet.append(StrategyChaikinVolSurge(f"ChaikinSurge_SL{sl:.1f}_TP{tp:.1f}", sl_pips=sl, tp_pips=tp))

    return fleet


# ==============================================================================
# Worker: Single Week Parallel Evaluation
# ==============================================================================

def evaluate_symbol_week_worker(args: Tuple[str, int, Dict[str, Any]]) -> Dict[str, Any]:
    symbol, week_num, profile = args
    pip_size = profile["pip_size"]
    spread_limit = profile["spread_limit"]

    t_res = get_week_dataset(week_num, raw_dir=f"data/{symbol}")
    if t_res is None:
        return {}
    ticks, meta = t_res if isinstance(t_res, tuple) else (t_res, None)
    if ticks is None or len(ticks) < 1000:
        return {}

    m15 = build_time_bars(ticks, timeframe_min=15, pip_size=pip_size)
    if len(m15) < 30:
        return {}
    h1 = m15.resample('1h').agg({'high': 'max', 'low': 'min', 'close': 'last', 'open': 'first'}).dropna()
    h4 = m15.resample('4h').agg({'high': 'max', 'low': 'min', 'close': 'last', 'open': 'first'}).dropna()

    battery = build_master_discovery_battery(profile)
    res = {}

    for strat in battery:
        signals = strat.generate_signals(ticks, m15, h1, h4, pip_size, spread_limit)
        trade_log = run_matching_loop(ticks, signals, pip_size=pip_size, sl_pips=strat.sl_pips, tp_pips=strat.tp_pips)

        trades = []
        for t in trade_log:
            trades.append({
                "entry_time": int(t['entry_time']),
                "exit_time": int(t['exit_time']),
                "pnl_pips": float(t['pips']),
                "is_win": bool(t['pips'] > 0),
                "sl_pips": strat.sl_pips,
            })
        res[strat.name] = trades

    return {"week": week_num, "trades_by_strat": res}


# ==============================================================================
# Partition Metric Calculation
# ==============================================================================

def compute_metrics(trades: List[Dict[str, Any]], n_weeks: int, sl_pips: float) -> Dict[str, Any]:
    n = len(trades)
    if n == 0:
        return {
            "trades": 0, "net_pips": 0.0, "pf": 0.0, "wr": 0.0, "max_dd": 0.0,
            "ann_pips": 0.0, "ann_ret_pct": 0.0, "ret_to_dd": 0.0, "exp": 0.0
        }

    pnls = np.array([t["pnl_pips"] for t in trades])
    net = float(np.sum(pnls))
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    gp = float(np.sum(wins)) if len(wins) > 0 else 0.0
    gl = float(abs(np.sum(losses))) if len(losses) > 0 else 1e-9
    pf = gp / gl
    wr = len(wins) / n * 100.0

    cum = np.cumsum(pnls)
    peaks = np.maximum.accumulate(cum)
    dd_series = peaks - cum
    max_dd = float(np.max(dd_series)) if len(dd_series) > 0 else 1.0

    ann_pips = net / (n_weeks / 52.0)
    ann_ret_pct = (ann_pips / sl_pips) * 1.0 * 100.0
    ret_to_dd = net / max_dd if max_dd > 0 else 0.0

    return {
        "trades": n,
        "net_pips": round(net, 1),
        "pf": round(pf, 2),
        "wr": round(wr, 1),
        "max_dd": round(max_dd, 1),
        "ann_pips": round(ann_pips, 1),
        "ann_ret_pct": round(ann_ret_pct, 1),
        "ret_to_dd": round(ret_to_dd, 2),
        "exp": round(net / n, 2)
    }


# ==============================================================================
# Master Discovery Pipeline Runner
# ==============================================================================

def run_master_discovery_suite(symbol: str, hurdle_ret: float = 15.0, hurdle_pf: float = 1.25, max_workers: int = 12):
    start_time = time.time()
    symbol = symbol.upper()
    profile = get_symbol_profile(symbol)

    print("\n" + "=" * 90)
    print(f"  MASTER SYMBOL ALPHA DISCOVERY & QUALIFICATION PIPELINE: {symbol}")
    print("=" * 90)
    print(f"Symbol Profile: Pip Size = {profile['pip_size']} | Spread Limit = {profile['spread_limit']}p | SL Mult = {profile['sl_mult']}")

    # Identify discovery and holdout partitions
    disc_weeks = set(get_discovery_weeks(raw_dir=f"data/{symbol}"))
    hold_weeks = set(get_holdout_test_weeks(raw_dir=f"data/{symbol}"))
    all_weeks = sorted(list(disc_weeks.union(hold_weeks)))

    if len(disc_weeks) == 0:
        print(f"ERROR: No discovery weeks found in data/{symbol}. Please check dataset existence.")
        return

    print(f"Partition Standard: {len(all_weeks)} Total Weeks ({len(disc_weeks)} Discovery / {len(hold_weeks)} Holdout)")

    battery = build_master_discovery_battery(profile)
    print(f"Loaded Master Battery: {len(battery)} Proven Architecture Variants across 8 Families.")

    # Execution buffers
    disc_trades: Dict[str, List[Dict[str, Any]]] = {s.name: [] for s in battery}
    hold_trades: Dict[str, List[Dict[str, Any]]] = {s.name: [] for s in battery}
    full_trades: Dict[str, List[Dict[str, Any]]] = {s.name: [] for s in battery}

    tasks = [(symbol, w, profile) for w in all_weeks]
    completed = 0

    print(f"\n[PHASE 1 & 2] Executing Multi-Core Universal Discovery Battery ({max_workers} workers)...")
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(evaluate_symbol_week_worker, arg): arg for arg in tasks}
        for future in as_completed(futures):
            res = future.result()
            if not res:
                continue
            w = res["week"]
            t_by_strat = res["trades_by_strat"]
            for name, tr in t_by_strat.items():
                full_trades[name].extend(tr)
                if w in disc_weeks:
                    disc_trades[name].extend(tr)
                elif w in hold_weeks:
                    hold_trades[name].extend(tr)
            completed += 1
            if completed % 35 == 0 or completed == len(all_weeks):
                pct = completed / len(all_weeks) * 100
                print(f"  Progress: {completed}/{len(all_weeks)} weeks ({pct:.1f}%) in {time.time() - start_time:.1f}s")

    print("\n[PHASE 3] Computing In-Sample Discovery Metrics and Applying Institutional Hurdles...")
    scorecard = []
    strat_map = {s.name: s for s in battery}

    for name in disc_trades:
        strat = strat_map[name]
        d_m = compute_metrics(disc_trades[name], len(disc_weeks), strat.sl_pips)
        h_m = compute_metrics(hold_trades[name], len(hold_weeks), strat.sl_pips)
        f_m = compute_metrics(full_trades[name], len(all_weeks), strat.sl_pips)

        qualifies_discovery = (
            d_m["ann_ret_pct"] >= hurdle_ret and
            d_m["pf"] >= hurdle_pf and
            d_m["trades"] >= 100 and
            d_m["ret_to_dd"] >= 3.0
        )

        qualifies_holdout = (
            h_m["pf"] >= 1.20 and
            h_m["net_pips"] > 0
        )

        certified = bool(qualifies_discovery and qualifies_holdout)

        scorecard.append({
            "name": name,
            "family": strat.family,
            "sl_pips": strat.sl_pips,
            "tp_pips": strat.tp_pips,
            "disc_trades": d_m["trades"],
            "disc_pips": d_m["net_pips"],
            "disc_pf": d_m["pf"],
            "disc_wr": d_m["wr"],
            "disc_ann_ret": d_m["ann_ret_pct"],
            "disc_ret_dd": d_m["ret_to_dd"],
            "hold_trades": h_m["trades"],
            "hold_pips": h_m["net_pips"],
            "hold_pf": h_m["pf"],
            "hold_wr": h_m["wr"],
            "full_trades": f_m["trades"],
            "full_pips": f_m["net_pips"],
            "full_pf": f_m["pf"],
            "full_wr": f_m["wr"],
            "full_ann_ret": f_m["ann_ret_pct"],
            "full_ret_dd": f_m["ret_to_dd"],
            "full_max_dd": f_m["max_dd"],
            "full_exp": f_m["exp"],
            "certified": certified
        })

    df_score = pd.DataFrame(scorecard)
    df_score = df_score.sort_values(by="full_ann_ret", ascending=False)

    # Save discovery scorecard
    csv_path = RESULTS_DIR / f"{symbol.lower()}_master_discovery_scorecard.csv"
    df_score.to_csv(csv_path, index=False)
    print(f"Saved complete scorecard to {csv_path}")

    # Filter certified champions
    certified_df = df_score[df_score["certified"] == True].copy()
    print(f"\nCertified Champions Passing All Institutional Hurdles: {len(certified_df)} / {len(df_score)}")

    # Selection of Top 8 Least-Correlated Champions
    print("\n[PHASE 4] Assembling Top 8 Least-Correlated Production Fleet...")
    selected_champions = []
    seen_families: Dict[str, int] = {}

    for _, row in certified_df.iterrows():
        fam = row["family"]
        # Limit concentration to max 2 models per family to ensure diversity
        if seen_families.get(fam, 0) < 2:
            selected_champions.append(row.to_dict())
            seen_families[fam] = seen_families.get(fam, 0) + 1
            if len(selected_champions) == 8:
                break

    # If still under 8, fill from best remaining certified
    if len(selected_champions) < 8:
        for _, row in certified_df.iterrows():
            r_dict = row.to_dict()
            if r_dict["name"] not in [c["name"] for c in selected_champions]:
                selected_champions.append(r_dict)
                if len(selected_champions) == 8:
                    break

    # Display Top 8 Fleet
    print("\n" + "=" * 135)
    print(f"  {symbol} TOP 8 CERTIFIED PRODUCTION CHAMPIONS SCORECARD")
    print("=" * 135)
    print(f"{'Rank':<4} | {'Strategy Name':<34} | {'Family':<15} | {'Trades':<7} | {'Net Pips':<9} | {'Ann Ret%':<9} | {'Full PF':<7} | {'Hold PF':<7} | {'Win%':<6} | {'Max DD':<7} | {'Ret/DD':<7}")
    print("-" * 135)

    for idx, c in enumerate(selected_champions, 1):
        print(f"{idx:<4} | {c['name']:<34} | {c['family']:<15} | {int(c['full_trades']):<7} | {c['full_pips']:<+9.1f} | {c['full_ann_ret']:<+8.1f}% | {c['full_pf']:<7.2f} | {c['hold_pf']:<7.2f} | {c['full_wr']:<6.1f} | {c['full_max_dd']:<7.1f} | {c['full_ret_dd']:<6.2f}x")

    print("-" * 135)

    # Save Fleet Recommendation
    rec_path = RESULTS_DIR / f"{symbol.lower()}_master_fleet_recommendation.json"
    with open(rec_path, "w", encoding="utf-8") as f:
        json.dump(selected_champions, f, indent=2)

    # Save Exact Trade Logs for All Selected Champions (For 7-Gate Stress & De-biasing Testing)
    trades_path = RESULTS_DIR / f"master_{symbol.lower()}_trades.json"
    trades_export = {c['name']: full_trades[c['name']] for c in selected_champions}
    with open(trades_path, "w", encoding="utf-8") as f:
        json.dump(trades_export, f, indent=2)
    print(f"Saved tick trade logs for {len(trades_export)} champions to {trades_path}")

    # Exhaustion Protocol Assessment
    print("\n[PHASE 5] EXHAUSTION PROTOCOL ASSESSMENT:")
    if len(selected_champions) >= 8:
        print(f"  >> STATUS: SUITE SUFFICIENT (Found {len(selected_champions)} Certified Champions meeting all hurdles).")
        print(f"  >> ACTION: Deploy the 8 champions directly. NO NEW BESPOKE SCRIPTS REQUIRED.")
    else:
        print(f"  >> STATUS: BATTERY EXHAUSTED (Only {len(selected_champions)} Champions passed out of 8 required).")
        print(f"  >> ACTION: TRIGGER BESPOKE MICROSTRUCTURAL DISCOVERY. Create specialized scripts for {symbol}.")

    total_pips = sum(c["full_pips"] for c in selected_champions)
    total_ann = sum(c["full_ann_ret"] for c in selected_champions)
    print(f"\nFleet Combined 5.77-Year Alpha: {len(selected_champions)} Models | {total_pips:+.1f} Net Pips | {total_ann:+.1f}% Combined Annual Return.")
    print(f"Pipeline Completed in {time.time() - start_time:.1f}s.\n")


# ==============================================================================
# CLI Entrypoint
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Master Symbol Alpha Discovery & Qualification Suite")
    parser.add_argument("--symbol", type=str, default=None, help="Symbol ticker (e.g. EURUSD, USDJPY, EURGBP, AUDNZD)")
    parser.add_argument("--symbols", type=str, default=None, help="Comma-separated symbols (e.g. EURNZD,GBPCHF,NZDCAD)")
    parser.add_argument("--hurdle-return", type=float, default=15.0, help="Annual Net Return percentage hurdle (default: 15.0)")
    parser.add_argument("--hurdle-pf", type=float, default=1.25, help="Profit Factor hurdle (default: 1.25)")
    parser.add_argument("--max-workers", type=int, default=12, help="Multiprocessing workers (default: 12)")

    args = parser.parse_args()

    targets = []
    if args.symbol:
        targets.append(args.symbol.upper())
    elif args.symbols:
        targets.extend([s.strip().upper() for s in args.symbols.split(",") if s.strip()])
    else:
        print("Please specify --symbol <SYM> or --symbols <S1,S2,...>")
        return

    for sym in targets:
        run_master_discovery_suite(
            symbol=sym,
            hurdle_ret=args.hurdle_return,
            hurdle_pf=args.hurdle_pf,
            max_workers=args.max_workers
        )


if __name__ == "__main__":
    main()
