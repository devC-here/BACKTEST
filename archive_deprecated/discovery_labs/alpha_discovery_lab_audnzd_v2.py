#!/usr/bin/env python3
"""
================================================================================
  ALPHA DISCOVERY LAB AUDNZD V2: HIGH-EDGE MULTI-SESSION DISCOVERY
================================================================================
Building on Discovery Lab V1 results where Macro Structure Retest / Asymmetric
Runners proved profitable on AUDNZD (PF 1.53 - 1.55):
  - Integrates Pacific/Sydney/Tokyo core trading hours (21:00 - 07:00 UTC).
  - Tests multiple touch tolerances (3.5p, 4.5p, 5.5p) and SL/TP configurations.
  - Tests Cointegrated Mean-Reversion architectures:
      * Daily Pivot Wall Rejection Fades (Sydney/Tokyo session)
      * H1 50-EMA Mean-Reversion Exhaustion Fades
      * Bollinger 2.5-SD Extreme Fades with RSI(14) Overbought/Oversold Filter
      * Sydney Open Range Breakout (21:00 - 01:00 UTC)

All models evaluated strictly on the 70% Quarantined Discovery Partition
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

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tick_engine.core.engine import BacktestEngine
from tick_engine.core.dataloader import get_week_dataset, get_discovery_weeks
from tick_engine.strategies.base import BaseStrategy, SIGNAL_LONG, SIGNAL_SHORT
from data_engineering_suite.bars import build_time_bars

RESULTS_DIR = ROOT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# Model Class 1: AUDNZD Multi-Session Macro Structure Engine
# ==============================================================================
class AUDNZDMacroStructureEngine(BaseStrategy):
    def __init__(
        self,
        name: str,
        session_mode: str = "asian_london", # 'asian_london', 'asian_only', '24h', 'london_only'
        lookback_h1: int = 20,
        dist_pips: float = 4.5,
        min_wick: float = 0.45,
        sl_pips: float = 12.0,
        tp_pips: float = 24.0,
        max_spread: float = 2.8,
        pip_size: float = 0.0001,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.name = name
        self.session_mode = session_mode
        self.lookback_h1 = int(lookback_h1)
        self.dist_pips = float(dist_pips)
        self.min_wick = float(min_wick)
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.max_spread = float(max_spread)
        self.pip_size = float(pip_size)

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
            if self.session_mode == 'london_only' and not (7 <= hr <= 16):
                continue
            elif self.session_mode == 'asian_london' and not (hr >= 21 or hr <= 16):
                continue
            elif self.session_mode == 'asian_only' and not (hr >= 21 or hr <= 7):
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


# ==============================================================================
# Model Class 2: AUDNZD Daily Pivot Wall Fade (Mean Reversion)
# ==============================================================================
class AUDNZDDailyPivotFade(BaseStrategy):
    def __init__(
        self,
        name: str,
        sl_pips: float = 14.0,
        tp_pips: float = 24.0,
        min_wick: float = 0.45,
        max_dist_pips: float = 5.0,
        max_spread: float = 2.8,
        pip_size: float = 0.0001,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.name = name
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.min_wick = float(min_wick)
        self.max_dist_pips = float(max_dist_pips)
        self.max_spread = float(max_spread)
        self.pip_size = float(pip_size)

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

        # Daily bars from 00:00 to 24:00 UTC
        ts = pd.to_datetime(ticks['timestamp'], unit='ms')
        mid = 0.5 * (ticks['bid'] + ticks['ask'])
        base = pd.DataFrame({'close': mid, 'high': mid, 'low': mid}, index=ts)
        daily = base.resample('1D').agg({'high': 'max', 'low': 'min', 'close': 'last'}).dropna()

        if len(daily) < 2:
            return signals, sl_pips_arr, tp_pips_arr

        # Classical Floor Pivots: P = (H+L+C)/3, R1 = 2P-L, S1 = 2P-H, R2 = P+(H-L), S2 = P-(H-L)
        p = (daily['high'] + daily['low'] + daily['close']) / 3.0
        r1 = 2 * p - daily['low']
        s1 = 2 * p - daily['high']
        r2 = p + (daily['high'] - daily['low'])
        s2 = p - (daily['high'] - daily['low'])

        daily['s1'] = s1.shift(1).bfill()
        daily['r1'] = r1.shift(1).bfill()
        daily['s2'] = s2.shift(1).bfill()
        daily['r2'] = r2.shift(1).bfill()

        s1_m15 = daily['s1'].reindex(m15.index, method='ffill').values
        r1_m15 = daily['r1'].reindex(m15.index, method='ffill').values
        s2_m15 = daily['s2'].reindex(m15.index, method='ffill').values
        r2_m15 = daily['r2'].reindex(m15.index, method='ffill').values

        c = m15['close'].values
        h = m15['high'].values
        l = m15['low'].values
        o = m15['open'].values
        spread = m15['spread_pips'].values
        dates = m15.index.date
        time_ms = m15.index.astype('datetime64[ms]').view('int64')
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        last_trade_date = None

        for i in range(15, len(m15)):
            dt = dates[i]
            if dt == last_trade_date:
                continue
            if spread[i] > self.max_spread:
                continue

            bar_rng = (h[i] - l[i]) / self.pip_size
            if bar_rng < 2.5:
                continue

            sig = 0
            # S1 or S2 Support Rebound -> BUY
            dist_s1 = abs(l[i] - s1_m15[i]) / self.pip_size
            dist_s2 = abs(l[i] - s2_m15[i]) / self.pip_size
            lower_wick = (min(o[i], c[i]) - l[i]) / self.pip_size
            if (dist_s1 <= self.max_dist_pips or dist_s2 <= self.max_dist_pips) and (lower_wick / bar_rng) >= self.min_wick and c[i] > o[i]:
                sig = SIGNAL_LONG

            # R1 or R2 Resistance Rejection -> SELL
            dist_r1 = abs(r1_m15[i] - h[i]) / self.pip_size
            dist_r2 = abs(r2_m15[i] - h[i]) / self.pip_size
            upper_wick = (h[i] - max(o[i], c[i])) / self.pip_size
            if (dist_r1 <= self.max_dist_pips or dist_r2 <= self.max_dist_pips) and (upper_wick / bar_rng) >= self.min_wick and c[i] < o[i]:
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


# ==============================================================================
# Model Class 3: AUDNZD Bollinger 2.5-SD Extreme Reversion with RSI
# ==============================================================================
class AUDNZDBollingerRSIFade(BaseStrategy):
    def __init__(
        self,
        name: str,
        bb_period: int = 20,
        bb_std: float = 2.5,
        rsi_period: int = 14,
        sl_pips: float = 12.0,
        tp_pips: float = 20.0,
        max_spread: float = 2.8,
        pip_size: float = 0.0001,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.name = name
        self.bb_period = int(bb_period)
        self.bb_std = float(bb_std)
        self.rsi_period = int(rsi_period)
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.max_spread = float(max_spread)
        self.pip_size = float(pip_size)

    def generate_signals(self, ticks: np.ndarray):
        n_ticks = len(ticks)
        signals = np.zeros(n_ticks, dtype=np.int8)
        sl_pips_arr = np.zeros(n_ticks, dtype=np.float64)
        tp_pips_arr = np.zeros(n_ticks, dtype=np.float64)

        if n_ticks < 500:
            return signals, sl_pips_arr, tp_pips_arr

        m15 = build_time_bars(ticks, timeframe_min=15, pip_size=self.pip_size)
        if len(m15) < max(self.bb_period, self.rsi_period) + 15:
            return signals, sl_pips_arr, tp_pips_arr

        c = m15['close'].values
        h = m15['high'].values
        l = m15['low'].values
        o = m15['open'].values
        spread = m15['spread_pips'].values
        dates = m15.index.date
        time_ms = m15.index.astype('datetime64[ms]').view('int64')
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        # Calculate Bollinger Bands
        s_c = pd.Series(c)
        ma = s_c.rolling(self.bb_period).mean().values
        std = s_c.rolling(self.bb_period).std().values
        upper = ma + self.bb_std * std
        lower = ma - self.bb_std * std

        # Calculate RSI
        delta = s_c.diff()
        gain = delta.clip(lower=0).rolling(self.rsi_period).mean()
        loss = (-delta.clip(upper=0)).rolling(self.rsi_period).mean()
        rs = gain / (loss + 1e-9)
        rsi = (100.0 - (100.0 / (1.0 + rs))).fillna(50.0).values

        last_trade_date = None

        for i in range(self.bb_period + 5, len(m15)):
            dt = dates[i]
            if dt == last_trade_date:
                continue
            if spread[i] > self.max_spread:
                continue

            bar_rng = (h[i] - l[i]) / self.pip_size
            if bar_rng < 2.5:
                continue

            sig = 0
            # Lower Band Extreme Fade + Oversold RSI (<25) + Bullish Rejection
            lower_wick = (min(o[i], c[i]) - l[i]) / self.pip_size
            if l[i] <= lower[i] and rsi[i] < 28.0 and (lower_wick / bar_rng) >= 0.40 and c[i] > o[i]:
                sig = SIGNAL_LONG

            # Upper Band Extreme Fade + Overbought RSI (>75) + Bearish Rejection
            upper_wick = (h[i] - max(o[i], c[i])) / self.pip_size
            if h[i] >= upper[i] and rsi[i] > 72.0 and (upper_wick / bar_rng) >= 0.40 and c[i] < o[i]:
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


# ==============================================================================
# Model Class 4: AUDNZD H1 50-EMA Mean-Reversion Exhaustion Fade
# ==============================================================================
class AUDNZDH1EMADeviationFade(BaseStrategy):
    def __init__(
        self,
        name: str,
        dev_threshold_pips: float = 25.0,
        sl_pips: float = 12.0,
        tp_pips: float = 20.0,
        max_spread: float = 2.8,
        pip_size: float = 0.0001,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.name = name
        self.dev_threshold_pips = float(dev_threshold_pips)
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.max_spread = float(max_spread)
        self.pip_size = float(pip_size)

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
        base = pd.DataFrame({'close': mid}, index=ts)
        h1 = base.resample('1h').last().dropna()
        if len(h1) < 55:
            return signals, sl_pips_arr, tp_pips_arr

        h1['ema50'] = h1['close'].ewm(span=50, adjust=False).mean()
        ema50_m15 = h1['ema50'].reindex(m15.index, method='ffill').values

        c = m15['close'].values
        h = m15['high'].values
        l = m15['low'].values
        o = m15['open'].values
        spread = m15['spread_pips'].values
        dates = m15.index.date
        time_ms = m15.index.astype('datetime64[ms]').view('int64')
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        last_trade_date = None

        for i in range(25, len(m15)):
            dt = dates[i]
            if dt == last_trade_date:
                continue
            if spread[i] > self.max_spread:
                continue

            bar_rng = (h[i] - l[i]) / self.pip_size
            if bar_rng < 2.5:
                continue

            sig = 0
            # Deviation below H1 EMA50 >= threshold + bullish wick -> BUY fade back
            dev = (c[i] - ema50_m15[i]) / self.pip_size
            lower_wick = (min(o[i], c[i]) - l[i]) / self.pip_size
            upper_wick = (h[i] - max(o[i], c[i])) / self.pip_size

            if dev <= -self.dev_threshold_pips and (lower_wick / bar_rng) >= 0.45 and c[i] > o[i]:
                sig = SIGNAL_LONG
            elif dev >= self.dev_threshold_pips and (upper_wick / bar_rng) >= 0.45 and c[i] < o[i]:
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


# ==============================================================================
# Model Class 5: AUDNZD Sydney Range Breakout (21:00 - 01:00 UTC)
# ==============================================================================
class AUDNZDSydneyRangeBreakout(BaseStrategy):
    def __init__(
        self,
        name: str,
        sl_pips: float = 12.0,
        tp_pips: float = 22.0,
        max_spread: float = 2.8,
        pip_size: float = 0.0001,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.name = name
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.max_spread = float(max_spread)
        self.pip_size = float(pip_size)

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

        c = m15['close'].values
        h = m15['high'].values
        l = m15['low'].values
        hours = m15.index.hour.values
        spread = m15['spread_pips'].values
        dates = m15.index.date
        time_ms = m15.index.astype('datetime64[ms]').view('int64')
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        # Determine Sydney opening range: first 2 M15 bars between 21:00 and 21:30 UTC
        last_trade_date = None
        sydney_high = None
        sydney_low = None
        current_dt = None

        for i in range(len(m15)):
            dt = dates[i]
            hr = hours[i]

            if dt != current_dt:
                current_dt = dt
                sydney_high = None
                sydney_low = None

            if hr == 21:
                if sydney_high is None:
                    sydney_high = h[i]
                    sydney_low = l[i]
                else:
                    sydney_high = max(sydney_high, h[i])
                    sydney_low = min(sydney_low, l[i])

            if (22 <= hr <= 23 or 0 <= hr <= 3) and sydney_high is not None:
                if dt == last_trade_date or spread[i] > self.max_spread:
                    continue

                sig = 0
                if c[i] > sydney_high:
                    sig = SIGNAL_LONG
                elif c[i] < sydney_low:
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


def build_lab_v2_candidates() -> List[Tuple[str, Any]]:
    """Builds comprehensive test matrix for Lab V2."""
    models = [
        # --- 1. Macro Structure Engine: Multi-Session Variations ---
        ("MacroRetest_AsianLondon_SL10_TP18", AUDNZDMacroStructureEngine(
            name="MacroRetest_AsianLondon_SL10_TP18", session_mode="asian_london",
            dist_pips=4.5, min_wick=0.45, sl_pips=10.0, tp_pips=18.0
        )),
        ("MacroRetest_AsianLondon_SL12_TP20", AUDNZDMacroStructureEngine(
            name="MacroRetest_AsianLondon_SL12_TP20", session_mode="asian_london",
            dist_pips=4.5, min_wick=0.45, sl_pips=12.0, tp_pips=20.0
        )),
        ("MacroRetest_AsianLondon_SL12_TP24", AUDNZDMacroStructureEngine(
            name="MacroRetest_AsianLondon_SL12_TP24", session_mode="asian_london",
            dist_pips=4.5, min_wick=0.45, sl_pips=12.0, tp_pips=24.0
        )),
        ("MacroRetest_AsianLondon_SL14_TP28", AUDNZDMacroStructureEngine(
            name="MacroRetest_AsianLondon_SL14_TP28", session_mode="asian_london",
            dist_pips=5.0, min_wick=0.40, sl_pips=14.0, tp_pips=28.0
        )),

        # --- 2. Macro Runner (2.5R - 3.0R): Multi-Session Variations ---
        ("MacroRunner_AsianLondon_SL10_TP25", AUDNZDMacroStructureEngine(
            name="MacroRunner_AsianLondon_SL10_TP25", session_mode="asian_london",
            dist_pips=4.5, min_wick=0.45, sl_pips=10.0, tp_pips=25.0
        )),
        ("MacroRunner_AsianLondon_SL12_TP30", AUDNZDMacroStructureEngine(
            name="MacroRunner_AsianLondon_SL12_TP30", session_mode="asian_london",
            dist_pips=4.5, min_wick=0.45, sl_pips=12.0, tp_pips=30.0
        )),
        ("MacroRunner_AsianLondon_SL14_TP35", AUDNZDMacroStructureEngine(
            name="MacroRunner_AsianLondon_SL14_TP35", session_mode="asian_london",
            dist_pips=5.0, min_wick=0.40, sl_pips=14.0, tp_pips=35.0
        )),
        ("MacroRunner_AsianLondon_SL12_TP36", AUDNZDMacroStructureEngine(
            name="MacroRunner_AsianLondon_SL12_TP36", session_mode="asian_london",
            dist_pips=4.5, min_wick=0.45, sl_pips=12.0, tp_pips=36.0 # 3.0R
        )),

        # --- 3. Macro Structure: 24h & Asian-Only Modes ---
        ("MacroRetest_24h_SL12_TP20", AUDNZDMacroStructureEngine(
            name="MacroRetest_24h_SL12_TP20", session_mode="24h",
            dist_pips=4.5, min_wick=0.45, sl_pips=12.0, tp_pips=20.0
        )),
        ("MacroRunner_24h_SL12_TP30", AUDNZDMacroStructureEngine(
            name="MacroRunner_24h_SL12_TP30", session_mode="24h",
            dist_pips=4.5, min_wick=0.45, sl_pips=12.0, tp_pips=30.0
        )),
        ("MacroRetest_AsianOnly_SL12_TP20", AUDNZDMacroStructureEngine(
            name="MacroRetest_AsianOnly_SL12_TP20", session_mode="asian_only",
            dist_pips=4.5, min_wick=0.45, sl_pips=12.0, tp_pips=20.0
        )),
        ("MacroRunner_AsianOnly_SL12_TP30", AUDNZDMacroStructureEngine(
            name="MacroRunner_AsianOnly_SL12_TP30", session_mode="asian_only",
            dist_pips=4.5, min_wick=0.45, sl_pips=12.0, tp_pips=30.0
        )),

        # --- 4. Daily Pivot Wall Fades (Mean Reversion) ---
        ("DailyPivot_SL12_TP24", AUDNZDDailyPivotFade(
            name="DailyPivot_SL12_TP24", sl_pips=12.0, tp_pips=24.0, min_wick=0.45, max_dist_pips=5.0
        )),
        ("DailyPivot_SL14_TP28", AUDNZDDailyPivotFade(
            name="DailyPivot_SL14_TP28", sl_pips=14.0, tp_pips=28.0, min_wick=0.45, max_dist_pips=5.5
        )),
        ("DailyPivot_SL16_TP32", AUDNZDDailyPivotFade(
            name="DailyPivot_SL16_TP32", sl_pips=16.0, tp_pips=32.0, min_wick=0.40, max_dist_pips=6.0
        )),

        # --- 5. Bollinger 2.5-SD Extreme Reversion with RSI Filter ---
        ("BollingerRSI_SL10_TP18", AUDNZDBollingerRSIFade(
            name="BollingerRSI_SL10_TP18", sl_pips=10.0, tp_pips=18.0
        )),
        ("BollingerRSI_SL12_TP22", AUDNZDBollingerRSIFade(
            name="BollingerRSI_SL12_TP22", sl_pips=12.0, tp_pips=22.0
        )),
        ("BollingerRSI_SL14_TP28", AUDNZDBollingerRSIFade(
            name="BollingerRSI_SL14_TP28", sl_pips=14.0, tp_pips=28.0
        )),

        # --- 6. H1 50-EMA Deviation Fades ---
        ("H1EMA_Dev20_SL12_TP20", AUDNZDH1EMADeviationFade(
            name="H1EMA_Dev20_SL12_TP20", dev_threshold_pips=20.0, sl_pips=12.0, tp_pips=20.0
        )),
        ("H1EMA_Dev25_SL12_TP22", AUDNZDH1EMADeviationFade(
            name="H1EMA_Dev25_SL12_TP22", dev_threshold_pips=25.0, sl_pips=12.0, tp_pips=22.0
        )),
        ("H1EMA_Dev30_SL14_TP26", AUDNZDH1EMADeviationFade(
            name="H1EMA_Dev30_SL14_TP26", dev_threshold_pips=30.0, sl_pips=14.0, tp_pips=26.0
        )),

        # --- 7. Sydney Range Breakout ---
        ("SydneyBreakout_SL12_TP22", AUDNZDSydneyRangeBreakout(
            name="SydneyBreakout_SL12_TP22", sl_pips=12.0, tp_pips=22.0
        )),
        ("SydneyBreakout_SL14_TP28", AUDNZDSydneyRangeBreakout(
            name="SydneyBreakout_SL14_TP28", sl_pips=14.0, tp_pips=28.0
        )),
    ]
    return models


def evaluate_week_v2(week_num: int) -> Dict[str, Any]:
    engine = BacktestEngine(pip_size=0.0001, use_slippage=True)
    try:
        ticks, meta = get_week_dataset(week_num, raw_dir="data/AUDNZD")
        if len(ticks) == 0:
            return {"week": week_num, "trades": {}}
    except Exception as e:
        return {"week": week_num, "trades": {}, "error": str(e)}

    models = build_lab_v2_candidates()
    week_trades = {}

    for model_name, model in models:
        try:
            trades = engine.run(ticks, model)
            pips_list = [float(t['pips']) for t in trades]
            week_trades[model_name] = pips_list
        except Exception:
            week_trades[model_name] = []

    return {"week": week_num, "trades": week_trades}


def run_lab_v2():
    t_start = time.time()
    print("=" * 80)
    print("  ALPHA DISCOVERY LAB AUDNZD V2: HIGH-EDGE MULTI-SESSION DISCOVERY")
    print("=" * 80)

    disc_weeks = get_discovery_weeks("AUDNZD")
    total_disc_weeks = len(disc_weeks)
    candidates = build_lab_v2_candidates()
    print(f"[*] Quarantined Discovery Weeks: {total_disc_weeks} (4.04 Years)")
    print(f"[*] Candidate Model Variants:    {len(candidates)}")
    print(f"[*] Quarantined Holdout (30%):   Strictly Isolated & Preserved")

    models_trades: Dict[str, List[float]] = {name: [] for name, _ in candidates}

    max_workers = min(os.cpu_count() or 4, 12)
    print(f"[*] Launching parallel evaluation across {max_workers} worker processes...")

    completed = 0
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(evaluate_week_v2, w): w for w in disc_weeks}
        for future in as_completed(futures):
            res = future.result()
            w = res["week"]
            trades_dict = res.get("trades", {})

            for model_name, pips_list in trades_dict.items():
                models_trades[model_name].extend(pips_list)

            completed += 1
            if completed % 35 == 0 or completed == total_disc_weeks:
                elapsed = time.time() - t_start
                rate = completed / max(elapsed, 0.001)
                print(f"  --> Progress: {completed:03d}/{total_disc_weeks} discovery weeks processed ({rate:.1f} w/s)...")

    # Score and rank all models
    print("\n" + "=" * 80)
    print("  AUDNZD LAB V2 RESULTS SCORECARD (70% DISCOVERY / 210 WEEKS)")
    print("=" * 80)

    scorecard = []
    years_equiv = total_disc_weeks / 52.0

    for model_name, pips in models_trades.items():
        n_trades = len(pips)
        if n_trades == 0:
            continue

        p_arr = np.array(pips)
        net_pips = float(np.sum(p_arr))
        wins = p_arr[p_arr > 0]
        losses = -p_arr[p_arr < 0]
        sum_wins = float(np.sum(wins))
        sum_losses = float(np.sum(losses))

        wr = len(wins) / n_trades * 100.0 if n_trades > 0 else 0.0
        pf = sum_wins / sum_losses if sum_losses > 0 else (99.0 if sum_wins > 0 else 0.0)

        equity = np.cumsum(p_arr)
        peak = np.maximum.accumulate(equity)
        dd = peak - equity
        max_dd = float(np.max(dd)) if len(dd) > 0 else 0.0

        ret_dd = net_pips / max_dd if max_dd > 0 else (net_pips if net_pips > 0 else 0.0)
        annual_pips = net_pips / years_equiv

        scorecard.append({
            "model": model_name,
            "trades": n_trades,
            "win_rate": round(wr, 1),
            "profit_factor": round(pf, 2),
            "net_pips": round(net_pips, 1),
            "annual_pips": round(annual_pips, 1),
            "max_dd_pips": round(max_dd, 1),
            "return_to_dd": round(ret_dd, 2),
            "expectancy_pips": round(net_pips / n_trades, 2)
        })

    scorecard.sort(key=lambda x: x["net_pips"], reverse=True)

    header = f"{'Rank':<4} {'Model Identifier':<36} {'Trades':<7} {'WinRate':<8} {'PF':<6} {'Net Pips':<10} {'Pips/Yr':<10} {'Max DD':<9} {'Ret/DD':<8} {'Exp(p)':<7}"
    print(header)
    print("-" * len(header))

    profitable_count = 0
    elite_count = 0

    for i, row in enumerate(scorecard, 1):
        is_elite = row["profit_factor"] >= 1.25 and row["return_to_dd"] >= 3.0 and row["net_pips"] > 0
        is_prof = row["profit_factor"] >= 1.15 and row["net_pips"] > 0
        tag = "[ELITE]" if is_elite else ("[PASS]" if is_prof else "      ")

        if is_elite: elite_count += 1
        if is_prof: profitable_count += 1

        print(f"{i:<4} {row['model']:<36} {row['trades']:<7} {row['win_rate']:>5.1f}%  {row['profit_factor']:>5.2f} {row['net_pips']:>9.1f}p {row['annual_pips']:>9.1f}p {row['max_dd_pips']:>7.1f}p {row['return_to_dd']:>7.2f}x {row['expectancy_pips']:>6.2f}p  {tag}")

    print("-" * len(header))
    print(f"Summary: Evaluated {len(scorecard)} models across {total_disc_weeks} discovery weeks ({years_equiv:.2f} yrs).")
    print(f"Profitable Models (PF >= 1.15): {profitable_count} | High-Conviction Elite: {elite_count}")

    out_file = RESULTS_DIR / "audnzd_discovery_screening_v2.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "asset": "AUDNZD",
            "partition": "discovery_70",
            "total_weeks": total_disc_weeks,
            "years_equivalent": round(years_equiv, 2),
            "timestamp_utc": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "scorecard": scorecard
        }, f, indent=2)

    print(f"[+] Detailed discovery results saved to {out_file.resolve()}")
    print(f"[+] Total execution time: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    run_lab_v2()

