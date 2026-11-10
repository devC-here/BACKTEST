#!/usr/bin/env python3
"""
================================================================================
  ALPHA DISCOVERY LAB V17: LUNCH SQUEEZE, ORDER FLOW RETEST, VWAP BANDS, KELTNER SQUEEZE
================================================================================
Evaluates 4 Quantitative Microstructure Architectures across 70% Discovery Partitions:
  1. DISC82_LONDON_LUNCH_EXPANSION:
     European midday (11:30 - 13:15 UTC) compression range.
     Contraction: Midday range < 65% of Asian range.
     Breakout during Pre-NY liquidity surge (13:30 - 15:00 UTC) with H4 20-EMA trend agreement.
     Target: 2.0R.
  2. DISC83_SWING_BREAKOUT_RETEST:
     M15 20-bar swing high/low breakout.
     Retest of broken swing level within next 2-6 bars.
     Rejection wick >= 40% on M15 retest candle in trend direction.
     Active: 07:00 - 16:30 UTC. Target: 1.8R.
  3. DISC84_SESSION_VWAP_BAND_FADE:
     Session VWAP anchored at 00:00 UTC with 2.0-SD Bands.
     Price stretches beyond 2.0-SD outer band and closes back inside in direction of H4 trend.
     Active: 07:00 - 16:00 UTC. Target: 2.0R.
  4. DISC85_KELTNER_BOLLINGER_SQUEEZE:
     Bollinger Bands (20, 2.0) squeeze completely inside Keltner Channels (20, 1.5 ATR).
     First bar expanding outside Keltner in direction of H4 20-EMA trend.
     Active: 07:00 - 15:30 UTC. Target: 2.0R.

Tested across EURUSD, EURGBP, and USDJPY on 70% Discovery Partition ONLY.
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
# Model 1: London Lunch Compression Expansion (DISC82)
# ==============================================================================
class LondonLunchCompressionExpansion(BaseStrategy):
    """
    Operates during Pre-NY surge (13:30 - 15:00 UTC).
    Identifies midday European compression (11:30 - 13:15 UTC).
    Midday range < 65% of Asian range (00:00 - 06:30 UTC).
    Decisive M15 close breaking midday high/low in H4 trend direction.
    Target: 2.0R.
    """
    def __init__(self, asset: str, sl_pips: float, tp_pips: float, pip_size: float, max_spread: float, **kwargs):
        super().__init__(**kwargs)
        self.asset = asset
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.pip_size = float(pip_size)
        self.max_spread = float(max_spread)
        self.name = f"{asset}_London_Lunch_Expansion"

    def generate_signals(self, ticks: np.ndarray):
        n = len(ticks)
        signals = np.zeros(n, dtype=np.int8)
        sl_arr = np.zeros(n, dtype=np.float64)
        tp_arr = np.zeros(n, dtype=np.float64)
        if n < 500:
            return signals, sl_arr, tp_arr

        m15 = build_time_bars(ticks, timeframe_min=15, pip_size=self.pip_size)
        h4 = build_time_bars(ticks, timeframe_min=240, pip_size=self.pip_size)
        if len(m15) < 30 or len(h4) < 10:
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
        spread = m15_df['spread_pips'].values
        h4_trend = m15_df['h4_bias'].fillna(0).values.astype(int)
        hours = m15_df.index.hour.values
        minutes = m15_df.index.minute.values
        dates = m15_df.index.date
        time_ms = m15_df.index.astype('datetime64[ms]').view('int64')
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        # Precompute Asian ranges and Lunch ranges per date
        unique_dates = np.unique(dates)
        lunch_hi = {}
        lunch_lo = {}
        valid_dates = set()

        for d in unique_dates:
            mask_asia = (dates == d) & (hours >= 0) & ((hours < 6) | ((hours == 6) & (minutes <= 30)))
            mask_lunch = (dates == d) & (((hours == 11) & (minutes >= 30)) | (hours == 12) | ((hours == 13) & (minutes <= 15)))

            if np.any(mask_asia) and np.any(mask_lunch):
                asia_rng = (np.max(h[mask_asia]) - np.min(l[mask_asia])) / self.pip_size
                l_hi = np.max(h[mask_lunch])
                l_lo = np.min(l[mask_lunch])
                lunch_rng = (l_hi - l_lo) / self.pip_size

                # Lunch range compressed compared to Asian range
                if asia_rng >= 12.0 and (lunch_rng <= 0.65 * asia_rng) and (lunch_rng >= 5.0):
                    lunch_hi[d] = l_hi
                    lunch_lo[d] = l_lo
                    valid_dates.add(d)

        last_trade_date = None
        for i in range(1, len(m15_df)):
            dt = dates[i]
            hr = hours[i]
            mn = minutes[i]

            # Entry window 13:30 to 15:00 UTC
            in_window = ((hr == 13) and (mn >= 30)) or (hr == 14) or ((hr == 15) and (mn == 0))
            if not in_window:
                continue
            if dt == last_trade_date or dt not in valid_dates:
                continue
            if spread[i] > self.max_spread:
                continue

            macro = h4_trend[i]
            if macro == 0:
                continue

            hi_lvl = lunch_hi[dt]
            lo_lvl = lunch_lo[dt]

            sig = 0
            if macro == 1 and c[i] > hi_lvl and c[i-1] <= hi_lvl:
                sig = SIGNAL_LONG
            elif macro == -1 and c[i] < lo_lvl and c[i-1] >= lo_lvl:
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


# ==============================================================================
# Model 2: Swing Breakout Retest (DISC83)
# ==============================================================================
class SwingBreakoutRetest(BaseStrategy):
    """
    Operates during 07:00 - 16:30 UTC.
    Detects M15 20-bar swing breakout.
    Waits for price to pull back to within 2.5 pips of broken swing level within 2-6 bars.
    Confirms retest rejection wick >= 35% in trend direction.
    Target: 1.8R.
    """
    def __init__(self, asset: str, sl_pips: float, tp_pips: float, pip_size: float, max_spread: float, **kwargs):
        super().__init__(**kwargs)
        self.asset = asset
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.pip_size = float(pip_size)
        self.max_spread = float(max_spread)
        self.name = f"{asset}_Swing_Breakout_Retest"

    def generate_signals(self, ticks: np.ndarray):
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
        dates = m15_df.index.date
        time_ms = m15_df.index.astype('datetime64[ms]').view('int64')
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        # 20-bar rolling high / low
        roll_hi = pd.Series(h).shift(1).rolling(20).max().values
        roll_lo = pd.Series(l).shift(1).rolling(20).min().values

        # Track broken levels: (bar_idx, level, direction)
        active_breaks = []  # list of [break_bar, level, direction, expire_bar]

        last_trade_date = None
        tol = 2.5 * self.pip_size

        for i in range(25, len(m15_df)):
            dt = dates[i]
            hr = hours[i]
            macro = h4_trend[i]

            # Clean expired breaks
            active_breaks = [b for b in active_breaks if b[3] >= i]

            # Detect new breakout
            if macro == 1 and c[i] > roll_hi[i] and c[i-1] <= roll_hi[i]:
                active_breaks.append([i, roll_hi[i], 1, i + 6])
            elif macro == -1 and c[i] < roll_lo[i] and c[i-1] >= roll_lo[i]:
                active_breaks.append([i, roll_lo[i], -1, i + 6])

            if not (7 <= hr <= 16):
                continue
            if dt == last_trade_date:
                continue
            if spread[i] > self.max_spread:
                continue

            bar_rng = h[i] - l[i]
            if bar_rng < 4.0 * self.pip_size:
                continue

            sig = 0
            for b in active_breaks:
                brk_bar, level, direction, expire = b
                # Retest must be at least 2 bars after the breakout
                if i - brk_bar < 2:
                    continue

                if direction == 1 and macro == 1:
                    # Low tests broken resistance (now support)
                    if abs(l[i] - level) <= tol and c[i] > level and c[i] > o[i]:
                        lower_wick = (min(o[i], c[i]) - l[i]) / bar_rng
                        if lower_wick >= 0.35:
                            sig = SIGNAL_LONG
                            break
                elif direction == -1 and macro == -1:
                    # High tests broken support (now resistance)
                    if abs(h[i] - level) <= tol and c[i] < level and c[i] < o[i]:
                        upper_wick = (h[i] - max(o[i], c[i])) / bar_rng
                        if upper_wick >= 0.35:
                            sig = SIGNAL_SHORT
                            break

            if sig != 0:
                bar_close_ms = time_ms[i] + dur_ms
                idx = np.searchsorted(tick_ts, bar_close_ms)
                if idx < n:
                    signals[idx] = sig
                    sl_arr[idx] = self.sl_pips
                    tp_arr[idx] = self.tp_pips
                    last_trade_date = dt

        return signals, sl_arr, tp_arr


# ==============================================================================
# Model 3: Session VWAP Band Fade (DISC84)
# ==============================================================================
class SessionVWAPBandFade(BaseStrategy):
    """
    Operates during European/US session (07:00 - 16:00 UTC).
    Computes daily VWAP from 00:00 UTC using volume proxy (tick count / spread).
    Calculates 2.0-SD volatility band around VWAP.
    When price exceeds the 2.0-SD band in counter-trend, and then snaps back inside
    in the direction of the H4 20-EMA trend.
    Target: 2.0R.
    """
    def __init__(self, asset: str, sl_pips: float, tp_pips: float, pip_size: float, max_spread: float, **kwargs):
        super().__init__(**kwargs)
        self.asset = asset
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.pip_size = float(pip_size)
        self.max_spread = float(max_spread)
        self.name = f"{asset}_VWAP_Band_Fade"

    def generate_signals(self, ticks: np.ndarray):
        n = len(ticks)
        signals = np.zeros(n, dtype=np.int8)
        sl_arr = np.zeros(n, dtype=np.float64)
        tp_arr = np.zeros(n, dtype=np.float64)
        if n < 500:
            return signals, sl_arr, tp_arr

        m15 = build_time_bars(ticks, timeframe_min=15, pip_size=self.pip_size)
        h4 = build_time_bars(ticks, timeframe_min=240, pip_size=self.pip_size)
        if len(m15) < 30 or len(h4) < 10:
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
        spread = m15_df['spread_pips'].values
        h4_trend = m15_df['h4_bias'].fillna(0).values.astype(int)
        hours = m15_df.index.hour.values
        dates = m15_df.index.date
        time_ms = m15_df.index.astype('datetime64[ms]').view('int64')
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        # Approximate VWAP per date using typical price and ticks count
        typ = (h + l + c) / 3.0
        vols = m15_df['tick_count'].values if 'tick_count' in m15_df.columns else np.ones(len(m15_df))

        vwap = np.zeros(len(m15_df))
        band_hi = np.zeros(len(m15_df))
        band_lo = np.zeros(len(m15_df))

        unique_dates = np.unique(dates)
        for d in unique_dates:
            mask = (dates == d)
            idx_sub = np.where(mask)[0]
            if len(idx_sub) == 0:
                continue
            t_sub = typ[idx_sub]
            v_sub = vols[idx_sub]
            cum_v = np.cumsum(v_sub)
            cum_pv = np.cumsum(t_sub * v_sub)
            vwap_sub = cum_pv / np.maximum(cum_v, 1)

            # Variance
            cum_p2v = np.cumsum((t_sub ** 2) * v_sub)
            var_sub = np.maximum(0, (cum_p2v / np.maximum(cum_v, 1)) - (vwap_sub ** 2))
            std_sub = np.sqrt(var_sub)

            vwap[idx_sub] = vwap_sub
            band_hi[idx_sub] = vwap_sub + 2.0 * std_sub
            band_lo[idx_sub] = vwap_sub - 2.0 * std_sub

        last_trade_date = None
        for i in range(1, len(m15_df)):
            dt = dates[i]
            hr = hours[i]
            if not (7 <= hr <= 16):
                continue
            if dt == last_trade_date:
                continue
            if spread[i] > self.max_spread:
                continue

            macro = h4_trend[i]
            if macro == 0 or band_hi[i] == band_lo[i]:
                continue

            sig = 0
            # Bullish trend: price dipped below lower VWAP band on prev bar, closes back above band
            if macro == 1 and l[i-1] < band_lo[i-1] and c[i] > band_lo[i] and c[i] > (h[i] + l[i])/2.0:
                sig = SIGNAL_LONG
            # Bearish trend: price poked above upper VWAP band on prev bar, closes back below band
            elif macro == -1 and h[i-1] > band_hi[i-1] and c[i] < band_hi[i] and c[i] < (h[i] + l[i])/2.0:
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


# ==============================================================================
# Model 4: Keltner-Bollinger Squeeze Explosion (DISC85)
# ==============================================================================
class KeltnerBollingerSqueezeExplosion(BaseStrategy):
    """
    Operates during European/US morning (07:00 - 15:30 UTC).
    Computes Bollinger Bands (20, 2.0) and Keltner Channel (20, 1.5 ATR).
    Squeeze state: BB upper < KC upper AND BB lower > KC lower (Bollinger Bands inside Keltner).
    Explosion state: Squeeze was active within last 4 bars; current bar breaks out of Keltner Channel
    in the direction of H4 20-EMA trend.
    Target: 2.0R.
    """
    def __init__(self, asset: str, sl_pips: float, tp_pips: float, pip_size: float, max_spread: float, **kwargs):
        super().__init__(**kwargs)
        self.asset = asset
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.pip_size = float(pip_size)
        self.max_spread = float(max_spread)
        self.name = f"{asset}_Keltner_Squeeze_Explosion"

    def generate_signals(self, ticks: np.ndarray):
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
        spread = m15_df['spread_pips'].values
        h4_trend = m15_df['h4_bias'].fillna(0).values.astype(int)
        hours = m15_df.index.hour.values
        dates = m15_df.index.date
        time_ms = m15_df.index.astype('datetime64[ms]').view('int64')
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        # Bollinger Bands (20, 2.0)
        c_s = pd.Series(c)
        bb_mid = c_s.rolling(20).mean().values
        bb_std = c_s.rolling(20).std().values
        bb_up = bb_mid + 2.0 * bb_std
        bb_lo = bb_mid - 2.0 * bb_std

        # ATR 20
        tr = np.maximum(h[1:] - l[1:], np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))
        tr = np.insert(tr, 0, h[0] - l[0])
        atr = pd.Series(tr).rolling(20).mean().values

        # Keltner Channel (20, 1.5 ATR)
        kc_mid = c_s.ewm(span=20, adjust=False).mean().values
        kc_up = kc_mid + 1.5 * atr
        kc_lo = kc_mid - 1.5 * atr

        # Squeeze indicator: BB inside KC
        squeeze = (bb_up < kc_up) & (bb_lo > kc_lo)
        squeeze_recent = pd.Series(squeeze).rolling(4).max().fillna(0).values.astype(bool)

        last_trade_date = None
        for i in range(25, len(m15_df)):
            dt = dates[i]
            hr = hours[i]
            if not (7 <= hr <= 15):
                continue
            if dt == last_trade_date:
                continue
            if spread[i] > self.max_spread:
                continue

            macro = h4_trend[i]
            if macro == 0 or np.isnan(kc_up[i]):
                continue

            # Squeeze must have been present in past 4 bars
            if not squeeze_recent[i-1]:
                continue

            sig = 0
            if macro == 1 and c[i] > kc_up[i] and c[i-1] <= kc_up[i-1]:
                sig = SIGNAL_LONG
            elif macro == -1 and c[i] < kc_lo[i] and c[i-1] >= kc_lo[i-1]:
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


# ==============================================================================
# Discovery Worker for Alpha Discovery Lab V17
# ==============================================================================
def evaluate_v17_discovery_week(task: Tuple[str, int]) -> Dict[str, Any]:
    asset, week_num = task
    pip_size = 0.01 if asset == "USDJPY" else 0.0001
    engine = BacktestEngine(pip_size=pip_size, use_slippage=True)

    ticks, meta = get_week_dataset(week_num, raw_dir=f"data/{asset}")
    if ticks is None or len(ticks) < 500:
        return {"asset": asset, "week": week_num, "trades": {}}

    models = []
    if asset == "EURUSD":
        models.append(("LUNCH_EXPANSION", LondonLunchCompressionExpansion(asset=asset, sl_pips=9.0, tp_pips=18.0, pip_size=0.0001, max_spread=0.6)))
        models.append(("SWING_RETEST", SwingBreakoutRetest(asset=asset, sl_pips=9.0, tp_pips=16.2, pip_size=0.0001, max_spread=0.6)))
        models.append(("VWAP_BAND_FADE", SessionVWAPBandFade(asset=asset, sl_pips=9.0, tp_pips=18.0, pip_size=0.0001, max_spread=0.6)))
        models.append(("KELTNER_SQUEEZE", KeltnerBollingerSqueezeExplosion(asset=asset, sl_pips=9.0, tp_pips=18.0, pip_size=0.0001, max_spread=0.6)))
    elif asset == "USDJPY":
        models.append(("LUNCH_EXPANSION", LondonLunchCompressionExpansion(asset=asset, sl_pips=12.0, tp_pips=24.0, pip_size=0.01, max_spread=0.8)))
        models.append(("SWING_RETEST", SwingBreakoutRetest(asset=asset, sl_pips=12.0, tp_pips=21.6, pip_size=0.01, max_spread=0.8)))
        models.append(("VWAP_BAND_FADE", SessionVWAPBandFade(asset=asset, sl_pips=12.0, tp_pips=24.0, pip_size=0.01, max_spread=0.8)))
        models.append(("KELTNER_SQUEEZE", KeltnerBollingerSqueezeExplosion(asset=asset, sl_pips=12.0, tp_pips=24.0, pip_size=0.01, max_spread=0.8)))
    elif asset == "EURGBP":
        models.append(("LUNCH_EXPANSION", LondonLunchCompressionExpansion(asset=asset, sl_pips=8.0, tp_pips=16.0, pip_size=0.0001, max_spread=0.8)))
        models.append(("SWING_RETEST", SwingBreakoutRetest(asset=asset, sl_pips=8.0, tp_pips=14.4, pip_size=0.0001, max_spread=0.8)))
        models.append(("VWAP_BAND_FADE", SessionVWAPBandFade(asset=asset, sl_pips=8.0, tp_pips=16.0, pip_size=0.0001, max_spread=0.8)))
        models.append(("KELTNER_SQUEEZE", KeltnerBollingerSqueezeExplosion(asset=asset, sl_pips=8.0, tp_pips=16.0, pip_size=0.0001, max_spread=0.8)))

    week_trades = {}
    for m_name, model in models:
        try:
            raw_trades = engine.run(ticks, model)
            t_list = []
            for i in range(len(raw_trades)):
                row = raw_trades[i]
                t_list.append({
                    "strategy": m_name,
                    "asset": asset,
                    "week": week_num,
                    "pips": float(row["pips"]),
                    "direction": int(row["direction"])
                })
            week_trades[m_name] = t_list
        except Exception:
            week_trades[m_name] = []

    return {"asset": asset, "week": week_num, "trades": week_trades}


def run_discovery_v17():
    t_start = time.time()
    print("=" * 80, flush=True)
    print("  LAUNCHING ALPHA DISCOVERY V17 EXPEDITION (70% DISCOVERY PARTITIONS)", flush=True)
    print("=" * 80, flush=True)

    assets = [os.environ["DISCOVERY_TARGET_SYMBOL"]] if "DISCOVERY_TARGET_SYMBOL" in os.environ else ["USDJPY", "EURGBP", "EURUSD"]
    all_tasks = []
    for asset in assets:
        disc_weeks = get_discovery_weeks(f"data/{asset}")
        print(f"[*] Queued {len(disc_weeks)} discovery weeks for {asset}...", flush=True)
        for w in disc_weeks:
            all_tasks.append((asset, w))

    print(f"\n[*] Total discovery tasks across all assets: {len(all_tasks)} weeks", flush=True)

    results_by_key: Dict[Tuple[str, str], List[float]] = {}
    completed = 0

    with ProcessPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(evaluate_v17_discovery_week, t): t for t in all_tasks}
        for future in as_completed(futures):
            res = future.result()
            completed += 1
            asset = res["asset"]
            for m_name, t_list in res["trades"].items():
                key = (asset, m_name)
                if key not in results_by_key:
                    results_by_key[key] = []
                results_by_key[key].extend([t["pips"] for t in t_list])

            if completed % 100 == 0 or completed == len(all_tasks):
                print(f"  [+] Processed {completed:3d}/{len(all_tasks)} discovery weeks in {time.time() - t_start:.1f}s...", flush=True)

    # Scorecard
    print("\n" + "=" * 90, flush=True)
    print(f"{'ASSET':<8} {'MODEL NAME':<26} {'TRADES':<8} {'WIN RATE':<10} {'PROFIT FACTOR':<15} {'NET PIPS':<12} {'STATUS'}", flush=True)
    print("=" * 90, flush=True)

    summary_rows = []
    for (asset, m_name), pips in sorted(results_by_key.items()):
        n_t = len(pips)
        if n_t == 0:
            print(f"{asset:<8} {m_name:<26} {'0':<8} {'0.0%':<10} {'0.00':<15} {'+0.0p':<12} {'NO TRADES'}", flush=True)
            continue

        wins = [p for p in pips if p > 0]
        losses = [p for p in pips if p < 0]
        wr = len(wins) / n_t * 100.0
        gross_win = sum(wins) if wins else 0.0
        gross_loss = abs(sum(losses)) if losses else 0.0
        pf = gross_win / gross_loss if gross_loss > 0 else 999.0
        net = sum(pips)

        status = "REJECTED"
        if pf >= 1.25 and net >= 500.0 and n_t >= 250:
            status = "PASS_HOLDOUT_CANDIDATE"

        summary_rows.append({
            "asset": asset,
            "model": m_name,
            "trades": n_t,
            "win_rate": round(wr, 1),
            "profit_factor": round(pf, 2),
            "net_pips": round(net, 1),
            "status": status
        })

        sign = "+" if net >= 0 else ""
        print(f"{asset:<8} {m_name:<26} {n_t:<8} {wr:5.1f}%    {pf:<15.2f} {sign}{net:.1f}p       {status}", flush=True)

    print("=" * 90, flush=True)
    elapsed = time.time() - t_start
    print(f"[*] Discovery Lab V17 complete in {elapsed:.2f} seconds.", flush=True)

    out_file = RESULTS_DIR / "alpha_discovery_lab_v17_summary.json"
    with open(out_file, "w") as f:
        json.dump(summary_rows, f, indent=2)
    print(f"[*] Results saved to {out_file}", flush=True)


if __name__ == "__main__":
    run_discovery_v17()
