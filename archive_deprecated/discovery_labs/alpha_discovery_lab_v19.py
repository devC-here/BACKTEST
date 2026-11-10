#!/usr/bin/env python3
"""
================================================================================
  ALPHA DISCOVERY LAB V19: LIQUIDITY RETEST, MTF MICRO-SCALPING & ASYMMETRIC FADE
================================================================================
Explores 4 novel algorithmic architectures across the 70% In-Sample Discovery
partition (Seed 42) across EURUSD, USDJPY, and EURGBP (603 discovery weeks):

  1. ASIAN_HIGH_LOW_RETEST:
     - Measures Asian Range (00:00 - 06:30 UTC).
     - At London morning (07:30 - 11:30 UTC), price breaks Asian High/Low,
       then retests the broken level with a rejection wick (>= 40%) in H4 trend direction.
     - Target: 2.0R.

  2. H4_STOCH_M5_WICK_SCALPER:
     - H4 Stochastic(14,3,3) extreme (< 20 oversold or > 80 overbought).
     - M5 liquidity absorption wick (shadow >= 50% of bar) during active hours (08:00 - 16:00 UTC).
     - Target: 2.0R.

  3. NY_AFTERNOON_RANGE_FADE:
     - Post-London fix session (16:30 - 19:30 UTC).
     - Intraday High/Low set between 12:00 - 16:00 UTC is swept by 2-6 pips, then closes back inside.
     - Mean reversion fade towards session median.
     - Target: 1.5R.

  4. ADAPTIVE_ATR_ASIAN_MEAN_REVERSION:
     - Asian quiet hours (21:00 - 04:30 UTC).
     - Price extends >= 1.75 ATR(14) away from M15 20-EMA with RSI(14) < 25 or > 75.
     - Counter-trend fade back to 20-EMA.
     - Target: 1.5R.
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
# Architecture 1: Asian High/Low Retest (DISC71)
# ==============================================================================
class AsianHighLowRetest(BaseStrategy):
    """
    Measures Asian Range (00:00 - 06:30 UTC).
    Between 07:30 - 11:30 UTC, if price has broken Asian High/Low by >= 2 pips,
    waits for a retest of the broken level (+/- 2.5 pips) with a rejection wick >= 40%.
    H4 20-EMA trend filter.
    Target: 2.0R.
    """
    def __init__(self, asset: str, sl_pips: float, tp_pips: float, pip_size: float, max_spread: float, **kwargs):
        super().__init__(**kwargs)
        self.asset = asset
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.pip_size = float(pip_size)
        self.max_spread = float(max_spread)
        self.name = f"{asset}_Asian_HL_Retest"

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
        minutes = m15_df.index.minute.values
        dates = m15_df.index.date
        time_ms = m15_df.index.astype('datetime64[ms]').view('int64')
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        last_trade_date = None

        # Precompute Asian ranges for each date
        asian_ranges: Dict[Any, Tuple[float, float]] = {}
        for d in np.unique(dates):
            mask = (dates == d) & (hours >= 0) & ((hours < 6) | ((hours == 6) & (minutes <= 30)))
            if np.sum(mask) >= 6:
                asian_ranges[d] = (float(np.max(h[mask])), float(np.min(l[mask])))

        # Track breakout status per day
        broke_hi_map: Dict[Any, bool] = {}
        broke_lo_map: Dict[Any, bool] = {}

        for i in range(15, len(m15_df)):
            dt = dates[i]
            hr = hours[i]
            mn = minutes[i]

            if dt not in asian_ranges:
                continue

            asia_hi, asia_lo = asian_ranges[dt]
            asia_rng = asia_hi - asia_lo
            if asia_rng < 8.0 * self.pip_size:
                continue

            # Check if breakout occurred prior to or at current bar
            if h[i] > asia_hi + 2.0 * self.pip_size:
                broke_hi_map[dt] = True
            if l[i] < asia_lo - 2.0 * self.pip_size:
                broke_lo_map[dt] = True

            in_window = (hr >= 7 and mn >= 30) if hr == 7 else (8 <= hr <= 11)
            if not in_window:
                continue
            if dt == last_trade_date:
                continue
            if spread[i] > self.max_spread:
                continue

            macro = h4_trend[i]
            if macro == 0:
                continue

            bar_rng = h[i] - l[i]
            if bar_rng < 2.5 * self.pip_size:
                continue

            sig = 0
            tol = 2.5 * self.pip_size

            # Bullish retest: broke high earlier, pulls back to touch asia_hi +/- tol, prints rejection wick
            if macro == 1 and broke_hi_map.get(dt, False):
                tested_level = (l[i] <= asia_hi + tol) and (h[i] >= asia_hi - tol)
                closed_above = c[i] > asia_hi
                lower_wick = (min(o[i], c[i]) - l[i]) / bar_rng
                if tested_level and closed_above and lower_wick >= 0.40:
                    sig = SIGNAL_LONG

            # Bearish retest: broke low earlier, pulls back to touch asia_lo +/- tol, prints rejection wick
            elif macro == -1 and broke_lo_map.get(dt, False):
                tested_level = (h[i] >= asia_lo - tol) and (l[i] <= asia_lo + tol)
                closed_below = c[i] < asia_lo
                upper_wick = (h[i] - max(o[i], c[i])) / bar_rng
                if tested_level and closed_below and upper_wick >= 0.40:
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
# Architecture 2: H4 Stoch + M5 Absorption Wick Scalper (DISC72)
# ==============================================================================
class H4StochM5WickScalper(BaseStrategy):
    """
    H4 Stochastic(14,3,3) extreme condition (<20 oversold or >80 overbought).
    During 08:00 - 16:00 UTC, M5 bar prints strong absorption wick (>= 50%)
    confirming institutional buyer/seller liquidity sweep.
    Target: 2.0R.
    """
    def __init__(self, asset: str, sl_pips: float, tp_pips: float, pip_size: float, max_spread: float, **kwargs):
        super().__init__(**kwargs)
        self.asset = asset
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.pip_size = float(pip_size)
        self.max_spread = float(max_spread)
        self.name = f"{asset}_H4_Stoch_M5_Wick"

    def generate_signals(self, ticks: np.ndarray):
        n = len(ticks)
        signals = np.zeros(n, dtype=np.int8)
        sl_arr = np.zeros(n, dtype=np.float64)
        tp_arr = np.zeros(n, dtype=np.float64)
        if n < 500:
            return signals, sl_arr, tp_arr

        m5 = build_time_bars(ticks, timeframe_min=5, pip_size=self.pip_size)
        h4 = build_time_bars(ticks, timeframe_min=240, pip_size=self.pip_size)
        if len(m5) < 60 or len(h4) < 15:
            return signals, sl_arr, tp_arr

        # H4 Stochastic (14, 3, 3)
        h4_h = pd.Series(h4['high'].values)
        h4_l = pd.Series(h4['low'].values)
        h4_c = pd.Series(h4['close'].values)

        lowest_l = h4_l.rolling(14).min()
        highest_h = h4_h.rolling(14).max()
        fast_k = 100.0 * ((h4_c - lowest_l) / np.maximum(highest_h - lowest_l, 1e-6))
        slow_k = fast_k.rolling(3).mean()
        slow_d = slow_k.rolling(3).mean()

        h4_bias = np.zeros(len(h4), dtype=int)
        for j in range(len(h4)):
            if slow_k.iloc[j] < 25 and slow_k.iloc[j] > slow_d.iloc[j]:
                h4_bias[j] = 1  # Bullish hook from oversold
            elif slow_k.iloc[j] > 75 and slow_k.iloc[j] < slow_d.iloc[j]:
                h4_bias[j] = -1  # Bearish hook from overbought

        h4_df = pd.DataFrame({'h4_bias': h4_bias}, index=h4.index)

        m5_df = pd.merge_asof(
            m5.sort_index(),
            h4_df.sort_index(),
            left_index=True,
            right_index=True,
            direction='backward'
        )

        c = m5_df['close'].values
        h = m5_df['high'].values
        l = m5_df['low'].values
        o = m5_df['open'].values
        spread = m5_df['spread_pips'].values
        h4_trend = m5_df['h4_bias'].fillna(0).values.astype(int)
        hours = m5_df.index.hour.values
        dates = m5_df.index.date
        time_ms = m5_df.index.astype('datetime64[ms]').view('int64')
        dur_ms = 5 * 60 * 1000
        tick_ts = ticks['timestamp']

        last_trade_date = None

        for i in range(20, len(m5_df)):
            dt = dates[i]
            hr = hours[i]
            if not (8 <= hr <= 16):
                continue
            if dt == last_trade_date:
                continue
            if spread[i] > self.max_spread:
                continue

            bias = h4_trend[i]
            if bias == 0:
                continue

            bar_rng = h[i] - l[i]
            if bar_rng < 2.5 * self.pip_size:
                continue

            sig = 0
            if bias == 1:
                lower_wick = (min(o[i], c[i]) - l[i]) / bar_rng
                if lower_wick >= 0.50 and c[i] > o[i]:
                    sig = SIGNAL_LONG
            elif bias == -1:
                upper_wick = (h[i] - max(o[i], c[i])) / bar_rng
                if upper_wick >= 0.50 and c[i] < o[i]:
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
# Architecture 3: New York Post-Fix Liquidity Sweep Fade (DISC73)
# ==============================================================================
class NYAfternoonRangeFade(BaseStrategy):
    """
    Between 16:30 - 19:30 UTC (post-London fix).
    Price sweeps the 12:00 - 16:00 UTC intraday High/Low by 2.0 to 6.0 pips,
    then closes back inside the range with a rejection wick >= 35%.
    Counter-trend fade towards session midpoint.
    Target: 1.5R.
    """
    def __init__(self, asset: str, sl_pips: float, tp_pips: float, pip_size: float, max_spread: float, **kwargs):
        super().__init__(**kwargs)
        self.asset = asset
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.pip_size = float(pip_size)
        self.max_spread = float(max_spread)
        self.name = f"{asset}_NY_Afternoon_Fade"

    def generate_signals(self, ticks: np.ndarray):
        n = len(ticks)
        signals = np.zeros(n, dtype=np.int8)
        sl_arr = np.zeros(n, dtype=np.float64)
        tp_arr = np.zeros(n, dtype=np.float64)
        if n < 500:
            return signals, sl_arr, tp_arr

        m15 = build_time_bars(ticks, timeframe_min=15, pip_size=self.pip_size)
        if len(m15) < 40:
            return signals, sl_arr, tp_arr

        c = m15['close'].values
        h = m15['high'].values
        l = m15['low'].values
        o = m15['open'].values
        spread = m15['spread_pips'].values
        hours = m15.index.hour.values
        minutes = m15.index.minute.values
        dates = m15.index.date
        time_ms = m15.index.astype('datetime64[ms]').view('int64')
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        last_trade_date = None

        # Precompute 12:00-16:00 range per date
        ny_ranges: Dict[Any, Tuple[float, float]] = {}
        for d in np.unique(dates):
            mask = (dates == d) & (hours >= 12) & (hours < 16)
            if np.sum(mask) >= 8:
                ny_ranges[d] = (float(np.max(h[mask])), float(np.min(l[mask])))

        for i in range(25, len(m15)):
            dt = dates[i]
            hr = hours[i]
            mn = minutes[i]

            in_window = (hr == 16 and mn >= 30) or (17 <= hr <= 19)
            if not in_window:
                continue
            if dt == last_trade_date:
                continue
            if dt not in ny_ranges:
                continue
            if spread[i] > self.max_spread:
                continue

            ny_hi, ny_lo = ny_ranges[dt]
            ny_rng = ny_hi - ny_lo
            if ny_rng < 10.0 * self.pip_size:
                continue

            bar_rng = h[i] - l[i]
            if bar_rng < 2.5 * self.pip_size:
                continue

            sig = 0
            # False upside breakout: high pierced ny_hi by 2-6 pips, but closed back inside
            pierce_up = h[i] - ny_hi
            if (2.0 * self.pip_size <= pierce_up <= 6.0 * self.pip_size) and (c[i] < ny_hi):
                upper_wick = (h[i] - max(o[i], c[i])) / bar_rng
                if upper_wick >= 0.35:
                    sig = SIGNAL_SHORT

            # False downside breakout: low pierced ny_lo by 2-6 pips, but closed back inside
            pierce_dn = ny_lo - l[i]
            if (2.0 * self.pip_size <= pierce_dn <= 6.0 * self.pip_size) and (c[i] > ny_lo):
                lower_wick = (min(o[i], c[i]) - l[i]) / bar_rng
                if lower_wick >= 0.35:
                    sig = SIGNAL_LONG

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
# Architecture 4: Asian Quiet Hours Adaptive ATR Mean Reversion (DISC74)
# ==============================================================================
class AdaptiveATRBounce(BaseStrategy):
    """
    Operates during Asian quiet hours (21:00 - 04:30 UTC).
    Detects when price moves >= 1.75 ATR(14) away from M15 20-EMA.
    RSI(14) confirmed oversold (<25) or overbought (>75).
    Enters counter-trend mean reversion back towards 20-EMA.
    Target: 1.5R.
    """
    def __init__(self, asset: str, sl_pips: float, tp_pips: float, pip_size: float, max_spread: float, **kwargs):
        super().__init__(**kwargs)
        self.asset = asset
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.pip_size = float(pip_size)
        self.max_spread = float(max_spread)
        self.name = f"{asset}_Asian_ATR_Mean_Reversion"

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
        spread = m15['spread_pips'].values
        hours = m15.index.hour.values
        dates = m15.index.date
        time_ms = m15.index.astype('datetime64[ms]').view('int64')
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        c_s = pd.Series(c)
        ema20 = c_s.ewm(span=20, adjust=False).mean().values

        # ATR 14
        tr = np.maximum(h[1:] - l[1:], np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))
        tr = np.insert(tr, 0, h[0] - l[0])
        atr = pd.Series(tr).rolling(14).mean().values

        # RSI 14
        diff = np.diff(c, prepend=c[0])
        gain = np.where(diff > 0, diff, 0.0)
        loss = np.where(diff < 0, -diff, 0.0)
        avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean().values
        avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean().values
        rs = np.where(avg_loss == 0, 100.0, avg_gain / np.maximum(avg_loss, 1e-9))
        rsi = 100.0 - (100.0 / (1.0 + rs))

        last_trade_date = None

        for i in range(25, len(m15)):
            dt = dates[i]
            hr = hours[i]

            in_asia = (hr >= 21 or hr <= 4)
            if not in_asia:
                continue
            if dt == last_trade_date:
                continue
            if spread[i] > self.max_spread:
                continue
            if np.isnan(atr[i]) or atr[i] < 2.0 * self.pip_size:
                continue

            dist = c[i] - ema20[i]
            atr_mult = dist / atr[i]

            sig = 0
            # Stretched below 20-EMA by >= 1.75 ATR with oversold RSI
            if atr_mult <= -1.75 and rsi[i] < 28.0:
                sig = SIGNAL_LONG
            # Stretched above 20-EMA by >= 1.75 ATR with overbought RSI
            elif atr_mult >= 1.75 and rsi[i] > 72.0:
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
# Execution Harness & Multiprocessing Orchestrator
# ==============================================================================
def build_model(model_name: str, asset: str):
    if model_name == "ASIAN_HL_RETEST":
        if asset == "EURUSD":
            return AsianHighLowRetest(asset=asset, sl_pips=9.0, tp_pips=18.0, pip_size=0.0001, max_spread=0.6)
        elif asset == "USDJPY":
            return AsianHighLowRetest(asset=asset, sl_pips=12.0, tp_pips=24.0, pip_size=0.01, max_spread=0.8)
        elif asset == "EURGBP":
            return AsianHighLowRetest(asset=asset, sl_pips=8.0, tp_pips=16.0, pip_size=0.0001, max_spread=0.8)
    elif model_name == "H4_STOCH_M5_WICK":
        if asset == "EURUSD":
            return H4StochM5WickScalper(asset=asset, sl_pips=7.5, tp_pips=15.0, pip_size=0.0001, max_spread=0.6)
        elif asset == "USDJPY":
            return H4StochM5WickScalper(asset=asset, sl_pips=10.0, tp_pips=20.0, pip_size=0.01, max_spread=0.8)
        elif asset == "EURGBP":
            return H4StochM5WickScalper(asset=asset, sl_pips=6.5, tp_pips=13.0, pip_size=0.0001, max_spread=0.8)
    elif model_name == "NY_AFTERNOON_FADE":
        if asset == "EURUSD":
            return NYAfternoonRangeFade(asset=asset, sl_pips=9.0, tp_pips=13.5, pip_size=0.0001, max_spread=0.6)
        elif asset == "USDJPY":
            return NYAfternoonRangeFade(asset=asset, sl_pips=12.0, tp_pips=18.0, pip_size=0.01, max_spread=0.8)
        elif asset == "EURGBP":
            return NYAfternoonRangeFade(asset=asset, sl_pips=7.5, tp_pips=11.25, pip_size=0.0001, max_spread=0.8)
    elif model_name == "ASIAN_ATR_MEAN_REV":
        if asset == "EURUSD":
            return AdaptiveATRBounce(asset=asset, sl_pips=8.0, tp_pips=12.0, pip_size=0.0001, max_spread=0.6)
        elif asset == "USDJPY":
            return AdaptiveATRBounce(asset=asset, sl_pips=10.0, tp_pips=15.0, pip_size=0.01, max_spread=0.8)
        elif asset == "EURGBP":
            return AdaptiveATRBounce(asset=asset, sl_pips=7.0, tp_pips=10.5, pip_size=0.0001, max_spread=0.8)
    return None


def run_single_week_task(task: Tuple[str, str, int]) -> Dict[str, Any]:
    asset, model_name, week_num = task
    pip_size = 0.01 if asset == "USDJPY" else 0.0001
    engine = BacktestEngine(pip_size=pip_size, use_slippage=True)

    ticks, meta = get_week_dataset(week_num, raw_dir=f"data/{asset}")
    if ticks is None or len(ticks) < 500:
        return {"asset": asset, "model": model_name, "week": week_num, "trades": []}

    strat = build_model(model_name, asset)
    if strat is None:
        return {"asset": asset, "model": model_name, "week": week_num, "trades": []}

    try:
        raw_trades = engine.run(ticks, strat)
        t_list = []
        for i in range(len(raw_trades)):
            row = raw_trades[i]
            t_list.append({
                "pips": float(row["pips"]),
                "direction": int(row["direction"])
            })
        return {"asset": asset, "model": model_name, "week": week_num, "trades": t_list}
    except Exception:
        return {"asset": asset, "model": model_name, "week": week_num, "trades": []}


def calculate_metrics(pips: List[float]) -> Dict[str, Any]:
    if not pips:
        return {"trades": 0, "win_rate": 0.0, "pf": 0.0, "net_pips": 0.0, "max_dd": 0.0, "ret_dd": 0.0, "expectancy": 0.0}
    arr = np.array(pips, dtype=np.float64)
    trades = len(arr)
    wins = arr[arr > 0]
    losses = arr[arr < 0]
    wr = len(wins) / trades * 100.0 if trades > 0 else 0.0
    gross_win = float(np.sum(wins)) if len(wins) > 0 else 0.0
    gross_loss = float(abs(np.sum(losses))) if len(losses) > 0 else 0.0
    pf = gross_win / gross_loss if gross_loss > 0 else 999.0
    net_pips = float(np.sum(arr))
    expectancy = net_pips / trades if trades > 0 else 0.0

    cumsum = np.cumsum(arr)
    running_max = np.maximum.accumulate(cumsum)
    drawdowns = running_max - cumsum
    max_dd = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0
    ret_dd = net_pips / max_dd if max_dd > 0 else 0.0

    return {
        "trades": trades,
        "win_rate": round(wr, 1),
        "pf": round(pf, 2),
        "net_pips": round(net_pips, 1),
        "max_dd": round(max_dd, 1),
        "ret_dd": round(ret_dd, 2),
        "expectancy": round(expectancy, 2)
    }


def main():
    t_start = time.time()
    print("=" * 80, flush=True)
    print("  LAUNCHING ALPHA DISCOVERY LAB V19: LIQUIDITY RETEST & ADAPTIVE FADE", flush=True)
    print("=" * 80, flush=True)

    assets = [os.environ["DISCOVERY_TARGET_SYMBOL"]] if "DISCOVERY_TARGET_SYMBOL" in os.environ else ["EURUSD", "USDJPY", "EURGBP"]
    models = [
        "ASIAN_HL_RETEST",
        "H4_STOCH_M5_WICK",
        "NY_AFTERNOON_FADE",
        "ASIAN_ATR_MEAN_REV"
    ]

    all_tasks = []
    for asset in assets:
        disc_weeks = get_discovery_weeks(f"data/{asset}")
        for m_name in models:
            for w in disc_weeks:
                all_tasks.append((asset, m_name, w))

    print(f"[*] Total week-tasks across 3 assets & 4 models: {len(all_tasks)}", flush=True)

    trades_by_key: Dict[Tuple[str, str], List[float]] = {}
    completed = 0

    with ProcessPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(run_single_week_task, t): t for t in all_tasks}
        for future in as_completed(futures):
            res = future.result()
            completed += 1
            if completed % 300 == 0 or completed == len(all_tasks):
                elapsed = time.time() - t_start
                print(f"  [+] Audited {completed}/{len(all_tasks)} week-tasks in {elapsed:.1f}s...", flush=True)

            k = (res["asset"], res["model"])
            for tr in res["trades"]:
                trades_by_key.setdefault(k, []).append(tr["pips"])

    print("\n" + "=" * 90, flush=True)
    print(f"{'ASSET':<8} | {'MODEL':<20} | {'TRADES':<6} | {'WIN RATE':<8} | {'PF':<5} | {'NET PIPS':<10} | {'MAX DD':<8} | {'RET/DD':<7}", flush=True)
    print("=" * 90, flush=True)

    summary_records = []
    for asset in assets:
        for model in models:
            pips = trades_by_key.get((asset, model), [])
            m = calculate_metrics(pips)
            rec = {
                "asset": asset,
                "model": model,
                **m
            }
            summary_records.append(rec)
            print(f"{asset:<8} | {model:<20} | {m['trades']:<6} | {m['win_rate']:<7}% | {m['pf']:<5.2f} | {m['net_pips']:<10.1f} | {m['max_dd']:<8.1f} | {m['ret_dd']:<6.2f}x", flush=True)
        print("-" * 90, flush=True)

    out_file = RESULTS_DIR / "lab_v19_discovery_results.json"
    with open(out_file, "w") as f:
        json.dump(summary_records, f, indent=2)
    print(f"\n[+] Discovery results saved to: {out_file}", flush=True)
    print(f"[+] Total execution time: {time.time() - t_start:.2f}s", flush=True)
    print("=" * 90, flush=True)


if __name__ == "__main__":
    main()
