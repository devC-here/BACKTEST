#!/usr/bin/env python3
"""
================================================================================
  ALPHA DISCOVERY LAB V5: PIVOT EXTENSIONS, FIBONACCI PULLBACKS & NY REVERSALS
================================================================================
Evaluates 5 New Quantitative Alpha Archetypes across the 70% Discovery Partition:
  1. WEEKLY_PIVOT_WALL: Weekly S1/R1 Institutional Wall Defense (EURUSD, EURGBP, USDJPY)
  2. FIB_GOLDEN_POCKET: Intraday 61.8%-78.6% Retracement Defense in H4 Trend
  3. NY_LUNCH_BREAKOUT: 16:30 - 19:30 UTC Post-London Consolidation Breakout
  4. DAILY_OPEN_REVERSION: Extreme deviation from 00:00 Daily Open mean-reversion
  5. MACRO_ASYMMETRIC_RUNNER: DISC49 entry with 1:2.5 R:R asymmetric profit target

Tested across EURUSD, EURGBP, and USDJPY on 70% Discovery Partition ONLY.
================================================================================
"""

from __future__ import annotations
import sys
import os
import time
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
# Model 1: Weekly Pivot Wall Defense (EURUSD, EURGBP, USDJPY)
# ==============================================================================
class WeeklyPivotWallDefense(BaseStrategy):
    """
    Weekly Pivot, S1, and R1 levels.
    When price tests Weekly S1 (support) or Weekly R1 (resistance) with an M15 rejection
    wick (>= 45%) aligned with H4 20-EMA trend, enter reversal.
    """
    def __init__(self, asset: str, sl_pips: float, tp_pips: float, pip_size: float, max_spread: float, **kwargs):
        super().__init__(**kwargs)
        self.asset = asset
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.pip_size = float(pip_size)
        self.max_spread = float(max_spread)
        self.name = f"{asset}_Weekly_Pivot_Wall"

    def generate_signals(self, ticks: np.ndarray):
        n = len(ticks)
        signals = np.zeros(n, dtype=np.int8)
        sl_arr = np.zeros(n, dtype=np.float64)
        tp_arr = np.zeros(n, dtype=np.float64)
        if n < 500:
            return signals, sl_arr, tp_arr

        m15 = build_time_bars(ticks, timeframe_min=15, pip_size=self.pip_size)
        if len(m15) < 30:
            return signals, sl_arr, tp_arr

        ts = pd.to_datetime(ticks['timestamp'], unit='ms')
        mid = 0.5 * (ticks['bid'] + ticks['ask'])
        base = pd.DataFrame({'close': mid, 'high': mid, 'low': mid}, index=ts)

        # Weekly aggregation
        weekly = base.resample('W-FRI').agg({'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
        if len(weekly) < 2:
            return signals, sl_arr, tp_arr

        prev_wh = weekly['high'].shift(1)
        prev_wl = weekly['low'].shift(1)
        prev_wc = weekly['close'].shift(1)

        pivot = (prev_wh + prev_wl + prev_wc) / 3.0
        r1 = 2.0 * pivot - prev_wl
        s1 = 2.0 * pivot - prev_wh

        r1_m15 = r1.reindex(m15.index, method='ffill').values
        s1_m15 = s1.reindex(m15.index, method='ffill').values

        # H4 Trend Filter
        h4 = base.resample('4h').agg({'close': 'last'}).dropna()
        h4['ema20'] = h4['close'].ewm(span=20, adjust=False).mean()
        h4_trend = np.where(h4['close'] > h4['ema20'], 1, -1)
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

        traded_days = set()
        dist_thresh = 5.0 if self.asset == "USDJPY" else 3.5

        for i in range(15, len(m15)):
            if not (7 <= hours[i] <= 16):
                continue
            d = dates[i]
            if d in traded_days:
                continue
            if spread[i] > self.max_spread:
                continue

            bar_rng = (h[i] - l[i]) / self.pip_size
            if bar_rng < 3.0:
                continue

            bias = h4_bias[i]
            sig = 0

            # Bullish test of Weekly S1
            if bias > 0:
                dist_s1 = abs(l[i] - s1_m15[i]) / self.pip_size
                lower_wick = (min(o[i], c[i]) - l[i]) / self.pip_size
                if dist_s1 <= dist_thresh and (lower_wick / bar_rng) >= 0.45 and c[i] > o[i]:
                    sig = SIGNAL_LONG

            # Bearish test of Weekly R1
            elif bias < 0:
                dist_r1 = abs(r1_m15[i] - h[i]) / self.pip_size
                upper_wick = (h[i] - max(o[i], c[i])) / self.pip_size
                if dist_r1 <= dist_thresh and (upper_wick / bar_rng) >= 0.45 and c[i] < o[i]:
                    sig = SIGNAL_SHORT

            if sig != 0:
                exec_t = time_ms[i] + dur_ms
                idx = int(np.searchsorted(tick_ts, exec_t))
                if idx < n and signals[idx] == 0:
                    signals[idx] = sig
                    sl_arr[idx] = self.sl_pips
                    tp_arr[idx] = self.tp_pips
                    traded_days.add(d)

        return signals, sl_arr, tp_arr


# ==============================================================================
# Model 2: Fibonacci Golden Pocket Retracement (61.8% - 78.6%)
# ==============================================================================
class FibGoldenPocketRetracement(BaseStrategy):
    """
    Computes 24-hour swing high and low on H1 bars.
    When market pulls back into the 61.8% to 78.6% Fibonacci zone of the swing
    aligned with H4 50-EMA trend, triggers entry on M15 rejection candle.
    Target: 1.8R.
    """
    def __init__(self, asset: str, sl_pips: float, tp_pips: float, pip_size: float, max_spread: float, **kwargs):
        super().__init__(**kwargs)
        self.asset = asset
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.pip_size = float(pip_size)
        self.max_spread = float(max_spread)
        self.name = f"{asset}_Fib_Golden_Pocket"

    def generate_signals(self, ticks: np.ndarray):
        n = len(ticks)
        signals = np.zeros(n, dtype=np.int8)
        sl_arr = np.zeros(n, dtype=np.float64)
        tp_arr = np.zeros(n, dtype=np.float64)
        if n < 500:
            return signals, sl_arr, tp_arr

        m15 = build_time_bars(ticks, timeframe_min=15, pip_size=self.pip_size)
        if len(m15) < 30:
            return signals, sl_arr, tp_arr

        ts = pd.to_datetime(ticks['timestamp'], unit='ms')
        mid = 0.5 * (ticks['bid'] + ticks['ask'])
        base = pd.DataFrame({'close': mid, 'high': mid, 'low': mid}, index=ts)

        h1 = base.resample('1h').agg({'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
        if len(h1) < 25:
            return signals, sl_arr, tp_arr

        h1['swing_h'] = h1['high'].rolling(24).max().shift(1)
        h1['swing_l'] = h1['low'].rolling(24).min().shift(1)
        h1['swing_rng'] = h1['swing_h'] - h1['swing_l']

        # Fib levels
        h1['fib618_up'] = h1['swing_h'] - 0.618 * h1['swing_rng']
        h1['fib786_up'] = h1['swing_h'] - 0.786 * h1['swing_rng']

        h1['fib618_dn'] = h1['swing_l'] + 0.618 * h1['swing_rng']
        h1['fib786_dn'] = h1['swing_l'] + 0.786 * h1['swing_rng']

        # H4 trend
        h4 = base.resample('4h').agg({'close': 'last'}).dropna()
        h4['ema50'] = h4['close'].ewm(span=50, adjust=False).mean()
        h4_trend = np.where(h4['close'] > h4['ema50'], 1, -1)
        h4_bias = pd.Series(h4_trend, index=h4.index).shift(1).reindex(m15.index, method='ffill').fillna(0).values

        fib618_up_m15 = h1['fib618_up'].reindex(m15.index, method='ffill').values
        fib786_up_m15 = h1['fib786_up'].reindex(m15.index, method='ffill').values
        fib618_dn_m15 = h1['fib618_dn'].reindex(m15.index, method='ffill').values
        fib786_dn_m15 = h1['fib786_dn'].reindex(m15.index, method='ffill').values

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

        traded_days = set()

        for i in range(25, len(m15)):
            if not (7 <= hours[i] <= 16):
                continue
            d = dates[i]
            if d in traded_days:
                continue
            if spread[i] > self.max_spread:
                continue

            bar_rng = (h[i] - l[i]) / self.pip_size
            if bar_rng < 3.0:
                continue

            bias = h4_bias[i]
            sig = 0

            # Uptrend pullback into 61.8% - 78.6% zone
            if bias > 0:
                f618 = fib618_up_m15[i]
                f786 = fib786_up_m15[i]
                if f786 <= l[i] <= f618 and c[i] > o[i]:
                    lower_wick = (min(o[i], c[i]) - l[i]) / self.pip_size
                    if (lower_wick / bar_rng) >= 0.40:
                        sig = SIGNAL_LONG

            # Downtrend rally into 61.8% - 78.6% zone
            elif bias < 0:
                f618 = fib618_dn_m15[i]
                f786 = fib786_dn_m15[i]
                if f618 <= h[i] <= f786 and c[i] < o[i]:
                    upper_wick = (h[i] - max(o[i], c[i])) / self.pip_size
                    if (upper_wick / bar_rng) >= 0.40:
                        sig = SIGNAL_SHORT

            if sig != 0:
                exec_t = time_ms[i] + dur_ms
                idx = int(np.searchsorted(tick_ts, exec_t))
                if idx < n and signals[idx] == 0:
                    signals[idx] = sig
                    sl_arr[idx] = self.sl_pips
                    tp_arr[idx] = self.tp_pips
                    traded_days.add(d)

        return signals, sl_arr, tp_arr


# ==============================================================================
# Model 3: NY Afternoon Squeeze Breakout (16:45 - 19:30 UTC)
# ==============================================================================
class NYAfternoonBreakout(BaseStrategy):
    """
    Measures 14:30 - 16:30 UTC consolidation range (post-London close).
    If range is tightly compressed (< 18 pips), and breaks out with strong momentum
    candle between 16:45 and 19:00 UTC in direction of daily trend, enters expansion.
    """
    def __init__(self, asset: str, sl_pips: float, tp_pips: float, pip_size: float, max_spread: float, **kwargs):
        super().__init__(**kwargs)
        self.asset = asset
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.pip_size = float(pip_size)
        self.max_spread = float(max_spread)
        self.name = f"{asset}_NY_Afternoon_Breakout"

    def generate_signals(self, ticks: np.ndarray):
        n = len(ticks)
        signals = np.zeros(n, dtype=np.int8)
        sl_arr = np.zeros(n, dtype=np.float64)
        tp_arr = np.zeros(n, dtype=np.float64)
        if n < 500:
            return signals, sl_arr, tp_arr

        m15 = build_time_bars(ticks, timeframe_min=15, pip_size=self.pip_size)
        if len(m15) < 30:
            return signals, sl_arr, tp_arr

        hours = m15.index.hour.values
        dates = m15.index.date
        c = m15['close'].values
        h = m15['high'].values
        l = m15['low'].values
        o = m15['open'].values
        spread = m15['spread_pips'].values
        time_ms = m15.index.astype('datetime64[ms]').view('int64')
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        # Compute 14:30 to 16:30 range per day
        box_ranges = {}
        for d in np.unique(dates):
            mask = (dates == d) & (hours >= 14) & (hours <= 16)
            if np.any(mask):
                bh = np.max(h[mask])
                bl = np.min(l[mask])
                br = (bh - bl) / self.pip_size
                if 5.0 <= br <= (25.0 if self.asset == "USDJPY" else 18.0):
                    box_ranges[d] = (bh, bl)

        traded_days = set()

        for i in range(len(m15)):
            hr = hours[i]
            if hr not in {17, 18, 19}:
                continue
            d = dates[i]
            if d not in box_ranges or d in traded_days:
                continue
            if spread[i] > self.max_spread:
                continue

            bh, bl = box_ranges[d]
            bar_rng = (h[i] - l[i]) / self.pip_size
            body = abs(c[i] - o[i]) / self.pip_size
            if bar_rng < 3.5 or (body / bar_rng) < 0.55:
                continue

            sig = 0
            if c[i] > bh and c[i] > o[i]:
                sig = SIGNAL_LONG
            elif c[i] < bl and c[i] < o[i]:
                sig = SIGNAL_SHORT

            if sig != 0:
                exec_t = time_ms[i] + dur_ms
                idx = int(np.searchsorted(tick_ts, exec_t))
                if idx < n and signals[idx] == 0:
                    signals[idx] = sig
                    sl_arr[idx] = self.sl_pips
                    tp_arr[idx] = self.tp_pips
                    traded_days.add(d)

        return signals, sl_arr, tp_arr


# ==============================================================================
# Model 4: Daily Open Extreme Deviation Mean Reversion
# ==============================================================================
class DailyOpenDeviationReversion(BaseStrategy):
    """
    Measures 00:00 UTC Daily Open price.
    When market expands > 35 pips from Daily Open during London/NY (08:00 - 15:00 UTC)
    and prints a clear exhaustion pin bar (wick >= 50%), enters reversion targeting Daily Open.
    """
    def __init__(self, asset: str, sl_pips: float, tp_pips: float, pip_size: float, max_spread: float, **kwargs):
        super().__init__(**kwargs)
        self.asset = asset
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.pip_size = float(pip_size)
        self.max_spread = float(max_spread)
        self.name = f"{asset}_Daily_Open_Reversion"

    def generate_signals(self, ticks: np.ndarray):
        n = len(ticks)
        signals = np.zeros(n, dtype=np.int8)
        sl_arr = np.zeros(n, dtype=np.float64)
        tp_arr = np.zeros(n, dtype=np.float64)
        if n < 500:
            return signals, sl_arr, tp_arr

        m15 = build_time_bars(ticks, timeframe_min=15, pip_size=self.pip_size)
        if len(m15) < 30:
            return signals, sl_arr, tp_arr

        hours = m15.index.hour.values
        dates = m15.index.date
        c = m15['close'].values
        h = m15['high'].values
        l = m15['low'].values
        o = m15['open'].values
        spread = m15['spread_pips'].values
        time_ms = m15.index.astype('datetime64[ms]').view('int64')
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        # Determine 00:00 Open for each day
        open_prices = {}
        for d in np.unique(dates):
            mask = (dates == d)
            if np.any(mask):
                open_prices[d] = o[mask][0]

        traded_days = set()
        dist_thresh = 45.0 if self.asset == "USDJPY" else 30.0

        for i in range(len(m15)):
            if not (8 <= hours[i] <= 15):
                continue
            d = dates[i]
            if d not in open_prices or d in traded_days:
                continue
            if spread[i] > self.max_spread:
                continue

            day_o = open_prices[d]
            dev_pips = (c[i] - day_o) / self.pip_size
            bar_rng = (h[i] - l[i]) / self.pip_size
            if bar_rng < 3.5:
                continue

            sig = 0
            # Fading upper extreme: price is > dist_thresh above open, prints upper rejection wick >= 50%
            if dev_pips >= dist_thresh and c[i] < o[i]:
                upper_wick = (h[i] - max(o[i], c[i])) / self.pip_size
                if (upper_wick / bar_rng) >= 0.50:
                    sig = SIGNAL_SHORT

            # Fading lower extreme: price is < -dist_thresh below open, prints lower rejection wick >= 50%
            elif dev_pips <= -dist_thresh and c[i] > o[i]:
                lower_wick = (min(o[i], c[i]) - l[i]) / self.pip_size
                if (lower_wick / bar_rng) >= 0.50:
                    sig = SIGNAL_LONG

            if sig != 0:
                exec_t = time_ms[i] + dur_ms
                idx = int(np.searchsorted(tick_ts, exec_t))
                if idx < n and signals[idx] == 0:
                    signals[idx] = sig
                    sl_arr[idx] = self.sl_pips
                    tp_arr[idx] = self.tp_pips
                    traded_days.add(d)

        return signals, sl_arr, tp_arr


# ==============================================================================
# Model 5: Macro Asymmetric Trend Runner (DISC49 with 1:2.5 R:R Target)
# ==============================================================================
class MacroAsymmetricRunner(BaseStrategy):
    """
    Extends DISC49 H1/M15 Donchian retest to capture larger asymmetric runners.
    Risk: 1.0R (tight micro stop).
    Reward: 2.5R target (25.0p on EURUSD, 30.0p on USDJPY, 22.0p on EURGBP).
    """
    def __init__(self, asset: str, sl_pips: float, tp_pips: float, pip_size: float, max_spread: float, **kwargs):
        super().__init__(**kwargs)
        self.asset = asset
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.pip_size = float(pip_size)
        self.max_spread = float(max_spread)
        self.name = f"{asset}_Macro_Asymmetric_Runner"

    def generate_signals(self, ticks: np.ndarray):
        n = len(ticks)
        signals = np.zeros(n, dtype=np.int8)
        sl_arr = np.zeros(n, dtype=np.float64)
        tp_arr = np.zeros(n, dtype=np.float64)
        if n < 500:
            return signals, sl_arr, tp_arr

        m15 = build_time_bars(ticks, timeframe_min=15, pip_size=self.pip_size)
        if len(m15) < 30:
            return signals, sl_arr, tp_arr

        ts = pd.to_datetime(ticks['timestamp'], unit='ms')
        mid = 0.5 * (ticks['bid'] + ticks['ask'])
        base = pd.DataFrame({'close': mid, 'high': mid, 'low': mid}, index=ts)
        h1 = base.resample('1h').agg({'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
        if len(h1) < 25:
            return signals, sl_arr, tp_arr

        h1['ema20'] = h1['close'].ewm(span=20, adjust=False).mean()
        h1['bias'] = np.where(h1['close'] > h1['ema20'], 1, -1)
        h1['roll_h'] = h1['high'].rolling(20).max().shift(1).bfill()
        h1['roll_l'] = h1['low'].rolling(20).min().shift(1).bfill()

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

        traded_days = set()
        valid_hours = {1, 2, 3, 7, 8, 9, 12, 13, 14, 15} if self.asset == "USDJPY" else set(range(7, 17))
        dist_thresh = 4.0 if self.asset == "USDJPY" else 3.5

        for i in range(15, len(m15)):
            if hours[i] not in valid_hours:
                continue
            d = dates[i]
            if d in traded_days:
                continue
            if spread[i] > self.max_spread:
                continue

            bar_rng = (h[i] - l[i]) / self.pip_size
            if bar_rng < (4.0 if self.asset == "USDJPY" else 3.0):
                continue

            b = bias_m15[i]
            sig = 0

            # Bullish Retest of H1 Support
            if b > 0:
                dist_supp = abs(l[i] - supp_m15[i]) / self.pip_size
                lower_wick = (min(o[i], c[i]) - l[i]) / self.pip_size
                if dist_supp <= dist_thresh and (lower_wick / bar_rng) >= 0.45 and c[i] > o[i]:
                    sig = SIGNAL_LONG

            # Bearish Retest of H1 Resistance
            elif b < 0:
                dist_res = abs(res_m15[i] - h[i]) / self.pip_size
                upper_wick = (h[i] - max(o[i], c[i])) / self.pip_size
                if dist_res <= dist_thresh and (upper_wick / bar_rng) >= 0.45 and c[i] < o[i]:
                    sig = SIGNAL_SHORT

            if sig != 0:
                exec_t = time_ms[i] + dur_ms
                idx = int(np.searchsorted(tick_ts, exec_t))
                if idx < n and signals[idx] == 0:
                    signals[idx] = sig
                    sl_arr[idx] = self.sl_pips
                    tp_arr[idx] = self.tp_pips
                    traded_days.add(d)

        return signals, sl_arr, tp_arr


# ==============================================================================
# Discovery V5 Multi-Asset Harness
# ==============================================================================
def evaluate_v5_discovery_week(args: Tuple[str, int]) -> Dict[str, Any]:
    asset, week_num = args
    pip_size = 0.01 if asset == "USDJPY" else 0.0001
    engine = BacktestEngine(pip_size=pip_size, use_slippage=True)

    ticks, meta = get_week_dataset(week_num, raw_dir=f"data/{asset}")
    if len(ticks) == 0:
        return {"asset": asset, "week": week_num, "trades": {}}

    models = []
    if asset == "USDJPY":
        models.append(("WEEKLY_PIVOT_WALL", WeeklyPivotWallDefense(asset=asset, sl_pips=14.0, tp_pips=24.0, pip_size=0.01, max_spread=0.8)))
        models.append(("FIB_POCKET", FibGoldenPocketRetracement(asset=asset, sl_pips=12.0, tp_pips=22.0, pip_size=0.01, max_spread=0.8)))
        models.append(("NY_BREAKOUT", NYAfternoonBreakout(asset=asset, sl_pips=14.0, tp_pips=22.0, pip_size=0.01, max_spread=0.8)))
        models.append(("DAILY_OPEN_REV", DailyOpenDeviationReversion(asset=asset, sl_pips=14.0, tp_pips=24.0, pip_size=0.01, max_spread=0.8)))
        models.append(("ASYMMETRIC_RUNNER", MacroAsymmetricRunner(asset=asset, sl_pips=12.0, tp_pips=30.0, pip_size=0.01, max_spread=0.8)))
    elif asset == "EURGBP":
        models.append(("WEEKLY_PIVOT_WALL", WeeklyPivotWallDefense(asset=asset, sl_pips=8.0, tp_pips=14.0, pip_size=0.0001, max_spread=2.0)))
        models.append(("FIB_POCKET", FibGoldenPocketRetracement(asset=asset, sl_pips=8.0, tp_pips=14.0, pip_size=0.0001, max_spread=2.0)))
        models.append(("NY_BREAKOUT", NYAfternoonBreakout(asset=asset, sl_pips=8.0, tp_pips=13.0, pip_size=0.0001, max_spread=2.0)))
        models.append(("DAILY_OPEN_REV", DailyOpenDeviationReversion(asset=asset, sl_pips=8.0, tp_pips=14.0, pip_size=0.0001, max_spread=2.0)))
        models.append(("ASYMMETRIC_RUNNER", MacroAsymmetricRunner(asset=asset, sl_pips=8.0, tp_pips=20.0, pip_size=0.0001, max_spread=2.0)))
    elif asset == "EURUSD":
        models.append(("WEEKLY_PIVOT_WALL", WeeklyPivotWallDefense(asset=asset, sl_pips=9.0, tp_pips=16.0, pip_size=0.0001, max_spread=0.6)))
        models.append(("FIB_POCKET", FibGoldenPocketRetracement(asset=asset, sl_pips=9.0, tp_pips=16.0, pip_size=0.0001, max_spread=0.6)))
        models.append(("NY_BREAKOUT", NYAfternoonBreakout(asset=asset, sl_pips=9.0, tp_pips=15.0, pip_size=0.0001, max_spread=0.6)))
        models.append(("DAILY_OPEN_REV", DailyOpenDeviationReversion(asset=asset, sl_pips=9.0, tp_pips=16.0, pip_size=0.0001, max_spread=0.6)))
        models.append(("ASYMMETRIC_RUNNER", MacroAsymmetricRunner(asset=asset, sl_pips=9.0, tp_pips=24.0, pip_size=0.0001, max_spread=0.6)))

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


def run_discovery_v5():
    t_start = time.time()
    print("=" * 80, flush=True)
    print("  LAUNCHING ALPHA DISCOVERY V5 EXPEDITION (70% DISCOVERY PARTITIONS)", flush=True)
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
        futures = {executor.submit(evaluate_v5_discovery_week, t): t for t in all_tasks}
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

    # Compute Scorecard
    rows = []
    for (asset, m_name), pips in results_by_key.items():
        n = len(pips)
        if n == 0:
            continue
        p_arr = np.array(pips)
        net_pips = float(np.sum(p_arr))
        wins = p_arr[p_arr > 0]
        losses = -p_arr[p_arr < 0]
        wr = float(len(wins) / n * 100.0)
        sum_w = float(np.sum(wins))
        sum_l = float(np.sum(losses))
        pf = round(sum_w / sum_l, 2) if sum_l > 0 else (99.0 if sum_w > 0 else 0.0)

        equity = np.cumsum(p_arr)
        peak = np.maximum.accumulate(equity)
        dd = peak - equity
        max_dd = float(np.max(dd)) if len(dd) > 0 else 0.0
        ret_dd = round(net_pips / max_dd, 2) if max_dd > 0 else 0.0
        expectancy = round(net_pips / n, 2)

        rows.append({
            "Asset": asset,
            "Strategy": m_name,
            "Discovery_Trades": n,
            "Win_Rate_Pct": round(wr, 1),
            "Profit_Factor": pf,
            "Net_Pips": round(net_pips, 1),
            "Max_DD_Pips": round(max_dd, 1),
            "Ret_DD_Ratio": ret_dd,
            "Expectancy_pips": expectancy,
            "Discovery_Status": "🌟 VIABLE CANDIDATE" if net_pips > 100.0 and pf >= 1.35 and expectancy >= 1.5 else ("✅ POSITIVE" if net_pips > 0 and pf >= 1.15 else "❌ REJECT")
        })

    df = pd.DataFrame(rows).sort_values(by="Net_Pips", ascending=False)
    csv_out = RESULTS_DIR / "alpha_discovery_v5_scorecard.csv"
    df.to_csv(csv_out, index=False)
    print(f"\n[+] Saved Discovery V5 Scorecard to: {csv_out.resolve()}", flush=True)
    print("\n" + "=" * 80, flush=True)
    print(f"DISCOVERY V5 EXPEDITION COMPLETED IN {time.time() - t_start:.1f}s!", flush=True)
    print("=" * 80, flush=True)
    return df


if __name__ == "__main__":
    run_discovery_v5()
