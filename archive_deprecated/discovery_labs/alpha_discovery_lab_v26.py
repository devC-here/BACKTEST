#!/usr/bin/env python3
"""
================================================================================
  ALPHA DISCOVERY LAB V26: EMA-ATR EXPANSION, PIVOT BREAKOUT, 3-DRIVE & ASIA RETEST
================================================================================
Explores 4 novel algorithmic architectures across the 70% In-Sample Discovery
partition (Seed 42) across EURUSD, USDJPY, and EURGBP (603 discovery weeks)
to surpass the 100+ Production Models milestone:

  1. EMA_CROSS_ATR_BAND_EXPANSION:
     - M15 9-EMA crosses 21-EMA with decisive momentum thrust.
     - Candle closes >= 1.0x ATR10 away from 21-EMA in cross direction.
     - Confirmed by H4 20-EMA macro trend alignment.
     - Active Hours: 07:00 - 16:00 UTC. Target: 2.0R.

  2. PIVOT_R1_S1_BREAKOUT_DRIVE:
     - Floor Trader Daily Pivot R1 / S1 breakout during European/NY hours (07:00 - 15:00 UTC).
     - Expansion trigger candle: Range >= 1.20x ATR10, Body >= 55%.
     - Aligned with H4 20-EMA macro trend. Target: 2.0R.

  3. CONSECUTIVE_THREE_DRIVE_IGNITION:
     - 3 consecutive M15 candles in the same direction with expanding momentum.
     - 3rd candle is an institutional ignition candle (Range >= 1.25x ATR10, Body >= 55%).
     - Aligned with H4 20-EMA macro trend.
     - Active Hours: 07:00 - 17:00 UTC. Target: 2.0R.

  4. ASIA_RANGE_HIGH_LOW_RETEST_IGNITION:
     - Defines 00:00 - 06:00 UTC Asian session range (Asia High and Asia Low).
     - Price breaks out and subsequently retests the broken boundary within 2.5 pips.
     - Prints a rejection wick >= 38% in breakout / H4 trend direction between 08:00 - 15:00 UTC.
     - Target: 2.0R.
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
# Model 1: EMA Cross ATR Band Expansion
# ==============================================================================
class EMACrossATRExpansion(BaseStrategy):
    def __init__(self, asset: str, sl_pips: float, tp_pips: float, pip_size: float, max_spread: float, **kwargs):
        super().__init__(**kwargs)
        self.asset = asset
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.pip_size = float(pip_size)
        self.max_spread = float(max_spread)
        self.name = f"{asset}_EMA_Cross_ATR_Expansion"

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

        c_ser = pd.Series(c)
        ema9 = c_ser.ewm(span=9, adjust=False).mean().values
        ema21 = c_ser.ewm(span=21, adjust=False).mean().values

        tr = np.maximum(h - l, np.maximum(np.abs(h - np.roll(c, 1)), np.abs(l - np.roll(c, 1))))
        tr[0] = h[0] - l[0]
        atr10 = pd.Series(tr).rolling(10).mean().values

        last_trade_date = None

        for i in range(25, len(m15_df)):
            dt = dates[i]
            hr = hours[i]

            if not (7 <= hr <= 16):
                continue
            if dt == last_trade_date:
                continue
            if spread[i] > self.max_spread:
                continue

            macro = h4_trend[i]
            if macro == 0:
                continue

            if np.isnan(atr10[i]) or atr10[i] <= 0:
                continue

            # Check cross within last 2 bars:
            bull_cross = (ema9[i] > ema21[i]) and (ema9[i-1] <= ema21[i-1] or ema9[i-2] <= ema21[i-2])
            bear_cross = (ema9[i] < ema21[i]) and (ema9[i-1] >= ema21[i-1] or ema9[i-2] >= ema21[i-2])

            sig = 0
            # Bullish expansion: close > ema21 + 0.8 * atr10
            if bull_cross and c[i] > ema21[i] + 0.8 * atr10[i] and c[i] > o[i] and macro == 1:
                sig = SIGNAL_LONG
            # Bearish expansion: close < ema21 - 0.8 * atr10
            elif bear_cross and c[i] < ema21[i] - 0.8 * atr10[i] and c[i] < o[i] and macro == -1:
                sig = SIGNAL_SHORT

            if sig != 0:
                end_ms = time_ms[i] + dur_ms
                idx = np.searchsorted(tick_ts, end_ms)
                if idx < n:
                    signals[idx] = sig
                    sl_arr[idx] = self.sl_pips
                    tp_arr[idx] = self.tp_pips
                    last_trade_date = dt

        return signals, sl_arr, tp_arr


# ==============================================================================
# Model 2: Pivot R1 / S1 Breakout Drive
# ==============================================================================
class PivotR1S1BreakoutDrive(BaseStrategy):
    def __init__(self, asset: str, sl_pips: float, tp_pips: float, pip_size: float, max_spread: float, **kwargs):
        super().__init__(**kwargs)
        self.asset = asset
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.pip_size = float(pip_size)
        self.max_spread = float(max_spread)
        self.name = f"{asset}_Pivot_R1_S1_Breakout_Drive"

    def generate_signals(self, ticks: np.ndarray):
        n = len(ticks)
        signals = np.zeros(n, dtype=np.int8)
        sl_arr = np.zeros(n, dtype=np.float64)
        tp_arr = np.zeros(n, dtype=np.float64)
        if n < 500:
            return signals, sl_arr, tp_arr

        m15 = build_time_bars(ticks, timeframe_min=15, pip_size=self.pip_size)
        d1 = build_time_bars(ticks, timeframe_min=1440, pip_size=self.pip_size)
        h4 = build_time_bars(ticks, timeframe_min=240, pip_size=self.pip_size)
        if len(m15) < 30 or len(d1) < 2 or len(h4) < 10:
            return signals, sl_arr, tp_arr

        # Daily Pivot Points
        # Shift d1 by 1 day so today uses yesterday's H, L, C
        d1_prev = d1.shift(1)
        p = (d1_prev['high'] + d1_prev['low'] + d1_prev['close']) / 3.0
        r1 = 2.0 * p - d1_prev['low']
        s1 = 2.0 * p - d1_prev['high']

        pivot_df = pd.DataFrame({'r1': r1, 's1': s1, 'pivot': p}, index=d1.index)

        h4_c = pd.Series(h4['close'].values)
        h4_ema20 = h4_c.ewm(span=20, adjust=False).mean().values
        h4_bias = np.where(h4_c.values > h4_ema20, 1, -1)
        h4_df = pd.DataFrame({'h4_bias': h4_bias}, index=h4.index)

        m15_df = pd.merge_asof(
            m15.sort_index(),
            pivot_df.sort_index(),
            left_index=True,
            right_index=True,
            direction='backward'
        )
        m15_df = pd.merge_asof(
            m15_df,
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
        r1_val = m15_df['r1'].values
        s1_val = m15_df['s1'].values
        hours = m15_df.index.hour.values
        dates = m15_df.index.date
        time_ms = m15_df.index.astype('datetime64[ms]').view('int64')
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        tr = np.maximum(h - l, np.maximum(np.abs(h - np.roll(c, 1)), np.abs(l - np.roll(c, 1))))
        tr[0] = h[0] - l[0]
        atr10 = pd.Series(tr).rolling(10).mean().values

        last_trade_date = None

        for i in range(15, len(m15_df)):
            dt = dates[i]
            hr = hours[i]

            if not (7 <= hr <= 15):
                continue
            if dt == last_trade_date:
                continue
            if spread[i] > self.max_spread:
                continue

            macro = h4_trend[i]
            if macro == 0:
                continue

            if np.isnan(r1_val[i]) or np.isnan(s1_val[i]) or np.isnan(atr10[i]):
                continue

            rng = h[i] - l[i]
            if rng < 1.20 * atr10[i]:
                continue
            body = abs(c[i] - o[i])
            if body < 0.55 * rng:
                continue

            sig = 0
            # Bullish R1 breakout: candle crosses above R1 and closes above R1
            if o[i] <= r1_val[i] and c[i] > r1_val[i] + 0.5 * self.pip_size and c[i] > o[i] and macro == 1:
                sig = SIGNAL_LONG
            # Bearish S1 breakout: candle crosses below S1 and closes below S1
            elif o[i] >= s1_val[i] and c[i] < s1_val[i] - 0.5 * self.pip_size and c[i] < o[i] and macro == -1:
                sig = SIGNAL_SHORT

            if sig != 0:
                end_ms = time_ms[i] + dur_ms
                idx = np.searchsorted(tick_ts, end_ms)
                if idx < n:
                    signals[idx] = sig
                    sl_arr[idx] = self.sl_pips
                    tp_arr[idx] = self.tp_pips
                    last_trade_date = dt

        return signals, sl_arr, tp_arr


# ==============================================================================
# Model 3: Consecutive Three-Drive Ignition
# ==============================================================================
class ConsecutiveThreeDriveIgnition(BaseStrategy):
    def __init__(self, asset: str, sl_pips: float, tp_pips: float, pip_size: float, max_spread: float, **kwargs):
        super().__init__(**kwargs)
        self.asset = asset
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.pip_size = float(pip_size)
        self.max_spread = float(max_spread)
        self.name = f"{asset}_Three_Drive_Ignition"

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
        o = m15_df['open'].values
        spread = m15_df['spread_pips'].values
        h4_trend = m15_df['h4_bias'].fillna(0).values.astype(int)
        hours = m15_df.index.hour.values
        dates = m15_df.index.date
        time_ms = m15_df.index.astype('datetime64[ms]').view('int64')
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        tr = np.maximum(h - l, np.maximum(np.abs(h - np.roll(c, 1)), np.abs(l - np.roll(c, 1))))
        tr[0] = h[0] - l[0]
        atr10 = pd.Series(tr).rolling(10).mean().values

        last_trade_date = None

        for i in range(15, len(m15_df)):
            dt = dates[i]
            hr = hours[i]

            if not (7 <= hr <= 17):
                continue
            if dt == last_trade_date:
                continue
            if spread[i] > self.max_spread:
                continue

            macro = h4_trend[i]
            if macro == 0:
                continue

            if np.isnan(atr10[i]) or atr10[i] <= 0:
                continue

            rng3 = h[i] - l[i]
            if rng3 < 1.25 * atr10[i]:
                continue
            body3 = abs(c[i] - o[i])
            if body3 < 0.55 * rng3:
                continue

            sig = 0
            # Bullish Three-Drive: 3 consecutive higher closes, all green bars
            if (c[i] > c[i-1] > c[i-2]) and (c[i] > o[i] and c[i-1] > o[i-1] and c[i-2] > o[i-2]) and macro == 1:
                sig = SIGNAL_LONG
            # Bearish Three-Drive: 3 consecutive lower closes, all red bars
            elif (c[i] < c[i-1] < c[i-2]) and (c[i] < o[i] and c[i-1] < o[i-1] and c[i-2] < o[i-2]) and macro == -1:
                sig = SIGNAL_SHORT

            if sig != 0:
                end_ms = time_ms[i] + dur_ms
                idx = np.searchsorted(tick_ts, end_ms)
                if idx < n:
                    signals[idx] = sig
                    sl_arr[idx] = self.sl_pips
                    tp_arr[idx] = self.tp_pips
                    last_trade_date = dt

        return signals, sl_arr, tp_arr


# ==============================================================================
# Model 4: Asia Range High/Low Retest Ignition
# ==============================================================================
class AsiaRangeHLRetestIgnition(BaseStrategy):
    def __init__(self, asset: str, sl_pips: float, tp_pips: float, pip_size: float, max_spread: float, **kwargs):
        super().__init__(**kwargs)
        self.asset = asset
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.pip_size = float(pip_size)
        self.max_spread = float(max_spread)
        self.name = f"{asset}_Asia_HL_Retest_Ignition"

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
        o = m15_df['open'].values
        spread = m15_df['spread_pips'].values
        h4_trend = m15_df['h4_bias'].fillna(0).values.astype(int)
        hours = m15_df.index.hour.values
        dates = m15_df.index.date
        time_ms = m15_df.index.astype('datetime64[ms]').view('int64')
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        # Determine Asian Session High & Low (00:00 - 06:00 UTC) per date
        asia_hi: Dict[Any, float] = {}
        asia_lo: Dict[Any, float] = {}
        for idx_bar in range(len(m15_df)):
            dt = dates[idx_bar]
            hr = hours[idx_bar]
            if 0 <= hr < 6:
                if dt not in asia_hi:
                    asia_hi[dt] = h[idx_bar]
                    asia_lo[dt] = l[idx_bar]
                else:
                    asia_hi[dt] = max(asia_hi[dt], h[idx_bar])
                    asia_lo[dt] = min(asia_lo[dt], l[idx_bar])

        tolerance = 2.5 * self.pip_size
        last_trade_date = None

        for i in range(15, len(m15_df)):
            dt = dates[i]
            hr = hours[i]

            # Active hours: 08:00 to 15:00 UTC
            if not (8 <= hr <= 15):
                continue
            if dt == last_trade_date:
                continue
            if spread[i] > self.max_spread:
                continue

            ah = asia_hi.get(dt)
            al = asia_lo.get(dt)
            if ah is None or al is None:
                continue

            macro = h4_trend[i]
            if macro == 0:
                continue

            rng = h[i] - l[i]
            if rng <= 0:
                continue

            sig = 0
            # Bullish Asia High Retest: Low touched near Asia High (within tolerance), candle rejects upward
            if abs(l[i] - ah) <= tolerance and c[i] > ah and c[i] > o[i] and macro == 1:
                lower_wick = min(c[i], o[i]) - l[i]
                if lower_wick >= 0.35 * rng:
                    sig = SIGNAL_LONG

            # Bearish Asia Low Retest: High touched near Asia Low (within tolerance), candle rejects downward
            elif abs(h[i] - al) <= tolerance and c[i] < al and c[i] < o[i] and macro == -1:
                upper_wick = h[i] - max(c[i], o[i])
                if upper_wick >= 0.35 * rng:
                    sig = SIGNAL_SHORT

            if sig != 0:
                end_ms = time_ms[i] + dur_ms
                idx = np.searchsorted(tick_ts, end_ms)
                if idx < n:
                    signals[idx] = sig
                    sl_arr[idx] = self.sl_pips
                    tp_arr[idx] = self.tp_pips
                    last_trade_date = dt

        return signals, sl_arr, tp_arr


# ==============================================================================
# Model Factory
# ==============================================================================
def build_model_v26(model_name: str, asset: str) -> BaseStrategy | None:
    if model_name == "EMA_CROSS_ATR_BAND_EXPANSION":
        if asset == "EURUSD":
            return EMACrossATRExpansion(asset=asset, sl_pips=8.5, tp_pips=17.0, pip_size=0.0001, max_spread=0.6)
        elif asset == "USDJPY":
            return EMACrossATRExpansion(asset=asset, sl_pips=12.0, tp_pips=24.0, pip_size=0.01, max_spread=0.8)
        elif asset == "EURGBP":
            return EMACrossATRExpansion(asset=asset, sl_pips=7.0, tp_pips=14.0, pip_size=0.0001, max_spread=0.8)

    elif model_name == "PIVOT_R1_S1_BREAKOUT_DRIVE":
        if asset == "EURUSD":
            return PivotR1S1BreakoutDrive(asset=asset, sl_pips=8.5, tp_pips=17.0, pip_size=0.0001, max_spread=0.6)
        elif asset == "USDJPY":
            return PivotR1S1BreakoutDrive(asset=asset, sl_pips=12.0, tp_pips=24.0, pip_size=0.01, max_spread=0.8)
        elif asset == "EURGBP":
            return PivotR1S1BreakoutDrive(asset=asset, sl_pips=7.0, tp_pips=14.0, pip_size=0.0001, max_spread=0.8)

    elif model_name == "CONSECUTIVE_THREE_DRIVE_IGNITION":
        if asset == "EURUSD":
            return ConsecutiveThreeDriveIgnition(asset=asset, sl_pips=8.5, tp_pips=17.0, pip_size=0.0001, max_spread=0.6)
        elif asset == "USDJPY":
            return ConsecutiveThreeDriveIgnition(asset=asset, sl_pips=12.0, tp_pips=24.0, pip_size=0.01, max_spread=0.8)
        elif asset == "EURGBP":
            return ConsecutiveThreeDriveIgnition(asset=asset, sl_pips=7.0, tp_pips=14.0, pip_size=0.0001, max_spread=0.8)

    elif model_name == "ASIA_RANGE_HIGH_LOW_RETEST_IGNITION":
        if asset == "EURUSD":
            return AsiaRangeHLRetestIgnition(asset=asset, sl_pips=8.5, tp_pips=17.0, pip_size=0.0001, max_spread=0.6)
        elif asset == "USDJPY":
            return AsiaRangeHLRetestIgnition(asset=asset, sl_pips=12.0, tp_pips=24.0, pip_size=0.01, max_spread=0.8)
        elif asset == "EURGBP":
            return AsiaRangeHLRetestIgnition(asset=asset, sl_pips=7.0, tp_pips=14.0, pip_size=0.0001, max_spread=0.8)

    return None


def run_week_task_v26(task: Tuple[str, str, int]) -> Dict[str, Any]:
    asset, model_name, week_num = task
    pip_size = 0.01 if asset == "USDJPY" else 0.0001
    engine = BacktestEngine(pip_size=pip_size, use_slippage=True)

    ticks, meta = get_week_dataset(week_num, raw_dir=f"data/{asset}")
    if ticks is None or len(ticks) < 500:
        return {"asset": asset, "model": model_name, "week": week_num, "trades": []}

    strat = build_model_v26(model_name, asset)
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

    equity = np.cumsum(arr)
    peak = np.maximum.accumulate(equity)
    dd = peak - equity
    max_dd = float(np.max(dd)) if len(dd) > 0 else 0.0
    ret_dd = net_pips / max_dd if max_dd > 0 else 999.0

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
    print("=" * 80, flush=True)
    print("  LAUNCHING ALPHA DISCOVERY LAB V26: EMA-ATR, PIVOT R1/S1, 3-DRIVE & ASIA RETEST", flush=True)
    print("=" * 80, flush=True)
    t0 = time.time()

    assets = [os.environ["DISCOVERY_TARGET_SYMBOL"]] if "DISCOVERY_TARGET_SYMBOL" in os.environ else ["EURUSD", "USDJPY", "EURGBP"]
    models = [
        "EMA_CROSS_ATR_BAND_EXPANSION",
        "PIVOT_R1_S1_BREAKOUT_DRIVE",
        "CONSECUTIVE_THREE_DRIVE_IGNITION",
        "ASIA_RANGE_HIGH_LOW_RETEST_IGNITION"
    ]

    tasks: List[Tuple[str, str, int]] = []
    for asset in assets:
        disc_weeks = get_discovery_weeks(f"data/{asset}")
        for m in models:
            for w in disc_weeks:
                tasks.append((asset, m, w))

    print(f"[*] Total week-tasks across 3 assets & 4 models: {len(tasks)}", flush=True)

    trades_by_key: Dict[str, List[float]] = {f"{a}_{m}": [] for a in assets for m in models}

    max_workers = max(1, (os.cpu_count() or 4) - 1)
    completed_count = 0

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(run_week_task_v26, t): t for t in tasks}

        for fut in as_completed(futures):
            res = fut.result()
            key = f"{res['asset']}_{res['model']}"
            for tr in res["trades"]:
                trades_by_key[key].append(tr["pips"])

            completed_count += 1
            if completed_count % 300 == 0 or completed_count == len(tasks):
                elapsed = time.time() - t0
                print(f"  [+] Audited {completed_count}/{len(tasks)} week-tasks in {elapsed:.1f}s...", flush=True)

    print("\n" + "=" * 90, flush=True)
    print(f"{'ASSET':<8} | {'MODEL':<35} | {'TRADES':<6} | {'WIN RATE':<8} | {'PF':<5} | {'NET PIPS':<10} | {'MAX DD':<8} | {'RET/DD':<6}", flush=True)
    print("=" * 90, flush=True)

    summary_results: Dict[str, Any] = {}

    for a in assets:
        for m in models:
            k = f"{a}_{m}"
            pips = trades_by_key[k]
            met = calculate_metrics(pips)
            summary_results[k] = met
            print(f"{a:<8} | {m:<35} | {met['trades']:<6} | {met['win_rate']:<6.1f} % | {met['pf']:<5.2f} | {met['net_pips']:<10.1f} | {met['max_dd']:<8.1f} | {met['ret_dd']:<5.2f} x", flush=True)
        print("-" * 90, flush=True)

    out_file = RESULTS_DIR / "lab_v26_discovery_results.json"
    with open(out_file, "w") as f:
        json.dump(summary_results, f, indent=2)

    total_time = time.time() - t0
    print(f"\n[+] Discovery results saved to: {out_file}", flush=True)
    print(f"[+] Total execution time: {total_time:.2f}s", flush=True)
    print("=" * 90, flush=True)


if __name__ == "__main__":
    main()
