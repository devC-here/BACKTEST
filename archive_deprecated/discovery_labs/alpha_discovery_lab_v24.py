#!/usr/bin/env python3
"""
================================================================================
  ALPHA DISCOVERY LAB V24: SAR ACCELERATION, LONDON FIX REVERSION & CHAIKIN SURGE
================================================================================
Explores 4 novel algorithmic architectures across the 70% In-Sample Discovery
partition (Seed 42) across EURUSD, USDJPY, and EURGBP (603 discovery weeks):

  1. PARABOLIC_SAR_ACCELERATION_FLIP:
     - M15 Parabolic SAR (0.02, 0.20) flips dot from above to below (bullish) or below to above (bearish).
     - Flip candle range >= 1.2x ATR(10), body >= 50% in H4 20-EMA trend direction.
     - Active Session: 07:00 - 16:00 UTC. Target: 2.0R.

  2. LONDON_PRE_FIX_MEAN_REVERSION:
     - Active Hours: 14:00 - 16:00 UTC (pre-London 16:00 Fix rebalancing window).
     - Price stretched >= 1.6x ATR(10) from M15 20-EMA.
     - Rejection wick >= 45% pointing away from 20-EMA (exhaustion fade back to mean).
     - Target: 1.67R.

  3. CHAIKIN_VOLATILITY_EXPANSION_SURGE:
     - Chaikin Volatility: 10-bar EMA of (High - Low) percentage change over 10 bars.
     - Volatility surge >= +30% indicating institutional liquidity ignition.
     - Breakout past 4-bar extreme in H4 20-EMA trend direction.
     - Active Session: 07:00 - 16:00 UTC. Target: 2.0R.

  4. LONDON_30M_ORB_CONTINUATION:
     - London Cash Open 30-Minute Range: 07:00 - 07:30 UTC High and Low.
     - Breakout past 30m ORB extreme by >= 2.0 pips between 07:30 - 11:30 UTC in H4 trend direction.
     - Active Session: 07:30 - 11:30 UTC. Target: 2.0R.
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
# Model 1: Parabolic SAR Acceleration Flip (DISC79)
# ==============================================================================
class ParabolicSARAccelerationFlip(BaseStrategy):
    def __init__(self, asset: str, sl_pips: float, tp_pips: float, pip_size: float, max_spread: float, **kwargs):
        super().__init__(**kwargs)
        self.asset = asset
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.pip_size = float(pip_size)
        self.max_spread = float(max_spread)
        self.name = f"{asset}_Parabolic_SAR_Flip"

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

        # Vectorized / Iterative Parabolic SAR
        length = len(m15_df)
        sar = np.zeros(length, dtype=np.float64)
        ep = np.zeros(length, dtype=np.float64)
        af = np.zeros(length, dtype=np.float64)
        trend = np.zeros(length, dtype=np.int8)  # 1 bull, -1 bear

        # Initialize SAR
        trend[0] = 1 if c[0] >= o[0] else -1
        sar[0] = l[0] if trend[0] == 1 else h[0]
        ep[0] = h[0] if trend[0] == 1 else l[0]
        af[0] = 0.02

        for i in range(1, length):
            prev_sar = sar[i-1]
            prev_ep = ep[i-1]
            prev_af = af[i-1]
            prev_trend = trend[i-1]

            cur_sar = prev_sar + prev_af * (prev_ep - prev_sar)

            if prev_trend == 1:
                cur_sar = min(cur_sar, l[i-1], l[max(0, i-2)])
                if l[i] < cur_sar:
                    cur_trend = -1
                    cur_sar = prev_ep
                    cur_ep = l[i]
                    cur_af = 0.02
                else:
                    cur_trend = 1
                    cur_ep = max(prev_ep, h[i])
                    cur_af = min(0.20, prev_af + 0.02) if cur_ep > prev_ep else prev_af
            else:
                cur_sar = max(cur_sar, h[i-1], h[max(0, i-2)])
                if h[i] > cur_sar:
                    cur_trend = 1
                    cur_sar = prev_ep
                    cur_ep = h[i]
                    cur_af = 0.02
                else:
                    cur_trend = -1
                    cur_ep = min(prev_ep, l[i])
                    cur_af = min(0.20, prev_af + 0.02) if cur_ep < prev_ep else prev_af

            sar[i] = cur_sar
            ep[i] = cur_ep
            af[i] = cur_af
            trend[i] = cur_trend

        # ATR 10
        tr = np.maximum(h - l, np.maximum(np.abs(h - np.roll(c, 1)), np.abs(l - np.roll(c, 1))))
        tr[0] = h[0] - l[0]
        atr10 = pd.Series(tr).rolling(10).mean().values

        last_trade_date = None

        for i in range(15, length):
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

            # SAR Flip condition: trend changed on this exact bar
            is_flip = (trend[i] != trend[i-1])
            if not is_flip:
                continue

            bar_rng = h[i] - l[i]
            cur_atr = atr10[i]
            if np.isnan(cur_atr) or bar_rng < 1.15 * cur_atr:
                continue

            body = abs(c[i] - o[i])
            if body / bar_rng < 0.50:
                continue

            sig = 0
            # Bullish SAR flip agreeing with H4 uptrend
            if macro == 1 and trend[i] == 1 and c[i] > o[i]:
                sig = SIGNAL_LONG
            # Bearish SAR flip agreeing with H4 downtrend
            elif macro == -1 and trend[i] == -1 and c[i] < o[i]:
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
# Model 2: London Pre-Fix Mean Reversion (DISC80)
# ==============================================================================
class LondonPreFixMeanReversion(BaseStrategy):
    def __init__(self, asset: str, sl_pips: float, tp_pips: float, pip_size: float, max_spread: float, **kwargs):
        super().__init__(**kwargs)
        self.asset = asset
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.pip_size = float(pip_size)
        self.max_spread = float(max_spread)
        self.name = f"{asset}_London_Pre_Fix_Reversion"

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
        o = m15['open'].values
        spread = m15['spread_pips'].values
        hours = m15.index.hour.values
        dates = m15.index.date
        time_ms = m15.index.astype('datetime64[ms]').view('int64')
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        c_s = pd.Series(c)
        m15_ema20 = c_s.ewm(span=20, adjust=False).mean().values

        tr = np.maximum(h - l, np.maximum(np.abs(h - np.roll(c, 1)), np.abs(l - np.roll(c, 1))))
        tr[0] = h[0] - l[0]
        atr10 = pd.Series(tr).rolling(10).mean().values

        last_trade_date = None

        for i in range(25, len(m15)):
            dt = dates[i]
            hr = hours[i]

            # Restricted to pre-fix rebalancing window (14:00 - 15:45 UTC)
            if not (14 <= hr <= 15):
                continue
            if dt == last_trade_date:
                continue
            if spread[i] > self.max_spread:
                continue

            bar_rng = h[i] - l[i]
            cur_atr = atr10[i]
            if np.isnan(cur_atr) or bar_rng < 1.0 * cur_atr:
                continue

            dist_ema = c[i] - m15_ema20[i]
            stretch = abs(dist_ema) / cur_atr

            if stretch < 1.50:
                continue

            sig = 0
            # Downward stretched overextension with bullish rejection wick back toward 20-EMA
            if dist_ema < 0 and c[i] > o[i]:
                lower_wick = (o[i] - l[i]) / bar_rng
                if lower_wick >= 0.45:
                    sig = SIGNAL_LONG

            # Upward stretched overextension with bearish rejection wick back toward 20-EMA
            elif dist_ema > 0 and c[i] < o[i]:
                upper_wick = (h[i] - o[i]) / bar_rng
                if upper_wick >= 0.45:
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
# Model 3: Chaikin Volatility Expansion Surge (DISC81)
# ==============================================================================
class ChaikinVolatilityExpansionSurge(BaseStrategy):
    def __init__(self, asset: str, sl_pips: float, tp_pips: float, pip_size: float, max_spread: float, **kwargs):
        super().__init__(**kwargs)
        self.asset = asset
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.pip_size = float(pip_size)
        self.max_spread = float(max_spread)
        self.name = f"{asset}_Chaikin_Volatility_Surge"

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

        # Chaikin Volatility: 10-period EMA of (High - Low)
        hl_range = pd.Series(h - l)
        hl_ema = hl_range.ewm(span=10, adjust=False).mean()
        chaikin_vol = (hl_ema - hl_ema.shift(10)) / hl_ema.shift(10) * 100.0
        chaikin = chaikin_vol.values

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

            cv = chaikin[i]
            if np.isnan(cv) or cv < 30.0:
                continue

            # Breakout past 4-bar extreme
            prev_high4 = np.max(h[i-4:i])
            prev_low4 = np.min(l[i-4:i])

            bar_rng = h[i] - l[i]
            if bar_rng < 2.5 * self.pip_size:
                continue

            sig = 0
            if macro == 1 and c[i] > prev_high4 and c[i] > o[i]:
                body = c[i] - o[i]
                if body / bar_rng >= 0.50:
                    sig = SIGNAL_LONG
            elif macro == -1 and c[i] < prev_low4 and c[i] < o[i]:
                body = o[i] - c[i]
                if body / bar_rng >= 0.50:
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
# Model 4: London 30M Opening Range Breakout Continuation (DISC82)
# ==============================================================================
class London30MORBContinuation(BaseStrategy):
    def __init__(self, asset: str, sl_pips: float, tp_pips: float, pip_size: float, max_spread: float, **kwargs):
        super().__init__(**kwargs)
        self.asset = asset
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.pip_size = float(pip_size)
        self.max_spread = float(max_spread)
        self.name = f"{asset}_London_30M_ORB"

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

        # Determine daily London 30M ORB High and Low (07:00 - 07:30 UTC = two 15m bars: 07:00 and 07:15)
        orb_high: Dict[Any, float] = {}
        orb_low: Dict[Any, float] = {}

        for i in range(len(m15_df)):
            dt = dates[i]
            hr = hours[i]
            mn = minutes[i]
            if hr == 7 and (mn == 0 or mn == 15):
                if dt not in orb_high:
                    orb_high[dt] = h[i]
                    orb_low[dt] = l[i]
                else:
                    orb_high[dt] = max(orb_high[dt], h[i])
                    orb_low[dt] = min(orb_low[dt], l[i])

        last_trade_date = None

        for i in range(10, len(m15_df)):
            dt = dates[i]
            hr = hours[i]
            mn = minutes[i]

            # Trading window: 07:30 to 11:30 UTC
            time_val = hr * 60 + mn
            if not (7 * 60 + 30 <= time_val <= 11 * 60 + 30):
                continue
            if dt == last_trade_date:
                continue
            if spread[i] > self.max_spread:
                continue

            if dt not in orb_high or dt not in orb_low:
                continue

            oh = orb_high[dt]
            ol = orb_low[dt]
            orb_rng = oh - ol
            if orb_rng < 4.0 * self.pip_size:
                continue

            macro = h4_trend[i]
            if macro == 0:
                continue

            sig = 0
            # Bullish ORB Breakout
            if macro == 1 and c[i] > oh + 1.5 * self.pip_size and c[i] > o[i]:
                sig = SIGNAL_LONG
            # Bearish ORB Breakout
            elif macro == -1 and c[i] < ol - 1.5 * self.pip_size and c[i] < o[i]:
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
# Model Factory & Multiprocessing Orchestrator
# ==============================================================================
def build_model_v24(model_name: str, asset: str):
    if model_name == "PARABOLIC_SAR_ACCELERATION_FLIP":
        if asset == "EURUSD":
            return ParabolicSARAccelerationFlip(asset=asset, sl_pips=9.0, tp_pips=18.0, pip_size=0.0001, max_spread=0.6)
        elif asset == "USDJPY":
            return ParabolicSARAccelerationFlip(asset=asset, sl_pips=12.0, tp_pips=24.0, pip_size=0.01, max_spread=0.8)
        elif asset == "EURGBP":
            return ParabolicSARAccelerationFlip(asset=asset, sl_pips=8.0, tp_pips=16.0, pip_size=0.0001, max_spread=0.8)

    elif model_name == "LONDON_PRE_FIX_MEAN_REVERSION":
        if asset == "EURUSD":
            return LondonPreFixMeanReversion(asset=asset, sl_pips=9.0, tp_pips=15.0, pip_size=0.0001, max_spread=0.6)
        elif asset == "USDJPY":
            return LondonPreFixMeanReversion(asset=asset, sl_pips=12.0, tp_pips=20.0, pip_size=0.01, max_spread=0.8)
        elif asset == "EURGBP":
            return LondonPreFixMeanReversion(asset=asset, sl_pips=8.0, tp_pips=13.0, pip_size=0.0001, max_spread=0.8)

    elif model_name == "CHAIKIN_VOLATILITY_EXPANSION_SURGE":
        if asset == "EURUSD":
            return ChaikinVolatilityExpansionSurge(asset=asset, sl_pips=9.0, tp_pips=18.0, pip_size=0.0001, max_spread=0.6)
        elif asset == "USDJPY":
            return ChaikinVolatilityExpansionSurge(asset=asset, sl_pips=12.0, tp_pips=24.0, pip_size=0.01, max_spread=0.8)
        elif asset == "EURGBP":
            return ChaikinVolatilityExpansionSurge(asset=asset, sl_pips=8.0, tp_pips=16.0, pip_size=0.0001, max_spread=0.8)

    elif model_name == "LONDON_30M_ORB_CONTINUATION":
        if asset == "EURUSD":
            return London30MORBContinuation(asset=asset, sl_pips=9.0, tp_pips=18.0, pip_size=0.0001, max_spread=0.6)
        elif asset == "USDJPY":
            return London30MORBContinuation(asset=asset, sl_pips=12.0, tp_pips=24.0, pip_size=0.01, max_spread=0.8)
        elif asset == "EURGBP":
            return London30MORBContinuation(asset=asset, sl_pips=8.0, tp_pips=16.0, pip_size=0.0001, max_spread=0.8)

    return None


def run_week_task_v24(task: Tuple[str, str, int]) -> Dict[str, Any]:
    asset, model_name, week_num = task
    pip_size = 0.01 if asset == "USDJPY" else 0.0001
    engine = BacktestEngine(pip_size=pip_size, use_slippage=True)

    ticks, meta = get_week_dataset(week_num, raw_dir=f"data/{asset}")
    if ticks is None or len(ticks) < 500:
        return {"asset": asset, "model": model_name, "week": week_num, "trades": []}

    strat = build_model_v24(model_name, asset)
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
    print("  LAUNCHING ALPHA DISCOVERY LAB V24: SAR FLIP, LONDON PRE-FIX & CHAIKIN SURGE", flush=True)
    print("=" * 80, flush=True)
    t0 = time.time()

    assets = [os.environ["DISCOVERY_TARGET_SYMBOL"]] if "DISCOVERY_TARGET_SYMBOL" in os.environ else ["EURUSD", "USDJPY", "EURGBP"]
    models = [
        "PARABOLIC_SAR_ACCELERATION_FLIP",
        "LONDON_PRE_FIX_MEAN_REVERSION",
        "CHAIKIN_VOLATILITY_EXPANSION_SURGE",
        "LONDON_30M_ORB_CONTINUATION"
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
        futures = {executor.submit(run_week_task_v24, t): t for t in tasks}

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

    out_file = RESULTS_DIR / "lab_v24_discovery_results.json"
    with open(out_file, "w") as f:
        json.dump(summary_results, f, indent=2)

    total_time = time.time() - t0
    print(f"\n[+] Discovery results saved to: {out_file}", flush=True)
    print(f"[+] Total execution time: {total_time:.2f}s", flush=True)
    print("=" * 90, flush=True)


if __name__ == "__main__":
    main()
