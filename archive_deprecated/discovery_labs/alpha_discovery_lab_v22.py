#!/usr/bin/env python3
"""
================================================================================
  ALPHA DISCOVERY LAB V22: WEEKLY PIVOTS, VWAP 1-SD DISCOUNT & KELTNER BOUNCE
================================================================================
Explores 4 novel algorithmic architectures across the 70% In-Sample Discovery
partition (Seed 42) across EURUSD, USDJPY, and EURGBP (603 discovery weeks):

  1. WEEKLY_PIVOT_TREND_PULLBACK:
     - Classical Weekly Pivot (H+L+C)/3 calculated from previous calendar week.
     - In H4 20-EMA trend direction, price pulls back to test Weekly Pivot P or S1/R1.
     - Rejection wick >= 40% on M15 closing in trend direction.
     - Target: 2.5R.

  2. VWAP_1SD_TREND_REBOUND:
     - Intraday Anchored VWAP (00:00 UTC) with 1.0-SD bands.
     - In H4 uptrend: price dips to touch lower 1.0-SD band (institutional discount) and prints bullish wick >= 40%.
     - In H4 downtrend: price rallies to touch upper 1.0-SD band (institutional premium) and prints bearish wick >= 40%.
     - Target: 2.0R.

  3. KELTNER_MIDLINE_EXPANSION:
     - M15 Keltner Channel (20-EMA +/- 1.5 ATR).
     - In H4 trend direction, price pulls back to touch the 20-EMA midline.
     - Prints an expansion bar closing in trend direction with range >= 1.25x 10-bar ATR and body >= 55% of range.
     - Target: 2.0R.

  4. THREE_BAR_INSIDE_CONFIRMATION:
     - M15 Inside Bar (Bar 2 high < Bar 1 high and Bar 2 low > Bar 1 low).
     - Bar 3 closes decisively beyond Bar 1's high (bullish) or low (bearish) in H4 20-EMA trend direction.
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
# Model 1: Weekly Pivot Trend Pullback (DISC75)
# ==============================================================================
class WeeklyPivotTrendPullback(BaseStrategy):
    def __init__(self, asset: str, sl_pips: float, tp_pips: float, pip_size: float, max_spread: float, **kwargs):
        super().__init__(**kwargs)
        self.asset = asset
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.pip_size = float(pip_size)
        self.max_spread = float(max_spread)
        self.name = f"{asset}_Weekly_Pivot_Pullback"

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

        # Weekly Pivot approximate from first 24h of week if no external prior week
        # Use initial Monday 00:00 - 12:00 range as early anchor
        wh = np.max(h[:min(48, len(h))])
        wl = np.min(l[:min(48, len(l))])
        wc = c[min(48, len(c)) - 1]
        wp = (wh + wl + wc) / 3.0
        ws1 = 2.0 * wp - wh
        wr1 = 2.0 * wp - wl

        last_trade_date = None

        for i in range(48, len(m15_df)):
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

            bar_rng = h[i] - l[i]
            if bar_rng < 2.5 * self.pip_size:
                continue

            sig = 0
            tol_up = 2.0 * self.pip_size
            tol_dn = 4.0 * self.pip_size

            # Bullish trend: test weekly pivot P or S1
            if macro == 1:
                touched = (l[i] <= wp + tol_up and l[i] >= wp - tol_dn) or \
                          (l[i] <= ws1 + tol_up and l[i] >= ws1 - tol_dn)
                if touched and c[i] > o[i]:
                    lower_wick = (o[i] - l[i]) / bar_rng
                    if lower_wick >= 0.40:
                        sig = SIGNAL_LONG

            # Bearish trend: test weekly pivot P or R1
            elif macro == -1:
                touched = (h[i] >= wp - tol_up and h[i] <= wp + tol_dn) or \
                          (h[i] >= wr1 - tol_up and h[i] <= wr1 + tol_dn)
                if touched and c[i] < o[i]:
                    upper_wick = (h[i] - o[i]) / bar_rng
                    if upper_wick >= 0.40:
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
# Model 2: VWAP 1-SD Discount Trend Rebound (DISC76)
# ==============================================================================
class VWAP1SDTrendRebound(BaseStrategy):
    def __init__(self, asset: str, sl_pips: float, tp_pips: float, pip_size: float, max_spread: float, **kwargs):
        super().__init__(**kwargs)
        self.asset = asset
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.pip_size = float(pip_size)
        self.max_spread = float(max_spread)
        self.name = f"{asset}_VWAP_1SD_Rebound"

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
        vol = m15_df['tick_count'].values if 'tick_count' in m15_df.columns else np.ones(len(m15_df))
        h4_trend = m15_df['h4_bias'].fillna(0).values.astype(int)
        hours = m15_df.index.hour.values
        dates = m15_df.index.date
        time_ms = m15_df.index.astype('datetime64[ms]').view('int64')
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        typ_price = (h + l + c) / 3.0
        pv = typ_price * vol

        # Calculate daily anchored VWAP and running 1-SD
        vwap = np.zeros(len(m15))
        vwap_std = np.zeros(len(m15))

        cur_date = None
        cum_pv = 0.0
        cum_vol = 0.0
        day_prices: List[float] = []

        for i in range(len(m15)):
            d = dates[i]
            if d != cur_date:
                cur_date = d
                cum_pv = 0.0
                cum_vol = 0.0
                day_prices = []

            cum_pv += pv[i]
            cum_vol += vol[i]
            day_prices.append(typ_price[i])

            v_val = cum_pv / max(cum_vol, 1e-6)
            vwap[i] = v_val
            vwap_std[i] = np.std(day_prices) if len(day_prices) > 2 else (h[i] - l[i])

        last_trade_date = None

        for i in range(25, len(m15_df)):
            dt = dates[i]
            hr = hours[i]

            if not (8 <= hr <= 16):
                continue
            if dt == last_trade_date:
                continue
            if spread[i] > self.max_spread:
                continue

            macro = h4_trend[i]
            if macro == 0:
                continue

            cur_vwap = vwap[i]
            cur_std = vwap_std[i]
            if cur_std < 1.5 * self.pip_size:
                continue

            upper_1sd = cur_vwap + 1.0 * cur_std
            lower_1sd = cur_vwap - 1.0 * cur_std

            bar_rng = h[i] - l[i]
            if bar_rng < 2.5 * self.pip_size:
                continue

            sig = 0
            # Bullish trend: dip to touch lower 1-SD band, reject upwards
            if macro == 1 and l[i] <= lower_1sd and c[i] > lower_1sd:
                wick = (min(o[i], c[i]) - l[i]) / bar_rng
                if wick >= 0.40 and c[i] > o[i]:
                    sig = SIGNAL_LONG

            # Bearish trend: rally to touch upper 1-SD band, reject downwards
            elif macro == -1 and h[i] >= upper_1sd and c[i] < upper_1sd:
                wick = (h[i] - max(o[i], c[i])) / bar_rng
                if wick >= 0.40 and c[i] < o[i]:
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
# Model 3: Keltner Midline Trend Expansion (DISC77)
# ==============================================================================
class KeltnerMidlineExpansion(BaseStrategy):
    def __init__(self, asset: str, sl_pips: float, tp_pips: float, pip_size: float, max_spread: float, **kwargs):
        super().__init__(**kwargs)
        self.asset = asset
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.pip_size = float(pip_size)
        self.max_spread = float(max_spread)
        self.name = f"{asset}_Keltner_Midline_Expansion"

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

        c_s = pd.Series(c)
        m15_ema20 = c_s.ewm(span=20, adjust=False).mean().values
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

            bar_rng = h[i] - l[i]
            cur_atr = atr10[i]
            if np.isnan(cur_atr) or bar_rng < 1.20 * cur_atr:
                continue

            body = abs(c[i] - o[i])
            if body / bar_rng < 0.55:
                continue

            sig = 0
            tol = 1.5 * self.pip_size

            # Bullish expansion candle after touching midline
            if macro == 1 and l[i] <= m15_ema20[i] + tol and c[i] > o[i] and c[i] > m15_ema20[i]:
                sig = SIGNAL_LONG

            # Bearish expansion candle after touching midline
            elif macro == -1 and h[i] >= m15_ema20[i] - tol and c[i] < o[i] and c[i] < m15_ema20[i]:
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
# Model 4: Three-Bar Inside Confirmation (DISC78)
# ==============================================================================
class ThreeBarInsideConfirmation(BaseStrategy):
    def __init__(self, asset: str, sl_pips: float, tp_pips: float, pip_size: float, max_spread: float, **kwargs):
        super().__init__(**kwargs)
        self.asset = asset
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.pip_size = float(pip_size)
        self.max_spread = float(max_spread)
        self.name = f"{asset}_Three_Bar_Inside"

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

            # Bar i-1 is inside bar relative to Bar i-2
            is_inside = (h[i-1] <= h[i-2]) and (l[i-1] >= l[i-2])
            if not is_inside:
                continue

            sig = 0
            # Bullish Three-Inside-Up
            if macro == 1 and c[i] > h[i-2] and c[i] > o[i]:
                sig = SIGNAL_LONG
            # Bearish Three-Inside-Down
            elif macro == -1 and c[i] < l[i-2] and c[i] < o[i]:
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
def build_model_v22(model_name: str, asset: str):
    if model_name == "WEEKLY_PIVOT_TREND_PULLBACK":
        if asset == "EURUSD":
            return WeeklyPivotTrendPullback(asset=asset, sl_pips=9.0, tp_pips=22.5, pip_size=0.0001, max_spread=0.6)
        elif asset == "USDJPY":
            return WeeklyPivotTrendPullback(asset=asset, sl_pips=12.0, tp_pips=30.0, pip_size=0.01, max_spread=0.8)
        elif asset == "EURGBP":
            return WeeklyPivotTrendPullback(asset=asset, sl_pips=8.0, tp_pips=20.0, pip_size=0.0001, max_spread=0.8)

    elif model_name == "VWAP_1SD_TREND_REBOUND":
        if asset == "EURUSD":
            return VWAP1SDTrendRebound(asset=asset, sl_pips=9.0, tp_pips=18.0, pip_size=0.0001, max_spread=0.6)
        elif asset == "USDJPY":
            return VWAP1SDTrendRebound(asset=asset, sl_pips=12.0, tp_pips=24.0, pip_size=0.01, max_spread=0.8)
        elif asset == "EURGBP":
            return VWAP1SDTrendRebound(asset=asset, sl_pips=8.0, tp_pips=16.0, pip_size=0.0001, max_spread=0.8)

    elif model_name == "KELTNER_MIDLINE_EXPANSION":
        if asset == "EURUSD":
            return KeltnerMidlineExpansion(asset=asset, sl_pips=9.0, tp_pips=18.0, pip_size=0.0001, max_spread=0.6)
        elif asset == "USDJPY":
            return KeltnerMidlineExpansion(asset=asset, sl_pips=12.0, tp_pips=24.0, pip_size=0.01, max_spread=0.8)
        elif asset == "EURGBP":
            return KeltnerMidlineExpansion(asset=asset, sl_pips=8.0, tp_pips=16.0, pip_size=0.0001, max_spread=0.8)

    elif model_name == "THREE_BAR_INSIDE_CONFIRMATION":
        if asset == "EURUSD":
            return ThreeBarInsideConfirmation(asset=asset, sl_pips=9.0, tp_pips=18.0, pip_size=0.0001, max_spread=0.6)
        elif asset == "USDJPY":
            return ThreeBarInsideConfirmation(asset=asset, sl_pips=12.0, tp_pips=24.0, pip_size=0.01, max_spread=0.8)
        elif asset == "EURGBP":
            return ThreeBarInsideConfirmation(asset=asset, sl_pips=8.0, tp_pips=16.0, pip_size=0.0001, max_spread=0.8)

    return None


def run_week_task_v22(task: Tuple[str, str, int]) -> Dict[str, Any]:
    asset, model_name, week_num = task
    pip_size = 0.01 if asset == "USDJPY" else 0.0001
    engine = BacktestEngine(pip_size=pip_size, use_slippage=True)

    ticks, meta = get_week_dataset(week_num, raw_dir=f"data/{asset}")
    if ticks is None or len(ticks) < 500:
        return {"asset": asset, "model": model_name, "week": week_num, "trades": []}

    strat = build_model_v22(model_name, asset)
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
    print("  LAUNCHING ALPHA DISCOVERY LAB V22: WEEKLY PIVOTS, VWAP 1-SD & KELTNER", flush=True)
    print("=" * 80, flush=True)
    t0 = time.time()

    assets = [os.environ["DISCOVERY_TARGET_SYMBOL"]] if "DISCOVERY_TARGET_SYMBOL" in os.environ else ["EURUSD", "USDJPY", "EURGBP"]
    models = [
        "WEEKLY_PIVOT_TREND_PULLBACK",
        "VWAP_1SD_TREND_REBOUND",
        "KELTNER_MIDLINE_EXPANSION",
        "THREE_BAR_INSIDE_CONFIRMATION"
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
        futures = {executor.submit(run_week_task_v22, t): t for t in tasks}

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
    print(f"{'ASSET':<8} | {'MODEL':<30} | {'TRADES':<6} | {'WIN RATE':<8} | {'PF':<5} | {'NET PIPS':<10} | {'MAX DD':<8} | {'RET/DD':<6}", flush=True)
    print("=" * 90, flush=True)

    summary_results: Dict[str, Any] = {}

    for a in assets:
        for m in models:
            k = f"{a}_{m}"
            pips = trades_by_key[k]
            met = calculate_metrics(pips)
            summary_results[k] = met
            print(f"{a:<8} | {m:<30} | {met['trades']:<6} | {met['win_rate']:<6.1f} % | {met['pf']:<5.2f} | {met['net_pips']:<10.1f} | {met['max_dd']:<8.1f} | {met['ret_dd']:<5.2f} x", flush=True)
        print("-" * 90, flush=True)

    out_file = RESULTS_DIR / "lab_v22_discovery_results.json"
    with open(out_file, "w") as f:
        json.dump(summary_results, f, indent=2)

    total_time = time.time() - t0
    print(f"\n[+] Discovery results saved to: {out_file}", flush=True)
    print(f"[+] Total execution time: {total_time:.2f}s", flush=True)
    print("=" * 90, flush=True)


if __name__ == "__main__":
    main()
