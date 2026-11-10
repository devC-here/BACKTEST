#!/usr/bin/env python3
"""
================================================================================
  ALPHA DISCOVERY LAB V7: INSTITUTIONAL COMPRESSION, RETESTS & OVERLAPS
================================================================================
Evaluates 4 New Quantitative Alpha Models across the 70% Discovery Partition:
  1. DISC53_LONDON_NY_OVERLAP_PIN: Reversal pin bar at 12:30 - 15:30 UTC when London/NY overlap
     drives price into H1 20-period Donchian highs/lows with >= 50% wick rejection.
  2. DISC54_M15_3_BAR_PULLBACK: Trend continuation after 3 consecutive counter-trend M15 bars
     pulling back to the M15 20-EMA with H4/H1 trend alignment.
  3. DISC55_DAILY_CPR_BREAKOUT: Central Pivot Range (CPR) narrow compression breakout during
     European open (07:00 - 10:00 UTC).
  4. DISC56_TOKYO_PIVOT_REVERSAL_CALIBRATED: Asian session (00:00 - 05:00 UTC) Daily Pivot / S1 / R1
     defense with strict >= 50% rejection pin bar filter.

Tested across EURUSD, EURGBP, and USDJPY on 70% Discovery Partition ONLY.
Strict 70/30 deterministic holdout isolation is maintained.
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
# Model 1: London / NY Overlap Structural Liquidity Reversal Pin (DISC53)
# ==============================================================================
class LondonNYOverlapPin(BaseStrategy):
    """
    Operates at 12:30 - 15:30 UTC (peak global FX volume overlap).
    Triggers when price touches or penetrates the 20-period H1 Donchian High/Low
    and rejects with an M15 candle printing >= 50% wick back inside the structure.
    """
    def __init__(self, asset: str, sl_pips: float, tp_pips: float, pip_size: float, max_spread: float, **kwargs):
        super().__init__(**kwargs)
        self.asset = asset
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.pip_size = float(pip_size)
        self.max_spread = float(max_spread)
        self.name = f"{asset}_London_NY_Overlap_Pin"

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
        if len(h1) < 22:
            return signals, sl_arr, tp_arr

        h1['don_h20'] = h1['high'].rolling(20).max().shift(1)
        h1['don_l20'] = h1['low'].rolling(20).min().shift(1)

        don_h_m15 = h1['don_h20'].reindex(m15.index, method='ffill').values
        don_l_m15 = h1['don_l20'].reindex(m15.index, method='ffill').values

        c = m15['close'].values
        h = m15['high'].values
        l = m15['low'].values
        o = m15['open'].values
        spread = m15['spread_pips'].values
        hours = m15.index.hour.values
        mins = m15.index.minute.values
        dates = m15.index.date
        time_ms = m15.index.astype('datetime64[ms]').view('int64')
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        last_trade_date = None

        min_rng_pips = 5.0 if self.asset == "USDJPY" else 3.5

        for i in range(25, len(m15)):
            # Active window: 12:30 to 15:30 UTC
            h_val = hours[i]
            m_val = mins[i]
            in_window = (h_val == 12 and m_val >= 30) or (h_val in {13, 14}) or (h_val == 15 and m_val <= 30)
            if not in_window:
                continue

            dt = dates[i]
            if dt == last_trade_date:
                continue
            if spread[i] > self.max_spread:
                continue

            bar_rng = (h[i] - l[i]) / self.pip_size
            if bar_rng < min_rng_pips:
                continue

            dh = don_h_m15[i]
            dl = don_l_m15[i]
            if np.isnan(dh) or np.isnan(dl):
                continue

            sig = 0
            # Test Donchian Low & reject with lower wick >= 50%
            if l[i] <= dl and c[i] > dl:
                lower_wick = (min(o[i], c[i]) - l[i]) / self.pip_size
                if (lower_wick / bar_rng) >= 0.50 and c[i] > o[i]:
                    sig = SIGNAL_LONG

            # Test Donchian High & reject with upper wick >= 50%
            elif h[i] >= dh and c[i] < dh:
                upper_wick = (h[i] - max(o[i], c[i])) / self.pip_size
                if (upper_wick / bar_rng) >= 0.50 and c[i] < o[i]:
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
# Model 2: Trend Continuation 3-Bar Counter-Trend Exhaustion (DISC54)
# ==============================================================================
class M15ThreeBarPullback(BaseStrategy):
    """
    Operates in active European & US sessions (07:00 - 17:00 UTC).
    Requires H4 and H1 20-EMA trend agreement.
    Waits for 3 consecutive counter-trend M15 bars that touch or test the M15 20-EMA,
    followed by a momentum expansion candle in the direction of the trend.
    """
    def __init__(self, asset: str, sl_pips: float, tp_pips: float, pip_size: float, max_spread: float, **kwargs):
        super().__init__(**kwargs)
        self.asset = asset
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.pip_size = float(pip_size)
        self.max_spread = float(max_spread)
        self.name = f"{asset}_M15_3_Bar_Pullback"

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

        h1 = base.resample('1h').agg({'close': 'last'}).dropna()
        h4 = base.resample('4h').agg({'close': 'last'}).dropna()
        if len(h1) < 25 or len(h4) < 15:
            return signals, sl_arr, tp_arr

        h1['ema20'] = h1['close'].ewm(span=20, adjust=False).mean()
        h4['ema20'] = h4['close'].ewm(span=20, adjust=False).mean()

        h1_trend = np.where(h1['close'] > h1['ema20'], 1, -1)
        h4_trend = np.where(h4['close'] > h4['ema20'], 1, -1)

        h1_bias = pd.Series(h1_trend, index=h1.index).shift(1).reindex(m15.index, method='ffill').fillna(0).values
        h4_bias = pd.Series(h4_trend, index=h4.index).shift(1).reindex(m15.index, method='ffill').fillna(0).values

        m15['ema20'] = m15['close'].ewm(span=20, adjust=False).mean()
        m15_ema = m15['ema20'].values

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
        min_body_pips = 4.0 if self.asset == "USDJPY" else 2.5

        for i in range(25, len(m15)):
            if not (7 <= hours[i] <= 17):
                continue
            dt = dates[i]
            if dt == last_trade_date:
                continue
            if spread[i] > self.max_spread:
                continue

            bias_h1 = h1_bias[i]
            bias_h4 = h4_bias[i]
            if bias_h1 != bias_h4 or bias_h1 == 0:
                continue

            bar_rng = (h[i] - l[i]) / self.pip_size
            body = abs(c[i] - o[i]) / self.pip_size
            if bar_rng < min_body_pips or (body / bar_rng) < 0.50:
                continue

            sig = 0

            # Bullish trend: Check if previous 3 bars (i-3, i-2, i-1) were pulling back
            if bias_h1 > 0:
                # 3 consecutive lower highs or bearish closes
                three_pb = (h[i-1] <= h[i-2] and h[i-2] <= h[i-3]) or (c[i-1] <= o[i-1] and c[i-2] <= o[i-2] and c[i-3] <= o[i-3])
                # Touched or came within 2 pips of M15 EMA20
                tested_ema = any(abs(l[k] - m15_ema[k]) / self.pip_size <= 2.5 or l[k] <= m15_ema[k] for k in [i-3, i-2, i-1])
                # Trigger bar is strong bullish expansion engulfing high of i-1
                if three_pb and tested_ema and c[i] > o[i] and c[i] > h[i-1]:
                    sig = SIGNAL_LONG

            # Bearish trend: Check if previous 3 bars were pulling up
            elif bias_h1 < 0:
                # 3 consecutive higher lows or bullish closes
                three_pb = (l[i-1] >= l[i-2] and l[i-2] >= l[i-3]) or (c[i-1] >= o[i-1] and c[i-2] >= o[i-2] and c[i-3] >= o[i-3])
                tested_ema = any(abs(h[k] - m15_ema[k]) / self.pip_size <= 2.5 or h[k] >= m15_ema[k] for k in [i-3, i-2, i-1])
                # Trigger bar is strong bearish expansion breaking low of i-1
                if three_pb and tested_ema and c[i] < o[i] and c[i] < l[i-1]:
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
# Model 3: Daily Central Pivot Range (CPR) Narrow Compression Breakout (DISC55)
# ==============================================================================
class DailyCPRBreakout(BaseStrategy):
    """
    Calculates Daily Central Pivot Range (CPR):
      P = (H + L + C) / 3
      BC = (H + L) / 2
      TC = 2*P - BC
      CPR Width = abs(TC - BC)
    Filters for narrow CPR days (compression) and enters when London Open (07:00 - 10:00 UTC)
    breaks out of the CPR band with momentum expansion.
    """
    def __init__(self, asset: str, sl_pips: float, tp_pips: float, pip_size: float, max_spread: float, max_cpr_width: float, **kwargs):
        super().__init__(**kwargs)
        self.asset = asset
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.pip_size = float(pip_size)
        self.max_spread = float(max_spread)
        self.max_cpr_width = float(max_cpr_width)
        self.name = f"{asset}_Daily_CPR_Breakout"

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

        daily = base.resample('1D').agg({'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
        if len(daily) < 2:
            return signals, sl_arr, tp_arr

        prev_d = daily.shift(1)
        p = (prev_d['high'] + prev_d['low'] + prev_d['close']) / 3.0
        bc = (prev_d['high'] + prev_d['low']) / 2.0
        tc = 2.0 * p - bc
        cpr_top = np.maximum(tc, bc)
        cpr_bot = np.minimum(tc, bc)
        cpr_width = (cpr_top - cpr_bot) / self.pip_size

        daily_cpr = pd.DataFrame({
            'cpr_top': cpr_top,
            'cpr_bot': cpr_bot,
            'cpr_width': cpr_width
        }, index=daily.index)

        cpr_top_m15 = daily_cpr['cpr_top'].reindex(m15.index, method='ffill').values
        cpr_bot_m15 = daily_cpr['cpr_bot'].reindex(m15.index, method='ffill').values
        cpr_width_m15 = daily_cpr['cpr_width'].reindex(m15.index, method='ffill').values

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

        for i in range(10, len(m15)):
            if hours[i] not in {7, 8, 9}:
                continue
            d = dates[i]
            if d in traded_days:
                continue
            if spread[i] > self.max_spread:
                continue

            width = cpr_width_m15[i]
            if np.isnan(width) or width > self.max_cpr_width or width <= 0:
                continue

            bar_rng = (h[i] - l[i]) / self.pip_size
            body = abs(c[i] - o[i]) / self.pip_size
            if bar_rng < 3.0 or (body / bar_rng) < 0.50:
                continue

            top = cpr_top_m15[i]
            bot = cpr_bot_m15[i]

            sig = 0
            # Bullish expansion candle closing decisively above CPR top
            if c[i] > top and o[i] <= top and c[i] > o[i]:
                sig = SIGNAL_LONG
            # Bearish expansion candle closing decisively below CPR bot
            elif c[i] < bot and o[i] >= bot and c[i] < o[i]:
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
# Model 4: Calibrated Tokyo Session Pivot Reversal Pin (DISC56)
# ==============================================================================
class CalibratedTokyoPivotReversal(BaseStrategy):
    """
    Operates during Tokyo Asian session (00:00 - 05:00 UTC).
    Identifies institutional absorption at Daily Pivot, Daily S1, or Daily R1
    with strict >= 50% rejection wick pin bar filter.
    """
    def __init__(self, asset: str, sl_pips: float, tp_pips: float, pip_size: float, max_spread: float, **kwargs):
        super().__init__(**kwargs)
        self.asset = asset
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.pip_size = float(pip_size)
        self.max_spread = float(max_spread)
        self.name = f"{asset}_Tokyo_Pivot_Rev_Calibrated"

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

        daily = base.resample('1D').agg({'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
        if len(daily) < 2:
            return signals, sl_arr, tp_arr

        prev_d = daily.shift(1)
        p = (prev_d['high'] + prev_d['low'] + prev_d['close']) / 3.0
        r1 = 2.0 * p - prev_d['low']
        s1 = 2.0 * p - prev_d['high']

        daily_pivots = pd.DataFrame({'pivot': p, 'r1': r1, 's1': s1}, index=daily.index)

        pivot_m15 = daily_pivots['pivot'].reindex(m15.index, method='ffill').values
        r1_m15 = daily_pivots['r1'].reindex(m15.index, method='ffill').values
        s1_m15 = daily_pivots['s1'].reindex(m15.index, method='ffill').values

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
        min_rng_pips = 4.0 if self.asset == "USDJPY" else 2.5
        tol = 3.0 * self.pip_size

        for i in range(10, len(m15)):
            if not (0 <= hours[i] <= 4):
                continue
            d = dates[i]
            if d in traded_days:
                continue
            if spread[i] > self.max_spread:
                continue

            bar_rng = (h[i] - l[i]) / self.pip_size
            if bar_rng < min_rng_pips:
                continue

            pv = pivot_m15[i]
            s1_val = s1_m15[i]
            r1_val = r1_m15[i]
            if np.isnan(pv) or np.isnan(s1_val) or np.isnan(r1_val):
                continue

            sig = 0

            # Long bounce off Pivot or S1 with >= 50% lower wick
            touched_support = (l[i] <= pv + tol and c[i] > pv) or (l[i] <= s1_val + tol and c[i] > s1_val)
            if touched_support and c[i] > o[i]:
                lower_wick = (min(o[i], c[i]) - l[i]) / self.pip_size
                if (lower_wick / bar_rng) >= 0.50:
                    sig = SIGNAL_LONG

            # Short bounce off Pivot or R1 with >= 50% upper wick
            touched_res = (h[i] >= pv - tol and c[i] < pv) or (h[i] >= r1_val - tol and c[i] < r1_val)
            if touched_res and c[i] < o[i]:
                upper_wick = (h[i] - max(o[i], c[i])) / self.pip_size
                if (upper_wick / bar_rng) >= 0.50:
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
# Discovery V7 Multi-Asset Parallel Worker
# ==============================================================================
def evaluate_v7_discovery_week(args: Tuple[str, int]) -> Dict[str, Any]:
    asset, week_num = args
    pip_size = 0.01 if asset == "USDJPY" else 0.0001
    engine = BacktestEngine(pip_size=pip_size, use_slippage=True)

    ticks, meta = get_week_dataset(week_num, raw_dir=f"data/{asset}")
    if len(ticks) == 0:
        return {"asset": asset, "week": week_num, "trades": {}}

    models = []
    if asset == "USDJPY":
        models.append(("LONDON_NY_PIN", LondonNYOverlapPin(asset=asset, sl_pips=12.0, tp_pips=22.0, pip_size=0.01, max_spread=0.8)))
        models.append(("M15_3BAR_PB", M15ThreeBarPullback(asset=asset, sl_pips=12.0, tp_pips=24.0, pip_size=0.01, max_spread=0.8)))
        models.append(("DAILY_CPR_BO", DailyCPRBreakout(asset=asset, sl_pips=14.0, tp_pips=28.0, pip_size=0.01, max_spread=0.8, max_cpr_width=15.0)))
        models.append(("TOKYO_PIVOT_CALIB", CalibratedTokyoPivotReversal(asset=asset, sl_pips=12.0, tp_pips=22.0, pip_size=0.01, max_spread=0.8)))
    elif asset == "EURGBP":
        models.append(("LONDON_NY_PIN", LondonNYOverlapPin(asset=asset, sl_pips=8.0, tp_pips=14.0, pip_size=0.0001, max_spread=2.0)))
        models.append(("M15_3BAR_PB", M15ThreeBarPullback(asset=asset, sl_pips=7.0, tp_pips=14.0, pip_size=0.0001, max_spread=2.0)))
        models.append(("DAILY_CPR_BO", DailyCPRBreakout(asset=asset, sl_pips=8.0, tp_pips=16.0, pip_size=0.0001, max_spread=2.0, max_cpr_width=8.0)))
        models.append(("TOKYO_PIVOT_CALIB", CalibratedTokyoPivotReversal(asset=asset, sl_pips=8.0, tp_pips=14.0, pip_size=0.0001, max_spread=2.0)))
    elif asset == "EURUSD":
        models.append(("LONDON_NY_PIN", LondonNYOverlapPin(asset=asset, sl_pips=9.0, tp_pips=16.0, pip_size=0.0001, max_spread=0.6)))
        models.append(("M15_3BAR_PB", M15ThreeBarPullback(asset=asset, sl_pips=9.0, tp_pips=18.0, pip_size=0.0001, max_spread=0.6)))
        models.append(("DAILY_CPR_BO", DailyCPRBreakout(asset=asset, sl_pips=10.0, tp_pips=20.0, pip_size=0.0001, max_spread=0.6, max_cpr_width=10.0)))
        models.append(("TOKYO_PIVOT_CALIB", CalibratedTokyoPivotReversal(asset=asset, sl_pips=9.0, tp_pips=16.0, pip_size=0.0001, max_spread=0.6)))

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


def run_discovery_v7():
    t_start = time.time()
    print("=" * 80, flush=True)
    print("  LAUNCHING ALPHA DISCOVERY V7 EXPEDITION (70% DISCOVERY PARTITIONS)", flush=True)
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
        futures = {executor.submit(evaluate_v7_discovery_week, t): t for t in all_tasks}
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
            "asset": asset,
            "strategy": m_name,
            "trades": n,
            "win_rate": wr,
            "net_pips": net_pips,
            "profit_factor": pf,
            "max_dd_pips": max_dd,
            "ret_dd": ret_dd,
            "expectancy": expectancy
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(by=["asset", "net_pips"], ascending=[True, False])
        out_csv = RESULTS_DIR / "alpha_discovery_lab_v7_scorecard.csv"
        df.to_csv(out_csv, index=False)

        print("\n" + "=" * 90, flush=True)
        print("  ALPHA DISCOVERY LAB V7 SCORECARD (70% DISCOVERY PARTITIONS ONLY)", flush=True)
        print("=" * 90, flush=True)
        print(f"{'Asset':<8} {'Strategy':<20} {'Trades':<8} {'Win Rate':<10} {'Net Pips':<12} {'PF':<8} {'Max DD':<10} {'Ret/DD':<8} {'Exp (p/t)':<10}")
        print("-" * 90, flush=True)
        for _, r in df.iterrows():
            print(f"{r['asset']:<8} {r['strategy']:<20} {int(r['trades']):<8} {r['win_rate']:>6.1f}%   {r['net_pips']:>10.1f}p  {r['profit_factor']:>6.2f}  {r['max_dd_pips']:>8.1f}p  {r['ret_dd']:>6.2f}x  {r['expectancy']:>8.2f}p")
        print("=" * 90, flush=True)

    print(f"\n[*] Alpha Discovery Lab V7 Complete in {time.time() - t_start:.2f}s.", flush=True)


if __name__ == "__main__":
    run_discovery_v7()
