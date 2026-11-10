#!/usr/bin/env python3
"""
================================================================================
  ALPHA OPTIMIZATION LAB V41:
  Tuning Parameters of "Almost Good" Previous Strategies (PF 1.18 - 1.21):
    1. EURGBP M15 200-EMA Institutional Bounce (Baseline: 1.21 PF, +215.4 norm pips)
    2. EURGBP Asian Compression Breakout (Baseline: 1.18 PF, +360.8 norm pips)
    3. EURUSD Keltner TTM Squeeze Breakout (Baseline: 1.19 PF, +319.1 pips)
    4. EURGBP Triple Screen Momentum (Baseline: 1.20 PF, +872.3 norm pips)
================================================================================
STRICT CONSTRAINT:
  Evaluated EXCLUSIVELY on the 70% In-Sample Discovery Partition (Seed 42).
  ZERO touch or leakage of the 30% quarantined holdout partition!

Passing Criteria:
  - Profit Factor >= 1.25
  - EURUSD-Equivalent Normalized Net Pips >= 150.0 pips
  - Expectancy >= 1.20 pips/trade
  - Return / Drawdown >= 4.0x
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

EURUSD_MULT = 1.0000
USDJPY_MULT = 0.7193
EURGBP_MULT = 1.2921


# ==============================================================================
# Model 1: EURGBP M15 200-EMA Bounce (Parametric Tuning)
# ==============================================================================
class TunedM15TwoHundredEMABounce(BaseStrategy):
    def __init__(
        self,
        asset: str,
        tol_pips: float,
        min_wick_pct: float,
        sl_pips: float,
        tp_pips: float,
        pip_size: float,
        max_spread: float,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.asset = asset
        self.tol_pips = float(tol_pips)
        self.min_wick_pct = float(min_wick_pct)
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.pip_size = float(pip_size)
        self.max_spread = float(max_spread)
        self.name = f"{asset}_200EMA_Tol{tol_pips}_W{int(min_wick_pct*100)}_R{tp_pips/sl_pips:.1f}"

    def generate_signals(self, ticks: np.ndarray):
        n = len(ticks)
        signals = np.zeros(n, dtype=np.int8)
        sl_arr = np.zeros(n, dtype=np.float64)
        tp_arr = np.zeros(n, dtype=np.float64)
        if n < 500:
            return signals, sl_arr, tp_arr

        m15 = build_time_bars(ticks, timeframe_min=15, pip_size=self.pip_size)
        h4 = build_time_bars(ticks, timeframe_min=240, pip_size=self.pip_size)
        if len(m15) < 220 or len(h4) < 15:
            return signals, sl_arr, tp_arr

        # H4 Trend Filter
        h4_c = pd.Series(h4['close'].values)
        h4_ema20 = h4_c.ewm(span=20, adjust=False).mean().values
        h4_trend = np.where(h4_c.values > h4_ema20, 1, -1)
        h4_df = pd.DataFrame({'h4_bias': h4_trend}, index=h4.index)

        # M15 200-EMA
        m15_c = pd.Series(m15['close'].values)
        ema200 = m15_c.ewm(span=200, adjust=False).mean().values

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
        h4_bias = m15_df['h4_bias'].fillna(0).values.astype(int)
        hours = m15_df.index.hour.values
        dates = m15_df.index.date
        time_ms = m15_df.index.astype('datetime64[ms]').view('int64')
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        last_trade_date = None
        tol = self.tol_pips * self.pip_size

        for i in range(200, len(m15_df)):
            # Active institutional liquidity drift (07:30 - 15:30 UTC)
            if not (7 <= hours[i] <= 15):
                continue

            dt = dates[i]
            if dt == last_trade_date:
                continue
            if spread[i] > self.max_spread:
                continue

            macro = h4_bias[i]
            if macro == 0:
                continue

            bar_rng = h[i] - l[i]
            if bar_rng <= 0:
                continue

            e = ema200[i]
            sig = 0
            # Bullish bounce off 200-EMA
            if macro == 1 and abs(l[i] - e) <= tol and c[i] > e and c[i] > o[i]:
                lower_wick = min(c[i], o[i]) - l[i]
                if lower_wick >= self.min_wick_pct * bar_rng:
                    sig = SIGNAL_LONG

            # Bearish bounce off 200-EMA
            elif macro == -1 and abs(h[i] - e) <= tol and c[i] < e and c[i] < o[i]:
                upper_wick = h[i] - max(c[i], o[i])
                if upper_wick >= self.min_wick_pct * bar_rng:
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
# Model 2: EURGBP Asian Compression Breakout (Tuned Box Size 14-18p)
# ==============================================================================
class TunedAsianCompressionBO(BaseStrategy):
    def __init__(
        self,
        asset: str,
        max_asia_pips: float,
        sl_pips: float,
        tp_pips: float,
        pip_size: float,
        max_spread: float,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.asset = asset
        self.max_asia_pips = float(max_asia_pips)
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.pip_size = float(pip_size)
        self.max_spread = float(max_spread)
        self.name = f"{asset}_AsiaComp_Box{int(max_asia_pips)}_R{tp_pips/sl_pips:.1f}"

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
        h4_trend = np.where(h4_c.values > h4_ema20, 1, -1)
        h4_df = pd.DataFrame({'h4_bias': h4_trend}, index=h4.index)

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
        h4_bias = m15_df['h4_bias'].fillna(0).values.astype(int)
        hours = m15_df.index.hour.values
        dates = m15_df.index.date
        time_ms = m15_df.index.astype('datetime64[ms]').view('int64')
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        # Measure Asian Box: 00:00 - 06:45 UTC
        asian_boxes = {}
        unique_dates = np.unique(dates)
        for ud in unique_dates:
            mask = (dates == ud) & (hours < 7)
            if np.sum(mask) >= 8:
                a_hi = np.max(h[mask])
                a_lo = np.min(l[mask])
                box_pips = (a_hi - a_lo) / self.pip_size
                if box_pips <= self.max_asia_pips:
                    asian_boxes[ud] = (a_hi, a_lo)

        last_trade_date = None
        for i in range(25, len(m15_df)):
            # London Opening Breakout window (07:00 - 10:00 UTC)
            if not (7 <= hours[i] <= 9):
                continue

            dt = dates[i]
            if dt == last_trade_date or dt not in asian_boxes:
                continue

            if spread[i] > self.max_spread:
                continue

            macro = h4_bias[i]
            if macro == 0:
                continue

            a_hi, a_lo = asian_boxes[dt]
            bar_rng = h[i] - l[i]
            if bar_rng <= 0:
                continue

            sig = 0
            if macro == 1 and c[i] > a_hi and o[i] < a_hi and (c[i] - o[i]) >= 0.50 * bar_rng:
                sig = SIGNAL_LONG
            elif macro == -1 and c[i] < a_lo and o[i] > a_lo and (o[i] - c[i]) >= 0.50 * bar_rng:
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
# Model 3: EURUSD Keltner TTM Squeeze Breakout (Parametric Tuning)
# ==============================================================================
class TunedKeltnerSqueezeBreakout(BaseStrategy):
    def __init__(
        self,
        asset: str,
        squeeze_bars: int,
        sl_pips: float,
        tp_pips: float,
        pip_size: float,
        max_spread: float,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.asset = asset
        self.squeeze_bars = int(squeeze_bars)
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.pip_size = float(pip_size)
        self.max_spread = float(max_spread)
        self.name = f"{asset}_Squeeze_SQ{squeeze_bars}_R{tp_pips/sl_pips:.1f}"

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
        bb_mid = c_ser.rolling(20).mean().values
        bb_std = c_ser.rolling(20).std().bfill().values
        bb_upper = bb_mid + 2.0 * bb_std
        bb_lower = bb_mid - 2.0 * bb_std

        tr = np.maximum(h - l, np.maximum(np.abs(h - np.roll(c, 1)), np.abs(l - np.roll(c, 1))))
        tr[0] = h[0] - l[0]
        atr20 = pd.Series(tr).rolling(20).mean().bfill().values
        kc_mid = c_ser.ewm(span=20, adjust=False).mean().values
        kc_upper = kc_mid + 1.5 * atr20
        kc_lower = kc_mid - 1.5 * atr20

        is_in_squeeze = (bb_upper <= kc_upper) & (bb_lower >= kc_lower)

        last_trade_date = None

        for i in range(25, len(m15_df)):
            dt = dates[i]
            hr = hours[i]

            # European / US active hours (07:30 - 15:30 UTC)
            if not (7 <= hr <= 15):
                continue
            if dt == last_trade_date:
                continue
            if spread[i] > self.max_spread:
                continue

            macro = h4_trend[i]
            if macro == 0:
                continue

            # Check if was in squeeze for required bars
            was_squeezed = True
            for k in range(1, self.squeeze_bars + 1):
                if not is_in_squeeze[i - k]:
                    was_squeezed = False
                    break

            is_firing = (bb_upper[i] > kc_upper[i]) or (bb_lower[i] < kc_lower[i])

            if not (was_squeezed and is_firing):
                continue

            bar_rng = h[i] - l[i]
            if bar_rng <= 0:
                continue

            sig = 0
            if macro == 1 and c[i] > bb_upper[i] and (c[i] - o[i]) >= 0.50 * bar_rng:
                sig = SIGNAL_LONG
            elif macro == -1 and c[i] < bb_lower[i] and (o[i] - c[i]) >= 0.50 * bar_rng:
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
# Model 4: EURGBP Triple Screen Momentum (Calibrated Wick & Body)
# ==============================================================================
class TunedTripleScreenMomentum(BaseStrategy):
    def __init__(
        self,
        asset: str,
        min_body_pct: float,
        sl_pips: float,
        tp_pips: float,
        pip_size: float,
        max_spread: float,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.asset = asset
        self.min_body_pct = float(min_body_pct)
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.pip_size = float(pip_size)
        self.max_spread = float(max_spread)
        self.name = f"{asset}_TripleScreen_B{int(min_body_pct*100)}_R{tp_pips/sl_pips:.1f}"

    def generate_signals(self, ticks: np.ndarray):
        n = len(ticks)
        signals = np.zeros(n, dtype=np.int8)
        sl_arr = np.zeros(n, dtype=np.float64)
        tp_arr = np.zeros(n, dtype=np.float64)
        if n < 500:
            return signals, sl_arr, tp_arr

        m15 = build_time_bars(ticks, timeframe_min=15, pip_size=self.pip_size)
        h1 = build_time_bars(ticks, timeframe_min=60, pip_size=self.pip_size)
        h4 = build_time_bars(ticks, timeframe_min=240, pip_size=self.pip_size)
        if len(m15) < 30 or len(h1) < 25 or len(h4) < 10:
            return signals, sl_arr, tp_arr

        # Screen 1: H4 20-EMA Macro Trend
        h4_c = pd.Series(h4['close'].values)
        h4_ema20 = h4_c.ewm(span=20, adjust=False).mean().values
        h4_bias = np.where(h4_c.values > h4_ema20, 1, -1)
        h4_df = pd.DataFrame({'h4_bias': h4_bias}, index=h4.index)

        # Screen 2: H1 MACD Histogram Slope
        h1_c = pd.Series(h1['close'].values)
        ema12 = h1_c.ewm(span=12, adjust=False).mean()
        ema26 = h1_c.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        sig = macd.ewm(span=9, adjust=False).mean()
        hist = macd - sig
        hist_slope = hist.diff().values
        h1_bias = np.where(hist_slope > 0, 1, -1)
        h1_df = pd.DataFrame({'h1_bias': h1_bias}, index=h1.index)

        # Screen 3: M15 EMA 13 Dynamic Test
        m15_c = pd.Series(m15['close'].values)
        ema13 = m15_c.ewm(span=13, adjust=False).mean().values

        m15_m1 = pd.merge_asof(
            m15.sort_index(),
            h1_df.sort_index(),
            left_index=True,
            right_index=True,
            direction='backward'
        )
        m15_df = pd.merge_asof(
            m15_m1.sort_index(),
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
        h1_trend = m15_df['h1_bias'].fillna(0).values.astype(int)
        hours = m15_df.index.hour.values
        dates = m15_df.index.date
        time_ms = m15_df.index.astype('datetime64[ms]').view('int64')
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        last_trade_date = None
        tol = 2.5 * self.pip_size

        for i in range(25, len(m15_df)):
            if not (7 <= hours[i] <= 16):
                continue

            dt = dates[i]
            if dt == last_trade_date:
                continue
            if spread[i] > self.max_spread:
                continue

            macro_h4 = h4_trend[i]
            macro_h1 = h1_trend[i]
            if macro_h4 != macro_h1 or macro_h4 == 0:
                continue

            rng = h[i] - l[i]
            if rng <= 0:
                continue

            body = abs(c[i] - o[i])
            if (body / rng) < self.min_body_pct:
                continue

            e = ema13[i]
            trade_sig = 0
            if macro_h4 == 1 and abs(l[i] - e) <= tol and c[i] > e and c[i] > o[i]:
                trade_sig = SIGNAL_LONG
            elif macro_h4 == -1 and abs(h[i] - e) <= tol and c[i] < e and c[i] < o[i]:
                trade_sig = SIGNAL_SHORT

            if trade_sig != 0:
                end_ms = time_ms[i] + dur_ms
                idx = np.searchsorted(tick_ts, end_ms)
                if idx < n:
                    signals[idx] = trade_sig
                    sl_arr[idx] = self.sl_pips
                    tp_arr[idx] = self.tp_pips
                    last_trade_date = dt

        return signals, sl_arr, tp_arr


# ==============================================================================
# Model Configurations to Audit on 70% In-Sample Discovery Weeks
# ==============================================================================
def get_lab_v41_models() -> List[Tuple[str, str, BaseStrategy]]:
    mods = []

    # 1. EURGBP M15 200-EMA Bounce Variants (2.0R vs 2.2R, Tol 2.0 vs 2.5p)
    mods.append(("EURGBP", "EMA200_TOL20_20R", TunedM15TwoHundredEMABounce("EURGBP", 2.0, 0.35, 6.5, 13.0, 0.0001, 0.6)))
    mods.append(("EURGBP", "EMA200_TOL20_22R", TunedM15TwoHundredEMABounce("EURGBP", 2.0, 0.35, 6.5, 14.3, 0.0001, 0.6)))
    mods.append(("EURGBP", "EMA200_TOL25_22R", TunedM15TwoHundredEMABounce("EURGBP", 2.5, 0.35, 6.5, 14.3, 0.0001, 0.6)))
    mods.append(("EURUSD", "EMA200_TOL25_22R", TunedM15TwoHundredEMABounce("EURUSD", 2.5, 0.35, 8.0, 17.6, 0.0001, 0.6)))

    # 2. EURGBP Asian Compression Breakout (Box 14p, 16p, 18p with 2.2R)
    mods.append(("EURGBP", "ASIA_COMP_BOX14_22R", TunedAsianCompressionBO("EURGBP", 14.0, 6.5, 14.3, 0.0001, 0.6)))
    mods.append(("EURGBP", "ASIA_COMP_BOX16_22R", TunedAsianCompressionBO("EURGBP", 16.0, 6.5, 14.3, 0.0001, 0.6)))
    mods.append(("EURGBP", "ASIA_COMP_BOX18_20R", TunedAsianCompressionBO("EURGBP", 18.0, 7.0, 14.0, 0.0001, 0.6)))

    # 3. EURUSD Keltner TTM Squeeze Breakout (Squeeze 1 vs 2 bars, 2.2R vs 2.5R)
    mods.append(("EURUSD", "SQUEEZE_SQ1_22R", TunedKeltnerSqueezeBreakout("EURUSD", 1, 8.0, 17.6, 0.0001, 0.6)))
    mods.append(("EURUSD", "SQUEEZE_SQ2_22R", TunedKeltnerSqueezeBreakout("EURUSD", 2, 8.0, 17.6, 0.0001, 0.6)))
    mods.append(("EURUSD", "SQUEEZE_SQ2_25R", TunedKeltnerSqueezeBreakout("EURUSD", 2, 7.5, 18.75, 0.0001, 0.6)))

    # 4. EURGBP Triple Screen Momentum (Body 50% vs 55%, 2.2R)
    mods.append(("EURGBP", "TS_MOM_B50_22R", TunedTripleScreenMomentum("EURGBP", 0.50, 6.5, 14.3, 0.0001, 0.6)))
    mods.append(("EURGBP", "TS_MOM_B55_22R", TunedTripleScreenMomentum("EURGBP", 0.55, 6.5, 14.3, 0.0001, 0.6)))

    return mods


def run_week_task(task: Tuple[str, str, int, BaseStrategy]) -> Dict[str, Any]:
    asset, model_name, week_num, strat = task
    pip_size = 0.01 if asset == "USDJPY" else 0.0001
    engine = BacktestEngine(pip_size=pip_size, use_slippage=True)

    try:
        ticks, _ = get_week_dataset(week_num, raw_dir=f"data/{asset}")
        if ticks is None or len(ticks) < 500:
            return {"asset": asset, "model": model_name, "week": week_num, "trades": []}

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


def calculate_metrics(pips: List[float], mult: float) -> Dict[str, Any]:
    if not pips:
        return {"trades": 0, "win_rate": 0.0, "pf": 0.0, "net_pips": 0.0, "norm_pips": 0.0, "max_dd": 0.0, "ret_dd": 0.0, "expectancy": 0.0}
    arr = np.array(pips, dtype=np.float64)
    trades = len(arr)
    wins = arr[arr > 0]
    losses = arr[arr < 0]
    wr = len(wins) / trades * 100.0 if trades > 0 else 0.0
    gross_win = float(np.sum(wins)) if len(wins) > 0 else 0.0
    gross_loss = float(abs(np.sum(losses))) if len(losses) > 0 else 0.0
    pf = gross_win / gross_loss if gross_loss > 0 else 999.0
    net_pips = float(np.sum(arr))
    norm_pips = net_pips * mult
    expectancy = net_pips / trades if trades > 0 else 0.0

    equity = np.cumsum(arr * mult)
    peak = np.maximum.accumulate(equity)
    dd = peak - equity
    max_dd = float(np.max(dd)) if len(dd) > 0 else 0.0
    ret_dd = norm_pips / max_dd if max_dd > 0 else 999.0

    return {
        "trades": trades,
        "win_rate": round(wr, 1),
        "pf": round(pf, 2),
        "net_pips": round(net_pips, 1),
        "norm_pips": round(norm_pips, 1),
        "max_dd": round(max_dd, 1),
        "ret_dd": round(ret_dd, 2),
        "expectancy": round(expectancy, 2)
    }


def main():
    print("=" * 100)
    print("  ALPHA OPTIMIZATION LAB V41: TUNING 'ALMOST GOOD' CANDIDATES TO PRODUCTION STATUS")
    print("  STRICT PARTITION: 70% In-Sample Discovery Weeks ONLY (Quarantined 30% Holdout Untouched)")
    print("=" * 100)

    models_config = get_lab_v41_models()
    tasks = []
    for asset, model_name, strat in models_config:
        disc_weeks = get_discovery_weeks(raw_dir=f"data/{asset}")
        for w in disc_weeks:
            tasks.append((asset, model_name, w, strat))

    print(f"Total tuning tasks: {len(tasks):,} across {len(models_config)} model configurations")

    results_by_key = {}
    t0 = time.time()

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(run_week_task, t) for t in tasks]
        done = 0
        total = len(futures)
        for fut in as_completed(futures):
            res = fut.result()
            key = f"{res['asset']}_{res['model']}"
            if key not in results_by_key:
                results_by_key[key] = []
            results_by_key[key].extend(res["trades"])
            done += 1
            if done % 350 == 0 or done == total:
                elapsed = time.time() - t0
                print(f"  Progress: [{done}/{total}] tasks ({done/total*100:.1f}%) in {elapsed:.1f}s")

    elapsed = time.time() - t0
    print(f"\nExecution phase completed in {elapsed:.1f}s!\n")

    summary_rows = []
    print("=" * 105)
    print(f"  {'Asset':<8} {'Model Variant':<25} {'Trades':<8} {'Win Rate':<10} {'PF':<8} {'Expectancy':<12} {'Raw Pips':<10} {'Norm Pips':<11} {'Max DD':<9} {'Verdict'}")
    print("  " + "-" * 102)

    for asset, model_name, _ in models_config:
        key = f"{asset}_{model_name}"
        trades = results_by_key.get(key, [])
        pips = [t["pips"] for t in trades]
        mult = EURUSD_MULT if asset == "EURUSD" else USDJPY_MULT if asset == "USDJPY" else EURGBP_MULT
        m = calculate_metrics(pips, mult)

        verdict = "[PRODUCTION VIABLE]" if (m["pf"] >= 1.25 and m["norm_pips"] >= 150.0 and m["expectancy"] >= 1.20) else "[PROMISING]" if m["norm_pips"] > 0 else "[SUB-OPTIMAL]"

        print(f"  {asset:<8} {model_name:<25} {m['trades']:<8d} {m['win_rate']:<9.1f}% {m['pf']:<7.2f} +{m['expectancy']:<10.2f}p +{m['net_pips']:<9.1f}p +{m['norm_pips']:<10.1f}p {m['max_dd']:<8.1f}p {verdict}")

        summary_rows.append({
            "asset": asset,
            "model_variant": model_name,
            "trades": m["trades"],
            "win_rate": m["win_rate"],
            "profit_factor": m["pf"],
            "expectancy": m["expectancy"],
            "raw_pips": m["net_pips"],
            "normalized_pips": m["norm_pips"],
            "max_dd_norm": m["max_dd"],
            "ret_dd": m["ret_dd"],
            "verdict": verdict
        })

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(RESULTS_DIR / "alpha_optimization_lab_v41_scorecard.csv", index=False)
    print(f"\nSaved Tuning Scorecard to: {RESULTS_DIR / 'alpha_optimization_lab_v41_scorecard.csv'}")


if __name__ == "__main__":
    main()

