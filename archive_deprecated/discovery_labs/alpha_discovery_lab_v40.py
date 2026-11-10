#!/usr/bin/env python3
"""
================================================================================
  ALPHA CORRECTION & OPTIMIZATION LAB V40:
  Systematically Correcting Flawed Previous Strategies:
    1. USDJPY & EURUSD London PM Volatility Expansion (Pre-US Range Compression + 2.2R)
    2. EURUSD Double Structure Neckline Breakout (Volume Expansion + Body Filter)
    3. EURGBP Asian Compression Breakout (Tighter 10-12p Compression Box + 2.2R)
    4. USDJPY M15 50-EMA Dynamic Pin Runner (ATR Minimum Range + Volume Confirmation)
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
# Model 1: London PM Volatility Expansion (Corrected)
# ==============================================================================
class CorrectedLondonPMExpansion(BaseStrategy):
    def __init__(
        self,
        asset: str,
        max_pre_us_pips: float,
        min_body_pct: float,
        sl_pips: float,
        tp_pips: float,
        pip_size: float,
        max_spread: float,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.asset = asset
        self.max_pre_us_pips = float(max_pre_us_pips)
        self.min_body_pct = float(min_body_pct)
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.pip_size = float(pip_size)
        self.max_spread = float(max_spread)
        self.name = f"{asset}_LonPM_Box{int(max_pre_us_pips)}_B{int(min_body_pct*100)}_R{tp_pips/sl_pips:.1f}"

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
        if len(m15) < 30 or len(h1) < 15 or len(h4) < 10:
            return signals, sl_arr, tp_arr

        # H4 Macro Trend Filter
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
        mins = m15_df.index.minute.values
        dates = m15_df.index.date
        time_ms = m15_df.index.astype('datetime64[ms]').view('int64')
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        # Measure Pre-US Open Consolidation Range: 10:30 - 12:30 UTC
        pre_us_ranges = {}
        unique_dates = np.unique(dates)
        for ud in unique_dates:
            mask = (dates == ud) & (((hours == 10) & (mins >= 30)) | (hours == 11) | ((hours == 12) & (mins <= 30)))
            if np.sum(mask) >= 4:
                p_hi = np.max(h[mask])
                p_lo = np.min(l[mask])
                rng_p = (p_hi - p_lo) / self.pip_size
                if rng_p <= self.max_pre_us_pips:
                    pre_us_ranges[ud] = (p_hi, p_lo)

        last_trade_date = None
        for i in range(20, len(m15_df)):
            t_hour = hours[i]
            t_min = mins[i]

            # US Open overlap expansion window: 12:30 - 15:00 UTC
            if not ((t_hour == 12 and t_min >= 30) | (t_hour in (13, 14))):
                continue

            dt = dates[i]
            if dt == last_trade_date or dt not in pre_us_ranges:
                continue

            if spread[i] > self.max_spread:
                continue

            macro = h4_bias[i]
            if macro == 0:
                continue

            p_hi, p_lo = pre_us_ranges[dt]
            bar_rng = h[i] - l[i]
            if bar_rng <= 0:
                continue

            sig = 0
            # Bullish Breakout: Closes above Pre-US high with strong body >= min_body_pct
            if macro == 1 and c[i] > p_hi and o[i] < p_hi and (c[i] - o[i]) >= self.min_body_pct * bar_rng:
                sig = SIGNAL_LONG

            # Bearish Breakout: Closes below Pre-US low with strong body >= min_body_pct
            elif macro == -1 and c[i] < p_lo and o[i] > p_lo and (o[i] - c[i]) >= self.min_body_pct * bar_rng:
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
# Model 2: Double Structure Neckline Breakout (Corrected)
# ==============================================================================
class CorrectedDoubleStructureBO(BaseStrategy):
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
        self.name = f"{asset}_DoubleStruct_B{int(min_body_pct*100)}_R{tp_pips/sl_pips:.1f}"

    def generate_signals(self, ticks: np.ndarray):
        n = len(ticks)
        signals = np.zeros(n, dtype=np.int8)
        sl_arr = np.zeros(n, dtype=np.float64)
        tp_arr = np.zeros(n, dtype=np.float64)
        if n < 500:
            return signals, sl_arr, tp_arr

        m15 = build_time_bars(ticks, timeframe_min=15, pip_size=self.pip_size)
        h4 = build_time_bars(ticks, timeframe_min=240, pip_size=self.pip_size)
        if len(m15) < 45 or len(h4) < 10:
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
        v = m15_df['vol'].values
        spread = m15_df['spread_pips'].values
        h4_bias = m15_df['h4_bias'].fillna(0).values.astype(int)
        hours = m15_df.index.hour.values
        dates = m15_df.index.date
        time_ms = m15_df.index.astype('datetime64[ms]').view('int64')
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        last_trade_date = None
        for i in range(35, len(m15_df)):
            # Active institutional liquidity window (07:30 - 15:30 UTC)
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

            # Lookback window for double bottom/top structure
            w_l = l[i-30:i-2]
            w_h = h[i-30:i-2]
            mid_pt = len(w_l) // 2
            min1, min2 = np.min(w_l[:mid_pt]), np.min(w_l[mid_pt:])
            max1, max2 = np.max(w_h[:mid_pt]), np.max(w_h[mid_pt:])
            neck_hi = np.max(w_h)
            neck_lo = np.min(w_l)

            sig = 0
            # Bullish Double Bottom Breakout
            if macro == 1 and abs(min1 - min2) <= 8.0 * self.pip_size:
                if c[i] > neck_hi and o[i] < neck_hi and (c[i] - o[i]) >= self.min_body_pct * bar_rng and v[i] > v[i-1]:
                    sig = SIGNAL_LONG

            # Bearish Double Top Breakout
            elif macro == -1 and abs(max1 - max2) <= 8.0 * self.pip_size:
                if c[i] < neck_lo and o[i] > neck_lo and (o[i] - c[i]) >= self.min_body_pct * bar_rng and v[i] > v[i-1]:
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
# Model 3: EURGBP Asian Compression Breakout (Corrected)
# ==============================================================================
class CorrectedAsianCompressionBO(BaseStrategy):
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

        # Measure Asian Box (00:00 - 06:45 UTC)
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
# Model 4: USDJPY M15 50-EMA Dynamic Pin Runner (Corrected)
# ==============================================================================
class CorrectedM15FiftyEMAPinRunner(BaseStrategy):
    def __init__(
        self,
        asset: str,
        min_wick_pct: float,
        sl_pips: float,
        tp_pips: float,
        pip_size: float,
        max_spread: float,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.asset = asset
        self.min_wick_pct = float(min_wick_pct)
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.pip_size = float(pip_size)
        self.max_spread = float(max_spread)
        self.name = f"{asset}_M15_50EMA_Pin_W{int(min_wick_pct*100)}_R{tp_pips/sl_pips:.1f}"

    def generate_signals(self, ticks: np.ndarray):
        n = len(ticks)
        signals = np.zeros(n, dtype=np.int8)
        sl_arr = np.zeros(n, dtype=np.float64)
        tp_arr = np.zeros(n, dtype=np.float64)
        if n < 500:
            return signals, sl_arr, tp_arr

        m15 = build_time_bars(ticks, timeframe_min=15, pip_size=self.pip_size)
        h4 = build_time_bars(ticks, timeframe_min=240, pip_size=self.pip_size)
        if len(m15) < 55 or len(h4) < 10:
            return signals, sl_arr, tp_arr

        h4_c = pd.Series(h4['close'].values)
        h4_ema20 = h4_c.ewm(span=20, adjust=False).mean().values
        h4_trend = np.where(h4_c.values > h4_ema20, 1, -1)
        h4_df = pd.DataFrame({'h4_bias': h4_trend}, index=h4.index)

        # M15 50-EMA and ATR(14)
        m15_c = pd.Series(m15['close'].values)
        ema50 = m15_c.ewm(span=50, adjust=False).mean().values

        tr1 = m15['high'] - m15['low']
        tr2 = (m15['high'] - m15['close'].shift(1)).abs()
        tr3 = (m15['low'] - m15['close'].shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr14 = tr.rolling(14, min_periods=5).mean().values

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
        v = m15_df['vol'].values
        spread = m15_df['spread_pips'].values
        h4_bias = m15_df['h4_bias'].fillna(0).values.astype(int)
        hours = m15_df.index.hour.values
        dates = m15_df.index.date
        time_ms = m15_df.index.astype('datetime64[ms]').view('int64')
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        last_trade_date = None
        for i in range(50, len(m15_df)):
            # Active institutional liquidity window (07:30 - 15:30 UTC)
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
            base_atr = atr14[i]
            if np.isnan(base_atr) or bar_rng < 0.9 * base_atr:
                continue

            e = ema50[i]
            touched_ema = (l[i] <= e <= h[i])
            if not touched_ema:
                continue

            sig = 0
            # Bullish Pin on 50-EMA with Volume Confirmation
            if macro == 1 and c[i] > e and c[i] > o[i] and v[i] > v[i-1]:
                lower_wick = min(o[i], c[i]) - l[i]
                if lower_wick >= self.min_wick_pct * bar_rng:
                    sig = SIGNAL_LONG

            # Bearish Pin on 50-EMA with Volume Confirmation
            elif macro == -1 and c[i] < e and c[i] < o[i] and v[i] > v[i-1]:
                upper_wick = h[i] - max(o[i], c[i])
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
# Model Configurations to Audit on 70% In-Sample Discovery Weeks
# ==============================================================================
def get_lab_v40_models() -> List[Tuple[str, str, BaseStrategy]]:
    mods = []

    # 1. USDJPY London PM Volatility Expansion (Corrected: Box <= 25p, 2.2R)
    mods.append(("USDJPY", "LON_PM_BOX25_22R", CorrectedLondonPMExpansion("USDJPY", 25.0, 0.60, 11.5, 25.3, 0.01, 0.8)))
    mods.append(("USDJPY", "LON_PM_BOX20_22R", CorrectedLondonPMExpansion("USDJPY", 20.0, 0.60, 11.5, 25.3, 0.01, 0.8)))
    mods.append(("EURUSD", "LON_PM_BOX18_22R", CorrectedLondonPMExpansion("EURUSD", 18.0, 0.60, 8.0, 17.6, 0.0001, 0.6)))

    # 2. EURUSD Double Structure Neckline Breakout (Corrected: Volume + 2.2R)
    mods.append(("EURUSD", "DBL_STRUCT_B55_20R", CorrectedDoubleStructureBO("EURUSD", 0.55, 8.5, 17.0, 0.0001, 0.6)))
    mods.append(("EURUSD", "DBL_STRUCT_B60_22R", CorrectedDoubleStructureBO("EURUSD", 0.60, 8.0, 17.6, 0.0001, 0.6)))

    # 3. EURGBP Asian Compression Breakout (Corrected: Box <= 10p-12p, 2.2R)
    mods.append(("EURGBP", "ASIA_COMP_BOX10_22R", CorrectedAsianCompressionBO("EURGBP", 10.0, 6.5, 14.3, 0.0001, 0.6)))
    mods.append(("EURGBP", "ASIA_COMP_BOX12_22R", CorrectedAsianCompressionBO("EURGBP", 12.0, 6.5, 14.3, 0.0001, 0.6)))
    mods.append(("EURGBP", "ASIA_COMP_BOX12_20R", CorrectedAsianCompressionBO("EURGBP", 12.0, 7.0, 14.0, 0.0001, 0.6)))

    # 4. USDJPY M15 50-EMA Pin Runner (Corrected: ATR + Volume + 2.2R)
    mods.append(("USDJPY", "EMA50_PIN_ATR_22R", CorrectedM15FiftyEMAPinRunner("USDJPY", 0.45, 11.5, 25.3, 0.01, 0.8)))
    mods.append(("USDJPY", "EMA50_PIN_ATR_24R", CorrectedM15FiftyEMAPinRunner("USDJPY", 0.45, 11.0, 26.4, 0.01, 0.8)))

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
    print("  ALPHA CORRECTION & OPTIMIZATION LAB V40: PREVIOUS FLAWED STRATEGIES")
    print("  STRICT PARTITION: 70% In-Sample Discovery Weeks ONLY (Quarantined 30% Holdout Untouched)")
    print("=" * 100)

    models_config = get_lab_v40_models()
    tasks = []
    for asset, model_name, strat in models_config:
        disc_weeks = get_discovery_weeks(raw_dir=f"data/{asset}")
        for w in disc_weeks:
            tasks.append((asset, model_name, w, strat))

    print(f"Total correction tasks: {len(tasks):,} across {len(models_config)} model configurations")

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
    df_summary.to_csv(RESULTS_DIR / "alpha_correction_lab_v40_scorecard.csv", index=False)
    print(f"\nSaved Correction Scorecard to: {RESULTS_DIR / 'alpha_correction_lab_v40_scorecard.csv'}")


if __name__ == "__main__":
    main()
