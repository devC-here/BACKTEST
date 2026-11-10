#!/usr/bin/env python3
"""
================================================================================
  ALPHA DISCOVERY & OPTIMIZATION LAB V44:
  Evaluating 6 Promising Architectures on 70% In-Sample Discovery Weeks
================================================================================
Target Candidates:
  1. USDJPY_DOUBLE_STRUCT_BO_20R & 22R
  2. EURUSD_DOUBLE_STRUCT_BO_22R
  3. USDJPY_LON_PM_EXPANSION_22R
  4. EURUSD_LONDON_IGNITION_CAUSAL_22R
  5. USDJPY_LONDON_IGNITION_CAUSAL_22R
  6. EURGBP_DAILY_PIVOT_WALL_M15_22R

Strict Methodology:
  - Executed EXCLUSIVELY on 70% In-Sample Discovery Weeks (Deterministic Seed 42).
  - Quarantined 30% Holdout is 100% UNTOUCHED during optimization!
  - 1.0% Risk Per Trade Model (Prop/Fund Standard).
  - Passing Bar: Annualized Return >= 15.0% AND Profit Factor >= 1.25.
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
# Model 1: Double Structure Neckline Breakout (Fully Causal MTF)
# ==============================================================================
class DoubleStructureNecklineBreakout(BaseStrategy):
    def __init__(self, asset: str, sl_pips: float, tp_pips: float, pip_size: float, max_spread: float, **kwargs):
        super().__init__(**kwargs)
        self.asset = asset
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.pip_size = float(pip_size)
        self.max_spread = float(max_spread)
        self.name = f"{asset}_Double_Structure_R{tp_pips/sl_pips:.1f}"

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
        spread = m15_df['spread_pips'].values
        h4_bias = m15_df['h4_bias'].fillna(0).values.astype(int)
        hours = m15_df.index.hour.values
        dates = m15_df.index.date
        time_ms = m15_df.index.astype('datetime64[ms]').view('int64')
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        last_trade_date = None
        for i in range(40, len(m15_df)):
            if not (7 <= hours[i] <= 16):
                continue
            dt = dates[i]
            if dt == last_trade_date:
                continue
            if spread[i] > self.max_spread:
                continue

            macro = h4_bias[i]
            if macro == 0:
                continue

            w_l = l[i-35:i-5]
            w_h = h[i-35:i-5]

            mid_pt = len(w_l) // 2
            min1 = np.min(w_l[:mid_pt])
            min2 = np.min(w_l[mid_pt:])
            neck_hi = np.max(w_h)

            diff_lo_pips = abs(min1 - min2) / self.pip_size
            neck_depth = (neck_hi - max(min1, min2)) / self.pip_size

            max1 = np.max(w_h[:mid_pt])
            max2 = np.max(w_h[mid_pt:])
            neck_lo = np.min(w_l)

            diff_hi_pips = abs(max1 - max2) / self.pip_size
            neck_height = (min(max1, max2) - neck_lo) / self.pip_size

            sig = 0
            if macro == 1 and (diff_lo_pips <= 4.0) and (neck_depth >= 8.0) and (c[i-1] <= neck_hi) and (c[i] > neck_hi):
                sig = SIGNAL_LONG
            elif macro == -1 and (diff_hi_pips <= 4.0) and (neck_height >= 8.0) and (c[i-1] >= neck_lo) and (c[i] < neck_lo):
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
# Model 2: London PM Volatility Expansion (Causal H1 MTF)
# ==============================================================================
class LondonPMVolatilityExpansion(BaseStrategy):
    def __init__(self, asset: str, sl_pips: float, tp_pips: float, pip_size: float, max_spread: float, **kwargs):
        super().__init__(**kwargs)
        self.asset = asset
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.pip_size = float(pip_size)
        self.max_spread = float(max_spread)
        self.name = f"{asset}_London_PM_Exp_R{tp_pips/sl_pips:.1f}"

    def generate_signals(self, ticks: np.ndarray):
        n = len(ticks)
        signals = np.zeros(n, dtype=np.int8)
        sl_arr = np.zeros(n, dtype=np.float64)
        tp_arr = np.zeros(n, dtype=np.float64)
        if n < 500:
            return signals, sl_arr, tp_arr

        m15 = build_time_bars(ticks, timeframe_min=15, pip_size=self.pip_size)
        h1 = build_time_bars(ticks, timeframe_min=60, pip_size=self.pip_size)
        if len(m15) < 30 or len(h1) < 15:
            return signals, sl_arr, tp_arr

        h1_c = pd.Series(h1['close'].values)
        h1_ema20 = h1_c.ewm(span=20, adjust=False).mean().values
        h1_trend = np.where(h1_c.values > h1_ema20, 1, -1)
        h1_df = pd.DataFrame({'h1_bias': h1_trend}, index=h1.index)

        m15_df = pd.merge_asof(
            m15.sort_index(),
            h1_df.sort_index(),
            left_index=True,
            right_index=True,
            direction='backward'
        )

        c = m15_df['close'].values
        h = m15_df['high'].values
        l = m15_df['low'].values
        o = m15_df['open'].values
        spread = m15_df['spread_pips'].values
        h1_bias = m15_df['h1_bias'].fillna(0).values.astype(int)
        hours = m15_df.index.hour.values
        mins = m15_df.index.minute.values
        dates = m15_df.index.date
        time_ms = m15_df.index.astype('datetime64[ms]').view('int64')
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        unique_dates = np.unique(dates)
        pre_us_ranges = {}
        for ud in unique_dates:
            mask = (dates == ud) & ((hours == 11) | ((hours == 12) & (mins < 30)))
            if np.any(mask):
                p_hi = np.max(h[mask])
                p_lo = np.min(l[mask])
                rng_p = (p_hi - p_lo) / self.pip_size
                if 4.0 <= rng_p <= 30.0:
                    pre_us_ranges[ud] = (p_hi, p_lo)

        last_trade_date = None
        for i in range(20, len(m15_df)):
            t_hour = hours[i]
            t_min = mins[i]
            if not ((t_hour == 12 and t_min >= 30) or (13 <= t_hour <= 15)):
                continue

            dt = dates[i]
            if dt == last_trade_date:
                continue
            if dt not in pre_us_ranges:
                continue
            if spread[i] > self.max_spread:
                continue

            macro = h1_bias[i]
            p_hi, p_lo = pre_us_ranges[dt]

            sig = 0
            if macro == 1 and (c[i] > p_hi) and (c[i] > o[i]):
                sig = SIGNAL_LONG
            elif macro == -1 and (c[i] < p_lo) and (c[i] < o[i]):
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
# Model 3: London Ignition Range Breakout (Strict Causal H4 MTF)
# ==============================================================================
class LondonIgnitionBreakoutCausal(BaseStrategy):
    """
    Measures the 07:00 - 07:30 UTC London opening 30-minute range.
    During 07:30 - 09:30 UTC:
    First M15 expansion candle closing decisively beyond ignition range in H4 20-EMA trend.
    Target: 2.2R.
    """
    def __init__(self, asset: str, sl_pips: float, tp_pips: float, pip_size: float, max_spread: float, **kwargs):
        super().__init__(**kwargs)
        self.asset = asset
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.pip_size = float(pip_size)
        self.max_spread = float(max_spread)
        self.name = f"{asset}_London_Ignition_R{tp_pips/sl_pips:.1f}"

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

        # Strictly Causal H4 Trend
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

        # Precompute London ignition range (07:00 to 07:30 UTC)
        ignition_ranges = {}
        for d in np.unique(dates):
            mask = (dates == d) & (hours == 7) & (mins < 30)
            if np.any(mask):
                ih = np.max(h[mask])
                il = np.min(l[mask])
                irng = (ih - il) / self.pip_size
                ignition_ranges[d] = (ih, il, irng)

        traded_days = set()
        max_irng = 30.0 if self.asset == "USDJPY" else 20.0

        for i in range(len(m15_df)):
            in_window = (hours[i] == 7 and mins[i] >= 30) or (hours[i] == 8) or (hours[i] == 9 and mins[i] <= 30)
            if not in_window:
                continue
            d = dates[i]
            if d not in ignition_ranges or d in traded_days:
                continue
            if spread[i] > self.max_spread:
                continue

            ih, il, irng = ignition_ranges[d]
            if irng < 3.5 or irng > max_irng:
                continue

            macro = h4_bias[i]
            if macro == 0:
                continue

            bar_rng = h[i] - l[i]
            if bar_rng <= 0:
                continue

            sig = 0
            if macro == 1 and c[i] > ih and c[i] > o[i]:
                sig = SIGNAL_LONG
            elif macro == -1 and c[i] < il and c[i] < o[i]:
                sig = SIGNAL_SHORT

            if sig != 0:
                bar_close_ms = time_ms[i] + dur_ms
                idx = np.searchsorted(tick_ts, bar_close_ms)
                if idx < n:
                    signals[idx] = sig
                    sl_arr[idx] = self.sl_pips
                    tp_arr[idx] = self.tp_pips
                    traded_days.add(d)

        return signals, sl_arr, tp_arr


# ==============================================================================
# Model 4: EURGBP Daily Pivot Wall M15 Rebound (2.2R Asymmetric Target)
# ==============================================================================
class EURGBPDailyPivotWallM15(BaseStrategy):
    """
    Computes prior day classic Pivot S1 and R1:
      Pivot = (High + Low + Close) / 3
      S1 = 2 * Pivot - High
      R1 = 2 * Pivot - Low
    When price tests S1 or R1 during 07:00 - 15:30 UTC:
      - Tests level within 2.5 pips and prints a strong rejection wick >= 35%.
      - 2.2R Asymmetric Target: SL 6.5 pips, TP 14.3 pips.
    """
    def __init__(self, asset: str = "EURGBP", sl_pips: float = 6.5, tp_pips: float = 14.3, pip_size: float = 0.0001, max_spread: float = 0.6, **kwargs):
        super().__init__(**kwargs)
        self.asset = asset
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.pip_size = float(pip_size)
        self.max_spread = float(max_spread)
        self.name = f"{asset}_Daily_Pivot_Wall_M15_R{tp_pips/sl_pips:.1f}"

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

        daily_bars = {}
        for i in range(len(m15)):
            d = dates[i]
            if d not in daily_bars:
                daily_bars[d] = {"high": h[i], "low": l[i], "close": c[i]}
            else:
                daily_bars[d]["high"] = max(daily_bars[d]["high"], h[i])
                daily_bars[d]["low"] = min(daily_bars[d]["low"], l[i])
                daily_bars[d]["close"] = c[i]

        unique_dates = sorted(list(daily_bars.keys()))
        date_to_pivots = {}
        for idx_d in range(1, len(unique_dates)):
            curr_d = unique_dates[idx_d]
            prev_d = unique_dates[idx_d - 1]
            p_h = daily_bars[prev_d]["high"]
            p_l = daily_bars[prev_d]["low"]
            p_c = daily_bars[prev_d]["close"]
            pivot = (p_h + p_l + p_c) / 3.0
            r1 = (2.0 * pivot) - p_l
            s1 = (2.0 * pivot) - p_h
            date_to_pivots[curr_d] = (s1, r1)

        last_trade_date = None
        tol = 2.5 * self.pip_size

        for i in range(15, len(m15)):
            dt = dates[i]
            hr = hours[i]
            if not (7 <= hr <= 15):
                continue
            if dt == last_trade_date:
                continue
            if dt not in date_to_pivots:
                continue
            if spread[i] > self.max_spread:
                continue

            s1, r1 = date_to_pivots[dt]
            bar_rng = h[i] - l[i]
            if bar_rng <= 0:
                continue

            sig = 0
            # Test S1 floor with lower wick rejection >= 35%
            dist_s1 = abs(l[i] - s1)
            lower_wick = min(c[i], o[i]) - l[i]
            if dist_s1 <= tol and c[i] > s1 and (lower_wick / bar_rng) >= 0.35 and c[i] > o[i]:
                sig = SIGNAL_LONG

            # Test R1 ceiling with upper wick rejection >= 35%
            dist_r1 = abs(h[i] - r1)
            upper_wick = h[i] - max(c[i], o[i])
            if dist_r1 <= tol and c[i] < r1 and (upper_wick / bar_rng) >= 0.35 and c[i] < o[i]:
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
# Model Registry for Lab V44
# ==============================================================================
def get_lab_v44_models() -> List[Tuple[str, str, BaseStrategy, float]]:
    mods = []

    # 1. USDJPY Double Structure Neckline Breakout (2.0R and 2.2R)
    mods.append(("USDJPY", "DOUBLE_STRUCT_20R", DoubleStructureNecklineBreakout("USDJPY", 12.0, 24.0, 0.01, 0.8), 12.0))
    mods.append(("USDJPY", "DOUBLE_STRUCT_22R", DoubleStructureNecklineBreakout("USDJPY", 11.5, 25.3, 0.01, 0.8), 11.5))

    # 2. EURUSD Double Structure Neckline Breakout (2.2R)
    mods.append(("EURUSD", "DOUBLE_STRUCT_22R", DoubleStructureNecklineBreakout("EURUSD", 8.0, 17.6, 0.0001, 0.6), 8.0))

    # 3. USDJPY London PM Volatility Expansion (2.0R and 2.2R)
    mods.append(("USDJPY", "LON_PM_EXP_20R", LondonPMVolatilityExpansion("USDJPY", 12.0, 24.0, 0.01, 0.8), 12.0))
    mods.append(("USDJPY", "LON_PM_EXP_22R", LondonPMVolatilityExpansion("USDJPY", 11.5, 25.3, 0.01, 0.8), 11.5))

    # 4. EURUSD & USDJPY London Ignition Breakout (Causal MTF)
    mods.append(("EURUSD", "LON_IGNITION_22R", LondonIgnitionBreakoutCausal("EURUSD", 8.0, 17.6, 0.0001, 0.6), 8.0))
    mods.append(("USDJPY", "LON_IGNITION_22R", LondonIgnitionBreakoutCausal("USDJPY", 11.5, 25.3, 0.01, 0.8), 11.5))

    # 5. EURGBP Daily Pivot Wall M15 Rebound (2.2R)
    mods.append(("EURGBP", "PIVOT_WALL_M15_22R", EURGBPDailyPivotWallM15("EURGBP", 6.5, 14.3, 0.0001, 0.6), 6.5))

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


def calculate_metrics(pips: List[float], mult: float, sl_pips: float) -> Dict[str, Any]:
    if not pips:
        return {"trades": 0, "win_rate": 0.0, "pf": 0.0, "net_pips": 0.0, "norm_pips": 0.0, "net_r": 0.0, "annual_return_pct": 0.0, "max_dd": 0.0, "expectancy": 0.0}
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

    net_r = net_pips / sl_pips if sl_pips > 0 else 0.0
    years_discovery = 3.5
    annual_r = net_r / years_discovery
    annual_return_pct = annual_r * 1.0  # 1% risk per trade -> 1R = 1%

    equity = np.cumsum(arr * mult)
    peak = np.maximum.accumulate(equity)
    dd = peak - equity
    max_dd = float(np.max(dd)) if len(dd) > 0 else 0.0

    return {
        "trades": trades,
        "win_rate": round(wr, 1),
        "pf": round(pf, 2),
        "net_pips": round(net_pips, 1),
        "norm_pips": round(norm_pips, 1),
        "net_r": round(net_r, 1),
        "annual_return_pct": round(annual_return_pct, 1),
        "max_dd": round(max_dd, 1),
        "expectancy": round(expectancy, 2)
    }


def main():
    t0 = time.time()
    models = get_lab_v44_models()
    print("=" * 115)
    print("  ALPHA DISCOVERY & OPTIMIZATION LAB V44: SUB-1.25 PF CANDIDATE OPTIMIZATION")
    print("  Strictly on 70% In-Sample Discovery Weeks | 1.0% Risk / Trade | Hurdle >= 15% Annual Ret & PF >= 1.25")
    print("=" * 115)

    all_tasks = []
    for asset, m_name, strat, sl_p in models:
        weeks = get_discovery_weeks(raw_dir=f"data/{asset}")
        for w in weeks:
            all_tasks.append((asset, m_name, w, strat))

    print(f"Total Model-Week Tasks: {len(all_tasks)} across {len(models)} configurations")

    trade_pool: Dict[str, List[float]] = {f"{a}_{m}": [] for a, m, _, _ in models}

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(run_week_task, t) for t in all_tasks]
        done = 0
        total = len(futures)
        for fut in as_completed(futures):
            res = fut.result()
            key = f"{res['asset']}_{res['model']}"
            for tr in res['trades']:
                trade_pool[key].append(tr['pips'])
            done += 1
            if done % 300 == 0 or done == total:
                print(f"  Progress: {done}/{total} tasks complete ({done/total*100:.1f}%) in {time.time()-t0:.1f}s", flush=True)

    print("\n" + "=" * 125)
    print(f"{'Asset':<8} {'Model Variant':<25} {'Trades':<8} {'WR%':<7} {'PF':<7} {'Exp(p)':<8} {'Raw Pips':<11} {'Norm Pips':<11} {'Net R':<9} {'Annual Ret%':<13} {'Max DD(p)':<10} {'Verdict'}")
    print("-" * 125)

    records = []
    mult_map = {"EURUSD": EURUSD_MULT, "USDJPY": USDJPY_MULT, "EURGBP": EURGBP_MULT}

    for asset, m_name, strat, sl_p in models:
        key = f"{asset}_{m_name}"
        pips = trade_pool[key]
        mult = mult_map[asset]
        metrics = calculate_metrics(pips, mult, sl_p)

        verdict = "[REJECT]"
        if metrics["pf"] >= 1.25 and metrics["annual_return_pct"] >= 15.0:
            verdict = "[>= 15% VIABLE CHAMPION]"
        elif metrics["pf"] >= 1.20 and metrics["annual_return_pct"] >= 10.0:
            verdict = "[PROMISING]"

        print(f"{asset:<8} {m_name:<25} {metrics['trades']:<8} {metrics['win_rate']:<7} {metrics['pf']:<7} {metrics['expectancy']:<8} {metrics['net_pips']:<11} {metrics['norm_pips']:<11} {metrics['net_r']:<9} {metrics['annual_return_pct']:<13} {metrics['max_dd']:<10} {verdict}")

        records.append({
            "asset": asset,
            "model_variant": m_name,
            "trades": metrics["trades"],
            "win_rate": metrics["win_rate"],
            "profit_factor": metrics["pf"],
            "expectancy": metrics["expectancy"],
            "raw_pips": metrics["net_pips"],
            "normalized_pips": metrics["norm_pips"],
            "net_r": metrics["net_r"],
            "annual_return_pct": metrics["annual_return_pct"],
            "max_dd_norm": metrics["max_dd"],
            "verdict": verdict
        })

    out_csv = RESULTS_DIR / "alpha_discovery_lab_v44_scorecard.csv"
    pd.DataFrame(records).to_csv(out_csv, index=False)
    print(f"\nSaved Scorecard to: {out_csv.resolve()}")
    print(f"Total Execution Time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()

