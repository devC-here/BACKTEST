#!/usr/bin/env python3
"""
================================================================================
  ALPHA DISCOVERY LAB AUDNZD: CROSS-CURRENCY SPECIALIST ALPHA MINING
================================================================================
Targets the unique structural properties of AUDNZD (low-beta cross, mean-reverting,
cointegrated fundamentals, Pacific/Asian & London liquidity) using proven
cross-specialist architectures from EURGBP and USDJPY:
  1. Daily Pivot Wall Rejection (DISC46 adapted)
  2. Weekly Pivot Wall Rejection (DISC47 adapted)
  3. Camarilla Floor / Ceiling Rejection (DISC48 adapted)
  4. Fibonacci Golden Pocket Pin Defense (DISC52 adapted)
  5. Multi-Period H4/H2 Structure Retests (DISC42/DISC111 variations)
  6. VWAP 2.2-SD Volatility Band Fade with Wick Filter (DISC65 adapted)
  7. High-Expectancy Triple Screen Filters (DISC56 adapted)

Evaluated strictly on the 70% Quarantined Discovery Partition
(210 weeks / 97.08M authentic Dukascopy ticks).
The 30% Holdout Test Partition (90 weeks) remains 100% strictly quarantined.
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

# OpenBLAS thread control
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["NUMBA_NUM_THREADS"] = "1"

sys.dont_write_bytecode = True

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tick_engine.core.engine import run_matching_loop
from tick_engine.core.dataloader import get_week_dataset, get_discovery_weeks
from tick_engine.strategies.base import BaseStrategy, SIGNAL_LONG, SIGNAL_SHORT
from data_engineering_suite.bars import build_time_bars

PIP_SIZE = 0.0001
MAX_SPREAD = 2.8
RESULTS_DIR = ROOT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


class BaseCrossStrategy(BaseStrategy):
    def __init__(self, name: str, sl_pips: float, tp_pips: float, **kwargs):
        super().__init__(**kwargs)
        self.name = name
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)

    def generate_signals(self, ticks: np.ndarray):
        m15 = build_time_bars(ticks, timeframe_min=15, pip_size=PIP_SIZE)
        if len(m15) < 30:
            n = len(ticks)
            return np.zeros(n, dtype=np.int8), np.zeros(n, dtype=np.float64), np.zeros(n, dtype=np.float64)
        h1 = m15.resample('1h').agg({'high': 'max', 'low': 'min', 'close': 'last', 'open': 'first'}).dropna()
        h4 = m15.resample('4h').agg({'high': 'max', 'low': 'min', 'close': 'last', 'open': 'first'}).dropna()
        return self.generate_signals_from_bars(ticks, m15, h1, h4)

    def generate_signals_from_bars(self, ticks: np.ndarray, m15: pd.DataFrame, h1: pd.DataFrame, h4: pd.DataFrame):
        raise NotImplementedError


# ==============================================================================
# 1. Daily Pivot Wall Rejection (DISC46 adapted)
# ==============================================================================
class AUDNZDDailyPivotWall(BaseCrossStrategy):
    def __init__(self, name: str, sl_pips: float, tp_pips: float, min_wick: float = 0.45, max_dist_pips: float = 4.5, **kwargs):
        super().__init__(name=name, sl_pips=sl_pips, tp_pips=tp_pips, **kwargs)
        self.min_wick = float(min_wick)
        self.max_dist_pips = float(max_dist_pips)

    def generate_signals_from_bars(self, ticks: np.ndarray, m15: pd.DataFrame, h1: pd.DataFrame, h4: pd.DataFrame):
        n = len(ticks)
        signals = np.zeros(n, dtype=np.int8)
        sl_arr = np.zeros(n, dtype=np.float64)
        tp_arr = np.zeros(n, dtype=np.float64)

        if len(m15) < 30 or len(h4) < 15:
            return signals, sl_arr, tp_arr

        # Daily resample from M15
        daily = m15.resample('1D').agg({'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
        if len(daily) < 2:
            return signals, sl_arr, tp_arr

        prev_h = daily['high'].shift(1)
        prev_l = daily['low'].shift(1)
        prev_c = daily['close'].shift(1)
        p = (prev_h + prev_l + prev_c) / 3.0
        r1 = (2 * p) - prev_l
        s1 = (2 * p) - prev_h

        daily['s1'] = s1
        daily['r1'] = r1
        daily['pivot'] = p

        h4_ema20 = h4['close'].ewm(span=20, adjust=False).mean()
        h4_macro = np.where(h4['close'] > h4_ema20, 1, -1)

        m15_s1 = daily['s1'].reindex(m15.index, method='ffill').values
        m15_r1 = daily['r1'].reindex(m15.index, method='ffill').values
        m15_macro = pd.Series(h4_macro, index=h4.index).shift(1).reindex(m15.index, method='ffill').fillna(0).values

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

        for i in range(15, len(m15)):
            hr = hours[i]
            if not (hr >= 21 or hr <= 16):
                continue
            if spread[i] > MAX_SPREAD:
                continue

            dt = dates[i]
            if dt == last_trade_date:
                continue

            m = m15_macro[i]
            bar_rng = (h[i] - l[i]) / PIP_SIZE
            if bar_rng < 2.5:
                continue

            sig = 0
            # S1 Floor Rejection in Macro Uptrend
            if m > 0 and not np.isnan(m15_s1[i]):
                dist_s1 = abs(l[i] - m15_s1[i]) / PIP_SIZE
                lower_wick = (min(o[i], c[i]) - l[i]) / PIP_SIZE
                if dist_s1 <= self.max_dist_pips and (lower_wick / bar_rng) >= self.min_wick and c[i] > o[i]:
                    sig = SIGNAL_LONG

            # R1 Ceiling Rejection in Macro Downtrend
            elif m < 0 and not np.isnan(m15_r1[i]):
                dist_r1 = abs(h[i] - m15_r1[i]) / PIP_SIZE
                upper_wick = (h[i] - max(o[i], c[i])) / PIP_SIZE
                if dist_r1 <= self.max_dist_pips and (upper_wick / bar_rng) >= self.min_wick and c[i] < o[i]:
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
# 2. Weekly Pivot Wall Rejection (DISC47 adapted)
# ==============================================================================
class AUDNZDWeeklyPivotWall(BaseCrossStrategy):
    def __init__(self, name: str, sl_pips: float, tp_pips: float, min_wick: float = 0.45, max_dist_pips: float = 5.0, **kwargs):
        super().__init__(name=name, sl_pips=sl_pips, tp_pips=tp_pips, **kwargs)
        self.min_wick = float(min_wick)
        self.max_dist_pips = float(max_dist_pips)

    def generate_signals_from_bars(self, ticks: np.ndarray, m15: pd.DataFrame, h1: pd.DataFrame, h4: pd.DataFrame):
        n = len(ticks)
        signals = np.zeros(n, dtype=np.int8)
        sl_arr = np.zeros(n, dtype=np.float64)
        tp_arr = np.zeros(n, dtype=np.float64)

        if len(m15) < 30 or len(h4) < 15:
            return signals, sl_arr, tp_arr

        # Resample Weekly from M15
        weekly = m15.resample('W').agg({'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
        if len(weekly) < 2:
            return signals, sl_arr, tp_arr

        prev_h = weekly['high'].shift(1)
        prev_l = weekly['low'].shift(1)
        prev_c = weekly['close'].shift(1)
        p = (prev_h + prev_l + prev_c) / 3.0
        r1 = (2 * p) - prev_l
        s1 = (2 * p) - prev_h

        weekly['s1'] = s1
        weekly['r1'] = r1
        weekly['pivot'] = p

        h4_ema20 = h4['close'].ewm(span=20, adjust=False).mean()
        h4_macro = np.where(h4['close'] > h4_ema20, 1, -1)

        m15_s1 = weekly['s1'].reindex(m15.index, method='ffill').values
        m15_r1 = weekly['r1'].reindex(m15.index, method='ffill').values
        m15_p = weekly['pivot'].reindex(m15.index, method='ffill').values
        m15_macro = pd.Series(h4_macro, index=h4.index).shift(1).reindex(m15.index, method='ffill').fillna(0).values

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

        for i in range(15, len(m15)):
            hr = hours[i]
            if not (hr >= 21 or hr <= 16):
                continue
            if spread[i] > MAX_SPREAD:
                continue

            dt = dates[i]
            if dt == last_trade_date:
                continue

            m = m15_macro[i]
            bar_rng = (h[i] - l[i]) / PIP_SIZE
            if bar_rng < 2.5:
                continue

            sig = 0
            if m > 0 and not np.isnan(m15_s1[i]):
                dist_s1 = abs(l[i] - m15_s1[i]) / PIP_SIZE
                dist_p = abs(l[i] - m15_p[i]) / PIP_SIZE
                lower_wick = (min(o[i], c[i]) - l[i]) / PIP_SIZE
                if (dist_s1 <= self.max_dist_pips or dist_p <= self.max_dist_pips) and (lower_wick / bar_rng) >= self.min_wick and c[i] > o[i]:
                    sig = SIGNAL_LONG

            elif m < 0 and not np.isnan(m15_r1[i]):
                dist_r1 = abs(h[i] - m15_r1[i]) / PIP_SIZE
                dist_p = abs(h[i] - m15_p[i]) / PIP_SIZE
                upper_wick = (h[i] - max(o[i], c[i])) / PIP_SIZE
                if (dist_r1 <= self.max_dist_pips or dist_p <= self.max_dist_pips) and (upper_wick / bar_rng) >= self.min_wick and c[i] < o[i]:
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
# 3. Camarilla Floor / Ceiling Rejection (DISC48 adapted)
# ==============================================================================
class AUDNZDCamarillaFloor(BaseCrossStrategy):
    def __init__(self, name: str, sl_pips: float, tp_pips: float, min_wick: float = 0.45, max_dist_pips: float = 4.0, **kwargs):
        super().__init__(name=name, sl_pips=sl_pips, tp_pips=tp_pips, **kwargs)
        self.min_wick = float(min_wick)
        self.max_dist_pips = float(max_dist_pips)

    def generate_signals_from_bars(self, ticks: np.ndarray, m15: pd.DataFrame, h1: pd.DataFrame, h4: pd.DataFrame):
        n = len(ticks)
        signals = np.zeros(n, dtype=np.int8)
        sl_arr = np.zeros(n, dtype=np.float64)
        tp_arr = np.zeros(n, dtype=np.float64)

        if len(m15) < 30 or len(h4) < 15:
            return signals, sl_arr, tp_arr

        daily = m15.resample('1D').agg({'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
        if len(daily) < 2:
            return signals, sl_arr, tp_arr

        prev_h = daily['high'].shift(1)
        prev_l = daily['low'].shift(1)
        prev_c = daily['close'].shift(1)
        rng = prev_h - prev_l

        # Camarilla L3 / H3
        daily['l3'] = prev_c - rng * (1.1 / 4.0)
        daily['h3'] = prev_c + rng * (1.1 / 4.0)

        h4_ema20 = h4['close'].ewm(span=20, adjust=False).mean()
        h4_macro = np.where(h4['close'] > h4_ema20, 1, -1)

        m15_l3 = daily['l3'].reindex(m15.index, method='ffill').values
        m15_h3 = daily['h3'].reindex(m15.index, method='ffill').values
        m15_macro = pd.Series(h4_macro, index=h4.index).shift(1).reindex(m15.index, method='ffill').fillna(0).values

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

        for i in range(15, len(m15)):
            hr = hours[i]
            if not (hr >= 21 or hr <= 16):
                continue
            if spread[i] > MAX_SPREAD:
                continue

            dt = dates[i]
            if dt == last_trade_date:
                continue

            m = m15_macro[i]
            bar_rng = (h[i] - l[i]) / PIP_SIZE
            if bar_rng < 2.5:
                continue

            sig = 0
            if m > 0 and not np.isnan(m15_l3[i]):
                dist_l3 = abs(l[i] - m15_l3[i]) / PIP_SIZE
                lower_wick = (min(o[i], c[i]) - l[i]) / PIP_SIZE
                if dist_l3 <= self.max_dist_pips and (lower_wick / bar_rng) >= self.min_wick and c[i] > o[i]:
                    sig = SIGNAL_LONG

            elif m < 0 and not np.isnan(m15_h3[i]):
                dist_h3 = abs(h[i] - m15_h3[i]) / PIP_SIZE
                upper_wick = (h[i] - max(o[i], c[i])) / PIP_SIZE
                if dist_h3 <= self.max_dist_pips and (upper_wick / bar_rng) >= self.min_wick and c[i] < o[i]:
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
# 4. Fibonacci Golden Pocket Pin Defense (DISC52 adapted)
# ==============================================================================
class AUDNZDFibGoldenPocket(BaseCrossStrategy):
    def __init__(self, name: str, sl_pips: float, tp_pips: float, lookback_h1: int = 36, min_wick: float = 0.45, **kwargs):
        super().__init__(name=name, sl_pips=sl_pips, tp_pips=tp_pips, **kwargs)
        self.lookback_h1 = int(lookback_h1)
        self.min_wick = float(min_wick)

    def generate_signals_from_bars(self, ticks: np.ndarray, m15: pd.DataFrame, h1: pd.DataFrame, h4: pd.DataFrame):
        n = len(ticks)
        signals = np.zeros(n, dtype=np.int8)
        sl_arr = np.zeros(n, dtype=np.float64)
        tp_arr = np.zeros(n, dtype=np.float64)

        if len(m15) < 30 or len(h1) < self.lookback_h1 + 5 or len(h4) < 15:
            return signals, sl_arr, tp_arr

        h4_ema20 = h4['close'].ewm(span=20, adjust=False).mean()
        h4_macro = np.where(h4['close'] > h4_ema20, 1, -1)

        # Rolling high and low on H1
        roll_h = h1['high'].rolling(self.lookback_h1).max().shift(1).bfill()
        roll_l = h1['low'].rolling(self.lookback_h1).min().shift(1).bfill()
        swing_range = roll_h - roll_l

        # Golden Pocket: 61.8% to 78.6%
        fib618_long = roll_h - 0.618 * swing_range
        fib786_long = roll_h - 0.786 * swing_range

        fib618_short = roll_l + 0.618 * swing_range
        fib786_short = roll_l + 0.786 * swing_range

        m15_fib618_l = fib618_long.reindex(m15.index, method='ffill').values
        m15_fib786_l = fib786_long.reindex(m15.index, method='ffill').values
        m15_fib618_s = fib618_short.reindex(m15.index, method='ffill').values
        m15_fib786_s = fib786_short.reindex(m15.index, method='ffill').values
        m15_macro = pd.Series(h4_macro, index=h4.index).shift(1).reindex(m15.index, method='ffill').fillna(0).values

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

        for i in range(15, len(m15)):
            hr = hours[i]
            if not (hr >= 21 or hr <= 16):
                continue
            if spread[i] > MAX_SPREAD:
                continue

            dt = dates[i]
            if dt == last_trade_date:
                continue

            m = m15_macro[i]
            bar_rng = (h[i] - l[i]) / PIP_SIZE
            if bar_rng < 2.5:
                continue

            sig = 0
            # Long Golden Pocket Defense
            if m > 0:
                f618 = m15_fib618_l[i]
                f786 = m15_fib786_l[i]
                # Price touches or dips between 61.8% and 78.6%
                if l[i] <= f618 and l[i] >= f786 - 3.0 * PIP_SIZE:
                    lower_wick = (min(o[i], c[i]) - l[i]) / PIP_SIZE
                    if (lower_wick / bar_rng) >= self.min_wick and c[i] > o[i]:
                        sig = SIGNAL_LONG

            # Short Golden Pocket Defense
            elif m < 0:
                f618 = m15_fib618_s[i]
                f786 = m15_fib786_s[i]
                if h[i] >= f618 and h[i] <= f786 + 3.0 * PIP_SIZE:
                    upper_wick = (h[i] - max(o[i], c[i])) / PIP_SIZE
                    if (upper_wick / bar_rng) >= self.min_wick and c[i] < o[i]:
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
# Model Factory
# ==============================================================================
def build_cross_specialist_candidates() -> List[Tuple[str, BaseCrossStrategy]]:
    candidates = []

    # 1. Daily Pivot Wall
    for sl, tp in [(10.0, 20.0), (12.0, 20.0), (12.0, 24.0), (14.0, 28.0)]:
        for wick in [0.45, 0.50]:
            name = f"AUDNZD_DailyPivotWall_SL{int(sl)}_TP{int(tp)}_Wick{int(wick*100)}"
            candidates.append((name, AUDNZDDailyPivotWall(name=name, sl_pips=sl, tp_pips=tp, min_wick=wick)))

    # 2. Weekly Pivot Wall
    for sl, tp in [(12.0, 24.0), (14.0, 28.0), (15.0, 30.0), (12.0, 20.0)]:
        for wick in [0.45, 0.50]:
            name = f"AUDNZD_WeeklyPivotWall_SL{int(sl)}_TP{int(tp)}_Wick{int(wick*100)}"
            candidates.append((name, AUDNZDWeeklyPivotWall(name=name, sl_pips=sl, tp_pips=tp, min_wick=wick)))

    # 3. Camarilla Floor / Ceiling
    for sl, tp in [(10.0, 20.0), (12.0, 20.0), (12.0, 24.0), (14.0, 28.0)]:
        for wick in [0.45, 0.50]:
            name = f"AUDNZD_CamarillaFloor_SL{int(sl)}_TP{int(tp)}_Wick{int(wick*100)}"
            candidates.append((name, AUDNZDCamarillaFloor(name=name, sl_pips=sl, tp_pips=tp, min_wick=wick)))

    # 4. Fibonacci Golden Pocket
    for sl, tp in [(12.0, 24.0), (14.0, 28.0), (12.0, 20.0), (10.0, 20.0)]:
        for lb in [24, 36, 48]:
            name = f"AUDNZD_FibGoldenPocket_SL{int(sl)}_TP{int(tp)}_LB{lb}"
            candidates.append((name, AUDNZDFibGoldenPocket(name=name, sl_pips=sl, tp_pips=tp, lookback_h1=lb)))

    return candidates


def evaluate_week_cross_lab(week_num: int) -> Dict[str, Any]:
    try:
        ticks, meta = get_week_dataset(week_num, raw_dir="data/AUDNZD")
        if len(ticks) == 0:
            return {"week": week_num, "trades": {}}
    except Exception as e:
        return {"week": week_num, "trades": {}, "error": str(e)}

    m15 = build_time_bars(ticks, timeframe_min=15, pip_size=PIP_SIZE)
    if len(m15) < 30:
        return {"week": week_num, "trades": {}}

    h1 = m15.resample('1h').agg({'high': 'max', 'low': 'min', 'close': 'last', 'open': 'first'}).dropna()
    h4 = m15.resample('4h').agg({'high': 'max', 'low': 'min', 'close': 'last', 'open': 'first'}).dropna()

    candidates = build_cross_specialist_candidates()
    week_trades = {}

    for name, model in candidates:
        try:
            signals, sl_arr, tp_arr = model.generate_signals_from_bars(ticks, m15, h1, h4)
            trades = run_matching_loop(
                ticks=ticks,
                signals=signals,
                pip_size=PIP_SIZE,
                sl_pips=sl_arr,
                tp_pips=tp_arr,
                use_slippage=True
            )
            week_trades[name] = [float(t['pips']) for t in trades]
        except Exception:
            week_trades[name] = []

    return {"week": week_num, "trades": week_trades}


def main():
    t_start = time.time()
    disc_weeks = get_discovery_weeks(raw_dir="data/AUDNZD")
    print(f"================================================================================")
    print(f"  ALPHA DISCOVERY LAB AUDNZD: SCREENING CROSS-CURRENCY SPECIALISTS")
    print(f"  Partition: 70% In-Sample Discovery ({len(disc_weeks)} weeks / ~97.08M ticks)")
    print(f"================================================================================")

    candidates = build_cross_specialist_candidates()
    print(f"Total Candidate Specialist Models: {len(candidates)}")

    all_week_trades: Dict[str, List[float]] = {name: [] for name, _ in candidates}

    max_workers = min(8, os.cpu_count() or 4)
    print(f"Launching evaluation across {max_workers} worker processes...")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(evaluate_week_cross_lab, w): w for w in disc_weeks}
        completed = 0
        for fut in as_completed(futures):
            res = fut.result()
            completed += 1
            if completed % 30 == 0 or completed == len(disc_weeks):
                elapsed = time.time() - t_start
                print(f"  [Progress] {completed}/{len(disc_weeks)} weeks completed ({elapsed:.1f}s elapsed)")
            trades_dict = res.get("trades", {})
            for m_name, pips in trades_dict.items():
                if m_name in all_week_trades:
                    all_week_trades[m_name].extend(pips)

    scorecard = []
    years = len(disc_weeks) / 52.0

    for name, pips in all_week_trades.items():
        if len(pips) == 0:
            continue
        pips_arr = np.array(pips)
        trades = len(pips_arr)
        wins = np.sum(pips_arr > 0)
        losses = np.sum(pips_arr < 0)
        win_rate = (wins / trades * 100.0) if trades > 0 else 0.0

        gross_win = np.sum(pips_arr[pips_arr > 0])
        gross_loss = abs(np.sum(pips_arr[pips_arr < 0]))
        pf = (gross_win / gross_loss) if gross_loss > 0 else (99.0 if gross_win > 0 else 0.0)

        net_pips = float(np.sum(pips_arr))
        ann_pips = net_pips / years if years > 0 else 0.0

        equity = np.cumsum(pips_arr)
        peak = np.maximum.accumulate(equity)
        dd = peak - equity
        max_dd = float(np.max(dd)) if len(dd) > 0 else 0.0
        ret_to_dd = (net_pips / max_dd) if max_dd > 0 else 0.0

        scorecard.append({
            "model": name,
            "trades": int(trades),
            "win_rate": round(float(win_rate), 1),
            "profit_factor": round(float(pf), 2),
            "net_pips": round(net_pips, 1),
            "annual_pips": round(ann_pips, 1),
            "max_dd_pips": round(max_dd, 1),
            "return_to_dd": round(ret_to_dd, 2),
            "expectancy_pips": round(float(np.mean(pips_arr)), 2)
        })

    scorecard.sort(key=lambda x: (x["profit_factor"], x["net_pips"]), reverse=True)

    print("\n" + "=" * 110)
    print(f"{'Model Name':<45} | {'Trades':<6} | {'WR %':<6} | {'PF':<5} | {'Net Pips':<9} | {'Ann Pips':<8} | {'Max DD':<7} | {'Ret/DD':<6} | {'Exp'}")
    print("-" * 110)
    for row in scorecard:
        print(f"{row['model']:<45} | {row['trades']:<6} | {row['win_rate']:<6.1f} | {row['profit_factor']:<5.2f} | {row['net_pips']:<+9.1f} | {row['annual_pips']:<+8.1f} | {row['max_dd_pips']:<7.1f} | {row['return_to_dd']:<6.2f} | {row['expectancy_pips']:+.2f}")
    print("=" * 110)

    out_file = RESULTS_DIR / "audnzd_cross_specialists_screening.json"
    with open(out_file, "w") as f:
        json.dump({
            "asset": "AUDNZD",
            "partition": "discovery_70",
            "total_weeks": len(disc_weeks),
            "years_equivalent": round(years, 2),
            "timestamp_utc": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "scorecard": scorecard
        }, f, indent=2)
    print(f"\n[Saved] Complete screening scorecard saved to: {out_file}")


if __name__ == "__main__":
    main()

