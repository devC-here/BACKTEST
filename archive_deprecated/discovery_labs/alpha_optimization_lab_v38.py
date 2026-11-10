#!/usr/bin/env python3
"""
================================================================================
  ALPHA OPTIMIZATION LAB V38:
  Systematic In-Sample Optimization of High-Potential Legacy Strategies
================================================================================
Target Architectures (Previously Generated +500 to +900 pips at 1.16 - 1.21 PF):
  1. OPENING_RANGE_IMPULSE (EURUSD baseline: +774.9 pips, 1.21 PF)
  2. M30_ENGULFING_RESUMPTION (USDJPY baseline: +926.6 pips, EURUSD: +522.2 pips)
  3. H1_INSIDE_BAR_EXPANSION (EURUSD baseline: +827.8 pips, 1.21 PF)

STRICT CONSTRAINT:
  Executes EXCLUSIVELY on the 70% In-Sample Discovery partition (get_discovery_weeks).
  ZERO touch or evaluation of the 30% quarantined holdout partition!

Goal:
  Optimize entry filters, session timing, volume confirmation, and asymmetric
  payoffs to elevate Profit Factor above 1.25 and maximize Net Pips.
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
# Model 1: Opening Range Impulse (Parametric Optimization)
# ==============================================================================
class ParametricOpeningRangeImpulse(BaseStrategy):
    def __init__(
        self,
        asset: str,
        session_mode: str,   # "LONDON", "NY", "BOTH"
        atr_mult: float,     # 1.3, 1.4, 1.5
        min_body_pct: float, # 0.55, 0.60
        sl_pips: float,
        tp_pips: float,
        pip_size: float,
        max_spread: float,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.asset = asset
        self.session_mode = session_mode
        self.atr_mult = float(atr_mult)
        self.min_body_pct = float(min_body_pct)
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.pip_size = float(pip_size)
        self.max_spread = float(max_spread)
        self.name = f"{asset}_ORI_{session_mode}_ATR{atr_mult}_R{tp_pips/sl_pips:.1f}"

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
        spread = m15_df['spread_pips'].values
        h4_trend = m15_df['h4_bias'].fillna(0).values.astype(int)
        hours = m15_df.index.hour.values
        minutes = m15_df.index.minute.values
        dates = m15_df.index.date
        time_ms = m15_df.index.astype('datetime64[ms]').view('int64')
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        last_trade_date = None

        for i in range(14, len(m15_df)):
            dt = dates[i]
            hr = hours[i]
            mn = minutes[i]

            is_london = (7 <= hr < 9)
            is_ny = (hr == 12 and mn >= 30) or (hr == 13) or (hr == 14 and mn == 0)

            if self.session_mode == "LONDON" and not is_london:
                continue
            elif self.session_mode == "NY" and not is_ny:
                continue
            elif self.session_mode == "BOTH" and not (is_london or is_ny):
                continue

            if dt == last_trade_date:
                continue
            if spread[i] > self.max_spread:
                continue

            macro = h4_trend[i]
            if macro == 0:
                continue

            bar_range = h[i] - l[i]
            base_atr = atr14[i]
            if np.isnan(base_atr) or base_atr <= 0:
                continue

            if bar_range < self.atr_mult * base_atr:
                continue

            sig = 0
            if c[i] > o[i] and macro == 1:
                upper_wick = h[i] - c[i]
                body = c[i] - o[i]
                if upper_wick <= 0.20 * bar_range and body >= self.min_body_pct * bar_range:
                    sig = SIGNAL_LONG

            elif c[i] < o[i] and macro == -1:
                lower_wick = c[i] - l[i]
                body = o[i] - c[i]
                if lower_wick <= 0.20 * bar_range and body >= self.min_body_pct * bar_range:
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
# Model 2: M30 Engulfing Trend Resumption (Parametric Optimization)
# ==============================================================================
class ParametricM30Engulfing(BaseStrategy):
    def __init__(
        self,
        asset: str,
        use_vol_filter: bool,
        min_body_pct: float,
        time_mode: str,       # "ALL_DAY" (07-17), "ACTIVE" (07:30-15:30)
        sl_pips: float,
        tp_pips: float,
        pip_size: float,
        max_spread: float,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.asset = asset
        self.use_vol_filter = use_vol_filter
        self.min_body_pct = float(min_body_pct)
        self.time_mode = time_mode
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.pip_size = float(pip_size)
        self.max_spread = float(max_spread)
        self.name = f"{asset}_M30Engulf_V{int(use_vol_filter)}_B{int(min_body_pct*100)}_R{tp_pips/sl_pips:.1f}"

    def generate_signals(self, ticks: np.ndarray):
        n = len(ticks)
        signals = np.zeros(n, dtype=np.int8)
        sl_arr = np.zeros(n, dtype=np.float64)
        tp_arr = np.zeros(n, dtype=np.float64)
        if n < 500:
            return signals, sl_arr, tp_arr

        m30 = build_time_bars(ticks, timeframe_min=30, pip_size=self.pip_size)
        h4 = build_time_bars(ticks, timeframe_min=240, pip_size=self.pip_size)
        if len(m30) < 20 or len(h4) < 10:
            return signals, sl_arr, tp_arr

        h4_c = pd.Series(h4['close'].values)
        h4_ema20 = h4_c.ewm(span=20, adjust=False).mean().values
        h4_trend = np.where(h4_c.values > h4_ema20, 1, -1)
        h4_df = pd.DataFrame({'h4_bias': h4_trend}, index=h4.index)

        m30_df = pd.merge_asof(
            m30.sort_index(),
            h4_df.sort_index(),
            left_index=True,
            right_index=True,
            direction='backward'
        )

        c = m30_df['close'].values
        h = m30_df['high'].values
        l = m30_df['low'].values
        o = m30_df['open'].values
        v = m30_df['vol'].values
        spread = m30_df['spread_pips'].values
        h4_bias = m30_df['h4_bias'].fillna(0).values.astype(int)
        hours = m30_df.index.hour.values
        dates = m30_df.index.date
        time_ms = m30_df.index.astype('datetime64[ms]').view('int64')
        dur_ms = 30 * 60 * 1000
        tick_ts = ticks['timestamp']

        last_trade_date = None
        for i in range(10, len(m30_df)):
            hr = hours[i]
            if self.time_mode == "ACTIVE":
                if not (7 <= hr <= 15):
                    continue
            else:
                if not (7 <= hr <= 17):
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

            # Volume filter
            if self.use_vol_filter and v[i] <= v[i-1]:
                continue

            # Bullish Engulfing
            is_bull_engulf = (c[i-1] < o[i-1]) and (h[i] > h[i-1]) and (l[i] <= l[i-1]) and (c[i] > h[i-1]) and (c[i] > o[i])
            # Bearish Engulfing
            is_bear_engulf = (c[i-1] > o[i-1]) and (l[i] < l[i-1]) and (h[i] >= h[i-1]) and (c[i] < l[i-1]) and (c[i] < o[i])

            sig = 0
            if macro == 1 and is_bull_engulf:
                body = c[i] - o[i]
                if body >= self.min_body_pct * bar_rng:
                    sig = SIGNAL_LONG
            elif macro == -1 and is_bear_engulf:
                body = o[i] - c[i]
                if body >= self.min_body_pct * bar_rng:
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
# Model 3: H1 Inside Bar Expansion (Parametric Optimization)
# ==============================================================================
class ParametricH1InsideBarExpansion(BaseStrategy):
    def __init__(
        self,
        asset: str,
        time_mode: str,       # "LONDON_ONLY" (07-12), "FULL_DAY" (07-16)
        min_body_pct: float,  # 0.50, 0.60
        sl_pips: float,
        tp_pips: float,
        pip_size: float,
        max_spread: float,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.asset = asset
        self.time_mode = time_mode
        self.min_body_pct = float(min_body_pct)
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.pip_size = float(pip_size)
        self.max_spread = float(max_spread)
        self.name = f"{asset}_H1Inside_{time_mode}_B{int(min_body_pct*100)}_R{tp_pips/sl_pips:.1f}"

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

        h4_c = pd.Series(h4['close'].values)
        h4_ema20 = h4_c.ewm(span=20, adjust=False).mean().values
        h4_bias = np.where(h4_c.values > h4_ema20, 1, -1)
        h4_df = pd.DataFrame({'h4_bias': h4_bias}, index=h4.index)

        # H1 Inside Bar detection
        h1_h = h1['high'].values
        h1_l = h1['low'].values
        is_inside = np.zeros(len(h1), dtype=bool)
        for j in range(1, len(h1)):
            if h1_h[j] <= h1_h[j-1] and h1_l[j] >= h1_l[j-1]:
                is_inside[j] = True

        h1_df = pd.DataFrame({
            'is_inside': is_inside,
            'mother_h': pd.Series(h1_h).shift(1).values,
            'mother_l': pd.Series(h1_l).shift(1).values
        }, index=h1.index)

        m15_merged = pd.merge_asof(
            m15.sort_index(),
            h1_df.sort_index(),
            left_index=True,
            right_index=True,
            direction='backward'
        )
        m15_df = pd.merge_asof(
            m15_merged.sort_index(),
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
        is_inside_bar = m15_df['is_inside'].fillna(False).values
        mother_hi = m15_df['mother_h'].values
        mother_lo = m15_df['mother_l'].values
        hours = m15_df.index.hour.values
        dates = m15_df.index.date
        time_ms = m15_df.index.astype('datetime64[ms]').view('int64')
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        last_trade_date = None

        for i in range(15, len(m15_df)):
            dt = dates[i]
            hr = hours[i]

            if self.time_mode == "LONDON_ONLY":
                if not (7 <= hr <= 11):
                    continue
            else:
                if not (7 <= hr <= 16):
                    continue

            if dt == last_trade_date:
                continue
            if spread[i] > self.max_spread:
                continue

            macro = h4_trend[i]
            if macro == 0 or not is_inside_bar[i]:
                continue

            m_hi = mother_hi[i]
            m_lo = mother_lo[i]
            if np.isnan(m_hi) or np.isnan(m_lo):
                continue

            rng = h[i] - l[i]
            if rng <= 0:
                continue

            sig = 0
            # Bullish Breakout of Mother Bar High
            if macro == 1 and c[i] > m_hi and o[i] < m_hi:
                body = c[i] - o[i]
                if body >= self.min_body_pct * rng:
                    sig = SIGNAL_LONG

            # Bearish Breakout of Mother Bar Low
            elif macro == -1 and c[i] < m_lo and o[i] > m_lo:
                body = o[i] - c[i]
                if body >= self.min_body_pct * rng:
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
def get_optimization_models() -> List[Tuple[str, str, BaseStrategy]]:
    mods = []
    
    # 1. EURUSD Opening Range Impulse Variants
    mods.append(("EURUSD", "ORI_LON_20R", ParametricOpeningRangeImpulse("EURUSD", "LONDON", 1.4, 0.60, 8.0, 16.0, 0.0001, 0.6)))
    mods.append(("EURUSD", "ORI_LON_22R", ParametricOpeningRangeImpulse("EURUSD", "LONDON", 1.4, 0.60, 8.0, 17.6, 0.0001, 0.6)))
    mods.append(("EURUSD", "ORI_LON_15ATR_22R", ParametricOpeningRangeImpulse("EURUSD", "LONDON", 1.5, 0.60, 8.0, 17.6, 0.0001, 0.6)))
    mods.append(("EURUSD", "ORI_BOTH_15ATR_22R", ParametricOpeningRangeImpulse("EURUSD", "BOTH", 1.5, 0.60, 8.0, 17.6, 0.0001, 0.6)))

    # 2. USDJPY & EURUSD M30 Engulfing Resumption Variants
    mods.append(("USDJPY", "M30_BASE_20R", ParametricM30Engulfing("USDJPY", False, 0.50, "ALL_DAY", 14.0, 28.0, 0.01, 0.8)))
    mods.append(("USDJPY", "M30_VOL_22R", ParametricM30Engulfing("USDJPY", True, 0.60, "ACTIVE", 12.0, 26.4, 0.01, 0.8)))
    mods.append(("USDJPY", "M30_ACTIVE_22R", ParametricM30Engulfing("USDJPY", False, 0.60, "ACTIVE", 12.0, 26.4, 0.01, 0.8)))
    mods.append(("EURUSD", "M30_ACTIVE_22R", ParametricM30Engulfing("EURUSD", False, 0.60, "ACTIVE", 8.5, 18.7, 0.0001, 0.6)))
    mods.append(("EURUSD", "M30_VOL_22R", ParametricM30Engulfing("EURUSD", True, 0.60, "ACTIVE", 8.5, 18.7, 0.0001, 0.6)))

    # 3. EURUSD & USDJPY H1 Inside Bar Expansion Variants
    mods.append(("EURUSD", "H1_INSIDE_LON_20R", ParametricH1InsideBarExpansion("EURUSD", "LONDON_ONLY", 0.55, 8.5, 17.0, 0.0001, 0.6)))
    mods.append(("EURUSD", "H1_INSIDE_LON_22R", ParametricH1InsideBarExpansion("EURUSD", "LONDON_ONLY", 0.60, 8.0, 17.6, 0.0001, 0.6)))
    mods.append(("USDJPY", "H1_INSIDE_LON_22R", ParametricH1InsideBarExpansion("USDJPY", "LONDON_ONLY", 0.60, 12.0, 26.4, 0.01, 0.8)))
    mods.append(("EURGBP", "H1_INSIDE_LON_20R", ParametricH1InsideBarExpansion("EURGBP", "LONDON_ONLY", 0.55, 7.0, 14.0, 0.0001, 0.6)))

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
    print("  ALPHA OPTIMIZATION LAB V38: OPTIMIZING PREVIOUS HIGH-POTENTIAL STRATEGIES")
    print("  STRICT PARTITION: 70% In-Sample Discovery Weeks ONLY (Quarantined 30% Holdout Untouched)")
    print("=" * 100)

    models_config = get_optimization_models()
    tasks = []
    for asset, model_name, strat in models_config:
        disc_weeks = get_discovery_weeks(raw_dir=f"data/{asset}")
        for w in disc_weeks:
            tasks.append((asset, model_name, w, strat))

    print(f"Total optimization backtest tasks: {len(tasks):,} across {len(models_config)} model configurations")

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
            if done % 400 == 0 or done == total:
                elapsed = time.time() - t0
                print(f"  Progress: [{done}/{total}] tasks ({done/total*100:.1f}%) in {elapsed:.1f}s")

    elapsed = time.time() - t0
    print(f"\nOptimization phase completed in {elapsed:.1f}s!\n")

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
    df_summary.to_csv(RESULTS_DIR / "alpha_optimization_lab_v38_scorecard.csv", index=False)
    print(f"\nSaved Optimization Scorecard to: {RESULTS_DIR / 'alpha_optimization_lab_v38_scorecard.csv'}")


if __name__ == "__main__":
    main()

