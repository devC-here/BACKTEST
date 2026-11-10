#!/usr/bin/env python3
"""
================================================================================
  ALPHA DISCOVERY & OPTIMIZATION LAB V45:
  Mean Reversion & Structural Pivot Walls for EURUSD & EURGBP
================================================================================
Target Architectures:
  1. EURUSD Woodie Pivot Reversion (2.0R and 2.2R)
  2. EURGBP Woodie Pivot Reversion (Calibrated tolerance, 2.0R and 2.2R)
  3. EURUSD Session Donchian Midline Bounce (2.0R and 2.2R)
  4. EURGBP Daily Pivot Wall S1/R1 (Calibrated tolerance 3.0p, 2.0R and 2.2R)

Strict Methodology:
  - Evaluated on 70% In-Sample Discovery Weeks (Deterministic Seed 42).
  - Quarantined 30% Holdout is 100% UNTOUCHED during discovery/tuning!
  - 1.0% Risk per trade model.
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
# Model 1: Woodie Pivot Reversion (EURUSD & EURGBP)
# ==============================================================================
class WoodiePivotReversion(BaseStrategy):
    def __init__(
        self,
        asset: str,
        tol_pips: float,
        sl_pips: float,
        tp_pips: float,
        pip_size: float,
        max_spread: float,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.asset = asset
        self.tol_pips = float(tol_pips)
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.pip_size = float(pip_size)
        self.max_spread = float(max_spread)
        self.name = f"{asset}_Woodie_Pivot_R{tp_pips/sl_pips:.1f}"

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
        date_to_woodie = {}
        for idx_d in range(1, len(unique_dates)):
            curr_d = unique_dates[idx_d]
            prev_d = unique_dates[idx_d - 1]
            p_h = daily_bars[prev_d]["high"]
            p_l = daily_bars[prev_d]["low"]
            p_c = daily_bars[prev_d]["close"]
            pivot = (p_h + p_l + 2.0 * p_c) / 4.0
            r1 = 2.0 * pivot - p_l
            s1 = 2.0 * pivot - p_h
            date_to_woodie[curr_d] = (r1, s1)

        last_trade_date = None
        tolerance = self.tol_pips * self.pip_size

        for i in range(len(m15)):
            dt = dates[i]
            hr = hours[i]

            if dt not in date_to_woodie:
                continue
            if not (7 <= hr <= 15):
                continue
            if dt == last_trade_date:
                continue
            if spread[i] > self.max_spread:
                continue

            r1, s1 = date_to_woodie[dt]
            rng = h[i] - l[i]
            if rng <= 0:
                continue

            sig = 0
            if abs(h[i] - r1) <= tolerance and c[i] < r1 and c[i] < o[i]:
                upper_wick = h[i] - max(c[i], o[i])
                if (upper_wick / rng) >= 0.35:
                    sig = SIGNAL_SHORT

            elif abs(l[i] - s1) <= tolerance and c[i] > s1 and c[i] > o[i]:
                lower_wick = min(c[i], o[i]) - l[i]
                if (lower_wick / rng) >= 0.35:
                    sig = SIGNAL_LONG

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
# Model 2: Session Donchian Midline Bounce (EURUSD)
# ==============================================================================
class SessionDonchianMidlineBounce(BaseStrategy):
    def __init__(self, asset: str, sl_pips: float, tp_pips: float, pip_size: float, max_spread: float, **kwargs):
        super().__init__(**kwargs)
        self.asset = asset
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.pip_size = float(pip_size)
        self.max_spread = float(max_spread)
        self.name = f"{asset}_Session_Donchian_Midline_R{tp_pips/sl_pips:.1f}"

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
        tolerance = 2.5 * self.pip_size

        for i in range(20, len(m15_df)):
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

            hi_20 = np.max(h[i-20:i])
            lo_20 = np.min(l[i-20:i])
            midline = (hi_20 + lo_20) / 2.0

            rng = h[i] - l[i]
            if rng <= 0:
                continue

            sig = 0
            if abs(l[i] - midline) <= tolerance and c[i] > midline and c[i] > o[i] and macro == 1:
                lower_wick = min(c[i], o[i]) - l[i]
                if lower_wick >= 0.35 * rng:
                    sig = SIGNAL_LONG

            elif abs(h[i] - midline) <= tolerance and c[i] < midline and c[i] < o[i] and macro == -1:
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
# Model 3: EURGBP Daily Pivot Wall S1/R1 (Calibrated Tolerance)
# ==============================================================================
class EURGBPDailyPivotWallCalibrated(BaseStrategy):
    def __init__(self, asset: str = "EURGBP", tol_pips: float = 3.0, sl_pips: float = 6.0, tp_pips: float = 13.2, pip_size: float = 0.0001, max_spread: float = 0.6, **kwargs):
        super().__init__(**kwargs)
        self.asset = asset
        self.tol_pips = float(tol_pips)
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.pip_size = float(pip_size)
        self.max_spread = float(max_spread)
        self.name = f"{asset}_Daily_Pivot_Wall_Calib_R{tp_pips/sl_pips:.1f}"

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
        tolerance = self.tol_pips * self.pip_size

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
            dist_s1 = abs(l[i] - s1)
            lower_wick = min(c[i], o[i]) - l[i]
            if dist_s1 <= tolerance and c[i] > s1 and (lower_wick / bar_rng) >= 0.35 and c[i] > o[i]:
                sig = SIGNAL_LONG

            dist_r1 = abs(h[i] - r1)
            upper_wick = h[i] - max(c[i], o[i])
            if dist_r1 <= tolerance and c[i] < r1 and (upper_wick / bar_rng) >= 0.35 and c[i] < o[i]:
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
# Model Configurations to Audit
# ==============================================================================
def get_lab_v45_models() -> List[Tuple[str, str, BaseStrategy, float]]:
    mods = []

    # 1. EURUSD Woodie Pivot Reversion (2.0R and 2.2R)
    mods.append(("EURUSD", "WOODIE_20R_SL85", WoodiePivotReversion("EURUSD", 2.5, 8.5, 17.0, 0.0001, 0.6), 8.5))
    mods.append(("EURUSD", "WOODIE_22R_SL80", WoodiePivotReversion("EURUSD", 2.5, 8.0, 17.6, 0.0001, 0.6), 8.0))

    # 2. EURGBP Woodie Pivot Reversion (2.0R and 2.2R)
    mods.append(("EURGBP", "WOODIE_20R_SL60", WoodiePivotReversion("EURGBP", 3.0, 6.0, 12.0, 0.0001, 0.6), 6.0))
    mods.append(("EURGBP", "WOODIE_22R_SL60", WoodiePivotReversion("EURGBP", 3.0, 6.0, 13.2, 0.0001, 0.6), 6.0))

    # 3. EURUSD Session Donchian Midline Bounce (2.0R and 2.2R)
    mods.append(("EURUSD", "DONCHIAN_MID_20R", SessionDonchianMidlineBounce("EURUSD", 8.5, 17.0, 0.0001, 0.6), 8.5))
    mods.append(("EURUSD", "DONCHIAN_MID_22R", SessionDonchianMidlineBounce("EURUSD", 8.0, 17.6, 0.0001, 0.6), 8.0))

    # 4. EURGBP Daily Pivot Wall Calibrated (2.0R and 2.2R)
    mods.append(("EURGBP", "PIVOT_WALL_CALIB_20R", EURGBPDailyPivotWallCalibrated("EURGBP", 3.0, 6.0, 12.0, 0.0001, 0.6), 6.0))
    mods.append(("EURGBP", "PIVOT_WALL_CALIB_22R", EURGBPDailyPivotWallCalibrated("EURGBP", 3.0, 6.0, 13.2, 0.0001, 0.6), 6.0))

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
    models = get_lab_v45_models()
    print("=" * 115)
    print("  ALPHA DISCOVERY & OPTIMIZATION LAB V45: EURUSD & EURGBP STRUCTURAL MODELS")
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

    out_csv = RESULTS_DIR / "alpha_discovery_lab_v45_scorecard.csv"
    pd.DataFrame(records).to_csv(out_csv, index=False)
    print(f"\nSaved Scorecard to: {out_csv.resolve()}")
    print(f"Total Execution Time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()

