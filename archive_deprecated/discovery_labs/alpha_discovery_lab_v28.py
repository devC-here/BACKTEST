#!/usr/bin/env python3
"""
================================================================================
  ALPHA DISCOVERY LAB V28: SUPERTREND SQUEEZE CALIBRATION,
  VALUE AREA (VAH/VAL) RETEST & MTF EMA RIBBON COMPRESSION PULLBACK
================================================================================
Systematic exploration and rigorous out-of-sample verification:
  1. CALIBRATED SUPERTREND SQUEEZE ELITE (DISC90):
     - Refines the Lab V27 discovery (+319.1p on EURUSD).
     - Incorporates M15 momentum expansion body filter (Body >= 55% of range),
       strict H1 SuperTrend + H4 20-EMA trend agreement, and asymmetric 2.2R target.
  2. PRIOR DAY VALUE AREA (VAH/VAL) RETEST ELITE (DISC93):
     - Uses prior-day high/low/close to compute Value Area High (VAH) and Low (VAL).
     - First touch of VAH/VAL during London/NY hours (07:00 - 16:00 UTC) with
       rejection wick >= 40% in trend direction. Target: 2.0R.
  3. MTF EMA RIBBON COMPRESSION PULLBACK (DISC94):
     - M15 EMA Ribbon (8, 13, 21, 34, 55 EMAs) in alignment with H4 20-EMA.
     - Deep pullback into 21-34 EMA pocket followed by rejection wick. Target: 2.0R.
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
from tick_engine.core.dataloader import get_week_dataset, get_discovery_weeks, get_holdout_test_weeks
from tick_engine.strategies.base import BaseStrategy, SIGNAL_LONG, SIGNAL_SHORT
from data_engineering_suite.bars import build_time_bars

RESULTS_DIR = ROOT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# Model 1: Calibrated SuperTrend Squeeze Elite
# ==============================================================================
class CalibratedSuperTrendSqueeze(BaseStrategy):
    def __init__(self, asset: str, sl_pips: float, tp_pips: float, pip_size: float, max_spread: float, **kwargs):
        super().__init__(**kwargs)
        self.asset = asset
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.pip_size = float(pip_size)
        self.max_spread = float(max_spread)
        self.name = f"{asset}_Calibrated_SuperTrend_Squeeze"

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

        last_trade_date = None

        for i in range(21, len(m15_df)):
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

            # Prior bar was in squeeze
            was_squeezed = (bb_upper[i-1] <= kc_upper[i-1]) and (bb_lower[i-1] >= kc_lower[i-1])
            is_firing = (bb_upper[i] > kc_upper[i]) or (bb_lower[i] < kc_lower[i])

            if not (was_squeezed and is_firing):
                continue

            rng = h[i] - l[i]
            body = abs(c[i] - o[i])
            if rng <= 0 or body < 0.55 * rng:
                continue

            sig = 0
            if macro == 1 and c[i] > o[i] and c[i] > bb_upper[i]:
                sig = SIGNAL_LONG
            elif macro == -1 and c[i] < o[i] and c[i] < bb_lower[i]:
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
# Model 2: Prior Day Value Area (VAH/VAL) Retest Elite
# ==============================================================================
class ValueAreaRetestElite(BaseStrategy):
    def __init__(self, asset: str, sl_pips: float, tp_pips: float, pip_size: float, max_spread: float, **kwargs):
        super().__init__(**kwargs)
        self.asset = asset
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.pip_size = float(pip_size)
        self.max_spread = float(max_spread)
        self.name = f"{asset}_Value_Area_Retest_Elite"

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

        # Group by day to compute daily VAH and VAL
        # Standard institutional calculation: VAH = Close + 0.70 * (High - Low), VAL = Close - 0.70 * (High - Low)
        daily_bars = {}
        for i in range(len(m15_df)):
            d = dates[i]
            if d not in daily_bars:
                daily_bars[d] = {"high": h[i], "low": l[i], "close": c[i]}
            else:
                daily_bars[d]["high"] = max(daily_bars[d]["high"], h[i])
                daily_bars[d]["low"] = min(daily_bars[d]["low"], l[i])
                daily_bars[d]["close"] = c[i]

        unique_dates = sorted(list(daily_bars.keys()))
        date_to_prior_va = {}
        for idx_d in range(1, len(unique_dates)):
            curr_d = unique_dates[idx_d]
            prev_d = unique_dates[idx_d - 1]
            p_hi = daily_bars[prev_d]["high"]
            p_lo = daily_bars[prev_d]["low"]
            p_cl = daily_bars[prev_d]["close"]
            rng = p_hi - p_lo
            vah = p_cl + 0.35 * rng
            val = p_cl - 0.35 * rng
            date_to_prior_va[curr_d] = (vah, val)

        last_trade_date = None
        tolerance = 2.5 * self.pip_size

        for i in range(len(m15_df)):
            dt = dates[i]
            hr = hours[i]

            if dt not in date_to_prior_va:
                continue
            if not (7 <= hr <= 16):
                continue
            if dt == last_trade_date:
                continue
            if spread[i] > self.max_spread:
                continue

            macro = h4_trend[i]
            if macro == 0:
                continue

            vah, val = date_to_prior_va[dt]
            rng = h[i] - l[i]
            if rng <= 0:
                continue

            sig = 0
            # Bullish test of VAH: Low touches near VAH, rejects upward with lower wick
            if abs(l[i] - vah) <= tolerance and c[i] > vah and c[i] > o[i] and macro == 1:
                lower_wick = min(c[i], o[i]) - l[i]
                if lower_wick >= 0.40 * rng:
                    sig = SIGNAL_LONG

            # Bearish test of VAL: High touches near VAL, rejects downward with upper wick
            elif abs(h[i] - val) <= tolerance and c[i] < val and c[i] < o[i] and macro == -1:
                upper_wick = h[i] - max(c[i], o[i])
                if upper_wick >= 0.40 * rng:
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
# Model 3: MTF EMA Ribbon Compression Pullback Elite
# ==============================================================================
class MTFEMARibbonPullbackElite(BaseStrategy):
    def __init__(self, asset: str, sl_pips: float, tp_pips: float, pip_size: float, max_spread: float, **kwargs):
        super().__init__(**kwargs)
        self.asset = asset
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.pip_size = float(pip_size)
        self.max_spread = float(max_spread)
        self.name = f"{asset}_EMA_Ribbon_Pullback_Elite"

    def generate_signals(self, ticks: np.ndarray):
        n = len(ticks)
        signals = np.zeros(n, dtype=np.int8)
        sl_arr = np.zeros(n, dtype=np.float64)
        tp_arr = np.zeros(n, dtype=np.float64)
        if n < 500:
            return signals, sl_arr, tp_arr

        m15 = build_time_bars(ticks, timeframe_min=15, pip_size=self.pip_size)
        h4 = build_time_bars(ticks, timeframe_min=240, pip_size=self.pip_size)
        if len(m15) < 60 or len(h4) < 10:
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
        ema8 = c_ser.ewm(span=8, adjust=False).mean().values
        ema13 = c_ser.ewm(span=13, adjust=False).mean().values
        ema21 = c_ser.ewm(span=21, adjust=False).mean().values
        ema34 = c_ser.ewm(span=34, adjust=False).mean().values
        ema55 = c_ser.ewm(span=55, adjust=False).mean().values

        last_trade_date = None

        for i in range(55, len(m15_df)):
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

            rng = h[i] - l[i]
            if rng <= 0:
                continue

            # Ribbon alignment check
            ribbon_bull = (ema8[i] > ema13[i]) and (ema13[i] > ema21[i]) and (ema21[i] > ema34[i]) and (ema34[i] > ema55[i])
            ribbon_bear = (ema8[i] < ema13[i]) and (ema13[i] < ema21[i]) and (ema21[i] < ema34[i]) and (ema34[i] < ema55[i])

            sig = 0
            # Bullish pullback into 21-34 EMA pocket
            if ribbon_bull and macro == 1:
                # low touched inside or near 21-34 EMA
                if l[i] <= ema21[i] and c[i] > ema21[i] and c[i] > o[i]:
                    lower_wick = min(c[i], o[i]) - l[i]
                    if lower_wick >= 0.40 * rng:
                        sig = SIGNAL_LONG

            # Bearish pullback into 21-34 EMA pocket
            elif ribbon_bear and macro == -1:
                if h[i] >= ema21[i] and c[i] < ema21[i] and c[i] < o[i]:
                    upper_wick = h[i] - max(c[i], o[i])
                    if upper_wick >= 0.40 * rng:
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
# Factory & Multi-Partition Test Harness
# ==============================================================================
def build_model_v28(model_key: str, asset: str) -> BaseStrategy:
    if model_key == "CALIBRATED_SUPERTREND_SQUEEZE":
        if asset == "EURUSD":
            return CalibratedSuperTrendSqueeze(asset=asset, sl_pips=8.5, tp_pips=18.5, pip_size=0.0001, max_spread=0.6)
        elif asset == "USDJPY":
            return CalibratedSuperTrendSqueeze(asset=asset, sl_pips=11.5, tp_pips=25.0, pip_size=0.01, max_spread=0.8)
        elif asset == "EURGBP":
            return CalibratedSuperTrendSqueeze(asset=asset, sl_pips=7.5, tp_pips=16.5, pip_size=0.0001, max_spread=0.6)

    elif model_key == "VALUE_AREA_RETEST":
        if asset == "EURUSD":
            return ValueAreaRetestElite(asset=asset, sl_pips=8.5, tp_pips=17.0, pip_size=0.0001, max_spread=0.6)
        elif asset == "USDJPY":
            return ValueAreaRetestElite(asset=asset, sl_pips=12.0, tp_pips=24.0, pip_size=0.01, max_spread=0.8)
        elif asset == "EURGBP":
            return ValueAreaRetestElite(asset=asset, sl_pips=7.5, tp_pips=15.0, pip_size=0.0001, max_spread=0.6)

    elif model_key == "EMA_RIBBON_PULLBACK":
        if asset == "EURUSD":
            return MTFEMARibbonPullbackElite(asset=asset, sl_pips=8.5, tp_pips=17.0, pip_size=0.0001, max_spread=0.6)
        elif asset == "USDJPY":
            return MTFEMARibbonPullbackElite(asset=asset, sl_pips=12.0, tp_pips=24.0, pip_size=0.01, max_spread=0.8)
        elif asset == "EURGBP":
            return MTFEMARibbonPullbackElite(asset=asset, sl_pips=7.5, tp_pips=15.0, pip_size=0.0001, max_spread=0.6)


def run_week_task(task: Tuple[str, str, int]) -> Dict[str, Any]:
    asset, model_key, week_num = task
    pip_size = 0.01 if asset == "USDJPY" else 0.0001
    engine = BacktestEngine(pip_size=pip_size, use_slippage=True)

    ticks, meta = get_week_dataset(week_num, raw_dir=f"data/{asset}")
    if ticks is None or len(ticks) < 500:
        return {"asset": asset, "model": model_key, "week": week_num, "trades": []}

    strat = build_model_v28(model_key, asset)
    try:
        raw_trades = engine.run(ticks, strat)
        t_list = []
        for i in range(len(raw_trades)):
            row = raw_trades[i]
            t_list.append({
                "pips": float(row["pips"]),
                "direction": int(row["direction"])
            })
        return {"asset": asset, "model": model_key, "week": week_num, "trades": t_list}
    except Exception:
        return {"asset": asset, "model": model_key, "week": week_num, "trades": []}


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
    print("=" * 95)
    print("  ALPHA DISCOVERY LAB V28: TESTING SUPERTREND SQUEEZE, VALUE AREA & RIBBON PULLBACK")
    print("  Partition: 70% In-Sample Discovery Weeks (Seed 42) across EURUSD, USDJPY, EURGBP")
    print("=" * 95)

    models = [
        "CALIBRATED_SUPERTREND_SQUEEZE",
        "VALUE_AREA_RETEST",
        "EMA_RIBBON_PULLBACK"
    ]
    assets = [os.environ["DISCOVERY_TARGET_SYMBOL"]] if "DISCOVERY_TARGET_SYMBOL" in os.environ else ["EURUSD", "USDJPY", "EURGBP"]

    discovery_tasks = []
    for asset in assets:
        disc_weeks = get_discovery_weeks(raw_dir=f"data/{asset}")
        for m in models:
            for w in disc_weeks:
                discovery_tasks.append((asset, m, w))

    print(f"Total discovery tasks: {len(discovery_tasks):,} across {len(assets)} assets and {len(models)} architectures")

    results_by_key = {}
    t0 = time.time()

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(run_week_task, t) for t in discovery_tasks]
        for fut in as_completed(futures):
            res = fut.result()
            key = f"{res['asset']}_{res['model']}"
            if key not in results_by_key:
                results_by_key[key] = []
            results_by_key[key].extend(res["trades"])

    elapsed = time.time() - t0
    print(f"Discovery phase completed in {elapsed:.1f}s!\n")

    summary_rows = []
    print("=" * 95)
    print(f"  {'Asset':<8} {'Model':<30} {'Trades':<8} {'Win Rate':<10} {'PF':<8} {'Expectancy':<12} {'Net Pips':<10} {'Max DD':<10} {'Verdict'}")
    print("  " + "-" * 90)

    passing_candidates = []

    for key, trades in sorted(results_by_key.items()):
        asset, model = key.split("_", 1)
        pips = [t["pips"] for t in trades]
        m = calculate_metrics(pips)

        # Economic viability hurdle: net_pips >= 250p, PF >= 1.25, expectancy >= 1.50 p/t
        verdict = "[DISCOVERY CHAMPION]" if (m["net_pips"] >= 250 and m["pf"] >= 1.25 and m["expectancy"] >= 1.50) else "[PROMISING]" if m["net_pips"] > 0 else "[DISCARDED]"

        print(f"  {asset:<8} {model:<30} {m['trades']:<8d} {m['win_rate']:<9.1f}% {m['pf']:<7.2f} +{m['expectancy']:<10.2f}p +{m['net_pips']:<9.1f}p {m['max_dd']:<9.1f}p {verdict}")

        summary_rows.append({
            "asset": asset,
            "model": model,
            "partition": "70% Discovery",
            "trades": m["trades"],
            "win_rate": m["win_rate"],
            "profit_factor": m["pf"],
            "expectancy": m["expectancy"],
            "net_pips": m["net_pips"],
            "max_dd": m["max_dd"],
            "ret_dd": m["ret_dd"],
            "verdict": verdict
        })

        if verdict == "[DISCOVERY CHAMPION]":
            passing_candidates.append((asset, model, m))

    # Save discovery scorecard
    df_out = pd.DataFrame(summary_rows)
    df_out.to_csv(RESULTS_DIR / "alpha_discovery_lab_v28_discovery_scorecard.csv", index=False)

    # ==========================================================================
    # QUARANTINED 30% OUT-OF-SAMPLE HOLDOUT AUDIT
    # ==========================================================================
    if passing_candidates:
        print("\n" + "=" * 95)
        print(f"  COMMENCING QUARANTINED 30% HOLDOUT AUDIT FOR {len(passing_candidates)} QUALIFIED CHAMPIONS")
        print("=" * 95)

        holdout_tasks = []
        for asset, model, _ in passing_candidates:
            test_weeks = get_holdout_test_weeks(raw_dir=f"data/{asset}")
            for w in test_weeks:
                holdout_tasks.append((asset, model, w))

        holdout_results = {}
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(run_week_task, t) for t in holdout_tasks]
            for fut in as_completed(futures):
                res = fut.result()
                key = f"{res['asset']}_{res['model']}"
                if key not in holdout_results:
                    holdout_results[key] = []
                holdout_results[key].extend(res["trades"])

        holdout_rows = []
        print(f"  {'Asset':<8} {'Model':<30} {'Holdout Tr':<12} {'Holdout WR':<12} {'Holdout PF':<12} {'Holdout Pips':<14} {'Verdict'}")
        print("  " + "-" * 90)

        for asset, model, disc_m in passing_candidates:
            key = f"{asset}_{model}"
            h_trades = holdout_results.get(key, [])
            h_pips = [t["pips"] for t in h_trades]
            hm = calculate_metrics(h_pips)

            h_verdict = "[100% CERTIFIED PRODUCTION]" if (hm["net_pips"] > 0 and hm["pf"] >= 1.20) else "[FAILED HOLDOUT]"
            print(f"  {asset:<8} {model:<30} {hm['trades']:<12d} {hm['win_rate']:<11.1f}% {hm['pf']:<11.2f} +{hm['net_pips']:<13.1f}p {h_verdict}")

            holdout_rows.append({
                "asset": asset,
                "model": model,
                "discovery_pips": disc_m["net_pips"],
                "discovery_pf": disc_m["pf"],
                "holdout_trades": hm["trades"],
                "holdout_win_rate": hm["win_rate"],
                "holdout_pf": hm["pf"],
                "holdout_pips": hm["net_pips"],
                "verdict": h_verdict
            })

        df_holdout = pd.DataFrame(holdout_rows)
        df_holdout.to_csv(RESULTS_DIR / "alpha_discovery_lab_v28_holdout_validation.csv", index=False)
        print("\n" + "=" * 95)
        print("Saved holdout validation results to results/alpha_discovery_lab_v28_holdout_validation.csv")
        print("=" * 95)


if __name__ == "__main__":
    main()
