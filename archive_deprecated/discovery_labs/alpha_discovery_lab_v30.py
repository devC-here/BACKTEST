#!/usr/bin/env python3
"""
================================================================================
  ALPHA DISCOVERY LAB V30: ASIAN RANGE LIQUIDITY SWEEP,
  WEEKLY PIVOT R1/S1 FIRST-TOUCH BOUNCE, AND OPENING RANGE IMPULSE EXPANSION
================================================================================
Evaluates 3 brand new, distinct structural quantitative architectures across
the 70% Quarantined Discovery Partition (210 weeks x 3 assets = 630 dataset weeks,
1,890 total backtest runs):

  1. ASIAN_RANGE_SWEEP_REVERSAL:
     - Asian liquidity pool purge (07:00 - 10:30 UTC).
     - False breakout rejection candle fading back inside the Asian boundary.
     - Asymmetric 2.0R target.

  2. WEEKLY_PIVOT_R1_S1_BOUNCE:
     - Multi-day institutional level first-touch mean reversion at Weekly R1 / S1.
     - M15 candle wick rejection >= 35%. First touch per week only.
     - Asymmetric 2.2R target.

  3. OPENING_RANGE_IMPULSE_EXPANSION:
     - High-momentum institutional ignition at London / NY Open (07:00 - 08:30 & 12:30 - 14:00 UTC).
     - Expansion candle >= 1.4x ATR(14) closing near extreme, aligned with H4 EMA20 trend.
     - Asymmetric 1.8R target.
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
# Model 1: Asian Range Liquidity Sweep & Reversal
# ==============================================================================
class AsianRangeSweepReversal(BaseStrategy):
    def __init__(self, asset: str, sl_pips: float, tp_pips: float, pip_size: float, max_spread: float, **kwargs):
        super().__init__(**kwargs)
        self.asset = asset
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.pip_size = float(pip_size)
        self.max_spread = float(max_spread)
        self.name = f"{asset}_Asian_Range_Sweep_Reversal"

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

        # Determine Asian High and Low for each day (00:00 to 06:45 UTC)
        asian_ranges = {}
        for i in range(len(m15)):
            d = dates[i]
            hr = hours[i]
            if 0 <= hr < 7:
                if d not in asian_ranges:
                    asian_ranges[d] = {"high": h[i], "low": l[i]}
                else:
                    asian_ranges[d]["high"] = max(asian_ranges[d]["high"], h[i])
                    asian_ranges[d]["low"] = min(asian_ranges[d]["low"], l[i])

        last_trade_date = None
        min_range = 10.0 * self.pip_size
        max_range = 50.0 * self.pip_size
        sweep_max = 15.0 * self.pip_size

        for i in range(len(m15)):
            dt = dates[i]
            hr = hours[i]

            if dt not in asian_ranges:
                continue
            if not (7 <= hr <= 10):
                continue
            if dt == last_trade_date:
                continue
            if spread[i] > self.max_spread:
                continue

            ar_hi = asian_ranges[dt]["high"]
            ar_lo = asian_ranges[dt]["low"]
            ar_width = ar_hi - ar_lo
            if not (min_range <= ar_width <= max_range):
                continue

            rng = h[i] - l[i]
            if rng <= 0:
                continue

            sig = 0
            # Bearish liquidity sweep: High pierced Asian High, but Close rejected below Asian High
            if h[i] > ar_hi and (h[i] - ar_hi) <= sweep_max and c[i] < ar_hi and c[i] < o[i]:
                upper_wick = h[i] - max(c[i], o[i])
                if upper_wick >= 0.35 * rng:
                    sig = SIGNAL_SHORT

            # Bullish liquidity sweep: Low pierced Asian Low, but Close rejected above Asian Low
            elif l[i] < ar_lo and (ar_lo - l[i]) <= sweep_max and c[i] > ar_lo and c[i] > o[i]:
                lower_wick = min(c[i], o[i]) - l[i]
                if lower_wick >= 0.35 * rng:
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
# Model 2: Weekly Pivot R1 / S1 First-Touch Bounce
# ==============================================================================
class WeeklyPivotR1S1Bounce(BaseStrategy):
    def __init__(self, asset: str, sl_pips: float, tp_pips: float, pip_size: float, max_spread: float, **kwargs):
        super().__init__(**kwargs)
        self.asset = asset
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.pip_size = float(pip_size)
        self.max_spread = float(max_spread)
        self.name = f"{asset}_Weekly_Pivot_R1_S1_Bounce"

    def generate_signals(self, ticks: np.ndarray):
        n = len(ticks)
        signals = np.zeros(n, dtype=np.int8)
        sl_arr = np.zeros(n, dtype=np.float64)
        tp_arr = np.zeros(n, dtype=np.float64)
        if n < 500:
            return signals, sl_arr, tp_arr

        m15 = build_time_bars(ticks, timeframe_min=15, pip_size=self.pip_size)
        if len(m15) < 50:
            return signals, sl_arr, tp_arr

        c = m15['close'].values
        h = m15['high'].values
        l = m15['low'].values
        o = m15['open'].values
        spread = m15['spread_pips'].values
        hours = m15.index.hour.values
        weekdays = m15.index.weekday.values
        time_ms = m15.index.astype('datetime64[ms]').view('int64')
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        # Group into calendar weeks (year, week_num)
        week_keys = [d.isocalendar()[:2] for d in m15.index]
        weekly_bars = {}
        for i in range(len(m15)):
            wk = week_keys[i]
            if wk not in weekly_bars:
                weekly_bars[wk] = {"high": h[i], "low": l[i], "close": c[i]}
            else:
                weekly_bars[wk]["high"] = max(weekly_bars[wk]["high"], h[i])
                weekly_bars[wk]["low"] = min(weekly_bars[wk]["low"], l[i])
                weekly_bars[wk]["close"] = c[i]

        unique_weeks = sorted(list(weekly_bars.keys()))
        week_to_pivots = {}
        for idx_w in range(1, len(unique_weeks)):
            curr_wk = unique_weeks[idx_w]
            prev_wk = unique_weeks[idx_w - 1]
            p_h = weekly_bars[prev_wk]["high"]
            p_l = weekly_bars[prev_wk]["low"]
            p_c = weekly_bars[prev_wk]["close"]
            pivot = (p_h + p_l + p_c) / 3.0
            r1 = 2.0 * pivot - p_l
            s1 = 2.0 * pivot - p_h
            week_to_pivots[curr_wk] = (r1, s1)

        long_traded_week = set()
        short_traded_week = set()
        tolerance = 2.5 * self.pip_size

        for i in range(len(m15)):
            wk = week_keys[i]
            hr = hours[i]
            wd = weekdays[i]

            if wk not in week_to_pivots:
                continue
            if wd >= 4 and hr >= 18:
                continue
            if not (7 <= hr <= 17):
                continue
            if spread[i] > self.max_spread:
                continue

            r1, s1 = week_to_pivots[wk]
            rng = h[i] - l[i]
            if rng <= 0:
                continue

            sig = 0
            # Test Weekly R1 from below -> Bearish Reversal (First Touch)
            if wk not in short_traded_week:
                if abs(h[i] - r1) <= tolerance and c[i] < r1 and c[i] < o[i]:
                    upper_wick = h[i] - max(c[i], o[i])
                    if upper_wick >= 0.35 * rng:
                        sig = SIGNAL_SHORT
                        short_traded_week.add(wk)

            # Test Weekly S1 from above -> Bullish Reversal (First Touch)
            if sig == 0 and wk not in long_traded_week:
                if abs(l[i] - s1) <= tolerance and c[i] > s1 and c[i] > o[i]:
                    lower_wick = min(c[i], o[i]) - l[i]
                    if lower_wick >= 0.35 * rng:
                        sig = SIGNAL_LONG
                        long_traded_week.add(wk)

            if sig != 0:
                end_ms = time_ms[i] + dur_ms
                idx = np.searchsorted(tick_ts, end_ms)
                if idx < n:
                    signals[idx] = sig
                    sl_arr[idx] = self.sl_pips
                    tp_arr[idx] = self.tp_pips

        return signals, sl_arr, tp_arr


# ==============================================================================
# Model 3: Opening Range Impulse Expansion
# ==============================================================================
class OpeningRangeImpulseExpansion(BaseStrategy):
    def __init__(self, asset: str, sl_pips: float, tp_pips: float, pip_size: float, max_spread: float, **kwargs):
        super().__init__(**kwargs)
        self.asset = asset
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.pip_size = float(pip_size)
        self.max_spread = float(max_spread)
        self.name = f"{asset}_Opening_Range_Impulse_Expansion"

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

            is_london = (7 <= hr < 8) or (hr == 8 and mn <= 30)
            is_ny = (hr == 12 and mn >= 30) or (hr == 13) or (hr == 14 and mn == 0)
            if not (is_london or is_ny):
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

            if bar_range < 1.4 * base_atr:
                continue

            sig = 0
            if c[i] > o[i] and macro == 1:
                upper_wick = h[i] - c[i]
                body = c[i] - o[i]
                if upper_wick <= 0.20 * bar_range and body >= 0.60 * bar_range:
                    sig = SIGNAL_LONG

            elif c[i] < o[i] and macro == -1:
                lower_wick = c[i] - l[i]
                body = o[i] - c[i]
                if lower_wick <= 0.20 * bar_range and body >= 0.60 * bar_range:
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
# Model Builder
# ==============================================================================
def build_model_v30(model_key: str, asset: str) -> BaseStrategy:
    if model_key == "ASIAN_RANGE_SWEEP":
        if asset == "EURUSD":
            return AsianRangeSweepReversal(asset=asset, sl_pips=8.5, tp_pips=17.0, pip_size=0.0001, max_spread=0.6)
        elif asset == "USDJPY":
            return AsianRangeSweepReversal(asset=asset, sl_pips=12.0, tp_pips=24.0, pip_size=0.01, max_spread=0.8)
        elif asset == "EURGBP":
            return AsianRangeSweepReversal(asset=asset, sl_pips=7.5, tp_pips=15.0, pip_size=0.0001, max_spread=0.6)

    elif model_key == "WEEKLY_PIVOT_R1_S1":
        if asset == "EURUSD":
            return WeeklyPivotR1S1Bounce(asset=asset, sl_pips=11.0, tp_pips=24.0, pip_size=0.0001, max_spread=0.6)
        elif asset == "USDJPY":
            return WeeklyPivotR1S1Bounce(asset=asset, sl_pips=14.0, tp_pips=31.0, pip_size=0.01, max_spread=0.8)
        elif asset == "EURGBP":
            return WeeklyPivotR1S1Bounce(asset=asset, sl_pips=8.0, tp_pips=18.0, pip_size=0.0001, max_spread=0.6)

    elif model_key == "OPENING_RANGE_IMPULSE":
        if asset == "EURUSD":
            return OpeningRangeImpulseExpansion(asset=asset, sl_pips=9.0, tp_pips=16.0, pip_size=0.0001, max_spread=0.6)
        elif asset == "USDJPY":
            return OpeningRangeImpulseExpansion(asset=asset, sl_pips=13.0, tp_pips=23.5, pip_size=0.01, max_spread=0.8)
        elif asset == "EURGBP":
            return OpeningRangeImpulseExpansion(asset=asset, sl_pips=7.5, tp_pips=13.5, pip_size=0.0001, max_spread=0.6)

    raise ValueError(f"Unknown model_key: {model_key}")


def run_week_task(task: Tuple[str, str, int]) -> Dict[str, Any]:
    asset, model_key, week_num = task
    pip_size = 0.01 if asset == "USDJPY" else 0.0001
    engine = BacktestEngine(pip_size=pip_size, use_slippage=True)

    try:
        ticks, meta = get_week_dataset(week_num, raw_dir=f"data/{asset}")
        if ticks is None or len(ticks) < 500:
            return {"asset": asset, "model": model_key, "week": week_num, "trades": []}

        strat = build_model_v30(model_key, asset)
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
    print("  ALPHA DISCOVERY LAB V30: ASIAN RANGE SWEEP, WEEKLY PIVOT R1/S1 & OPENING IMPULSE")
    print("  Partition: 70% In-Sample Discovery Weeks (Seed 42) across EURUSD, USDJPY, EURGBP")
    print("=" * 95)

    models = [
        "ASIAN_RANGE_SWEEP",
        "WEEKLY_PIVOT_R1_S1",
        "OPENING_RANGE_IMPULSE",
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
        done = 0
        total = len(futures)
        for fut in as_completed(futures):
            res = fut.result()
            key = f"{res['asset']}_{res['model']}"
            if key not in results_by_key:
                results_by_key[key] = []
            results_by_key[key].extend(res["trades"])
            done += 1
            if done % 300 == 0 or done == total:
                elapsed = time.time() - t0
                print(f"  Progress: [{done}/{total}] tasks completed ({done/total*100:.1f}%) in {elapsed:.1f}s")

    elapsed = time.time() - t0
    print(f"\nDiscovery phase completed in {elapsed:.1f}s!\n")

    summary_rows = []
    print("=" * 95)
    print(f"  {'Asset':<8} {'Model':<28} {'Trades':<8} {'Win Rate':<10} {'PF':<8} {'Expectancy':<12} {'Net Pips':<10} {'Max DD':<10} {'Verdict'}")
    print("  " + "-" * 90)

    passing_candidates = []

    for key, trades in sorted(results_by_key.items()):
        asset, model = key.split("_", 1)
        pips = [t["pips"] for t in trades]
        m = calculate_metrics(pips)

        verdict = "[DISCOVERY CHAMPION]" if (m["net_pips"] >= 150 and m["pf"] >= 1.25 and m["expectancy"] >= 1.50) else "[PROMISING]" if m["net_pips"] > 0 else "[DISCARDED]"

        print(f"  {asset:<8} {model:<28} {m['trades']:<8d} {m['win_rate']:<9.1f}% {m['pf']:<7.2f} +{m['expectancy']:<10.2f}p +{m['net_pips']:<9.1f}p {m['max_dd']:<9.1f}p {verdict}")

        summary_rows.append({
            "asset": asset,
            "model": model,
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

    df_out = pd.DataFrame(summary_rows)
    df_out.to_csv(RESULTS_DIR / "alpha_discovery_lab_v30_discovery_scorecard.csv", index=False)
    print(f"\nSaved Discovery Scorecard to: {RESULTS_DIR / 'alpha_discovery_lab_v30_discovery_scorecard.csv'}")

    # Holdout Validation
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
        print(f"  {'Asset':<8} {'Model':<28} {'Holdout Tr':<12} {'Holdout WR':<12} {'Holdout PF':<12} {'Holdout Pips':<14} {'Verdict'}")
        print("  " + "-" * 90)

        for asset, model, disc_m in passing_candidates:
            key = f"{asset}_{model}"
            h_trades = holdout_results.get(key, [])
            h_pips = [t["pips"] for t in h_trades]
            hm = calculate_metrics(h_pips)

            h_verdict = "[100% CERTIFIED PRODUCTION]" if (hm["net_pips"] > 0 and hm["pf"] >= 1.20) else "[FAILED HOLDOUT]"

            print(f"  {asset:<8} {model:<28} {hm['trades']:<12d} {hm['win_rate']:<11.1f}% {hm['pf']:<11.2f} +{hm['net_pips']:<13.1f}p {h_verdict}")

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
        df_holdout.to_csv(RESULTS_DIR / "alpha_discovery_lab_v30_holdout_validation.csv", index=False)
        print(f"\nSaved Holdout Results to: {RESULTS_DIR / 'alpha_discovery_lab_v30_holdout_validation.csv'}")


if __name__ == "__main__":
    main()
