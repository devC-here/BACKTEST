#!/usr/bin/env python3
"""
================================================================================
  ALPHA DISCOVERY LAB V32: H1-M15 EMA VALUE BOUNCE,
  SESSION DONCHIAN MIDLINE BOUNCE, AND ASIAN S/R FLIP RETEST
================================================================================
Evaluates 3 institutional trend & pullback architectures across the
70% Quarantined Discovery Partition (210 weeks x 3 assets = 630 dataset weeks,
1,809 total backtest evaluations):

  1. H1_EMA20_DYNAMIC_BOUNCE:
     - H1 20-EMA trend with M15 pullback touch within 2.5 pips.
     - M15 rejection wick >= 35% bouncing in trend direction.
     - Active Hours: 07:00 - 16:30 UTC. 2.0R target.

  2. SESSION_DONCHIAN_MIDLINE_BOUNCE:
     - M15 20-period Donchian midline test in alignment with H4 20-EMA trend.
     - M15 rejection wick >= 35% off the midline.
     - Active Hours: 07:30 - 15:30 UTC. 2.0R target.

  3. ASIAN_SR_FLIP_RETEST:
     - Post-breakout retest: Price breaks Asian High/Low, then pulls back to test
       the broken boundary from the outside (Old Resistance -> New Support).
     - Confirmed by H4 20-EMA trend and M15 rejection wick >= 35%.
     - Retest Window: 08:30 - 14:30 UTC. 2.2R target.
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
from tick_engine.core.dataloader import get_week_dataset, get_discovery_weeks, get_holdout_test_weeks
from tick_engine.strategies.base import BaseStrategy, SIGNAL_LONG, SIGNAL_SHORT
from data_engineering_suite.bars import build_time_bars

RESULTS_DIR = ROOT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# Model 1: H1 EMA20 Dynamic Value Bounce
# ==============================================================================
class H1EMA20DynamicBounce(BaseStrategy):
    def __init__(self, asset: str, sl_pips: float, tp_pips: float, pip_size: float, max_spread: float, **kwargs):
        super().__init__(**kwargs)
        self.asset = asset
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.pip_size = float(pip_size)
        self.max_spread = float(max_spread)
        self.name = f"{asset}_H1_EMA20_Dynamic_Bounce"

    def generate_signals(self, ticks: np.ndarray):
        n = len(ticks)
        signals = np.zeros(n, dtype=np.int8)
        sl_arr = np.zeros(n, dtype=np.float64)
        tp_arr = np.zeros(n, dtype=np.float64)
        if n < 500:
            return signals, sl_arr, tp_arr

        m15 = build_time_bars(ticks, timeframe_min=15, pip_size=self.pip_size)
        h1 = build_time_bars(ticks, timeframe_min=60, pip_size=self.pip_size)
        if len(m15) < 35 or len(h1) < 25:
            return signals, sl_arr, tp_arr

        h1_c = pd.Series(h1['close'].values)
        h1_ema20 = h1_c.ewm(span=20, adjust=False).mean().values
        h1_ema50 = h1_c.ewm(span=50, adjust=False).mean().values
        h1_bias = np.where(h1_ema20 > h1_ema50, 1, np.where(h1_ema20 < h1_ema50, -1, 0))

        h1_df = pd.DataFrame({
            'h1_ema20': h1_ema20,
            'h1_bias': h1_bias
        }, index=h1.index)

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
        h1_val = m15_df['h1_ema20'].values
        h1_trend = m15_df['h1_bias'].fillna(0).values.astype(int)
        hours = m15_df.index.hour.values
        dates = m15_df.index.date
        time_ms = m15_df.index.astype('datetime64[ms]').view('int64')
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        last_trade_date = None
        tolerance = 2.5 * self.pip_size

        for i in range(len(m15_df)):
            dt = dates[i]
            hr = hours[i]

            if not (7 <= hr <= 16):
                continue
            if dt == last_trade_date:
                continue
            if spread[i] > self.max_spread:
                continue

            macro = h1_trend[i]
            ema_lvl = h1_val[i]
            if macro == 0 or np.isnan(ema_lvl):
                continue

            rng = h[i] - l[i]
            if rng <= 0:
                continue

            sig = 0
            # Bullish bounce: Low tests H1 EMA20 within tolerance, rejects upward
            if abs(l[i] - ema_lvl) <= tolerance and c[i] > ema_lvl and c[i] > o[i] and macro == 1:
                lower_wick = min(c[i], o[i]) - l[i]
                if lower_wick >= 0.35 * rng:
                    sig = SIGNAL_LONG

            # Bearish bounce: High tests H1 EMA20 within tolerance, rejects downward
            elif abs(h[i] - ema_lvl) <= tolerance and c[i] < ema_lvl and c[i] < o[i] and macro == -1:
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
# Model 2: Session Donchian Midline Bounce
# ==============================================================================
class SessionDonchianMidlineBounce(BaseStrategy):
    def __init__(self, asset: str, sl_pips: float, tp_pips: float, pip_size: float, max_spread: float, **kwargs):
        super().__init__(**kwargs)
        self.asset = asset
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.pip_size = float(pip_size)
        self.max_spread = float(max_spread)
        self.name = f"{asset}_Session_Donchian_Midline_Bounce"

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

            # 20-period Donchian Midline
            hi_20 = np.max(h[i-20:i])
            lo_20 = np.min(l[i-20:i])
            midline = (hi_20 + lo_20) / 2.0

            rng = h[i] - l[i]
            if rng <= 0:
                continue

            sig = 0
            # Bullish bounce: Low tests midline, rejects upward in H4 uptrend
            if abs(l[i] - midline) <= tolerance and c[i] > midline and c[i] > o[i] and macro == 1:
                lower_wick = min(c[i], o[i]) - l[i]
                if lower_wick >= 0.35 * rng:
                    sig = SIGNAL_LONG

            # Bearish bounce: High tests midline, rejects downward in H4 downtrend
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
# Model 3: Asian S/R Flip Retest
# ==============================================================================
class AsianSRFlipRetest(BaseStrategy):
    def __init__(self, asset: str, sl_pips: float, tp_pips: float, pip_size: float, max_spread: float, **kwargs):
        super().__init__(**kwargs)
        self.asset = asset
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.pip_size = float(pip_size)
        self.max_spread = float(max_spread)
        self.name = f"{asset}_Asian_SR_Flip_Retest"

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
        max_range = 45.0 * self.pip_size
        tolerance = 2.5 * self.pip_size

        # Track whether morning breakout occurred
        day_broke_high = set()
        day_broke_low = set()

        for i in range(len(m15_df)):
            dt = dates[i]
            hr = hours[i]

            if dt not in asian_ranges:
                continue

            ar_hi = asian_ranges[dt]["high"]
            ar_lo = asian_ranges[dt]["low"]
            ar_width = ar_hi - ar_lo
            if not (min_range <= ar_width <= max_range):
                continue

            # Step 1: Detect morning breakout (07:00 - 09:30 UTC)
            if 7 <= hr <= 9:
                if c[i] > ar_hi:
                    day_broke_high.add(dt)
                if c[i] < ar_lo:
                    day_broke_low.add(dt)

            # Step 2: Retest Window (09:30 - 15:00 UTC)
            if not (9 <= hr <= 15):
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

            sig = 0
            # Bullish S/R Flip: Broke Asian High, now pulling back to test Asian High from ABOVE
            if dt in day_broke_high and macro == 1:
                if abs(l[i] - ar_hi) <= tolerance and c[i] > ar_hi and c[i] > o[i]:
                    lower_wick = min(c[i], o[i]) - l[i]
                    if lower_wick >= 0.35 * rng:
                        sig = SIGNAL_LONG

            # Bearish S/R Flip: Broke Asian Low, now pulling back to test Asian Low from BELOW
            elif dt in day_broke_low and macro == -1:
                if abs(h[i] - ar_lo) <= tolerance and c[i] < ar_lo and c[i] < o[i]:
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
# Model Builder
# ==============================================================================
def build_model_v32(model_key: str, asset: str) -> BaseStrategy:
    if model_key == "H1_EMA20_DYNAMIC_BOUNCE":
        if asset == "EURUSD":
            return H1EMA20DynamicBounce(asset=asset, sl_pips=8.5, tp_pips=17.0, pip_size=0.0001, max_spread=0.6)
        elif asset == "USDJPY":
            return H1EMA20DynamicBounce(asset=asset, sl_pips=12.0, tp_pips=24.0, pip_size=0.01, max_spread=0.8)
        elif asset == "EURGBP":
            return H1EMA20DynamicBounce(asset=asset, sl_pips=7.5, tp_pips=15.0, pip_size=0.0001, max_spread=0.6)

    elif model_key == "SESSION_DONCHIAN_MIDLINE":
        if asset == "EURUSD":
            return SessionDonchianMidlineBounce(asset=asset, sl_pips=8.5, tp_pips=17.0, pip_size=0.0001, max_spread=0.6)
        elif asset == "USDJPY":
            return SessionDonchianMidlineBounce(asset=asset, sl_pips=12.0, tp_pips=24.0, pip_size=0.01, max_spread=0.8)
        elif asset == "EURGBP":
            return SessionDonchianMidlineBounce(asset=asset, sl_pips=7.5, tp_pips=15.0, pip_size=0.0001, max_spread=0.6)

    elif model_key == "ASIAN_SR_FLIP_RETEST":
        if asset == "EURUSD":
            return AsianSRFlipRetest(asset=asset, sl_pips=9.0, tp_pips=20.0, pip_size=0.0001, max_spread=0.6)
        elif asset == "USDJPY":
            return AsianSRFlipRetest(asset=asset, sl_pips=13.0, tp_pips=28.5, pip_size=0.01, max_spread=0.8)
        elif asset == "EURGBP":
            return AsianSRFlipRetest(asset=asset, sl_pips=7.5, tp_pips=16.5, pip_size=0.0001, max_spread=0.6)

    raise ValueError(f"Unknown model_key: {model_key}")


def run_week_task(task: Tuple[str, str, int]) -> Dict[str, Any]:
    asset, model_key, week_num = task
    pip_size = 0.01 if asset == "USDJPY" else 0.0001
    engine = BacktestEngine(pip_size=pip_size, use_slippage=True)

    try:
        ticks, meta = get_week_dataset(week_num, raw_dir=f"data/{asset}")
        if ticks is None or len(ticks) < 500:
            return {"asset": asset, "model": model_key, "week": week_num, "trades": []}

        strat = build_model_v32(model_key, asset)
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
    print("  ALPHA DISCOVERY LAB V32: H1-M15 EMA VALUE, DONCHIAN MIDLINE & ASIAN S/R FLIP RETEST")
    print("  Partition: 70% In-Sample Discovery Weeks (Seed 42) across EURUSD, USDJPY, EURGBP")
    print("=" * 95)

    models = [
        "H1_EMA20_DYNAMIC_BOUNCE",
        "SESSION_DONCHIAN_MIDLINE",
        "ASIAN_SR_FLIP_RETEST",
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
    df_out.to_csv(RESULTS_DIR / "alpha_discovery_lab_v32_discovery_scorecard.csv", index=False)
    print(f"\nSaved Discovery Scorecard to: {RESULTS_DIR / 'alpha_discovery_lab_v32_discovery_scorecard.csv'}")

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
        df_holdout.to_csv(RESULTS_DIR / "alpha_discovery_lab_v32_holdout_validation.csv", index=False)
        print(f"\nSaved Holdout Results to: {RESULTS_DIR / 'alpha_discovery_lab_v32_holdout_validation.csv'}")


if __name__ == "__main__":
    main()
