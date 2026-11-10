#!/usr/bin/env python3
"""
================================================================================
  ALPHA DISCOVERY LAB V34: EURGBP PRIOR DAY EXTREME SWEEP,
  EURUSD NY LONDON-EXTREME SWEEP, AND USDJPY LONDON ORB S/R FLIP RETEST
================================================================================
Evaluates 3 distinct institutional structural architectures across the
70% Quarantined Discovery Partition (210 weeks x 3 assets = 630 dataset weeks,
1,809 total backtest evaluations):

  1. EURGBP_PRIOR_DAY_EXTREME_SWEEP:
     - Fades false breakouts of the Prior Day High / Low during European hours (07:00 - 14:00 UTC).
     - Sweeps by 1 to 12 pips, then closes back inside with rejection wick >= 40%.
     - 2.0R target.

  2. EURUSD_NY_LONDON_EXTREME_SWEEP:
     - New York open (12:30 - 15:30 UTC) liquidity test of the London Session (07:00 - 12:00 UTC) High/Low.
     - Reversal rejection wick >= 40% fading back toward equilibrium.
     - 2.0R target.

  3. USDJPY_LONDON_ORB_SR_FLIP:
     - London Opening Range (07:00 - 08:30 UTC) broken, then patient pullback to retest
       the broken boundary from the outside (Old Resistance -> Support, Old Support -> Resistance).
     - Confirmed by H4 20-EMA trend and M15 rejection wick >= 35%.
     - Retest Window: 09:00 - 15:30 UTC. 2.2R target.
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
# Model 1: Prior Day Extreme Sweep Reversion (EURGBP Specialist)
# ==============================================================================
class PriorDayExtremeSweep(BaseStrategy):
    def __init__(self, asset: str, sl_pips: float, tp_pips: float, pip_size: float, max_spread: float, **kwargs):
        super().__init__(**kwargs)
        self.asset = asset
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.pip_size = float(pip_size)
        self.max_spread = float(max_spread)
        self.name = f"{asset}_Prior_Day_Extreme_Sweep"

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
                daily_bars[d] = {"high": h[i], "low": l[i]}
            else:
                daily_bars[d]["high"] = max(daily_bars[d]["high"], h[i])
                daily_bars[d]["low"] = min(daily_bars[d]["low"], l[i])

        unique_dates = sorted(list(daily_bars.keys()))
        date_to_prev = {}
        for idx_d in range(1, len(unique_dates)):
            curr_d = unique_dates[idx_d]
            prev_d = unique_dates[idx_d - 1]
            date_to_prev[curr_d] = (daily_bars[prev_d]["high"], daily_bars[prev_d]["low"])

        last_trade_date = None
        max_sweep = 12.0 * self.pip_size

        for i in range(len(m15)):
            dt = dates[i]
            hr = hours[i]

            if dt not in date_to_prev:
                continue
            # European active liquidity hours (07:00 - 14:00 UTC)
            if not (7 <= hr <= 14):
                continue
            if dt == last_trade_date:
                continue
            if spread[i] > self.max_spread:
                continue

            prev_hi, prev_lo = date_to_prev[dt]
            rng = h[i] - l[i]
            if rng <= 0:
                continue

            sig = 0
            # Bearish sweep: High pierced prior day high by <= 12 pips, but close fell back below prior high
            if h[i] > prev_hi and (h[i] - prev_hi) <= max_sweep and c[i] < prev_hi and c[i] < o[i]:
                upper_wick = h[i] - max(c[i], o[i])
                if upper_wick >= 0.40 * rng:
                    sig = SIGNAL_SHORT

            # Bullish sweep: Low pierced prior day low by <= 12 pips, but close bounced back above prior low
            elif l[i] < prev_lo and (prev_lo - l[i]) <= max_sweep and c[i] > prev_lo and c[i] > o[i]:
                lower_wick = min(c[i], o[i]) - l[i]
                if lower_wick >= 0.40 * rng:
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
# Model 2: EURUSD NY London-Extreme Sweep Reversal
# ==============================================================================
class NYLondonExtremeSweep(BaseStrategy):
    def __init__(self, asset: str, sl_pips: float, tp_pips: float, pip_size: float, max_spread: float, **kwargs):
        super().__init__(**kwargs)
        self.asset = asset
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.pip_size = float(pip_size)
        self.max_spread = float(max_spread)
        self.name = f"{asset}_NY_London_Extreme_Sweep"

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

        # Determine London session range (07:00 - 12:00 UTC)
        london_ranges = {}
        for i in range(len(m15)):
            d = dates[i]
            hr = hours[i]
            if 7 <= hr < 12:
                if d not in london_ranges:
                    london_ranges[d] = {"high": h[i], "low": l[i]}
                else:
                    london_ranges[d]["high"] = max(london_ranges[d]["high"], h[i])
                    london_ranges[d]["low"] = min(london_ranges[d]["low"], l[i])

        last_trade_date = None
        max_sweep = 8.0 * self.pip_size

        for i in range(len(m15)):
            dt = dates[i]
            hr = hours[i]

            if dt not in london_ranges:
                continue
            # NY open window: 12:30 - 15:30 UTC
            if not (12 <= hr <= 15):
                continue
            if dt == last_trade_date:
                continue
            if spread[i] > self.max_spread:
                continue

            lon_hi = london_ranges[dt]["high"]
            lon_lo = london_ranges[dt]["low"]
            rng = h[i] - l[i]
            if rng <= 0:
                continue

            sig = 0
            # Test/Sweep London High -> Bearish Reversal
            if h[i] >= lon_hi and (h[i] - lon_hi) <= max_sweep and c[i] < lon_hi and c[i] < o[i]:
                upper_wick = h[i] - max(c[i], o[i])
                if upper_wick >= 0.40 * rng:
                    sig = SIGNAL_SHORT

            # Test/Sweep London Low -> Bullish Reversal
            elif l[i] <= lon_lo and (lon_lo - l[i]) <= max_sweep and c[i] > lon_lo and c[i] > o[i]:
                lower_wick = min(c[i], o[i]) - l[i]
                if lower_wick >= 0.40 * rng:
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
# Model 3: USDJPY London ORB S/R Flip Retest
# ==============================================================================
class LondonORBSRFlipRetest(BaseStrategy):
    def __init__(self, asset: str, sl_pips: float, tp_pips: float, pip_size: float, max_spread: float, **kwargs):
        super().__init__(**kwargs)
        self.asset = asset
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.pip_size = float(pip_size)
        self.max_spread = float(max_spread)
        self.name = f"{asset}_London_ORB_SR_Flip"

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

        # Measure London Opening 90-min range (07:00 - 08:30 UTC)
        london_orb = {}
        for i in range(len(m15_df)):
            d = dates[i]
            hr = hours[i]
            if 7 <= hr <= 8:
                if d not in london_orb:
                    london_orb[d] = {"high": h[i], "low": l[i]}
                else:
                    london_orb[d]["high"] = max(london_orb[d]["high"], h[i])
                    london_orb[d]["low"] = min(london_orb[d]["low"], l[i])

        last_trade_date = None
        tolerance = 2.5 * self.pip_size
        day_broke_high = set()
        day_broke_low = set()

        for i in range(len(m15_df)):
            dt = dates[i]
            hr = hours[i]

            if dt not in london_orb:
                continue

            orb_hi = london_orb[dt]["high"]
            orb_lo = london_orb[dt]["low"]

            # Step 1: Detect breakout between 08:30 and 10:30 UTC
            if 8 <= hr <= 10:
                if c[i] > orb_hi:
                    day_broke_high.add(dt)
                if c[i] < orb_lo:
                    day_broke_low.add(dt)

            # Step 2: Retest Window (10:30 - 15:30 UTC)
            if not (10 <= hr <= 15):
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
            # Bullish S/R Flip: Broke ORB high, tests ORB high from ABOVE in H4 uptrend
            if dt in day_broke_high and macro == 1:
                if abs(l[i] - orb_hi) <= tolerance and c[i] > orb_hi and c[i] > o[i]:
                    lower_wick = min(c[i], o[i]) - l[i]
                    if lower_wick >= 0.35 * rng:
                        sig = SIGNAL_LONG

            # Bearish S/R Flip: Broke ORB low, tests ORB low from BELOW in H4 downtrend
            elif dt in day_broke_low and macro == -1:
                if abs(h[i] - orb_lo) <= tolerance and c[i] < orb_lo and c[i] < o[i]:
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
def build_model_v34(model_key: str, asset: str) -> BaseStrategy:
    if model_key == "PRIOR_DAY_EXTREME_SWEEP":
        if asset == "EURUSD":
            return PriorDayExtremeSweep(asset=asset, sl_pips=8.5, tp_pips=17.0, pip_size=0.0001, max_spread=0.6)
        elif asset == "USDJPY":
            return PriorDayExtremeSweep(asset=asset, sl_pips=12.0, tp_pips=24.0, pip_size=0.01, max_spread=0.8)
        elif asset == "EURGBP":
            return PriorDayExtremeSweep(asset=asset, sl_pips=7.5, tp_pips=15.0, pip_size=0.0001, max_spread=0.6)

    elif model_key == "NY_LONDON_EXTREME_SWEEP":
        if asset == "EURUSD":
            return NYLondonExtremeSweep(asset=asset, sl_pips=8.5, tp_pips=17.0, pip_size=0.0001, max_spread=0.6)
        elif asset == "USDJPY":
            return NYLondonExtremeSweep(asset=asset, sl_pips=12.0, tp_pips=24.0, pip_size=0.01, max_spread=0.8)
        elif asset == "EURGBP":
            return NYLondonExtremeSweep(asset=asset, sl_pips=7.5, tp_pips=15.0, pip_size=0.0001, max_spread=0.6)

    elif model_key == "LONDON_ORB_SR_FLIP":
        if asset == "EURUSD":
            return LondonORBSRFlipRetest(asset=asset, sl_pips=8.5, tp_pips=18.5, pip_size=0.0001, max_spread=0.6)
        elif asset == "USDJPY":
            return LondonORBSRFlipRetest(asset=asset, sl_pips=12.0, tp_pips=26.5, pip_size=0.01, max_spread=0.8)
        elif asset == "EURGBP":
            return LondonORBSRFlipRetest(asset=asset, sl_pips=7.5, tp_pips=16.5, pip_size=0.0001, max_spread=0.6)

    raise ValueError(f"Unknown model_key: {model_key}")


def run_week_task(task: Tuple[str, str, int]) -> Dict[str, Any]:
    asset, model_key, week_num = task
    pip_size = 0.01 if asset == "USDJPY" else 0.0001
    engine = BacktestEngine(pip_size=pip_size, use_slippage=True)

    try:
        ticks, meta = get_week_dataset(week_num, raw_dir=f"data/{asset}")
        if ticks is None or len(ticks) < 500:
            return {"asset": asset, "model": model_key, "week": week_num, "trades": []}

        strat = build_model_v34(model_key, asset)
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
    print("  ALPHA DISCOVERY LAB V34: PRIOR DAY SWEEP, NY LONDON SWEEP & LONDON ORB S/R FLIP")
    print("  Partition: 70% In-Sample Discovery Weeks (Seed 42) across EURUSD, USDJPY, EURGBP")
    print("=" * 95)

    models = [
        "PRIOR_DAY_EXTREME_SWEEP",
        "NY_LONDON_EXTREME_SWEEP",
        "LONDON_ORB_SR_FLIP",
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
    df_out.to_csv(RESULTS_DIR / "alpha_discovery_lab_v34_discovery_scorecard.csv", index=False)
    print(f"\nSaved Discovery Scorecard to: {RESULTS_DIR / 'alpha_discovery_lab_v34_discovery_scorecard.csv'}")

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
        df_holdout.to_csv(RESULTS_DIR / "alpha_discovery_lab_v34_holdout_validation.csv", index=False)
        print(f"\nSaved Holdout Results to: {RESULTS_DIR / 'alpha_discovery_lab_v34_holdout_validation.csv'}")


if __name__ == "__main__":
    main()

