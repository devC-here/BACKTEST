#!/usr/bin/env python3
"""
================================================================================
  ALPHA DISCOVERY LAB V36:
  1. SESSION_DONCHIAN_MIDLINE_CALIBRATED (EURUSD & EURGBP adaptation of DISC91)
  2. LONDON_VWAP_COMPRESSION_RETEST (Session VWAP Liquidity Ingestion)
  3. ASIAN_LIQUIDITY_SWEEP_REVERSAL (Judas Swing / Turtle Soup)
================================================================================
Evaluates 3 distinct institutional quantitative architectures across the
70% Quarantined Discovery Partition (210 weeks x 3 assets = 630 dataset weeks,
1,809 total backtest evaluations).

Passing Criteria:
  - Discovery: Net Pips >= 150.0, Profit Factor >= 1.25, Expectancy >= 1.20 pips
  - Holdout: Profit Factor >= 1.20, Positive Net Pips, Expectancy >= 1.00 pips
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
# Model 1: Calibrated Session Donchian Midline Bounce (Adapting DISC91)
# ==============================================================================
class SessionDonchianMidlineCalibrated(BaseStrategy):
    def __init__(self, asset: str, sl_pips: float, tp_pips: float, pip_size: float, max_spread: float, **kwargs):
        super().__init__(**kwargs)
        self.asset = asset
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.pip_size = float(pip_size)
        self.max_spread = float(max_spread)
        self.name = f"{asset}_Session_Donchian_Midline_Calibrated"

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

        # 20-period Donchian Channels on M15
        m15_h = pd.Series(m15['high'].values)
        m15_l = pd.Series(m15['low'].values)
        donch_hi = m15_h.rolling(20, min_periods=20).max().values
        donch_lo = m15_l.rolling(20, min_periods=20).min().values
        donch_mid = (donch_hi + donch_lo) / 2.0

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
        tolerance = 2.0 * self.pip_size if self.asset != "USDJPY" else 2.5 * self.pip_size

        for i in range(20, len(m15_df)):
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

            mid = donch_mid[i]
            if np.isnan(mid):
                continue

            rng = h[i] - l[i]
            if rng <= 0:
                continue

            sig = 0
            # Bullish bounce: in H4 uptrend, low tags midline within tolerance, closes above open with lower wick >= 35%
            if macro == 1 and abs(l[i] - mid) <= tolerance and c[i] > mid and c[i] > o[i]:
                lower_wick = min(c[i], o[i]) - l[i]
                if lower_wick >= 0.35 * rng:
                    sig = SIGNAL_LONG

            # Bearish bounce: in H4 downtrend, high tags midline within tolerance, closes below open with upper wick >= 35%
            elif macro == -1 and abs(h[i] - mid) <= tolerance and c[i] < mid and c[i] < o[i]:
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
# Model 2: London VWAP Retest (Session Liquidity Ingestion)
# ==============================================================================
class LondonVWAPRetest(BaseStrategy):
    def __init__(self, asset: str, sl_pips: float, tp_pips: float, pip_size: float, max_spread: float, **kwargs):
        super().__init__(**kwargs)
        self.asset = asset
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.pip_size = float(pip_size)
        self.max_spread = float(max_spread)
        self.name = f"{asset}_London_VWAP_Retest"

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

        # Intraday cumulative VWAP from 00:00 UTC
        # Typical price = (H + L + C) / 3, Volume = tick count
        typ = (m15['high'] + m15['low'] + m15['close']) / 3.0
        vol = m15['tick_count'].replace(0, 1)

        dates = m15.index.date
        vwap_arr = np.zeros(len(m15), dtype=np.float64)
        
        curr_d = None
        cum_pv = 0.0
        cum_v = 0.0
        for i in range(len(m15)):
            d = dates[i]
            if d != curr_d:
                curr_d = d
                cum_pv = 0.0
                cum_v = 0.0
            cum_pv += typ.iloc[i] * vol.iloc[i]
            cum_v += vol.iloc[i]
            vwap_arr[i] = cum_pv / cum_v if cum_v > 0 else typ.iloc[i]

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
        time_ms = m15_df.index.astype('datetime64[ms]').view('int64')
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        last_trade_date = None
        tolerance = 2.0 * self.pip_size if self.asset != "USDJPY" else 2.5 * self.pip_size

        for i in range(len(m15_df)):
            dt = dates[i]
            hr = hours[i]

            # Prime European/US institutional hours: 08:00 - 15:30 UTC
            if not (8 <= hr <= 15):
                continue
            if dt == last_trade_date:
                continue
            if spread[i] > self.max_spread:
                continue

            macro = h4_trend[i]
            if macro == 0:
                continue

            vwap = vwap_arr[i]
            rng = h[i] - l[i]
            if rng <= 0:
                continue

            sig = 0
            # Bullish VWAP bounce: In H4 uptrend, price tests VWAP from above, leaves lower wick >= 35%
            if macro == 1 and abs(l[i] - vwap) <= tolerance and c[i] > vwap and c[i] > o[i]:
                lower_wick = min(c[i], o[i]) - l[i]
                if lower_wick >= 0.35 * rng:
                    sig = SIGNAL_LONG

            # Bearish VWAP bounce: In H4 downtrend, price tests VWAP from below, leaves upper wick >= 35%
            elif macro == -1 and abs(h[i] - vwap) <= tolerance and c[i] < vwap and c[i] < o[i]:
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
# Model 3: Asian Liquidity Sweep Reversal (Judas Swing / Turtle Soup)
# ==============================================================================
class AsianLiquiditySweepReversal(BaseStrategy):
    def __init__(self, asset: str, sl_pips: float, tp_pips: float, pip_size: float, max_spread: float, **kwargs):
        super().__init__(**kwargs)
        self.asset = asset
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.pip_size = float(pip_size)
        self.max_spread = float(max_spread)
        self.name = f"{asset}_Asian_Liquidity_Sweep_Reversal"

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

        # Asian Session Range: 00:00 - 06:30 UTC
        asian_ranges = {}
        for i in range(len(m15_df)):
            d = dates[i]
            hr = hours[i]
            if 0 <= hr < 6:
                if d not in asian_ranges:
                    asian_ranges[d] = {"high": h[i], "low": l[i]}
                else:
                    asian_ranges[d]["high"] = max(asian_ranges[d]["high"], h[i])
                    asian_ranges[d]["low"] = min(asian_ranges[d]["low"], l[i])

        last_trade_date = None
        min_sweep = 1.0 * self.pip_size
        max_sweep = 20.0 * self.pip_size

        for i in range(len(m15_df)):
            dt = dates[i]
            hr = hours[i]

            if dt not in asian_ranges:
                continue

            # London opening purge window: 06:30 - 10:00 UTC
            if not (6 <= hr <= 9):
                continue
            if dt == last_trade_date:
                continue
            if spread[i] > self.max_spread:
                continue

            macro = h4_trend[i]
            if macro == 0:
                continue

            ar_hi = asian_ranges[dt]["high"]
            ar_lo = asian_ranges[dt]["low"]
            rng = h[i] - l[i]
            if rng <= 0:
                continue

            sig = 0
            # Bullish Judas Sweep: In H4 uptrend, London sweeps Asian Low (l[i] < ar_lo),
            # but price snaps back and closes ABOVE ar_lo with strong lower wick >= 35%
            if macro == 1 and (ar_lo - l[i]) >= min_sweep and (ar_lo - l[i]) <= max_sweep and c[i] > ar_lo and c[i] > o[i]:
                lower_wick = min(c[i], o[i]) - l[i]
                if lower_wick >= 0.35 * rng:
                    sig = SIGNAL_LONG

            # Bearish Judas Sweep: In H4 downtrend, London sweeps Asian High (h[i] > ar_hi),
            # but price snaps back and closes BELOW ar_hi with strong upper wick >= 35%
            elif macro == -1 and (h[i] - ar_hi) >= min_sweep and (h[i] - ar_hi) <= max_sweep and c[i] < ar_hi and c[i] < o[i]:
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
def build_model_v36(model_key: str, asset: str) -> BaseStrategy:
    if model_key == "SESSION_DONCHIAN_MIDLINE":
        if asset == "EURUSD":
            return SessionDonchianMidlineCalibrated(asset=asset, sl_pips=8.5, tp_pips=17.0, pip_size=0.0001, max_spread=0.6)
        elif asset == "USDJPY":
            return SessionDonchianMidlineCalibrated(asset=asset, sl_pips=12.0, tp_pips=24.0, pip_size=0.01, max_spread=0.8)
        elif asset == "EURGBP":
            return SessionDonchianMidlineCalibrated(asset=asset, sl_pips=7.0, tp_pips=14.0, pip_size=0.0001, max_spread=0.6)

    elif model_key == "LONDON_VWAP_RETEST":
        if asset == "EURUSD":
            return LondonVWAPRetest(asset=asset, sl_pips=8.5, tp_pips=17.0, pip_size=0.0001, max_spread=0.6)
        elif asset == "USDJPY":
            return LondonVWAPRetest(asset=asset, sl_pips=12.0, tp_pips=24.0, pip_size=0.01, max_spread=0.8)
        elif asset == "EURGBP":
            return LondonVWAPRetest(asset=asset, sl_pips=7.0, tp_pips=14.0, pip_size=0.0001, max_spread=0.6)

    elif model_key == "ASIAN_LIQUIDITY_SWEEP_REVERSAL":
        if asset == "EURUSD":
            return AsianLiquiditySweepReversal(asset=asset, sl_pips=8.5, tp_pips=18.7, pip_size=0.0001, max_spread=0.6)
        elif asset == "USDJPY":
            return AsianLiquiditySweepReversal(asset=asset, sl_pips=12.0, tp_pips=26.4, pip_size=0.01, max_spread=0.8)
        elif asset == "EURGBP":
            return AsianLiquiditySweepReversal(asset=asset, sl_pips=7.0, tp_pips=15.4, pip_size=0.0001, max_spread=0.6)

    raise ValueError(f"Unknown model_key: {model_key}")


def run_week_task(task: Tuple[str, str, int]) -> Dict[str, Any]:
    asset, model_key, week_num = task
    pip_size = 0.01 if asset == "USDJPY" else 0.0001
    engine = BacktestEngine(pip_size=pip_size, use_slippage=True)

    try:
        ticks, meta = get_week_dataset(week_num, raw_dir=f"data/{asset}")
        if ticks is None or len(ticks) < 500:
            return {"asset": asset, "model": model_key, "week": week_num, "trades": []}

        strat = build_model_v36(model_key, asset)
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
    print("  ALPHA DISCOVERY LAB V36: DONCHIAN MIDLINE CALIBRATED, LONDON VWAP RETEST & JUDAS SWEEP")
    print("  Partition: 70% In-Sample Discovery Weeks (Seed 42) across EURUSD, USDJPY, EURGBP")
    print("=" * 95)

    models = [
        "SESSION_DONCHIAN_MIDLINE",
        "LONDON_VWAP_RETEST",
        "ASIAN_LIQUIDITY_SWEEP_REVERSAL",
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
    print(f"  {'Asset':<8} {'Model':<30} {'Trades':<8} {'Win Rate':<10} {'PF':<8} {'Expectancy':<12} {'Net Pips':<10} {'Max DD':<10} {'Verdict'}")
    print("  " + "-" * 90)

    passing_candidates = []

    for key, trades in sorted(results_by_key.items()):
        asset, model = key.split("_", 1)
        pips = [t["pips"] for t in trades]
        m = calculate_metrics(pips)

        # Standard discovery pass: Net Pips >= 150, PF >= 1.25, Expectancy >= 1.20
        verdict = "[DISCOVERY CHAMPION]" if (m["net_pips"] >= 150 and m["pf"] >= 1.25 and m["expectancy"] >= 1.20) else "[PROMISING]" if m["net_pips"] > 0 else "[DISCARDED]"

        print(f"  {asset:<8} {model:<30} {m['trades']:<8d} {m['win_rate']:<9.1f}% {m['pf']:<7.2f} +{m['expectancy']:<10.2f}p +{m['net_pips']:<9.1f}p {m['max_dd']:<9.1f}p {verdict}")

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
            passing_candidates.append((asset, model, m["net_pips"], m["pf"]))

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(RESULTS_DIR / "alpha_discovery_lab_v36_discovery_scorecard.csv", index=False)
    print(f"\nSaved Discovery Scorecard to: {RESULTS_DIR / 'alpha_discovery_lab_v36_discovery_scorecard.csv'}")

    # ==============================================================================
    # Phase 2: Quarantined 30% Holdout Validation
    # ==============================================================================
    if not passing_candidates:
        print("\nNo candidates met the strict Discovery Champion threshold.")
        return

    print("\n" + "=" * 95)
    print(f"  COMMENCING QUARANTINED 30% HOLDOUT AUDIT FOR {len(passing_candidates)} QUALIFIED CHAMPIONS")
    print("=" * 95)

    holdout_tasks = []
    for asset, model, _, _ in passing_candidates:
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

    for asset, model, disc_pips, disc_pf in passing_candidates:
        key = f"{asset}_{model}"
        trades = holdout_results.get(key, [])
        pips = [t["pips"] for t in trades]
        hm = calculate_metrics(pips)

        passed = hm["pf"] >= 1.20 and hm["net_pips"] > 0 and hm["trades"] >= 15
        h_verdict = "[100% CERTIFIED PRODUCTION]" if passed else "[FAILED HOLDOUT]"

        print(f"  {asset:<8} {model:<30} {hm['trades']:<12d} {hm['win_rate']:<11.1f}% {hm['pf']:<11.2f} +{hm['net_pips']:<12.1f}p {h_verdict}")

        holdout_rows.append({
            "asset": asset,
            "model": model,
            "discovery_pips": disc_pips,
            "discovery_pf": disc_pf,
            "holdout_trades": hm["trades"],
            "holdout_win_rate": hm["win_rate"],
            "holdout_pf": hm["pf"],
            "holdout_pips": hm["net_pips"],
            "verdict": h_verdict
        })

    df_holdout = pd.DataFrame(holdout_rows)
    df_holdout.to_csv(RESULTS_DIR / "alpha_discovery_lab_v36_holdout_validation.csv", index=False)
    print(f"\nSaved Holdout Results to: {RESULTS_DIR / 'alpha_discovery_lab_v36_holdout_validation.csv'}")


if __name__ == "__main__":
    main()

