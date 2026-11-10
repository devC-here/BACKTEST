#!/usr/bin/env python3
"""
================================================================================
  ALPHA DISCOVERY LAB V37:
  1. LONDON_VWAP_PULLBACK (Intraday Institutional VWAP Dynamic Bounce)
  2. DELTA_ABSORPTION_PULLBACK (Tick Order Flow Delta Absorption at 20 EMA)
  3. CAMARILLA_H4_L4_EXPANSION (Camarilla Breakout Momentum)
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
# Model 1: London VWAP Pullback
# ==============================================================================
class LondonVWAPPullback(BaseStrategy):
    def __init__(self, asset: str, sl_pips: float, tp_pips: float, pip_size: float, max_spread: float, **kwargs):
        super().__init__(**kwargs)
        self.asset = asset
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.pip_size = float(pip_size)
        self.max_spread = float(max_spread)
        self.name = f"{asset}_London_VWAP_Pullback"

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

        # Intraday cumulative VWAP using 'vol'
        typ = (m15['high'] + m15['low'] + m15['close']) / 3.0
        v = m15['vol'].values.astype(np.float64)
        v = np.where(v <= 0, 1.0, v)
        typ_v = typ.values * v

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
            cum_pv += typ_v[i]
            cum_v += v[i]
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

            # European & US overlap hours (08:00 - 15:30 UTC)
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
            # Bullish VWAP bounce: low tags VWAP within tolerance, bar closes above VWAP and green
            if macro == 1 and abs(l[i] - vwap) <= tolerance and c[i] > vwap and c[i] > o[i]:
                lower_wick = min(c[i], o[i]) - l[i]
                if lower_wick >= 0.35 * rng:
                    sig = SIGNAL_LONG

            # Bearish VWAP bounce: high tags VWAP within tolerance, bar closes below VWAP and red
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
# Model 2: Delta Absorption Pullback
# ==============================================================================
class DeltaAbsorptionPullback(BaseStrategy):
    def __init__(self, asset: str, sl_pips: float, tp_pips: float, pip_size: float, max_spread: float, **kwargs):
        super().__init__(**kwargs)
        self.asset = asset
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.pip_size = float(pip_size)
        self.max_spread = float(max_spread)
        self.name = f"{asset}_Delta_Absorption_Pullback"

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

        # M15 20 EMA
        m15_c = pd.Series(m15['close'].values)
        ema20 = m15_c.ewm(span=20, adjust=False).mean().values

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
        delta = m15_df['delta'].values
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

            # European & US active liquidity hours: 07:30 - 16:00 UTC
            if not (7 <= hr <= 16):
                continue
            if dt == last_trade_date:
                continue
            if spread[i] > self.max_spread:
                continue

            macro = h4_trend[i]
            if macro == 0:
                continue

            lvl = ema20[i]
            rng = h[i] - l[i]
            if rng <= 0:
                continue

            sig = 0
            # Bullish Delta Absorption:
            # Price pulls back to M15 20 EMA in H4 uptrend, leaves lower wick >= 35%,
            # and Delta > 0 (aggressive net buyer absorption), closing green.
            if macro == 1 and abs(l[i] - lvl) <= tolerance and c[i] > o[i] and delta[i] > 0:
                lower_wick = min(c[i], o[i]) - l[i]
                if lower_wick >= 0.35 * rng:
                    sig = SIGNAL_LONG

            # Bearish Delta Absorption:
            # Price rallies to M15 20 EMA in H4 downtrend, leaves upper wick >= 35%,
            # and Delta < 0 (aggressive net seller absorption), closing red.
            elif macro == -1 and abs(h[i] - lvl) <= tolerance and c[i] < o[i] and delta[i] < 0:
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
# Model 3: Camarilla H4 / L4 Breakout Expansion
# ==============================================================================
class CamarillaH4L4Expansion(BaseStrategy):
    def __init__(self, asset: str, sl_pips: float, tp_pips: float, pip_size: float, max_spread: float, **kwargs):
        super().__init__(**kwargs)
        self.asset = asset
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.pip_size = float(pip_size)
        self.max_spread = float(max_spread)
        self.name = f"{asset}_Camarilla_H4_L4_Expansion"

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
        date_to_camarilla = {}
        for idx_d in range(1, len(unique_dates)):
            curr_d = unique_dates[idx_d]
            prev_d = unique_dates[idx_d - 1]
            p_h = daily_bars[prev_d]["high"]
            p_l = daily_bars[prev_d]["low"]
            p_c = daily_bars[prev_d]["close"]
            rng = p_h - p_l
            h4_lvl = p_c + (rng * 1.1 / 2.0)
            l4_lvl = p_c - (rng * 1.1 / 2.0)
            date_to_camarilla[curr_d] = (h4_lvl, l4_lvl)

        last_trade_date = None

        for i in range(len(m15_df)):
            dt = dates[i]
            hr = hours[i]

            if dt not in date_to_camarilla:
                continue

            # Morning expansion window: 07:00 - 12:00 UTC
            if not (7 <= hr <= 12):
                continue
            if dt == last_trade_date:
                continue
            if spread[i] > self.max_spread:
                continue

            macro = h4_trend[i]
            if macro == 0:
                continue

            h4_lvl, l4_lvl = date_to_camarilla[dt]
            bar_rng = h[i] - l[i]
            if bar_rng <= 0:
                continue

            sig = 0
            # Bullish Breakout: Closes above H4 with strong green body >= 50%
            if macro == 1 and c[i] > h4_lvl and o[i] < h4_lvl and (c[i] - o[i]) >= 0.50 * bar_rng:
                sig = SIGNAL_LONG

            # Bearish Breakout: Closes below L4 with strong red body >= 50%
            elif macro == -1 and c[i] < l4_lvl and o[i] > l4_lvl and (o[i] - c[i]) >= 0.50 * bar_rng:
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
def build_model_v37(model_key: str, asset: str) -> BaseStrategy:
    if model_key == "LONDON_VWAP_PULLBACK":
        if asset == "EURUSD":
            return LondonVWAPPullback(asset=asset, sl_pips=8.5, tp_pips=17.0, pip_size=0.0001, max_spread=0.6)
        elif asset == "USDJPY":
            return LondonVWAPPullback(asset=asset, sl_pips=12.0, tp_pips=24.0, pip_size=0.01, max_spread=0.8)
        elif asset == "EURGBP":
            return LondonVWAPPullback(asset=asset, sl_pips=7.0, tp_pips=14.0, pip_size=0.0001, max_spread=0.6)

    elif model_key == "DELTA_ABSORPTION_PULLBACK":
        if asset == "EURUSD":
            return DeltaAbsorptionPullback(asset=asset, sl_pips=8.5, tp_pips=17.0, pip_size=0.0001, max_spread=0.6)
        elif asset == "USDJPY":
            return DeltaAbsorptionPullback(asset=asset, sl_pips=12.0, tp_pips=24.0, pip_size=0.01, max_spread=0.8)
        elif asset == "EURGBP":
            return DeltaAbsorptionPullback(asset=asset, sl_pips=7.0, tp_pips=14.0, pip_size=0.0001, max_spread=0.6)

    elif model_key == "CAMARILLA_H4_L4_EXPANSION":
        if asset == "EURUSD":
            return CamarillaH4L4Expansion(asset=asset, sl_pips=8.5, tp_pips=17.0, pip_size=0.0001, max_spread=0.6)
        elif asset == "USDJPY":
            return CamarillaH4L4Expansion(asset=asset, sl_pips=12.0, tp_pips=24.0, pip_size=0.01, max_spread=0.8)
        elif asset == "EURGBP":
            return CamarillaH4L4Expansion(asset=asset, sl_pips=7.0, tp_pips=14.0, pip_size=0.0001, max_spread=0.6)

    raise ValueError(f"Unknown model_key: {model_key}")


def run_week_task(task: Tuple[str, str, int]) -> Dict[str, Any]:
    asset, model_key, week_num = task
    pip_size = 0.01 if asset == "USDJPY" else 0.0001
    engine = BacktestEngine(pip_size=pip_size, use_slippage=True)

    try:
        ticks, meta = get_week_dataset(week_num, raw_dir=f"data/{asset}")
        if ticks is None or len(ticks) < 500:
            return {"asset": asset, "model": model_key, "week": week_num, "trades": []}

        strat = build_model_v37(model_key, asset)
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
    print("  ALPHA DISCOVERY LAB V37: LONDON VWAP PULLBACK, DELTA ABSORPTION & CAMARILLA H4/L4 EXPANSION")
    print("  Partition: 70% In-Sample Discovery Weeks (Seed 42) across EURUSD, USDJPY, EURGBP")
    print("=" * 95)

    models = [
        "LONDON_VWAP_PULLBACK",
        "DELTA_ABSORPTION_PULLBACK",
        "CAMARILLA_H4_L4_EXPANSION",
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
    df_summary.to_csv(RESULTS_DIR / "alpha_discovery_lab_v37_discovery_scorecard.csv", index=False)
    print(f"\nSaved Discovery Scorecard to: {RESULTS_DIR / 'alpha_discovery_lab_v37_discovery_scorecard.csv'}")

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
    df_holdout.to_csv(RESULTS_DIR / "alpha_discovery_lab_v37_holdout_validation.csv", index=False)
    print(f"\nSaved Holdout Results to: {RESULTS_DIR / 'alpha_discovery_lab_v37_holdout_validation.csv'}")


if __name__ == "__main__":
    main()

