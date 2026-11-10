#!/usr/bin/env python3
"""
================================================================================
  CROSS-ASSET HOLDOUT VALIDATION: M15 3-BAR COUNTER-TREND PULLBACK (2.0R)
================================================================================
Performs rigorous In-Sample (70% Discovery) vs Out-of-Sample (30% Quarantined Holdout)
validation for the newly discovered M15 3-Bar Counter-Trend Pullback strategy (DISC53).
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
from tick_engine.core.dataloader import get_week_dataset, get_discovery_weeks, get_holdout_test_weeks, load_manifest
from tick_engine.strategies.base import BaseStrategy, SIGNAL_LONG, SIGNAL_SHORT
from data_engineering_suite.bars import build_time_bars

RESULTS_DIR = ROOT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


class M15ThreeBarPullbackUniversal(BaseStrategy):
    """
    Operates in active European & US sessions (07:00 - 17:00 UTC).
    Requires H4 and H1 20-EMA trend agreement.
    Waits for 3 consecutive counter-trend M15 bars that touch or test the M15 20-EMA,
    followed by a momentum expansion candle in the direction of the trend.
    Target: 2.0R.
    """
    def __init__(self, asset: str, sl_pips: float, tp_pips: float, pip_size: float, max_spread: float, **kwargs):
        super().__init__(**kwargs)
        self.asset = asset
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.pip_size = float(pip_size)
        self.max_spread = float(max_spread)
        self.name = f"{asset}_M15_3_Bar_Pullback"

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

        ts = pd.to_datetime(ticks['timestamp'], unit='ms')
        mid = 0.5 * (ticks['bid'] + ticks['ask'])
        base = pd.DataFrame({'close': mid, 'high': mid, 'low': mid}, index=ts)

        h1 = base.resample('1h').agg({'close': 'last'}).dropna()
        h4 = base.resample('4h').agg({'close': 'last'}).dropna()
        if len(h1) < 25 or len(h4) < 15:
            return signals, sl_arr, tp_arr

        h1['ema20'] = h1['close'].ewm(span=20, adjust=False).mean()
        h4['ema20'] = h4['close'].ewm(span=20, adjust=False).mean()

        h1_trend = np.where(h1['close'] > h1['ema20'], 1, -1)
        h4_trend = np.where(h4['close'] > h4['ema20'], 1, -1)

        h1_bias = pd.Series(h1_trend, index=h1.index).shift(1).reindex(m15.index, method='ffill').fillna(0).values
        h4_bias = pd.Series(h4_trend, index=h4.index).shift(1).reindex(m15.index, method='ffill').fillna(0).values

        m15['ema20'] = m15['close'].ewm(span=20, adjust=False).mean()
        m15_ema = m15['ema20'].values

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
        min_body_pips = 4.0 if self.asset == "USDJPY" else 2.5

        for i in range(25, len(m15)):
            if not (7 <= hours[i] <= 17):
                continue
            dt = dates[i]
            if dt == last_trade_date:
                continue
            if spread[i] > self.max_spread:
                continue

            bias_h1 = h1_bias[i]
            bias_h4 = h4_bias[i]
            if bias_h1 != bias_h4 or bias_h1 == 0:
                continue

            bar_rng = (h[i] - l[i]) / self.pip_size
            body = abs(c[i] - o[i]) / self.pip_size
            if bar_rng < min_body_pips or (body / bar_rng) < 0.50:
                continue

            sig = 0

            # Bullish trend: Check if previous 3 bars (i-3, i-2, i-1) were pulling back
            if bias_h1 > 0:
                three_pb = (h[i-1] <= h[i-2] and h[i-2] <= h[i-3]) or (c[i-1] <= o[i-1] and c[i-2] <= o[i-2] and c[i-3] <= o[i-3])
                tested_ema = any(abs(l[k] - m15_ema[k]) / self.pip_size <= 2.5 or l[k] <= m15_ema[k] for k in [i-3, i-2, i-1])
                if three_pb and tested_ema and c[i] > o[i] and c[i] > h[i-1]:
                    sig = SIGNAL_LONG

            # Bearish trend: Check if previous 3 bars were pulling up
            elif bias_h1 < 0:
                three_pb = (l[i-1] >= l[i-2] and l[i-2] >= l[i-3]) or (c[i-1] >= o[i-1] and c[i-2] >= o[i-2] and c[i-3] >= o[i-3])
                tested_ema = any(abs(h[k] - m15_ema[k]) / self.pip_size <= 2.5 or h[k] >= m15_ema[k] for k in [i-3, i-2, i-1])
                if three_pb and tested_ema and c[i] < o[i] and c[i] < l[i-1]:
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


def evaluate_asset_week(args: Tuple[str, int]) -> Tuple[str, int, List[Dict[str, Any]]]:
    asset, week_num = args
    pip_size = 0.01 if asset == "USDJPY" else 0.0001
    engine = BacktestEngine(pip_size=pip_size, use_slippage=True)

    ticks, meta = get_week_dataset(week_num, raw_dir=f"data/{asset}")
    if len(ticks) == 0:
        return asset, week_num, []

    if asset == "USDJPY":
        model = M15ThreeBarPullbackUniversal(asset=asset, sl_pips=12.0, tp_pips=24.0, pip_size=0.01, max_spread=0.8)
    elif asset == "EURGBP":
        model = M15ThreeBarPullbackUniversal(asset=asset, sl_pips=7.0, tp_pips=14.0, pip_size=0.0001, max_spread=2.0)
    elif asset == "EURUSD":
        model = M15ThreeBarPullbackUniversal(asset=asset, sl_pips=9.0, tp_pips=18.0, pip_size=0.0001, max_spread=0.6)
    elif asset == "AUDNZD":
        model = M15ThreeBarPullbackUniversal(asset=asset, sl_pips=10.0, tp_pips=20.0, pip_size=0.0001, max_spread=2.5)

    trades = []
    try:
        raw_trades = engine.run(ticks, model)
        for i in range(len(raw_trades)):
            row = raw_trades[i]
            trades.append({
                "asset": asset,
                "week": week_num,
                "pips": float(row["pips"]),
                "direction": int(row["direction"])
            })
    except Exception:
        pass
    return asset, week_num, trades


def run_cross_holdout_validation():
    t_start = time.time()
    assets = ["EURUSD", "EURGBP", "USDJPY"]
    
    print("=" * 80)
    print("LAUNCHING INSTITUTIONAL 70/30 SPLIT AUDIT: M15 3-BAR COUNTER-TREND PULLBACK (2.0R)")
    print("=" * 80)

    all_tasks = []
    partition_map = {}
    for asset in assets:
        disc = set(get_discovery_weeks(f"data/{asset}"))
        hold = set(get_holdout_test_weeks(f"data/{asset}"))
        partition_map[asset] = {"disc": disc, "hold": hold}
        all_w = disc.union(hold)
        for w in all_w:
            all_tasks.append((asset, w))

    # Add AUDNZD full cycle
    audnzd_manifest = load_manifest("data/AUDNZD")
    for w in sorted(audnzd_manifest.keys()):
        all_tasks.append(("AUDNZD", w))

    print(f"[*] Queued {len(all_tasks)} total weeks across EURUSD, EURGBP, USDJPY, and AUDNZD...")

    trades_by_asset: Dict[str, Dict[str, List[Dict[str, Any]]]] = {
        "EURUSD": {"disc": [], "hold": [], "full": []},
        "EURGBP": {"disc": [], "hold": [], "full": []},
        "USDJPY": {"disc": [], "hold": [], "full": []},
        "AUDNZD": {"disc": [], "hold": [], "full": []},
    }

    with ProcessPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(evaluate_asset_week, t): t for t in all_tasks}
        completed = 0
        for f in as_completed(futures):
            asset, week_num, t_list = f.result()
            trades_by_asset[asset]["full"].extend(t_list)
            if asset in partition_map:
                if week_num in partition_map[asset]["disc"]:
                    trades_by_asset[asset]["disc"].extend(t_list)
                elif week_num in partition_map[asset]["hold"]:
                    trades_by_asset[asset]["hold"].extend(t_list)
            completed += 1
            if completed % 150 == 0 or completed == len(all_tasks):
                print(f"  [+] Audited {completed:3d}/{len(all_tasks)} weeks in {time.time() - t_start:.1f}s...")

    scorecard_rows = []
    for asset in ["EURUSD", "EURGBP", "USDJPY"]:
        for p_name, key in [("70% Discovery (In-Sample)", "disc"), ("30% Holdout (Out-of-Sample)", "hold"), ("100% Full Cycle (5 Years)", "full")]:
            t_list = trades_by_asset[asset][key]
            n = len(t_list)
            if n == 0:
                continue
            p_arr = np.array([t["pips"] for t in t_list])
            net_pips = float(np.sum(p_arr))
            wins = p_arr[p_arr > 0]
            losses = -p_arr[p_arr < 0]
            wr = float(len(wins) / n * 100.0)
            sum_w = float(np.sum(wins))
            sum_l = float(np.sum(losses))
            pf = round(sum_w / sum_l, 2) if sum_l > 0 else (99.0 if sum_w > 0 else 0.0)

            equity = np.cumsum(p_arr)
            peak = np.maximum.accumulate(equity)
            dd = peak - equity
            max_dd = float(np.max(dd)) if len(dd) > 0 else 0.0
            ret_dd = round(net_pips / max_dd, 2) if max_dd > 0 else 0.0
            expectancy = round(net_pips / n, 2)

            scorecard_rows.append({
                "Asset": asset,
                "Partition": p_name,
                "Trades": n,
                "Win_Rate": f"{wr:.1f}%",
                "Profit_Factor": pf,
                "Net_Pips": round(net_pips, 1),
                "Max_DD_Pips": round(max_dd, 1),
                "Ret_DD_Ratio": ret_dd,
                "Expectancy_pips": expectancy
            })

    # Add AUDNZD full cycle
    t_list_an = trades_by_asset["AUDNZD"]["full"]
    if len(t_list_an) > 0:
        p_arr = np.array([t["pips"] for t in t_list_an])
        n = len(p_arr)
        net_pips = float(np.sum(p_arr))
        wins = p_arr[p_arr > 0]
        losses = -p_arr[p_arr < 0]
        wr = float(len(wins) / n * 100.0)
        sum_w = float(np.sum(wins))
        sum_l = float(np.sum(losses))
        pf = round(sum_w / sum_l, 2) if sum_l > 0 else (99.0 if sum_w > 0 else 0.0)
        equity = np.cumsum(p_arr)
        peak = np.maximum.accumulate(equity)
        dd = peak - equity
        max_dd = float(np.max(dd)) if len(dd) > 0 else 0.0
        ret_dd = round(net_pips / max_dd, 2) if max_dd > 0 else 0.0
        expectancy = round(net_pips / n, 2)

        scorecard_rows.append({
            "Asset": "AUDNZD",
            "Partition": "100% Full Cycle (1 Year)",
            "Trades": n,
            "Win_Rate": f"{wr:.1f}%",
            "Profit_Factor": pf,
            "Net_Pips": round(net_pips, 1),
            "Max_DD_Pips": round(max_dd, 1),
            "Ret_DD_Ratio": ret_dd,
            "Expectancy_pips": expectancy
        })

    # Combined Holdouts across EURUSD + USDJPY
    hold_trades_majors = trades_by_asset["EURUSD"]["hold"] + trades_by_asset["USDJPY"]["hold"]
    if len(hold_trades_majors) > 0:
        p_arr = np.array([t["pips"] for t in hold_trades_majors])
        n = len(p_arr)
        net_pips = float(np.sum(p_arr))
        wins = p_arr[p_arr > 0]
        losses = -p_arr[p_arr < 0]
        wr = float(len(wins) / n * 100.0)
        sum_w = float(np.sum(wins))
        sum_l = float(np.sum(losses))
        pf = round(sum_w / sum_l, 2) if sum_l > 0 else 99.0
        equity = np.cumsum(p_arr)
        peak = np.maximum.accumulate(equity)
        dd = peak - equity
        max_dd = float(np.max(dd)) if len(dd) > 0 else 0.0
        ret_dd = round(net_pips / max_dd, 2) if max_dd > 0 else 0.0
        expectancy = round(net_pips / n, 2)

        scorecard_rows.append({
            "Asset": "MAJORS (EU+UJ)",
            "Partition": "30% Quarantined Holdout Total",
            "Trades": n,
            "Win_Rate": f"{wr:.1f}%",
            "Profit_Factor": pf,
            "Net_Pips": round(net_pips, 1),
            "Max_DD_Pips": round(max_dd, 1),
            "Ret_DD_Ratio": ret_dd,
            "Expectancy_pips": expectancy
        })

    # Combined 5-Year Full Cycle across EURUSD + USDJPY
    full_trades_majors = trades_by_asset["EURUSD"]["full"] + trades_by_asset["USDJPY"]["full"]
    if len(full_trades_majors) > 0:
        p_arr = np.array([t["pips"] for t in full_trades_majors])
        n = len(p_arr)
        net_pips = float(np.sum(p_arr))
        wins = p_arr[p_arr > 0]
        losses = -p_arr[p_arr < 0]
        wr = float(len(wins) / n * 100.0)
        sum_w = float(np.sum(wins))
        sum_l = float(np.sum(losses))
        pf = round(sum_w / sum_l, 2) if sum_l > 0 else 99.0
        equity = np.cumsum(p_arr)
        peak = np.maximum.accumulate(equity)
        dd = peak - equity
        max_dd = float(np.max(dd)) if len(dd) > 0 else 0.0
        ret_dd = round(net_pips / max_dd, 2) if max_dd > 0 else 0.0
        expectancy = round(net_pips / n, 2)

        scorecard_rows.append({
            "Asset": "MAJORS (EU+UJ)",
            "Partition": "5-Year Full Cycle Total",
            "Trades": n,
            "Win_Rate": f"{wr:.1f}%",
            "Profit_Factor": pf,
            "Net_Pips": round(net_pips, 1),
            "Max_DD_Pips": round(max_dd, 1),
            "Ret_DD_Ratio": ret_dd,
            "Expectancy_pips": expectancy
        })

    df_score = pd.DataFrame(scorecard_rows)
    out_csv = RESULTS_DIR / "m15_3bar_pullback_cross_holdout_scorecard.csv"
    df_score.to_csv(out_csv, index=False)

    print("\n" + "=" * 105)
    print("  INSTITUTIONAL 70/30 HOLDOUT SCORECARD: M15 3-BAR COUNTER-TREND PULLBACK (DISC53)")
    print("=" * 105)
    print(f"{'Asset':<16} {'Partition':<32} {'Trades':<8} {'Win Rate':<10} {'Profit Factor':<15} {'Net Pips':<12} {'Max DD':<10} {'Ret/DD':<8} {'Exp (p/t)':<10}")
    print("-" * 105)
    for _, r in df_score.iterrows():
        print(f"{r['Asset']:<16} {r['Partition']:<32} {int(r['Trades']):<8} {r['Win_Rate']:<10} {r['Profit_Factor']:>13.2f}  {r['Net_Pips']:>10.1f}p {r['Max_DD_Pips']:>8.1f}p {r['Ret_DD_Ratio']:>6.2f}x {r['Expectancy_pips']:>8.2f}p")
    print("=" * 105)
    print(f"\n[*] Full Validation Completed in {time.time() - t_start:.2f}s.")


if __name__ == "__main__":
    run_cross_holdout_validation()
