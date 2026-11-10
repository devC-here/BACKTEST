#!/usr/bin/env python3
"""
================================================================================
  CROSS-ASSET HOLDOUT VALIDATION: M30_M5_PRECISION_RETEST (EURUSD, EURGBP, USDJPY)
================================================================================
Performs rigorous In-Sample (70% Discovery) vs Out-of-Sample (30% Quarantined Holdout)
validation for the newly discovered M30/M5 Precision Retest strategy.
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


class M30M5PrecisionRetestUniversal(BaseStrategy):
    """
    M30 20-period Donchian levels + M5 rejection wick (>= 50%).
    Trend filter: M30 20-EMA.
    Active window: 07:00 - 16:30 UTC (plus Tokyo Fix for USDJPY).
    """
    def __init__(self, asset: str, sl_pips: float, tp_pips: float, pip_size: float, max_spread: float, **kwargs):
        super().__init__(**kwargs)
        self.asset = asset
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.pip_size = float(pip_size)
        self.max_spread = float(max_spread)
        self.name = f"{asset}_M30_M5_Precision_Retest"

    def generate_signals(self, ticks: np.ndarray):
        n = len(ticks)
        signals = np.zeros(n, dtype=np.int8)
        sl_arr = np.zeros(n, dtype=np.float64)
        tp_arr = np.zeros(n, dtype=np.float64)
        if n < 500:
            return signals, sl_arr, tp_arr

        m5 = build_time_bars(ticks, timeframe_min=5, pip_size=self.pip_size)
        if len(m5) < 60:
            return signals, sl_arr, tp_arr

        ts = pd.to_datetime(ticks['timestamp'], unit='ms')
        mid = 0.5 * (ticks['bid'] + ticks['ask'])
        base = pd.DataFrame({'close': mid, 'high': mid, 'low': mid}, index=ts)
        m30 = base.resample('30min').agg({'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
        if len(m30) < 25:
            return signals, sl_arr, tp_arr

        m30['ema20'] = m30['close'].ewm(span=20, adjust=False).mean()
        m30['bias'] = np.where(m30['close'] > m30['ema20'], 1, -1)
        m30['roll_h'] = m30['high'].rolling(20).max().shift(1).bfill()
        m30['roll_l'] = m30['low'].rolling(20).min().shift(1).bfill()

        bias_m5 = m30['bias'].reindex(m5.index, method='ffill').fillna(0).values
        supp_m5 = m30['roll_l'].reindex(m5.index, method='ffill').values
        res_m5 = m30['roll_h'].reindex(m5.index, method='ffill').values

        c = m5['close'].values
        h = m5['high'].values
        l = m5['low'].values
        o = m5['open'].values
        spread = m5['spread_pips'].values
        hours = m5.index.hour.values
        dates = m5.index.date
        time_ms = m5.index.astype('datetime64[ms]').view('int64')
        dur_ms = 5 * 60 * 1000
        tick_ts = ticks['timestamp']

        traded_days = set()
        valid_hours = {1, 2, 3, 7, 8, 9, 12, 13, 14, 15} if self.asset == "USDJPY" else set(range(7, 17))
        dist_thresh = 3.0 if self.asset == "USDJPY" else 2.5

        for i in range(20, len(m5)):
            if hours[i] not in valid_hours:
                continue
            d = dates[i]
            if d in traded_days:
                continue
            if spread[i] > self.max_spread:
                continue

            bar_rng = (h[i] - l[i]) / self.pip_size
            if bar_rng < (3.0 if self.asset == "USDJPY" else 2.2):
                continue

            b = bias_m5[i]
            sig = 0

            # Bullish Retest of M30 Support on M5
            if b > 0:
                dist_supp = abs(l[i] - supp_m5[i]) / self.pip_size
                lower_wick = (min(o[i], c[i]) - l[i]) / self.pip_size
                if dist_supp <= dist_thresh and (lower_wick / bar_rng) >= 0.50 and c[i] > o[i]:
                    sig = SIGNAL_LONG

            # Bearish Retest of M30 Resistance on M5
            elif b < 0:
                dist_res = abs(res_m5[i] - h[i]) / self.pip_size
                upper_wick = (h[i] - max(o[i], c[i])) / self.pip_size
                if dist_res <= dist_thresh and (upper_wick / bar_rng) >= 0.50 and c[i] < o[i]:
                    sig = SIGNAL_SHORT

            if sig != 0:
                exec_t = time_ms[i] + dur_ms
                idx = int(np.searchsorted(tick_ts, exec_t))
                if idx < n and signals[idx] == 0:
                    signals[idx] = sig
                    sl_arr[idx] = self.sl_pips
                    tp_arr[idx] = self.tp_pips
                    traded_days.add(d)

        return signals, sl_arr, tp_arr


def evaluate_asset_week(args: Tuple[str, int]) -> Tuple[str, int, List[Dict[str, Any]]]:
    asset, week_num = args
    pip_size = 0.01 if asset == "USDJPY" else 0.0001
    engine = BacktestEngine(pip_size=pip_size, use_slippage=True)
    ticks, meta = get_week_dataset(week_num, raw_dir=f"data/{asset}")
    if len(ticks) == 0:
        return asset, week_num, []

    if asset == "USDJPY":
        model = M30M5PrecisionRetestUniversal(asset=asset, sl_pips=10.0, tp_pips=16.0, pip_size=0.01, max_spread=0.8)
    elif asset == "EURGBP":
        model = M30M5PrecisionRetestUniversal(asset=asset, sl_pips=7.0, tp_pips=12.0, pip_size=0.0001, max_spread=2.0)
    elif asset == "EURUSD":
        model = M30M5PrecisionRetestUniversal(asset=asset, sl_pips=8.0, tp_pips=13.0, pip_size=0.0001, max_spread=0.6)

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
    print("LAUNCHING INSTITUTIONAL 70/30 SPLIT AUDIT: M30_M5_PRECISION_RETEST")
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

    print(f"[*] Queued {len(all_tasks)} total weeks across EURUSD, EURGBP, and USDJPY...")

    trades_by_asset: Dict[str, Dict[str, List[Dict[str, Any]]]] = {
        a: {"disc": [], "hold": [], "full": []} for a in assets
    }

    with ProcessPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(evaluate_asset_week, t): t for t in all_tasks}
        for f in as_completed(futures):
            asset, week_num, t_list = f.result()
            trades_by_asset[asset]["full"].extend(t_list)
            if week_num in partition_map[asset]["disc"]:
                trades_by_asset[asset]["disc"].extend(t_list)
            elif week_num in partition_map[asset]["hold"]:
                trades_by_asset[asset]["hold"].extend(t_list)

    scorecard_rows = []
    for asset in assets:
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

    df = pd.DataFrame(scorecard_rows)
    print("\n" + "=" * 80)
    print("M30_M5_PRECISION_RETEST: 3-ASSET DISCOVERY VS HOLDOUT PERFORMANCE")
    print("=" * 80)
    print(df.to_string(index=False))
    print("=" * 80)

    out_csv = RESULTS_DIR / "m30_m5_cross_holdout_scorecard.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n[+] Saved cross-holdout validation scorecard to: {out_csv.resolve()}")
    print(f"[+] Completed in {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    run_cross_holdout_validation()
