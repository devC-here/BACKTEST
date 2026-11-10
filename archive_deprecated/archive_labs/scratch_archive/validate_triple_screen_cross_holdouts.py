#!/usr/bin/env python3
"""
================================================================================
  CROSS-ASSET HOLDOUT VALIDATION: TRIPLE SCREEN M15 DYNAMIC IMPULSE (DISC56)
================================================================================
Performs rigorous In-Sample (70% Discovery) vs Out-of-Sample (30% Quarantined Holdout)
validation for the newly discovered Triple Screen M15 Dynamic Impulse strategy.
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


class TripleScreenM15ImpulseUniversal(BaseStrategy):
    """
    Elder Triple Screen Architecture adapted to FX tick microstructure:
      1. Screen 1 (H4): Macro Trend defined by H4 20-EMA.
      2. Screen 2 (H1): Wave Momentum defined by 12/26/9 MACD Histogram slope.
      3. Screen 3 (M15): Entry Ripple triggered when price pulls back to test M15 13-EMA
         and resumes trend with an expansion candle in H4/H1 direction.
      - Target: 2.0R.
      - Active Hours: 07:00 - 17:00 UTC.
    """
    def __init__(self, asset: str, sl_pips: float, tp_pips: float, pip_size: float, max_spread: float, **kwargs):
        super().__init__(**kwargs)
        self.asset = asset
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.pip_size = float(pip_size)
        self.max_spread = float(max_spread)
        self.name = f"{asset}_Triple_Screen_Impulse"

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
        if len(h1) < 35 or len(h4) < 15:
            return signals, sl_arr, tp_arr

        # Screen 1: H4 20-EMA trend
        h4['ema20'] = h4['close'].ewm(span=20, adjust=False).mean()
        h4_trend = np.where(h4['close'] > h4['ema20'], 1, -1)
        h4_bias = pd.Series(h4_trend, index=h4.index).shift(1).reindex(m15.index, method='ffill').fillna(0).values

        # Screen 2: H1 MACD Histogram slope
        h1['ema12'] = h1['close'].ewm(span=12, adjust=False).mean()
        h1['ema26'] = h1['close'].ewm(span=26, adjust=False).mean()
        h1['macd'] = h1['ema12'] - h1['ema26']
        h1['signal'] = h1['macd'].ewm(span=9, adjust=False).mean()
        h1['hist'] = h1['macd'] - h1['signal']
        h1['hist_slope'] = h1['hist'].diff()
        h1_slope_trend = np.where(h1['hist_slope'] > 0, 1, -1)
        h1_bias = pd.Series(h1_slope_trend, index=h1.index).shift(1).reindex(m15.index, method='ffill').fillna(0).values

        # Screen 3: M15 EMA 13 test
        m15['ema13'] = m15['close'].ewm(span=13, adjust=False).mean()
        ema13 = m15['ema13'].values

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
        tol = 2.5 * self.pip_size

        for i in range(25, len(m15)):
            if not (7 <= hours[i] <= 17):
                continue
            dt = dates[i]
            if dt == last_trade_date:
                continue
            if spread[i] > self.max_spread:
                continue

            bias_h4 = h4_bias[i]
            bias_h1 = h1_bias[i]
            if bias_h4 != bias_h1 or bias_h4 == 0:
                continue

            bar_rng = (h[i] - l[i]) / self.pip_size
            body = abs(c[i] - o[i]) / self.pip_size
            if bar_rng < 3.5 or (body / bar_rng) < 0.50:
                continue

            e13 = ema13[i]
            sig = 0

            # Bullish: Low touched near EMA13 and closed strongly above
            if bias_h4 > 0 and l[i] <= e13 + tol and c[i] > e13 and c[i] > o[i]:
                sig = SIGNAL_LONG
            # Bearish: High touched near EMA13 and closed strongly below
            elif bias_h4 < 0 and h[i] >= e13 - tol and c[i] < e13 and c[i] < o[i]:
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
        model = TripleScreenM15ImpulseUniversal(asset=asset, sl_pips=12.0, tp_pips=24.0, pip_size=0.01, max_spread=0.8)
    elif asset == "EURGBP":
        model = TripleScreenM15ImpulseUniversal(asset=asset, sl_pips=7.0, tp_pips=14.0, pip_size=0.0001, max_spread=2.0)
    elif asset == "EURUSD":
        model = TripleScreenM15ImpulseUniversal(asset=asset, sl_pips=9.0, tp_pips=18.0, pip_size=0.0001, max_spread=0.6)
    elif asset == "AUDNZD":
        model = TripleScreenM15ImpulseUniversal(asset=asset, sl_pips=10.0, tp_pips=20.0, pip_size=0.0001, max_spread=2.5)

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
    print("LAUNCHING INSTITUTIONAL 70/30 SPLIT AUDIT: TRIPLE SCREEN M15 IMPULSE (DISC56)")
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

    # All 3 Primary Assets Full Cycle Total
    all_3_full = trades_by_asset["EURUSD"]["full"] + trades_by_asset["EURGBP"]["full"] + trades_by_asset["USDJPY"]["full"]
    if len(all_3_full) > 0:
        p_arr = np.array([t["pips"] for t in all_3_full])
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
            "Asset": "3-PAIR GLOBAL",
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
    out_csv = RESULTS_DIR / "triple_screen_cross_holdout_scorecard.csv"
    df_score.to_csv(out_csv, index=False)

    print("\n" + "=" * 105)
    print("  INSTITUTIONAL 70/30 HOLDOUT SCORECARD: TRIPLE SCREEN M15 IMPULSE (DISC56)")
    print("=" * 105)
    print(f"{'Asset':<16} {'Partition':<32} {'Trades':<8} {'Win Rate':<10} {'Profit Factor':<15} {'Net Pips':<12} {'Max DD':<10} {'Ret/DD':<8} {'Exp (p/t)':<10}")
    print("-" * 105)
    for _, r in df_score.iterrows():
        print(f"{r['Asset']:<16} {r['Partition']:<32} {int(r['Trades']):<8} {r['Win_Rate']:<10} {r['Profit_Factor']:>13.2f}  {r['Net_Pips']:>10.1f}p {r['Max_DD_Pips']:>8.1f}p {r['Ret_DD_Ratio']:>6.2f}x {r['Expectancy_pips']:>8.2f}p")
    print("=" * 105)
    print(f"\n[*] Full Validation Completed in {time.time() - t_start:.2f}s.")


if __name__ == "__main__":
    run_cross_holdout_validation()
