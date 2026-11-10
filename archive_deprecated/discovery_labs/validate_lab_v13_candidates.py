#!/usr/bin/env python3
"""
================================================================================
  CROSS-HOLDOUT AUDIT FOR ALPHA DISCOVERY LAB V13 CANDIDATES
================================================================================
Performs institutional 30% Quarantined Out-of-Sample verification and 100% 5-Year
Full Cycle audit across EURUSD (262w), USDJPY (300w), and EURGBP (300w) (862 total weeks).

Evaluates Lab V13 architectures:
  - M15_50EMA_PIN: M15 50-EMA pin rejection runner
  - LON_PM_EXPANSION: London afternoon / Pre-US consolidation expansion
  - DOUBLE_STRUCT_BO: Double bottom/top neckline breakout
  - M30_ENGULF_RESUME: M30 trend resumption engulfing candle

Calculates exact:
  - Trades, Win Rate, Profit Factor, Net Pips, Max Drawdown, Ret/DD Ratio, Expectancy
  - Partition Breakdown: 70% In-Sample Discovery vs 30% Quarantined Out-of-Sample Holdout vs 100% Full Cycle
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
from scratch.alpha_discovery_lab_v13 import (
    M15FiftyEMAPinRunner,
    LondonPMVolatilityExpansion,
    DoubleStructureNecklineBreakout,
    M30EngulfingTrendResumption
)

RESULTS_DIR = ROOT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def build_models_for_asset(asset: str, model_name: str):
    if model_name == "M15_50EMA_PIN":
        if asset == "EURUSD":
            return M15FiftyEMAPinRunner(asset=asset, sl_pips=9.0, tp_pips=19.8, pip_size=0.0001, max_spread=0.6)
        elif asset == "USDJPY":
            return M15FiftyEMAPinRunner(asset=asset, sl_pips=12.0, tp_pips=26.4, pip_size=0.01, max_spread=0.8)
        elif asset == "EURGBP":
            return M15FiftyEMAPinRunner(asset=asset, sl_pips=8.0, tp_pips=17.6, pip_size=0.0001, max_spread=0.8)
    elif model_name == "LON_PM_EXPANSION":
        if asset == "EURUSD":
            return LondonPMVolatilityExpansion(asset=asset, sl_pips=9.0, tp_pips=18.0, pip_size=0.0001, max_spread=0.6)
        elif asset == "USDJPY":
            return LondonPMVolatilityExpansion(asset=asset, sl_pips=12.0, tp_pips=24.0, pip_size=0.01, max_spread=0.8)
        elif asset == "EURGBP":
            return LondonPMVolatilityExpansion(asset=asset, sl_pips=8.0, tp_pips=16.0, pip_size=0.0001, max_spread=0.8)
    elif model_name == "DOUBLE_STRUCT_BO":
        if asset == "EURUSD":
            return DoubleStructureNecklineBreakout(asset=asset, sl_pips=9.0, tp_pips=18.0, pip_size=0.0001, max_spread=0.6)
        elif asset == "USDJPY":
            return DoubleStructureNecklineBreakout(asset=asset, sl_pips=12.0, tp_pips=24.0, pip_size=0.01, max_spread=0.8)
        elif asset == "EURGBP":
            return DoubleStructureNecklineBreakout(asset=asset, sl_pips=8.0, tp_pips=16.0, pip_size=0.0001, max_spread=0.8)
    elif model_name == "M30_ENGULF_RESUME":
        if asset == "EURUSD":
            return M30EngulfingTrendResumption(asset=asset, sl_pips=10.0, tp_pips=20.0, pip_size=0.0001, max_spread=0.6)
        elif asset == "USDJPY":
            return M30EngulfingTrendResumption(asset=asset, sl_pips=14.0, tp_pips=28.0, pip_size=0.01, max_spread=0.8)
        elif asset == "EURGBP":
            return M30EngulfingTrendResumption(asset=asset, sl_pips=9.0, tp_pips=18.0, pip_size=0.0001, max_spread=0.8)
    return None


def evaluate_week_task(task: Tuple[str, str, int, str]) -> Dict[str, Any]:
    asset, model_name, week_num, partition = task
    pip_size = 0.01 if asset == "USDJPY" else 0.0001
    engine = BacktestEngine(pip_size=pip_size, use_slippage=True)

    ticks, meta = get_week_dataset(week_num, raw_dir=f"data/{asset}")
    if ticks is None or len(ticks) < 500:
        return {"asset": asset, "model": model_name, "week": week_num, "partition": partition, "trades": []}

    model = build_models_for_asset(asset, model_name)
    if model is None:
        return {"asset": asset, "model": model_name, "week": week_num, "partition": partition, "trades": []}

    try:
        raw_trades = engine.run(ticks, model)
        t_list = []
        for i in range(len(raw_trades)):
            row = raw_trades[i]
            t_list.append({
                "strategy": model_name,
                "asset": asset,
                "week": week_num,
                "partition": partition,
                "entry_time": int(row["entry_time"]),
                "exit_time": int(row["exit_time"]),
                "pips": float(row["pips"]),
                "direction": int(row["direction"])
            })
        return {"asset": asset, "model": model_name, "week": week_num, "partition": partition, "trades": t_list}
    except Exception:
        return {"asset": asset, "model": model_name, "week": week_num, "partition": partition, "trades": []}


def calculate_metrics(trades: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(trades)
    if n == 0:
        return {
            "trades": 0, "win_rate": 0.0, "profit_factor": 0.0,
            "net_pips": 0.0, "max_dd": 0.0, "ret_dd": 0.0, "expectancy": 0.0
        }
    pips = [t["pips"] for t in trades]
    wins = [p for p in pips if p > 0]
    losses = [p for p in pips if p < 0]
    wr = len(wins) / n * 100.0
    gp = sum(wins)
    gl = abs(sum(losses))
    pf = gp / gl if gl > 0 else (99.0 if gp > 0 else 0.0)
    net_p = sum(pips)

    # Max Drawdown
    equity = np.cumsum(pips)
    peak = np.maximum.accumulate(equity)
    dd = peak - equity
    max_dd = float(np.max(dd)) if len(dd) > 0 else 0.0
    ret_dd = net_p / max_dd if max_dd > 0 else (99.0 if net_p > 0 else 0.0)
    exp = net_p / n

    return {
        "trades": n,
        "win_rate": round(wr, 1),
        "profit_factor": round(pf, 2),
        "net_pips": round(net_p, 1),
        "max_dd": round(max_dd, 1),
        "ret_dd": round(ret_dd, 2),
        "expectancy": round(exp, 2)
    }


def run_candidate_audit(candidate_models: List[str] = None):
    if candidate_models is None:
        # Default all 4 models
        candidate_models = ["M15_50EMA_PIN", "LON_PM_EXPANSION", "DOUBLE_STRUCT_BO", "M30_ENGULF_RESUME"]

    t_start = time.time()
    print("=" * 90, flush=True)
    print("  COMMENCING 100% 5-YEAR FULL-CYCLE & QUARANTINED HOLDOUT AUDIT", flush=True)
    print("=" * 90, flush=True)

    assets = ["USDJPY", "EURGBP", "EURUSD"]
    all_tasks = []
    for model_name in candidate_models:
        for asset in assets:
            disc_weeks = set(get_discovery_weeks(f"data/{asset}"))
            hold_weeks = set(get_holdout_test_weeks(f"data/{asset}"))
            for w in sorted(disc_weeks):
                all_tasks.append((asset, model_name, w, "DISCOVERY"))
            for w in sorted(hold_weeks):
                all_tasks.append((asset, model_name, w, "HOLDOUT"))

    print(f"[*] Total validation tasks queued: {len(all_tasks)} week-runs across {len(assets)} assets...", flush=True)

    results_by_group: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
    completed = 0

    with ProcessPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(evaluate_week_task, t): t for t in all_tasks}
        for future in as_completed(futures):
            res = future.result()
            completed += 1
            key = (res["model"], res["asset"], res["partition"])
            if key not in results_by_group:
                results_by_group[key] = []
            results_by_group[key].extend(res["trades"])

            if completed % 200 == 0 or completed == len(all_tasks):
                print(f"  [+] Audited {completed:4d}/{len(all_tasks)} weeks in {time.time() - t_start:.1f}s...", flush=True)

    scorecard_rows = []
    print("\n" + "=" * 110, flush=True)
    print(f"{'MODEL':<18} {'ASSET':<8} {'PARTITION':<12} {'TRADES':<8} {'WIN RATE':<10} {'PROFIT FACTOR':<15} {'NET PIPS':<12} {'MAX DD':<10} {'RET/DD':<8}", flush=True)
    print("=" * 110, flush=True)

    for model_name in candidate_models:
        for asset in assets:
            disc_trades = results_by_group.get((model_name, asset, "DISCOVERY"), [])
            hold_trades = results_by_group.get((model_name, asset, "HOLDOUT"), [])
            full_trades = disc_trades + hold_trades

            m_disc = calculate_metrics(disc_trades)
            m_hold = calculate_metrics(hold_trades)
            m_full = calculate_metrics(full_trades)

            for part_name, m in [("Discovery", m_disc), ("Holdout", m_hold), ("Full Cycle", m_full)]:
                p_str = f"{'+' if m['net_pips'] >= 0 else ''}{m['net_pips']:.1f}p"
                dd_str = f"{m['max_dd']:.1f}p"
                r_str = f"{m['ret_dd']:.2f}x"
                print(f"{model_name:<18} {asset:<8} {part_name:<12} {m['trades']:<8} {m['win_rate']:5.1f}%     {m['profit_factor']:5.2f}           {p_str:<12} {dd_str:<10} {r_str:<8}", flush=True)
                scorecard_rows.append({
                    "model": model_name,
                    "asset": asset,
                    "partition": part_name,
                    **m
                })
            print("-" * 110, flush=True)

    df_out = pd.DataFrame(scorecard_rows)
    out_csv = RESULTS_DIR / "lab_v13_cross_holdout_scorecard.csv"
    df_out.to_csv(out_csv, index=False)
    print(f"\n[*] Full scorecard exported to {out_csv}")
    print(f"[*] Audit completed in {time.time() - t_start:.2f} seconds.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="*", default=None, help="Specific models to audit")
    args = parser.parse_args()
    run_candidate_audit(args.models)
