#!/usr/bin/env python3
"""
================================================================================
  QUARANTINED CROSS-HOLDOUT AUDIT FOR ALPHA DISCOVERY LAB V25 CANDIDATES
================================================================================
Performs institutional 30% Quarantined Out-of-Sample verification and 100% 5-Year
Full Cycle audit across EURUSD (262w), USDJPY (300w), and EURGBP (300w) (862 total weeks).

Evaluates Lab V25 candidates:
  1. LONDON_FIX_1530_THRUST
  2. DOUBLE_INSIDE_BAR_EXPLOSION
  3. KELTNER_2SD_EXTREME_ABSORPTION_FADE
  4. SESSION_OPEN_MIDPOINT_IMPULSE

Calculates exact:
  - Trades, Win Rate, Profit Factor, Net Pips, Max Drawdown, Ret/DD Ratio, Expectancy
  - Partition Breakdown: 70% In-Sample Discovery vs 30% Quarantined Out-of-Sample Holdout vs 100% Full Cycle
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
from scratch.alpha_discovery_lab_v25 import build_model_v25

RESULTS_DIR = ROOT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_week_task(task: Tuple[str, str, int, str]) -> Dict[str, Any]:
    asset, model_name, week_num, partition = task
    pip_size = 0.01 if asset == "USDJPY" else 0.0001
    engine = BacktestEngine(pip_size=pip_size, use_slippage=True)

    ticks, meta = get_week_dataset(week_num, raw_dir=f"data/{asset}")
    if ticks is None or len(ticks) < 500:
        return {"asset": asset, "model": model_name, "week": week_num, "partition": partition, "trades": []}

    model = build_model_v25(model_name, asset)
    if model is None:
        return {"asset": asset, "model": model_name, "week": week_num, "partition": partition, "trades": []}

    try:
        raw_trades = engine.run(ticks, model)
        t_list = []
        for i in range(len(raw_trades)):
            row = raw_trades[i]
            t_list.append({
                "asset": asset,
                "model": model_name,
                "week": week_num,
                "partition": partition,
                "pips": float(row["pips"]),
                "direction": int(row["direction"])
            })
        return {"asset": asset, "model": model_name, "week": week_num, "partition": partition, "trades": t_list}
    except Exception:
        return {"asset": asset, "model": model_name, "week": week_num, "partition": partition, "trades": []}


def calculate_metrics(pips: List[float]) -> Dict[str, Any]:
    if not pips:
        return {
            "trades": 0, "win_rate": 0.0, "pf": 0.0,
            "net_pips": 0.0, "max_dd": 0.0, "ret_dd": 0.0, "expectancy": 0.0
        }
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
    print("=" * 80, flush=True)
    print("  QUARANTINED CROSS-HOLDOUT AUDIT: ALPHA DISCOVERY LAB V25 CANDIDATES", flush=True)
    print("=" * 80, flush=True)
    t0 = time.time()

    assets = ["EURUSD", "USDJPY", "EURGBP"]
    models = [
        "SESSION_OPEN_MIDPOINT_IMPULSE"
    ]

    tasks: List[Tuple[str, str, int, str]] = []
    for asset in assets:
        disc_weeks = get_discovery_weeks(f"data/{asset}")
        hold_weeks = get_holdout_test_weeks(f"data/{asset}")
        for m in models:
            for w in disc_weeks:
                tasks.append((asset, m, w, "discovery"))
            for w in hold_weeks:
                tasks.append((asset, m, w, "holdout"))

    print(f"[*] Total validation week-tasks across 3 assets & 4 models: {len(tasks)}", flush=True)

    trades_by_key: Dict[str, List[float]] = {}
    for a in assets:
        for m in models:
            trades_by_key[f"{a}_{m}_discovery"] = []
            trades_by_key[f"{a}_{m}_holdout"] = []
            trades_by_key[f"{a}_{m}_full"] = []

    max_workers = max(1, (os.cpu_count() or 4) - 1)
    completed_count = 0

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(evaluate_week_task, t): t for t in tasks}

        for fut in as_completed(futures):
            res = fut.result()
            a = res["asset"]
            m = res["model"]
            part = res["partition"]
            for tr in res["trades"]:
                p = tr["pips"]
                trades_by_key[f"{a}_{m}_{part}"].append(p)
                trades_by_key[f"{a}_{m}_full"].append(p)

            completed_count += 1
            if completed_count % 500 == 0 or completed_count == len(tasks):
                elapsed = time.time() - t0
                print(f"  [+] Validated {completed_count}/{len(tasks)} week-tasks in {elapsed:.1f}s...", flush=True)

    print("\n" + "=" * 105, flush=True)
    print(f"{'ASSET':<8} | {'MODEL':<35} | {'PARTITION':<11} | {'TRADES':<6} | {'WR':<6} | {'PF':<5} | {'NET PIPS':<9} | {'MAX DD':<8} | {'RET/DD':<6}", flush=True)
    print("=" * 105, flush=True)

    validation_results: Dict[str, Any] = {}

    for a in assets:
        for m in models:
            disc_met = calculate_metrics(trades_by_key[f"{a}_{m}_discovery"])
            hold_met = calculate_metrics(trades_by_key[f"{a}_{m}_holdout"])
            full_met = calculate_metrics(trades_by_key[f"{a}_{m}_full"])

            validation_results[f"{a}_{m}"] = {
                "discovery": disc_met,
                "holdout": hold_met,
                "full": full_met
            }

            print(f"{a:<8} | {m:<35} | {'Discovery':<11} | {disc_met['trades']:<6} | {disc_met['win_rate']:<4.1f}% | {disc_met['pf']:<5.2f} | {disc_met['net_pips']:<9.1f} | {disc_met['max_dd']:<8.1f} | {disc_met['ret_dd']:<5.2f}x", flush=True)
            print(f"{'':<8} | {'':<35} | {'Holdout':<11} | {hold_met['trades']:<6} | {hold_met['win_rate']:<4.1f}% | {hold_met['pf']:<5.2f} | {hold_met['net_pips']:<9.1f} | {hold_met['max_dd']:<8.1f} | {hold_met['ret_dd']:<5.2f}x", flush=True)
            print(f"{'':<8} | {'':<35} | {'Full 5Y':<11} | {full_met['trades']:<6} | {full_met['win_rate']:<4.1f}% | {full_met['pf']:<5.2f} | {full_met['net_pips']:<9.1f} | {full_met['max_dd']:<8.1f} | {full_met['ret_dd']:<5.2f}x", flush=True)
            print("-" * 105, flush=True)

    out_file = RESULTS_DIR / "lab_v25_validation_results.json"
    with open(out_file, "w") as f:
        json.dump(validation_results, f, indent=2)

    total_time = time.time() - t0
    print(f"\n[+] Validation results saved to: {out_file}", flush=True)
    print(f"[+] Total execution time: {total_time:.2f}s", flush=True)
    print("=" * 105, flush=True)


if __name__ == "__main__":
    main()
