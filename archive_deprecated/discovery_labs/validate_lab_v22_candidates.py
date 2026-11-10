#!/usr/bin/env python3
"""
================================================================================
  CROSS-HOLDOUT AUDIT FOR ALPHA DISCOVERY LAB V22 CANDIDATES
================================================================================
Performs institutional 30% Quarantined Out-of-Sample verification and 100% 5-Year
Full Cycle audit across EURUSD (262w), USDJPY (300w), and EURGBP (300w) (862 total weeks).

Evaluates Lab V22 candidates:
  1. WEEKLY_PIVOT_TREND_PULLBACK
  2. VWAP_1SD_TREND_REBOUND
  3. KELTNER_MIDLINE_EXPANSION
  4. THREE_BAR_INSIDE_CONFIRMATION

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
from scratch.alpha_discovery_lab_v22 import build_model_v22

RESULTS_DIR = ROOT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_week_task(task: Tuple[str, str, int, str]) -> Dict[str, Any]:
    asset, model_name, week_num, partition = task
    pip_size = 0.01 if asset == "USDJPY" else 0.0001
    engine = BacktestEngine(pip_size=pip_size, use_slippage=True)

    ticks, meta = get_week_dataset(week_num, raw_dir=f"data/{asset}")
    if ticks is None or len(ticks) < 500:
        return {"asset": asset, "model": model_name, "week": week_num, "partition": partition, "trades": []}

    model = build_model_v22(model_name, asset)
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
    print("  QUARANTINED CROSS-HOLDOUT AUDIT: ALPHA DISCOVERY LAB V22 CANDIDATES", flush=True)
    print("=" * 80, flush=True)
    t0 = time.time()

    assets = ["EURUSD", "USDJPY", "EURGBP"]
    models = [
        "WEEKLY_PIVOT_TREND_PULLBACK",
        "VWAP_1SD_TREND_REBOUND",
        "KELTNER_MIDLINE_EXPANSION",
        "THREE_BAR_INSIDE_CONFIRMATION"
    ]

    tasks: List[Tuple[str, str, int, str]] = []

    for asset in assets:
        disc_weeks = set(get_discovery_weeks(f"data/{asset}"))
        hold_weeks = set(get_holdout_test_weeks(f"data/{asset}"))
        all_weeks = sorted(list(disc_weeks.union(hold_weeks)))

        for m in models:
            for w in all_weeks:
                part = "HOLDOUT" if w in hold_weeks else "DISCOVERY"
                tasks.append((asset, m, w, part))

    print(f"[*] Total week-tasks across 3 assets & {len(models)} models: {len(tasks)}", flush=True)

    trade_store: Dict[str, Dict[str, Dict[str, List[float]]]] = {
        m: {a: {"DISCOVERY": [], "HOLDOUT": [], "FULL": []} for a in assets}
        for m in models
    }

    max_workers = max(1, (os.cpu_count() or 4) - 1)
    completed_count = 0

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(evaluate_week_task, t): t for t in tasks}

        for fut in as_completed(futures):
            res = fut.result()
            m = res["model"]
            a = res["asset"]
            part = res["partition"]
            for tr in res["trades"]:
                p = tr["pips"]
                trade_store[m][a][part].append(p)
                trade_store[m][a]["FULL"].append(p)

            completed_count += 1
            if completed_count % 300 == 0 or completed_count == len(tasks):
                elapsed = time.time() - t0
                print(f"  [+] Audited {completed_count}/{len(tasks)} week-tasks in {elapsed:.1f}s...", flush=True)

    print("\n" + "=" * 80, flush=True)
    print("  EXHAUSTIVE METRICS & HOLDOUT COMPARISON MATRIX", flush=True)
    print("=" * 80, flush=True)

    summary_results: Dict[str, Any] = {}

    for m in models:
        print(f"\n>>> MODEL: {m}", flush=True)
        print(f"{'Asset':<8} | {'Partition':<10} | {'Trades':<6} | {'WinRate':<7} | {'PF':<6} | {'NetPips':<9} | {'MaxDD':<7} | {'Ret/DD':<6} | {'Expectancy':<6}", flush=True)
        print("-" * 80, flush=True)
        summary_results[m] = {}

        for a in assets:
            summary_results[m][a] = {}
            for part in ["DISCOVERY", "HOLDOUT", "FULL"]:
                pips = trade_store[m][a][part]
                met = calculate_metrics(pips)
                summary_results[m][a][part] = met
                print(f"{a:<8} | {part:<10} | {met['trades']:<6} | {met['win_rate']:<6.1f}% | {met['pf']:<6.2f} | {met['net_pips']:<9.1f} | {met['max_dd']:<7.1f} | {met['ret_dd']:<6.2f} | {met['expectancy']:<6.2f}", flush=True)

    out_file = RESULTS_DIR / "lab_v22_holdout_audit_results.json"
    with open(out_file, "w") as f:
        json.dump(summary_results, f, indent=2)

    total_time = time.time() - t0
    print(f"\n[OK] Validation complete in {total_time:.1f}s! Results saved to {out_file}", flush=True)


if __name__ == "__main__":
    main()
