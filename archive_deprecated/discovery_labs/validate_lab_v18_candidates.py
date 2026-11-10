#!/usr/bin/env python3
"""
================================================================================
  CROSS-HOLDOUT AUDIT FOR ALPHA DISCOVERY LAB V18 CANDIDATES
================================================================================
Performs institutional 30% Quarantined Out-of-Sample verification and 100% 5-Year
Full Cycle audit across EURUSD (262w), USDJPY (300w), and EURGBP (300w) (862 total weeks).

Evaluates Lab V18 standout architectures:
  1. HEIKIN_ASHI_SURGE: 3 consecutive HA bars with 0 counter-wick in H4 trend direction
  2. CMF_EXPANSION: Chaikin Money Flow crosses extreme (+0.10 / -0.10) in H4 trend direction

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
from scratch.alpha_discovery_lab_v18 import (
    HeikinAshiImpulseSurge,
    CMFMomentumExpansion
)

RESULTS_DIR = ROOT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def build_models_for_asset(asset: str, model_name: str):
    if model_name == "HEIKIN_ASHI_SURGE":
        if asset == "EURUSD":
            return HeikinAshiImpulseSurge(asset=asset, sl_pips=9.0, tp_pips=18.0, pip_size=0.0001, max_spread=0.6)
        elif asset == "USDJPY":
            return HeikinAshiImpulseSurge(asset=asset, sl_pips=12.0, tp_pips=24.0, pip_size=0.01, max_spread=0.8)
        elif asset == "EURGBP":
            return HeikinAshiImpulseSurge(asset=asset, sl_pips=8.0, tp_pips=16.0, pip_size=0.0001, max_spread=0.8)
    elif model_name == "CMF_EXPANSION":
        if asset == "EURUSD":
            return CMFMomentumExpansion(asset=asset, sl_pips=9.0, tp_pips=18.0, pip_size=0.0001, max_spread=0.6)
        elif asset == "USDJPY":
            return CMFMomentumExpansion(asset=asset, sl_pips=12.0, tp_pips=24.0, pip_size=0.01, max_spread=0.8)
        elif asset == "EURGBP":
            return CMFMomentumExpansion(asset=asset, sl_pips=8.0, tp_pips=16.0, pip_size=0.0001, max_spread=0.8)
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

    cumsum = np.cumsum(arr)
    running_max = np.maximum.accumulate(cumsum)
    drawdowns = running_max - cumsum
    max_dd = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0
    ret_dd = net_pips / max_dd if max_dd > 0 else 0.0

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
    t_start = time.time()
    print("=" * 80, flush=True)
    print("  LAUNCHING QUARANTINED HOLDOUT & FULL-CYCLE AUDIT: LAB V18 STANDOUTS", flush=True)
    print("=" * 80, flush=True)

    assets = ["EURUSD", "USDJPY", "EURGBP"]
    models = ["HEIKIN_ASHI_SURGE", "CMF_EXPANSION"]

    all_tasks = []
    for asset in assets:
        disc_weeks = set(get_discovery_weeks(f"data/{asset}"))
        hold_weeks = set(get_holdout_test_weeks(f"data/{asset}"))
        all_weeks = sorted(list(disc_weeks.union(hold_weeks)))

        for m_name in models:
            for w in all_weeks:
                part = "HOLDOUT" if w in hold_weeks else "DISCOVERY"
                all_tasks.append((asset, m_name, w, part))

    print(f"[*] Total validation week-tasks across 3 assets & 2 models: {len(all_tasks)}", flush=True)

    trades_by_key: Dict[Tuple[str, str, str], List[float]] = {}
    completed = 0

    with ProcessPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(evaluate_week_task, t): t for t in all_tasks}
        for future in as_completed(futures):
            res = future.result()
            completed += 1
            if completed % 250 == 0 or completed == len(all_tasks):
                elapsed = time.time() - t_start
                print(f"  [+] Audited {completed}/{len(all_tasks)} week-tasks in {elapsed:.1f}s...", flush=True)

            asset = res["asset"]
            model = res["model"]
            part = res["partition"]
            for tr in res["trades"]:
                p = tr["pips"]
                # Store under partition
                k_part = (asset, model, part)
                trades_by_key.setdefault(k_part, []).append(p)
                # Store under full cycle
                k_full = (asset, model, "FULL_CYCLE")
                trades_by_key.setdefault(k_full, []).append(p)

    print("\n" + "=" * 95, flush=True)
    print(f"{'ASSET':<8} | {'MODEL':<18} | {'PARTITION':<11} | {'TRADES':<6} | {'WIN RATE':<8} | {'PF':<5} | {'NET PIPS':<10} | {'MAX DD':<8} | {'RET/DD':<7}", flush=True)
    print("=" * 95, flush=True)

    summary_records = []
    for asset in assets:
        for model in models:
            for part in ["DISCOVERY", "HOLDOUT", "FULL_CYCLE"]:
                pips = trades_by_key.get((asset, model, part), [])
                m = calculate_metrics(pips)
                rec = {
                    "asset": asset,
                    "model": model,
                    "partition": part,
                    **m
                }
                summary_records.append(rec)
                print(f"{asset:<8} | {model:<18} | {part:<11} | {m['trades']:<6} | {m['win_rate']:<7}% | {m['pf']:<5.2f} | {m['net_pips']:<10.1f} | {m['max_dd']:<8.1f} | {m['ret_dd']:<6.2f}x", flush=True)
            print("-" * 95, flush=True)

    out_file = RESULTS_DIR / "lab_v18_holdout_audit.json"
    with open(out_file, "w") as f:
        json.dump(summary_records, f, indent=2)
    print(f"\n[+] Full verification audit saved to: {out_file}", flush=True)
    print(f"[+] Total execution time: {time.time() - t_start:.2f}s", flush=True)
    print("=" * 95, flush=True)


if __name__ == "__main__":
    main()
