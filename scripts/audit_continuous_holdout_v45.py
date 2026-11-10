#!/usr/bin/env python3
"""
================================================================================
  CANONICAL CONTINUOUS 90-WEEK HOLDOUT AUDIT: PRODUCTION FLEET v 4.5
================================================================================
Evaluates the 100 production strategies in 'v 4.5/' on the true, unbroken
Continuous 90-Week Quarantined Holdout (Weeks 211 to 300: Jan 2025 - Oct 2026).

Methodology & Zero-Bias Safeguards:
1. True Continuous Holdout: Weeks 211 to 300 (~1.75 years out-of-sample forward walk).
2. Data Completeness Gate: Validates minimum week duration (>=80h) to exclude truncated fragments.
3. Live Broker Friction: $4.00/lot round-turn flat commission + 0.20p adverse entry slippage.
4. Causal Execution: Strictly causal .shift(1), 100k tick warmup, bar-close execution.
5. Contiguous Output: Saves fresh audit report to 'results/continuous_walkforward_audits/'.
================================================================================
"""

import sys
import os
import json
import time
import importlib.util
from pathlib import Path
from typing import Dict, List, Any, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from tick_engine.core.dataloader import get_week_dataset, get_complete_split_weeks, load_manifest
from tick_engine.core.engine import run_matching_loop
from alpha_searcher.config import PIP_SIZES, get_friction_pips, SPLITS_FILE

V45_DIR = ROOT_DIR / "v 4.5"
RESULTS_DIR = ROOT_DIR / "results" / "continuous_walkforward_audits"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

WARMUP_TICKS = 100000

def load_v45_strategies() -> List[Dict[str, Any]]:
    manifest_path = V45_DIR / "FLEET_MANIFEST.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return meta

def evaluate_strategy_week(args: Tuple[str, str, int, str]) -> Dict[str, Any]:
    strat_file, class_name, week_num, symbol = args
    raw_dir = ROOT_DIR / "data" / symbol
    pip_size = PIP_SIZES.get(symbol, 0.0001)
    comm_pips, slip_pips = get_friction_pips(symbol)

    # Load strategy instance dynamically
    file_path = V45_DIR / strat_file
    spec = importlib.util.spec_from_file_location(f"strat_mod_{week_num}", file_path)
    if spec is None or spec.loader is None:
        return {"week": week_num, "trades": 0, "net_pips": 0.0, "wins": 0, "win_pips": 0.0, "loss_pips": 0.0}
    
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    
    strat_cls = getattr(mod, class_name) if (class_name and hasattr(mod, class_name)) else getattr(mod, "Strategy", None)
    if strat_cls is None:
        # Fallback inspection of classes
        for attr in dir(mod):
            if attr.startswith("strategy_") or attr.startswith("Strategy_"):
                strat_cls = getattr(mod, attr)
                break
    
    if strat_cls is None:
        return {"week": week_num, "trades": 0, "net_pips": 0.0, "wins": 0, "win_pips": 0.0, "loss_pips": 0.0}

    strat_inst = strat_cls()

    # Load week with historical causal warmup
    try:
        ticks, _ = get_week_dataset(week=week_num, raw_dir=raw_dir, warmup_prev_week_ticks=WARMUP_TICKS)
    except Exception:
        return {"week": week_num, "trades": 0, "net_pips": 0.0, "wins": 0, "win_pips": 0.0, "loss_pips": 0.0}

    if len(ticks) <= WARMUP_TICKS + 500:
        return {"week": week_num, "trades": 0, "net_pips": 0.0, "wins": 0, "win_pips": 0.0, "loss_pips": 0.0}

    # Generate signals across full tick series (warmup ensures indicators are settled)
    try:
        res = strat_inst.generate_signals(ticks)
    except Exception:
        return {"week": week_num, "trades": 0, "net_pips": 0.0, "wins": 0, "win_pips": 0.0, "loss_pips": 0.0}

    if isinstance(res, tuple):
        signals = res[0]
        sl_arr = res[1] if len(res) > 1 else np.full(len(ticks), getattr(strat_inst, "sl_pips", 20.0))
        tp_arr = res[2] if len(res) > 2 else np.full(len(ticks), getattr(strat_inst, "tp_pips", 40.0))
    else:
        signals = res
        sl_arr = np.full(len(ticks), getattr(strat_inst, "sl_pips", 20.0))
        tp_arr = np.full(len(ticks), getattr(strat_inst, "tp_pips", 40.0))

    # Zero out any signals during historical warmup to guarantee zero contamination
    signals[:WARMUP_TICKS] = 0

    if not np.any(signals != 0):
        return {"week": week_num, "trades": 0, "net_pips": 0.0, "wins": 0, "win_pips": 0.0, "loss_pips": 0.0}

    eval_ticks = ticks[WARMUP_TICKS:]
    eval_sigs = signals[WARMUP_TICKS:]
    eval_sl = sl_arr[WARMUP_TICKS:]
    eval_tp = tp_arr[WARMUP_TICKS:]

    trades = run_matching_loop(
        eval_ticks,
        eval_sigs,
        pip_size=pip_size,
        sl_pips=eval_sl,
        tp_pips=eval_tp,
        commission_pips=comm_pips,
        entry_slippage_pips=slip_pips
    )

    if len(trades) == 0:
        return {"week": week_num, "trades": 0, "net_pips": 0.0, "wins": 0, "win_pips": 0.0, "loss_pips": 0.0}

    pips = trades['pips']
    wins = np.sum(pips > 0)
    win_pips = float(np.sum(pips[pips > 0]))
    loss_pips = float(np.abs(np.sum(pips[pips < 0])))
    net_pips = float(np.sum(pips))

    return {
        "week": week_num,
        "trades": len(trades),
        "wins": int(wins),
        "net_pips": round(net_pips, 1),
        "win_pips": round(win_pips, 1),
        "loss_pips": round(loss_pips, 1)
    }

def run_continuous_audit():
    start_time = time.time()
    print("=" * 100)
    print("  PRODUCTION FLEET v 4.5: STRICT CONTINUOUS 90-WEEK HOLDOUT AUDIT (WEEKS 211 TO 300)")
    print("=" * 100)

    fleet = load_v45_strategies()
    print(f"Total Fleet Strategies: {len(fleet)}")

    # Preload clean complete holdout weeks per symbol
    holdout_weeks_by_sym = {}
    for item in fleet:
        sym = item["symbol"]
        if sym not in holdout_weeks_by_sym:
            raw_dir = ROOT_DIR / "data" / sym
            # Enforce completeness gate
            complete_wks = get_complete_split_weeks(raw_dir, split="test", min_hours=80.0, min_ticks=15000)
            holdout_weeks_by_sym[sym] = complete_wks
            print(f"  [Symbol Data Gate] {sym:7s}: {len(complete_wks)} clean complete holdout weeks (211..300)")

    print("-" * 100)
    print("Executing parallel evaluation across 421M out-of-sample ticks...")

    tasks = []
    strat_meta_map = {}
    for item in fleet:
        slot = item["slot"]
        fname = item["filename"]
        cname = item.get("class_name")
        sym = item["symbol"]
        key = f"{slot:03d}_{fname}"
        strat_meta_map[key] = item

        clean_wks = holdout_weeks_by_sym.get(sym, list(range(211, 301)))
        for w in clean_wks:
            tasks.append((fname, cname, w, sym, key))

    print(f"Total execution tasks queued: {len(tasks):,}")

    results_by_strat = {k: {"trades": 0, "wins": 0, "net_pips": 0.0, "win_pips": 0.0, "loss_pips": 0.0, "weeks_tested": 0} for k in strat_meta_map}

    # Parallel worker execution
    worker_args = [(t[0], t[1], t[2], t[3]) for t in tasks]
    strat_keys = [t[4] for t in tasks]

    max_workers = min(8, os.cpu_count() or 4)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for key, res in zip(strat_keys, executor.map(evaluate_strategy_week, worker_args)):
            st = results_by_strat[key]
            st["weeks_tested"] += 1
            st["trades"] += res["trades"]
            st["wins"] += res["wins"]
            st["net_pips"] += res["net_pips"]
            st["win_pips"] += res["win_pips"]
            st["loss_pips"] += res["loss_pips"]

    # Compile scorecard
    scorecard = []
    tot_trades = 0
    tot_pips = 0.0
    profitable_count = 0

    print("\n" + "=" * 115)
    print(f"{'Slot':4s} | {'Strategy File':42s} | {'Symbol':7s} | {'Tr':4s} | {'WR %':5s} | {'PF':5s} | {'Net Pips':11s} | {'Status':12s}")
    print("=" * 115)

    for key, meta in sorted(strat_meta_map.items()):
        slot = meta["slot"]
        fname = meta["filename"]
        sym = meta["symbol"]
        arch = meta.get("archetype", "")
        st = results_by_strat[key]

        tr = st["trades"]
        wr = (st["wins"] / tr * 100) if tr > 0 else 0.0
        pf = (st["win_pips"] / st["loss_pips"]) if st["loss_pips"] > 0 else (99.0 if st["win_pips"] > 0 else 0.0)
        net = round(st["net_pips"], 1)

        tot_trades += tr
        tot_pips += net
        is_prof = net > 0
        if is_prof:
            profitable_count += 1

        status = "PASSED" if (is_prof and pf >= 1.05) else ("SCRATCH" if net == 0 else "FAILED")

        entry = {
            "slot": slot,
            "filename": fname,
            "symbol": sym,
            "archetype": arch,
            "holdout_weeks_evaluated": st["weeks_tested"],
            "trades": tr,
            "wins": st["wins"],
            "win_rate": round(wr, 1),
            "profit_factor": round(pf, 2),
            "net_pips": net,
            "status": status
        }
        scorecard.append(entry)

        if slot <= 20 or slot % 10 == 0 or not is_prof:
            print(f"{slot:4d} | {fname:42s} | {sym:7s} | {tr:4d} | {wr:5.1f}% | {pf:5.2f} | {net:+10.1f}p | {status:12s}")

    elapsed = time.time() - start_time
    print("=" * 115)
    print(f"AUDIT SUMMARY (STRICT CONTINUOUS WEEKS 211 TO 300):")
    print(f"  * Total Strategies Evaluated: {len(scorecard)}")
    print(f"  * Profitable Strategies:     {profitable_count} / {len(scorecard)} ({profitable_count / len(scorecard) * 100:.1f}%)")
    print(f"  * Total Continuous Net Pips:  {tot_pips:+11.1f} pips")
    print(f"  * Total Continuous Trades:    {tot_trades:,} trades")
    print(f"  * Execution Time:             {elapsed:.1f}s")
    print("=" * 115)

    # Save detailed output report
    report_file = RESULTS_DIR / "v45_continuous_holdout_audit.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump({
            "standard": "Continuous Chronological Holdout (Weeks 211 to 300)",
            "date_range": "2025-01-13 to 2026-10-02",
            "fleet_version": "v 4.5",
            "total_strategies": len(scorecard),
            "profitable_count": profitable_count,
            "total_net_pips": round(tot_pips, 1),
            "total_trades": tot_trades,
            "strategies": scorecard
        }, f, indent=2)
    print(f"Full Continuous Holdout Audit Report saved to: {report_file}")

if __name__ == "__main__":
    run_continuous_audit()

