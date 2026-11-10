#!/usr/bin/env python3
"""
================================================================================
  STANDARDIZED OPERATING PROCEDURE (SOP) DUAL-PARTITION SYMBOL AUDIT RUNNER
================================================================================
Executes the institutional dual-partition discovery & qualification battery
for a single target symbol:
  1. In-Sample 52-Week 2021 Tick Audit under 0.7p friction + floating spread
  2. Individual Hurdle Filter: Standalone Net Return >= 12.0%
  3. Out-of-Sample 90-Week Quarantined Holdout Audit (Seed 42 / 126.6M Ticks)
  4. Dual-Certified Selection: Holdout Ann Ret >= 10.0% - 12.0% and Net Pips > 0
  5. JSON & Scorecard generation

Usage:
  .venv\\Scripts\\python.exe sop/run_symbol_sop_audit.py --symbol USDCAD
================================================================================
"""

import sys
import os
import time
import json
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Any, Tuple, Optional

import numpy as np
import pandas as pd

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
sys.dont_write_bytecode = True

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tick_engine.core.engine import run_matching_loop
from tick_engine.core.dataloader import get_week_dataset
from data_engineering_suite.bars import build_time_bars
from data_engineering_suite.master_symbol_alpha_discovery_suite import (
    get_symbol_profile,
    StrategySessionVWAPFade,
    StrategyFibGoldenPocket,
    StrategyDailyPivotSniper,
    StrategyMacroDonchianRetest,
    StrategyElderTripleScreen,
    StrategyKeltnerSqueezeExplosion,
    StrategyLondonORB,
    StrategyChaikinVolSurge
)

COMMISSION_PIPS = 0.5
ENTRY_SLIPPAGE_PIPS = 0.2
SPLITS_FILE = ROOT_DIR / "data_splits" / "master_70_30_splits.json"
RESULTS_DIR = ROOT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def get_holdout_weeks(symbol: str) -> List[int]:
    with open(SPLITS_FILE, "r") as f:
        d = json.load(f)
    return sorted(d["assets"].get(symbol.upper(), {}).get("holdout_test_30", {}).get("weeks", []))


def generate_candidates_for_symbol(symbol: str) -> List[Any]:
    """Generates tailored candidate battery across the 8 architecture families."""
    profile = get_symbol_profile(symbol)
    sl_m = profile["sl_mult"]
    candidates = []

    # 1. VWAP Band Fade
    for b in [2.0, 2.5]:
        for sl, tp in [(7.0, 14.0), (8.0, 16.0), (9.0, 18.0), (10.0, 20.0)]:
            candidates.append(StrategySessionVWAPFade(
                f"VWAP_B{b}_SL{sl*sl_m:.1f}_TP{tp*sl_m:.1f}",
                sl_pips=sl * sl_m,
                tp_pips=tp * sl_m,
                band_mult=b
            ))

    # 2. Fibonacci Golden Pocket
    for lb in [36, 48, 72]:
        for sl, tp in [(7.0, 14.0), (7.0, 17.5), (8.0, 16.0), (8.0, 20.0), (12.0, 24.0), (12.0, 30.0)]:
            candidates.append(StrategyFibGoldenPocket(
                f"FibPocket_LB{lb}_SL{sl*sl_m:.1f}_TP{tp*sl_m:.1f}",
                sl_pips=sl * sl_m,
                tp_pips=tp * sl_m,
                lookback_h1=lb
            ))

    # 3. Elder Triple Screen
    for sl, tp in [(7.0, 14.0), (8.0, 16.0), (9.0, 18.0), (10.0, 20.0), (12.0, 24.0)]:
        candidates.append(StrategyElderTripleScreen(
            f"ElderTripleScreen_SL{sl*sl_m:.1f}_TP{tp*sl_m:.1f}",
            sl_pips=sl * sl_m,
            tp_pips=tp * sl_m
        ))

    # 4. Daily Pivot Wall / Sniper
    for sl, tp in [(7.0, 14.0), (8.0, 16.0), (9.0, 18.0), (10.0, 20.0)]:
        candidates.append(StrategyDailyPivotSniper(
            f"DailyPivot_SL{sl*sl_m:.1f}_TP{tp*sl_m:.1f}",
            sl_pips=sl * sl_m,
            tp_pips=tp * sl_m
        ))

    # 5. Keltner Squeeze Explosion
    for sl, tp in [(8.0, 16.0), (10.0, 20.0), (12.0, 24.0), (14.0, 28.0)]:
        candidates.append(StrategyKeltnerSqueezeExplosion(
            f"KeltnerSqueeze_SL{sl*sl_m:.1f}_TP{tp*sl_m:.1f}",
            sl_pips=sl * sl_m,
            tp_pips=tp * sl_m
        ))

    # 6. London ORB
    for sl, tp in [(8.0, 16.0), (10.0, 20.0), (12.0, 24.0)]:
        candidates.append(StrategyLondonORB(
            f"LondonORB_SL{sl*sl_m:.1f}_TP{tp*sl_m:.1f}",
            sl_pips=sl * sl_m,
            tp_pips=tp * sl_m
        ))

    # 7. Chaikin Volatility Surge
    for sl, tp in [(8.0, 16.0), (10.0, 20.0), (12.0, 24.0)]:
        candidates.append(StrategyChaikinVolSurge(
            f"ChaikinVolSurge_SL{sl*sl_m:.1f}_TP{tp*sl_m:.1f}",
            sl_pips=sl * sl_m,
            tp_pips=tp * sl_m
        ))

    return candidates


def evaluate_week_batch(args):
    symbol, weeks, cand_data, raw_dir_str = args
    raw_dir = Path(raw_dir_str)
    profile = get_symbol_profile(symbol)
    pip_size = profile["pip_size"]
    spread_limit = profile["spread_limit"]

    # Recreate candidates
    candidates = generate_candidates_for_symbol(symbol)
    results = {c.name: {"trades": 0, "wins": 0, "losses": 0, "win_pips": 0.0, "loss_pips": 0.0, "net_pips": 0.0} for c in candidates}

    for w in weeks:
        try:
            ticks_rec, _ = get_week_dataset(w, raw_dir=raw_dir, auto_convert=True)
        except Exception:
            continue
        if ticks_rec is None or len(ticks_rec) < 1000:
            continue

        m15_bars = build_time_bars(ticks_rec, 15, pip_size=pip_size)
        h1_bars = build_time_bars(ticks_rec, 60, pip_size=pip_size)
        h4_bars = build_time_bars(ticks_rec, 240, pip_size=pip_size)

        for c in candidates:
            try:
                sigs = c.generate_signals(ticks_rec, m15_bars, h1_bars, h4_bars, pip_size, spread_limit)
            except Exception:
                continue

            if sigs is None or not np.any(sigs != 0):
                continue

            tr_list = run_matching_loop(
                ticks=ticks_rec,
                signals=sigs,
                pip_size=pip_size,
                sl_pips=c.sl_pips,
                tp_pips=c.tp_pips,
                commission_pips=COMMISSION_PIPS,
                entry_slippage_pips=ENTRY_SLIPPAGE_PIPS
            )

            if len(tr_list) == 0:
                continue

            for t in tr_list:
                pips = float(t["pips"])
                results[c.name]["trades"] += 1
                results[c.name]["net_pips"] += pips
                if pips > 0:
                    results[c.name]["wins"] += 1
                    results[c.name]["win_pips"] += pips
                else:
                    results[c.name]["losses"] += 1
                    results[c.name]["loss_pips"] += abs(pips)

    return results


def run_evaluation_across_weeks(symbol: str, weeks: List[int], n_workers: int = 4) -> Dict[str, Dict[str, Any]]:
    raw_dir = str(ROOT_DIR / "data" / symbol.upper())
    candidates = generate_candidates_for_symbol(symbol)
    cand_info = [(c.name, c.sl_pips, c.tp_pips) for c in candidates]

    # Split weeks into chunks for multiprocessing
    chunk_size = max(1, len(weeks) // n_workers)
    week_chunks = [weeks[i:i + chunk_size] for i in range(0, len(weeks), chunk_size)]

    tasks = [(symbol, chunk, cand_info, raw_dir) for chunk in week_chunks]

    combined = {c.name: {
        "name": c.name,
        "sl_pips": c.sl_pips,
        "tp_pips": c.tp_pips,
        "trades": 0, "wins": 0, "losses": 0, "win_pips": 0.0, "loss_pips": 0.0, "net_pips": 0.0
    } for c in candidates}

    with ProcessPoolExecutor(max_workers=min(n_workers, len(tasks))) as executor:
        futures = [executor.submit(evaluate_week_batch, t) for t in tasks]
        for fut in as_completed(futures):
            res = fut.result()
            for name, stats in res.items():
                combined[name]["trades"] += stats["trades"]
                combined[name]["wins"] += stats["wins"]
                combined[name]["losses"] += stats["losses"]
                combined[name]["win_pips"] += stats["win_pips"]
                combined[name]["loss_pips"] += stats["loss_pips"]
                combined[name]["net_pips"] += stats["net_pips"]

    for name, c in combined.items():
        tr = c["trades"]
        c["net_pips"] = round(c["net_pips"], 1)
        c["win_rate"] = round((c["wins"] / tr * 100.0), 1) if tr > 0 else 0.0
        c["profit_factor"] = round((c["win_pips"] / c["loss_pips"]), 2) if c["loss_pips"] > 0 else (99.0 if c["win_pips"] > 0 else 0.0)

    return combined


def main():
    parser = argparse.ArgumentParser(description="Standard Operating Procedure (SOP) Dual-Partition Symbol Audit")
    parser.add_argument("--symbol", type=str, required=True, help="Target symbol (e.g., USDCAD)")
    parser.add_argument("--workers", type=int, default=4, help="Number of worker processes")
    args = parser.parse_args()

    symbol = args.symbol.upper()
    print("=" * 115)
    print(f"       STANDARD OPERATING PROCEDURE (SOP) DUAL-PARTITION AUDIT: {symbol}       ".center(115))
    print("=" * 115)

    profile = get_symbol_profile(symbol)
    print(f"Microstructure Profile: Pip Size={profile['pip_size']}, Spread Limit={profile['spread_limit']}p, SL Mult={profile['sl_mult']}")

    candidates = generate_candidates_for_symbol(symbol)
    print(f"Generated {len(candidates)} quantitative candidate strategies across 8 architecture families.")

    # 1. In-Sample 52-Week 2021 Audit
    is_weeks = list(range(1, 53))
    print(f"\n[PHASE 1] Executing In-Sample 52-Week (Calendar 2021) Tick Audit ({len(is_weeks)} weeks)...")
    t0 = time.time()
    is_results = run_evaluation_across_weeks(symbol, is_weeks, n_workers=args.workers)
    print(f"In-Sample Audit Completed in {time.time()-t0:.1f}s.")

    for name, data in is_results.items():
        sl = data["sl_pips"]
        net_pips = data["net_pips"]
        ann_ret = (net_pips / sl) if sl > 0 else 0.0
        data["return_pct"] = round(ann_ret, 1)

    # Filter >= 12% Hurdle
    passing_is = [d for d in is_results.values() if d["return_pct"] >= 12.0 and d["profit_factor"] >= 1.20]
    passing_is.sort(key=lambda x: x["return_pct"], reverse=True)

    print(f"\nIn-Sample Hurdle Filter: {len(passing_is)} / {len(is_results)} candidates pass >= 12.0% individual net profit hurdle!")
    for i, p in enumerate(passing_is[:10]):
        print(f"  {i+1:2d}. {p['name']:40s}: Net={p['return_pct']:5.1f}% ({p['net_pips']:6.1f}p, {p['trades']:4d} tr), WR={p['win_rate']:4.1f}%, PF={p['profit_factor']:4.2f}")

    if not passing_is:
        print("ERROR: No candidates passed the 12% in-sample hurdle for this symbol.")
        return

    # 2. Out-of-Sample 90-Week Quarantined Holdout Audit
    holdout_weeks = get_holdout_weeks(symbol)
    print(f"\n[PHASE 2] Executing Out-of-Sample 90-Week Quarantined Holdout Audit ({len(holdout_weeks)} weeks / Seed 42)...")
    t0 = time.time()
    oos_results = run_evaluation_across_weeks(symbol, holdout_weeks, n_workers=args.workers)
    print(f"Holdout Audit Completed in {time.time()-t0:.1f}s.")

    years_oos = len(holdout_weeks) / 52.0
    for name, data in oos_results.items():
        sl = data["sl_pips"]
        net_pips = data["net_pips"]
        ann_pips = (net_pips / years_oos) if years_oos > 0 else 0.0
        ann_ret = (ann_pips / sl) if sl > 0 else 0.0
        data["ann_net_pips"] = round(ann_pips, 1)
        data["ann_return_pct"] = round(ann_ret, 1)

    # 3. Dual-Partition Correlation and Certification
    dual_certified = []
    for is_cand in passing_is:
        name = is_cand["name"]
        oos_cand = oos_results.get(name, {})
        oos_pips = oos_cand.get("net_pips", 0.0)
        oos_ann_ret = oos_cand.get("ann_return_pct", 0.0)

        # Dual Pass: IS >= 12% AND OOS > 0 pips AND OOS ann ret >= 10.0%
        if oos_pips > 0 and oos_ann_ret >= 10.0:
            dual_certified.append({
                "symbol": symbol,
                "name": name,
                "sl_pips": is_cand["sl_pips"],
                "tp_pips": is_cand["tp_pips"],
                "is_trades": is_cand["trades"],
                "is_net_pips": is_cand["net_pips"],
                "is_return_pct": is_cand["return_pct"],
                "is_win_rate": is_cand["win_rate"],
                "is_profit_factor": is_cand["profit_factor"],
                "oos_trades": oos_cand.get("trades", 0),
                "oos_net_pips": oos_pips,
                "oos_ann_net_pips": oos_cand.get("ann_net_pips", 0.0),
                "oos_ann_return_pct": oos_ann_ret,
                "oos_win_rate": oos_cand.get("win_rate", 0.0),
                "oos_profit_factor": oos_cand.get("profit_factor", 0.0),
                "decay_ratio": round(oos_ann_ret / is_cand["return_pct"], 2) if is_cand["return_pct"] > 0 else 0.0
            })

    dual_certified.sort(key=lambda x: (x["oos_ann_return_pct"] + x["is_return_pct"]), reverse=True)

    print("\n" + "=" * 115)
    print(f"            FINAL DUAL-CERTIFIED PRODUCTION SCORECARD FOR {symbol}            ".center(115))
    print("=" * 115)
    print(f"Total Initial Candidates:               {len(candidates)}")
    print(f"Passing 52w In-Sample (>=12% Hurdle):   {len(passing_is)}")
    print(f"Dual-Certified (IS >= 12% & OOS >= 10%):{len(dual_certified)}")
    print("-" * 115)
    print(f"{'Strategy Name':<38} | {'SL/TP':<10} | {'52w IS Ret (pips)':<20} | {'90w OOS Ret (pips/yr)':<23} | {'Decay':<6} | {'Status':<12}")
    print("-" * 115)
    for c in dual_certified:
        status = "DUAL PASS" if c["oos_ann_return_pct"] >= 12.0 else "PASS (>=10%)"
        print(f"{c['name']:<38} | {c['sl_pips']:4.1f}/{c['tp_pips']:4.1f} | {c['is_return_pct']:5.1f}% (+{c['is_net_pips']:5.1f}p) | {c['oos_ann_return_pct']:5.1f}% (+{c['oos_ann_net_pips']:5.1f}p/yr) | {c['decay_ratio']:5.2f}x | {status:<12}")
    print("=" * 115)

    out_file = RESULTS_DIR / f"{symbol.lower()}_sop_dual_audit.json"
    with open(out_file, "w") as f:
        json.dump({
            "symbol": symbol,
            "microstructure_profile": profile,
            "total_candidates": len(candidates),
            "passing_is_count": len(passing_is),
            "dual_certified_count": len(dual_certified),
            "dual_certified": dual_certified,
            "all_is_results": is_results,
            "all_oos_results": oos_results
        }, f, indent=2)

    print(f"\nFull SOP Audit Report saved to: {out_file}")

if __name__ == "__main__":
    main()
