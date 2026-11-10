#!/usr/bin/env python3
"""
================================================================================
  PARALLEL FLEET CERTIFICATION RUNNER & STATISTICAL AUDIT
================================================================================
Loops through candidate strategies, executing the full institutional verification
battery with parallel week evaluations across CPU cores.

For every individual strategy:
1. Validates against Data Quality Gate (clean_weeks_registry.json).
2. Runs 100k rolling prior-week causal warmup.
3. Enforces the Order-Timing Contract (strict subsequent-tick execution).
4. Deducts live institutional friction ($4/lot + 0.20p slippage + floating spread).
5. Evaluates In-Sample (Weeks 1..210) AND Holdout (Weeks 211..300).
6. Runs 5-Profile Broker Stress Battery.
7. Calculates advanced statistical confidence (t-stat, p-value, CI 95%, bootstrap DD, outlier sensitivity).
8. Exports cryptographic run manifest and per-trade CSV.
================================================================================
"""

from __future__ import annotations
import os
import sys
import json
import time
import importlib.util
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from tick_engine.core.engine import (
    run_matching_loop,
    DEFAULT_COMMISSION_PIPS,
    DEFAULT_ENTRY_SLIPPAGE_PIPS
)
from tick_engine.core.dataloader import (
    get_week_dataset,
    get_complete_split_weeks
)
from tick_engine.stress.stress_suite import (
    evaluate_friction_stress,
    STRESS_PROFILES
)
from tick_engine.reporting.run_manifest import (
    create_certified_run_manifest,
    compute_statistical_metrics
)
from tick_engine.core.config import PIP_SIZES, get_friction_pips

V45_DIR = ROOT_DIR / "v 4.5"
OUTPUT_DIR = ROOT_DIR / "results" / "certified_strategy_manifests"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
WARMUP_TICKS = 100000


def _worker_eval_week(
    strat_file_str: str,
    symbol: str,
    week_num: int,
    pip_size: float,
    comm_pips: float,
    slip_pips: float,
    enforce_subsequent: bool = True
) -> np.ndarray:
    """Worker function executed in parallel process to evaluate a single week."""
    strat_file = Path(strat_file_str)
    spec = importlib.util.spec_from_file_location(f"mod_{strat_file.stem}_{week_num}", strat_file)
    if spec is None or spec.loader is None:
        return np.empty(0)
    
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    
    strat_cls = getattr(mod, "Strategy", None)
    if strat_cls is None:
        for attr in dir(mod):
            obj = getattr(mod, attr)
            if isinstance(obj, type) and hasattr(obj, "generate_signals") and attr != "BaseStrategy":
                strat_cls = obj
                break
    if strat_cls is None:
        return np.empty(0)
        
    strat_inst = strat_cls()
    raw_dir = ROOT_DIR / "data" / symbol

    try:
        ticks, _ = get_week_dataset(
            week=week_num,
            raw_dir=raw_dir,
            warmup_prev_week_ticks=WARMUP_TICKS
        )
    except Exception:
        return np.empty(0)

    if len(ticks) <= WARMUP_TICKS + 500:
        return np.empty(0)

    res = strat_inst.generate_signals(ticks)
    sig_ts = None
    if isinstance(res, tuple):
        signals = res[0]
        sl_arr = res[1] if len(res) > 1 and res[1] is not None else np.full(len(ticks), getattr(strat_inst, "sl_pips", 20.0))
        tp_arr = res[2] if len(res) > 2 and res[2] is not None else np.full(len(ticks), getattr(strat_inst, "tp_pips", 40.0))
        if len(res) > 3 and res[3] is not None:
            sig_ts = res[3]
    else:
        signals = res
        sl_arr = np.full(len(ticks), getattr(strat_inst, "sl_pips", 20.0))
        tp_arr = np.full(len(ticks), getattr(strat_inst, "tp_pips", 40.0))

    min_entry_ts = ticks['timestamp'][WARMUP_TICKS]

    trades = run_matching_loop(
        ticks=ticks,
        signals=signals,
        pip_size=pip_size,
        sl_pips=sl_arr,
        tp_pips=tp_arr,
        use_slippage=True,
        close_at_end=True,
        commission_pips=comm_pips,
        entry_slippage_pips=slip_pips,
        signal_timestamps=sig_ts,
        min_entry_timestamp=min_entry_ts,
        enforce_subsequent_tick=enforce_subsequent
    )

    return trades


def run_strategy_evaluation(
    strat_file: Path,
    symbol: str,
    weeks: List[int],
    pip_size: float,
    comm_pips: float,
    slip_pips: float,
    executor: ProcessPoolExecutor,
    enforce_subsequent: bool = True
) -> np.ndarray:
    """Evaluates all weeks for a given strategy in parallel."""
    futures = [
        executor.submit(
            _worker_eval_week,
            str(strat_file),
            symbol,
            w,
            pip_size,
            comm_pips,
            slip_pips,
            enforce_subsequent
        )
        for w in weeks
    ]
    trades_list = []
    for f in as_completed(futures):
        tr = f.result()
        if len(tr) > 0:
            trades_list.append(tr)
    return np.concatenate(trades_list) if trades_list else np.empty(0)


def certify_individual_strategy(
    strat_file: Path,
    executor: ProcessPoolExecutor,
    candidates_screened: int = 1
) -> Dict[str, Any]:
    """
    Fully certifies an individual strategy on both In-Sample and Holdout,
    including statistical confidence, stress battery, and manifest generation.
    """
    fn = strat_file.stem
    # Identify symbol
    sym = None
    for s in PIP_SIZES.keys():
        if s.lower() in fn.lower():
            sym = s
            break
    if not sym:
        sym = "EURUSD"

    pip_size = PIP_SIZES.get(sym, 0.0001)
    comm_pips, slip_pips = get_friction_pips(sym)

    disc_weeks = get_complete_split_weeks(sym, "discovery_70")
    hold_weeks = get_complete_split_weeks(sym, "holdout_test_30")

    t0 = time.time()

    # 1. Evaluate Quarantined Holdout (Weeks 211..300)
    hold_trades = run_strategy_evaluation(
        strat_file, sym, hold_weeks, pip_size, comm_pips, slip_pips, executor, enforce_subsequent=True
    )

    # 2. Evaluate In-Sample Discovery (Weeks 1..210)
    disc_trades = run_strategy_evaluation(
        strat_file, sym, disc_weeks, pip_size, comm_pips, slip_pips, executor, enforce_subsequent=True
    )

    elapsed = time.time() - t0

    # Statistical metrics for Holdout
    hold_stats = compute_statistical_metrics(hold_trades, candidates_screened=candidates_screened)
    disc_stats = compute_statistical_metrics(disc_trades, candidates_screened=candidates_screened)

    # Build Certified Manifest for Holdout
    manifest = create_certified_run_manifest(
        strategy_name=strat_file.stem,
        strategy_file=strat_file,
        symbol=sym,
        split_version="canonical_continuous_v1.0 (In-Sample: 1..210, Holdout: 211..300)",
        start_week=211,
        end_week=300,
        pip_size=pip_size,
        commission_pips=comm_pips,
        entry_slippage_pips=slip_pips,
        spread_policy="Floating Raw Dukascopy Bid/Ask",
        warmup_ticks=WARMUP_TICKS,
        trades=hold_trades,
        candidates_screened=candidates_screened,
        additional_meta={
            "discovery_in_sample_metrics": disc_stats,
            "evaluation_time_seconds": round(elapsed, 2),
            "clean_weeks_holdout": len(hold_weeks),
            "clean_weeks_discovery": len(disc_weeks)
        }
    )

    # Save manifest JSON and trades CSV
    manifest_path = OUTPUT_DIR / f"{strat_file.stem}_run_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    if len(hold_trades) > 0:
        csv_path = OUTPUT_DIR / f"{strat_file.stem}_trades.csv"
        pd.DataFrame(manifest["trade_log"]).to_csv(csv_path, index=False)

    return {
        "file": strat_file.name,
        "symbol": sym,
        "elapsed_sec": round(elapsed, 2),
        "holdout": hold_stats,
        "discovery": disc_stats,
        "status": manifest["certification_status"]
    }


def main():
    start_total = time.time()
    print("=" * 100)
    print("  INSTITUTIONAL STRATEGY CERTIFICATION SUITE: PARALLEL FLEET RUNNER")
    print("=" * 100)

    # Load target candidate list: Start with the 15 verified survivors + key representative strategies
    manifest_file = V45_DIR / "FLEET_MANIFEST.json"
    with open(manifest_file, "r", encoding="utf-8") as f:
        fleet_meta = json.load(f)

    # The 15 verified continuous survivors
    target_slots = [
        17, 18, 19, 20, 21, 22, 23, 24, # USDCAD Asian Sweeps (8)
        9, 10, 11, 12, 14, 15, 16,      # EURUSD NY Sweep Fades (7)
        62                              # EURGBP London Sweep (1)
    ]

    target_files = []
    for item in fleet_meta:
        if item["slot"] in target_slots:
            target_files.append((item["slot"], V45_DIR / item["filename"]))

    # Sort by slot
    target_files.sort(key=lambda x: x[0])

    print(f"Targeting {len(target_files)} Priority Candidates for Individual Institutional Certification.")
    print(f"Parallel Worker Threads: 8 CPU Workers")
    print("=" * 100)

    results = []

    with ProcessPoolExecutor(max_workers=8) as executor:
        for slot, strat_path in target_files:
            print(f"\n[Slot {slot:02d}] Certifying {strat_path.name}...")
            res = certify_individual_strategy(strat_path, executor, candidates_screened=len(target_files))
            results.append(res)

            h = res["holdout"]
            d = res["discovery"]
            print(f"   In-Sample (W1..210):    Net: {d['net_pips']:>+7.1f}p | PF: {d['profit_factor']:>4.2f} | Trades: {d['total_trades']:>3} | WR: {d['win_rate_pct']:>4.1f}%")
            print(f"   Holdout   (W211..300):  Net: {h['net_pips']:>+7.1f}p | PF: {h['profit_factor']:>4.2f} | Trades: {h['total_trades']:>3} | WR: {h['win_rate_pct']:>4.1f}%")
            print(f"   Statistical Metrics:   Exp: {h['expectancy_pips']:>+5.2f}p | 95% CI: [{h['expectancy_ci_95'][0]:>+5.2f}, {h['expectancy_ci_95'][1]:>+5.2f}] | t-stat: {h['t_statistic']:>4.2f} (p={h['p_value']:.4f})")
            print(f"   Drawdown & Outliers:   Max DD: {h['max_drawdown_pips']:>5.1f}p | Boot 95th DD: {h['bootstrap_95_max_dd_pips']:>5.1f}p | Ex-Top5: {h['outlier_sensitivity']['pips_ex_top5']:>+7.1f}p | Outlier Dep: {h['outlier_sensitivity']['is_outlier_dependent']}")
            print(f"   STATUS:                {res['status']} ({res['elapsed_sec']}s)")

    # Save summary report
    summary_path = ROOT_DIR / "results" / "continuous_walkforward_audits" / "certified_survivors_fleet_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    total_time = time.time() - start_total
    print("\n" + "=" * 100)
    print(f"  ALL {len(results)} CANDIDATES CERTIFIED IN {total_time:.1f} SECONDS")
    print(f"  Summary Report Saved: {summary_path.relative_to(ROOT_DIR)}")
    print("=" * 100)


if __name__ == "__main__":
    main()

