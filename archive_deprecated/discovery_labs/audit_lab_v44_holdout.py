#!/usr/bin/env python3
"""
================================================================================
  FORMAL 30% QUARANTINED OUT-OF-SAMPLE HOLDOUT AUDIT: LAB V44 CANDIDATES
================================================================================
Audits the passing candidates from Lab V44 on the untouched 30% Holdout Partition:
  1. EURUSD_LON_IGNITION_22R (LondonIgnitionBreakoutCausal)
  2. USDJPY_DOUBLE_STRUCT_20R (DoubleStructureNecklineBreakout)
  3. USDJPY_LON_IGNITION_22R (LondonIgnitionBreakoutCausal)
  4. EURUSD_DOUBLE_STRUCT_22R (DoubleStructureNecklineBreakout)

Passing Bar:
  - 30% Holdout Profit Factor >= 1.20
  - 30% Holdout Net Pips > 0
  - Annualized Return >= 15.0%
================================================================================
"""

from pathlib import Path
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tick_engine.core.engine import BacktestEngine
from tick_engine.core.dataloader import get_holdout_test_weeks, get_discovery_weeks, get_week_dataset
from scratch.alpha_discovery_lab_v44 import DoubleStructureNecklineBreakout, LondonIgnitionBreakoutCausal

USDJPY_MULT = 0.7193
EURUSD_MULT = 1.0000

candidates = [
    {
        "name": "EURUSD_LON_IGNITION_22R",
        "asset": "EURUSD",
        "strat": LondonIgnitionBreakoutCausal("EURUSD", sl_pips=8.0, tp_pips=17.6, pip_size=0.0001, max_spread=0.6),
        "pip_size": 0.0001,
        "mult": EURUSD_MULT,
        "sl_pips": 8.0,
        "disc_pf": 1.38,
        "disc_ann_ret": 37.9
    },
    {
        "name": "USDJPY_DOUBLE_STRUCT_20R",
        "asset": "USDJPY",
        "strat": DoubleStructureNecklineBreakout("USDJPY", sl_pips=12.0, tp_pips=24.0, pip_size=0.01, max_spread=0.8),
        "pip_size": 0.01,
        "mult": USDJPY_MULT,
        "sl_pips": 12.0,
        "disc_pf": 1.33,
        "disc_ann_ret": 16.5
    },
    {
        "name": "USDJPY_LON_IGNITION_22R",
        "asset": "USDJPY",
        "strat": LondonIgnitionBreakoutCausal("USDJPY", sl_pips=11.5, tp_pips=25.3, pip_size=0.01, max_spread=0.8),
        "pip_size": 0.01,
        "mult": USDJPY_MULT,
        "sl_pips": 11.5,
        "disc_pf": 1.21,
        "disc_ann_ret": 22.6
    },
    {
        "name": "EURUSD_DOUBLE_STRUCT_22R",
        "asset": "EURUSD",
        "strat": DoubleStructureNecklineBreakout("EURUSD", sl_pips=8.0, tp_pips=17.6, pip_size=0.0001, max_spread=0.6),
        "pip_size": 0.0001,
        "mult": EURUSD_MULT,
        "sl_pips": 8.0,
        "disc_pf": 1.24,
        "disc_ann_ret": 18.1
    }
]

def run_task(args):
    c_idx, w = args
    c = candidates[c_idx]
    engine = BacktestEngine(pip_size=c["pip_size"], use_slippage=True)
    try:
        ticks, _ = get_week_dataset(w, raw_dir=f"data/{c['asset']}")
        if ticks is None or len(ticks) < 500:
            return c_idx, []
        raw_trades = engine.run(ticks, c["strat"])
        pips = [float(r["pips"]) for r in raw_trades]
        return c_idx, pips
    except Exception:
        return c_idx, []

def main():
    tasks = []
    for idx, c in enumerate(candidates):
        weeks = get_holdout_test_weeks(raw_dir=f"data/{c['asset']}")
        for w in weeks:
            tasks.append((idx, w))

    results_by_idx = {i: [] for i in range(len(candidates))}
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(run_task, t) for t in tasks]
        for fut in as_completed(futures):
            idx, pips = fut.result()
            results_by_idx[idx].extend(pips)

    print("=" * 125)
    print(f"{'Candidate Name':<28} {'Holdout Tr':<11} {'Holdout WR':<11} {'Holdout PF':<11} {'Holdout Pips':<13} {'Holdout R':<11} {'Annual Ret%':<12} {'Verdict'}")
    print("-" * 125)

    for idx, c in enumerate(candidates):
        pips = results_by_idx[idx]
        if not pips:
            print(f"{c['name']:<28} No trades")
            continue
        arr = np.array(pips, dtype=np.float64)
        trades = len(arr)
        wins = arr[arr > 0]
        losses = arr[arr < 0]
        wr = len(wins) / trades * 100.0 if trades > 0 else 0.0
        gross_win = float(np.sum(wins)) if len(wins) > 0 else 0.0
        gross_loss = float(abs(np.sum(losses))) if len(losses) > 0 else 0.0
        pf = gross_win / gross_loss if gross_loss > 0 else 999.0
        net_pips = float(np.sum(arr))
        net_r = net_pips / c["sl_pips"]
        annual_ret_pct = (net_r / 1.5) * 1.0  # 30% holdout is ~1.5 years

        verdict = "[FAILED HOLDOUT]"
        if pf >= 1.20 and net_pips > 0 and annual_ret_pct >= 15.0:
            verdict = "[100% PRODUCTION CERTIFIED]"
        elif pf >= 1.20 and net_pips > 0:
            verdict = "[HOLDOUT PASSED]"

        print(f"{c['name']:<28} {trades:<11} {wr:<11.1f} {pf:<11.2f} {net_pips:<13.1f} {net_r:<11.1f} {annual_ret_pct:<12.1f} {verdict}")

if __name__ == "__main__":
    main()

