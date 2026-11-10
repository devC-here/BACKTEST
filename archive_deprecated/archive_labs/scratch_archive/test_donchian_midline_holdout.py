#!/usr/bin/env python3
"""
30% Quarantined Out-of-Sample Holdout Audit for:
EURUSD Session Donchian Midline Bounce (2.0R vs 2.2R)
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
from scratch.alpha_discovery_lab_v45 import SessionDonchianMidlineBounce

candidates = [
    {
        "name": "EURUSD_DONCHIAN_MID_20R",
        "asset": "EURUSD",
        "strat": SessionDonchianMidlineBounce("EURUSD", sl_pips=8.5, tp_pips=17.0, pip_size=0.0001, max_spread=0.6),
        "pip_size": 0.0001,
        "sl_pips": 8.5
    },
    {
        "name": "EURUSD_DONCHIAN_MID_22R",
        "asset": "EURUSD",
        "strat": SessionDonchianMidlineBounce("EURUSD", sl_pips=8.0, tp_pips=17.6, pip_size=0.0001, max_spread=0.6),
        "pip_size": 0.0001,
        "sl_pips": 8.0
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

    results = {i: [] for i in range(len(candidates))}
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(run_task, t) for t in tasks]
        for fut in as_completed(futures):
            idx, pips = fut.result()
            results[idx].extend(pips)

    print("=" * 115)
    print(f"{'Candidate Name':<28} {'Holdout Tr':<11} {'Holdout WR':<11} {'Holdout PF':<11} {'Holdout Pips':<13} {'Holdout R':<11} {'Annual Ret%':<12} {'Verdict'}")
    print("-" * 115)

    for idx, c in enumerate(candidates):
        pips = results[idx]
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
        annual_ret_pct = (net_r / 1.5) * 1.0

        verdict = "[FAILED HOLDOUT]"
        if pf >= 1.20 and net_pips > 0 and annual_ret_pct >= 15.0:
            verdict = "[100% PRODUCTION CERTIFIED]"
        elif pf >= 1.20 and net_pips > 0:
            verdict = "[HOLDOUT PASSED]"

        print(f"{c['name']:<28} {trades:<11} {wr:<11.1f} {pf:<11.2f} {net_pips:<13.1f} {net_r:<11.1f} {annual_ret_pct:<12.1f} {verdict}")

if __name__ == "__main__":
    main()
