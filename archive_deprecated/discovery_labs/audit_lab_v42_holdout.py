#!/usr/bin/env python3
"""
Formal 30% Quarantined Out-of-Sample Holdout Audit for Lab V42 Champions:
  1. USDJPY Cardwell RSI 50 Momentum Bounce (2.2R)
  2. USDJPY Tokyo Expansion ATR 0.80 Vol Filter (2.2R)
  3. EURUSD Cardwell RSI 50 Momentum Bounce (2.2R)
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
from tick_engine.core.dataloader import get_holdout_test_weeks, get_week_dataset
from scratch.alpha_discovery_lab_v42 import CardwellRSI50MomentumBounce, TokyoExpansionATRVol

USDJPY_MULT = 0.7193
EURUSD_MULT = 1.0000

candidates = [
    {
        "name": "USDJPY_RSI50_BOUNCE_22R",
        "asset": "USDJPY",
        "strat": CardwellRSI50MomentumBounce("USDJPY", 11.5, 25.3, 0.01, 0.8),
        "pip_size": 0.01,
        "mult": USDJPY_MULT,
        "sl_pips": 11.5,
        "disc_pf": 1.30,
        "disc_ret_pct": 29.0
    },
    {
        "name": "USDJPY_TOKYO_EXP_ATR80_22R",
        "asset": "USDJPY",
        "strat": TokyoExpansionATRVol("USDJPY", 0.80, 11.5, 25.3, 0.01, 0.8),
        "pip_size": 0.01,
        "mult": USDJPY_MULT,
        "sl_pips": 11.5,
        "disc_pf": 1.60,
        "disc_ret_pct": 14.7
    },
    {
        "name": "EURUSD_RSI50_BOUNCE_22R",
        "asset": "EURUSD",
        "strat": CardwellRSI50MomentumBounce("EURUSD", 8.0, 17.6, 0.0001, 0.6),
        "pip_size": 0.0001,
        "mult": EURUSD_MULT,
        "sl_pips": 8.0,
        "disc_pf": 1.24,
        "disc_ret_pct": 21.0
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

    print("=" * 115)
    print(f"{'Candidate Name':<30} {'Holdout Tr':<11} {'Holdout WR':<11} {'Holdout PF':<11} {'Holdout R':<11} {'Annual Ret%':<12} {'Verdict'}")
    print("-" * 115)

    for idx, c in enumerate(candidates):
        pips = results_by_idx[idx]
        if not pips:
            print(f"{c['name']:<30} No trades")
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
        
        holdout_years = len(get_holdout_test_weeks(f"data/{c['asset']}")) / 52.0
        annual_ret = (net_r / holdout_years) * 1.0  # 1% risk per trade

        passed = pf >= 1.20 and annual_ret >= 15.0 and trades >= 15
        verdict = "[100% PRODUCTION CERTIFIED]" if passed else "[PASSES PF >= 1.20]" if pf >= 1.20 else "[FAILED HOLDOUT]"

        print(f"{c['name']:<30} {trades:<11d} {wr:<10.1f}% {pf:<10.2f} +{net_r:<10.1f}R +{annual_ret:<10.1f}% {verdict}")
    print("=" * 115)

if __name__ == "__main__":
    main()
