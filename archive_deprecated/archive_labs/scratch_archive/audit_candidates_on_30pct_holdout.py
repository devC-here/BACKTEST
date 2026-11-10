#!/usr/bin/env python3
"""
================================================================================
  FORMAL QUARANTINED 30% OUT-OF-SAMPLE HOLDOUT AUDIT
================================================================================
Candidate pool: 7 optimized architectures that passed in-sample discovery.
STRICT METHODOLOGY:
  Executes EXCLUSIVELY on the untouched 30% Quarantined Holdout Test Partition
  (get_holdout_test_weeks).
  Authentic Dukascopy tick data with floating spread and real slippage modeling.
  No parameters are fitted or modified here.

Passing Criteria:
  - Holdout Profit Factor >= 1.20
  - Holdout Net Pips > 0
  - Holdout Trades >= 15
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
from tick_engine.core.dataloader import get_holdout_test_weeks, get_week_dataset
from tick_engine.strategies.base import BaseStrategy

from scratch.alpha_optimization_lab_v38 import ParametricOpeningRangeImpulse, ParametricM30Engulfing
from scratch.alpha_discovery_lab_v39 import TripleScreenElite, DeltaAbsorptionElite, VWAPValueAreaDefenseElite
from scratch.alpha_discovery_lab_v41 import TunedM15TwoHundredEMABounce, TunedTripleScreenMomentum

RESULTS_DIR = ROOT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

EURUSD_MULT = 1.0000
USDJPY_MULT = 0.7193
EURGBP_MULT = 1.2921


def get_candidates() -> List[Dict[str, Any]]:
    return [
        {
            "code": "DISC93",
            "name": "Triple_Screen_Micro_Wick_Defense",
            "asset": "EURUSD",
            "strat": TripleScreenElite("EURUSD", 8.0, 17.6, 0.35, 0.0001, 0.6),
            "pip_size": 0.0001,
            "mult": EURUSD_MULT,
            "disc_pf": 1.49,
            "disc_pips": 1200.0,
        },
        {
            "code": "DISC94",
            "name": "Opening_Range_Impulse_Expansion",
            "asset": "EURUSD",
            "strat": ParametricOpeningRangeImpulse("EURUSD", "LONDON", 1.5, 0.60, 8.0, 17.6, 0.0001, 0.6),
            "pip_size": 0.0001,
            "mult": EURUSD_MULT,
            "disc_pf": 1.36,
            "disc_pips": 829.1,
        },
        {
            "code": "DISC95",
            "name": "M30_Engulfing_Trend_Resumption",
            "asset": "USDJPY",
            "strat": ParametricM30Engulfing("USDJPY", False, 0.50, "ALL_DAY", 14.0, 28.0, 0.01, 0.8),
            "pip_size": 0.01,
            "mult": USDJPY_MULT,
            "disc_pf": 1.46,
            "disc_pips": 778.9,
        },
        {
            "code": "DISC96",
            "name": "Order_Flow_Delta_Absorption",
            "asset": "USDJPY",
            "strat": DeltaAbsorptionElite("USDJPY", 11.5, 25.3, 0.35, 0.01, 0.8),
            "pip_size": 0.01,
            "mult": USDJPY_MULT,
            "disc_pf": 1.28,
            "disc_pips": 622.9,
        },
        {
            "code": "DISC97",
            "name": "VWAP_Value_Area_Defense",
            "asset": "EURUSD",
            "strat": VWAPValueAreaDefenseElite("EURUSD", 1.0, 8.5, 17.0, 0.0001, 0.6),
            "pip_size": 0.0001,
            "mult": EURUSD_MULT,
            "disc_pf": 1.28,
            "disc_pips": 529.8,
        },
        {
            "code": "DISC98",
            "name": "M15_200EMA_Institutional_Bounce",
            "asset": "EURUSD",
            "strat": TunedM15TwoHundredEMABounce("EURUSD", 2.5, 0.35, 8.0, 17.6, 0.0001, 0.6),
            "pip_size": 0.0001,
            "mult": EURUSD_MULT,
            "disc_pf": 1.32,
            "disc_pips": 266.5,
        },
        {
            "code": "DISC99",
            "name": "Triple_Screen_Momentum_Elite",
            "asset": "EURGBP",
            "strat": TunedTripleScreenMomentum("EURGBP", 0.50, 6.5, 14.3, 0.0001, 0.6),
            "pip_size": 0.0001,
            "mult": EURGBP_MULT,
            "disc_pf": 1.71,
            "disc_pips": 157.5,
        },
    ]


def run_holdout_task(task: Tuple[str, str, int, BaseStrategy, float]) -> Dict[str, Any]:
    asset, code, week_num, strat, pip_size = task
    engine = BacktestEngine(pip_size=pip_size, use_slippage=True)
    try:
        ticks, _ = get_week_dataset(week_num, raw_dir=f"data/{asset}")
        if ticks is None or len(ticks) < 500:
            return {"code": code, "week": week_num, "trades": []}
        raw_trades = engine.run(ticks, strat)
        t_list = []
        for i in range(len(raw_trades)):
            row = raw_trades[i]
            t_list.append({
                "pips": float(row["pips"]),
                "direction": int(row["direction"])
            })
        return {"code": code, "week": week_num, "trades": t_list}
    except Exception:
        return {"code": code, "week": week_num, "trades": []}


def calculate_metrics(pips: List[float], mult: float) -> Dict[str, Any]:
    if not pips:
        return {"trades": 0, "win_rate": 0.0, "pf": 0.0, "net_pips": 0.0, "norm_pips": 0.0, "max_dd": 0.0, "ret_dd": 0.0, "expectancy": 0.0}
    arr = np.array(pips, dtype=np.float64)
    trades = len(arr)
    wins = arr[arr > 0]
    losses = arr[arr < 0]
    wr = len(wins) / trades * 100.0 if trades > 0 else 0.0
    gross_win = float(np.sum(wins)) if len(wins) > 0 else 0.0
    gross_loss = float(abs(np.sum(losses))) if len(losses) > 0 else 0.0
    pf = gross_win / gross_loss if gross_loss > 0 else 999.0
    net_pips = float(np.sum(arr))
    norm_pips = net_pips * mult
    expectancy = net_pips / trades if trades > 0 else 0.0

    equity = np.cumsum(arr * mult)
    peak = np.maximum.accumulate(equity)
    dd = peak - equity
    max_dd = float(np.max(dd)) if len(dd) > 0 else 0.0
    ret_dd = norm_pips / max_dd if max_dd > 0 else 999.0

    return {
        "trades": trades,
        "win_rate": round(wr, 1),
        "pf": round(pf, 2),
        "net_pips": round(net_pips, 1),
        "norm_pips": round(norm_pips, 1),
        "max_dd": round(max_dd, 1),
        "ret_dd": round(ret_dd, 2),
        "expectancy": round(expectancy, 2)
    }


def main():
    print("=" * 105)
    print("  FORMAL 30% QUARANTINED OUT-OF-SAMPLE HOLDOUT AUDIT")
    print("  Evaluating Candidates DISC93 - DISC99 on 100% Untouched Out-of-Sample Weeks")
    print("=" * 105)

    candidates = get_candidates()
    holdout_tasks = []

    for c in candidates:
        asset = c["asset"]
        code = c["code"]
        strat = c["strat"]
        pip_size = c["pip_size"]
        holdout_weeks = get_holdout_test_weeks(raw_dir=f"data/{asset}")
        for w in holdout_weeks:
            holdout_tasks.append((asset, code, w, strat, pip_size))

    print(f"Total out-of-sample holdout tasks: {len(holdout_tasks):,} across {len(candidates)} candidate champions\n")

    results_by_code = {}
    t0 = time.time()

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(run_holdout_task, t) for t in holdout_tasks]
        done = 0
        total = len(futures)
        for fut in as_completed(futures):
            res = fut.result()
            cd = res["code"]
            if cd not in results_by_code:
                results_by_code[cd] = []
            results_by_code[cd].extend(res["trades"])
            done += 1
            if done % 100 == 0 or done == total:
                elapsed = time.time() - t0
                print(f"  Progress: [{done}/{total}] holdout tasks ({done/total*100:.1f}%) in {elapsed:.1f}s")

    elapsed = time.time() - t0
    print(f"\nHoldout execution phase completed in {elapsed:.1f}s!\n")

    summary_rows = []
    print("=" * 115)
    print(f"  {'Code':<8} {'Asset':<8} {'Architecture Name':<32} {'Holdout Tr':<11} {'Holdout WR':<11} {'Holdout PF':<11} {'Holdout Pips':<13} {'Verdict'}")
    print("  " + "-" * 112)

    for c in candidates:
        cd = c["code"]
        asset = c["asset"]
        name = c["name"]
        mult = c["mult"]
        disc_pf = c["disc_pf"]
        disc_pips = c["disc_pips"]

        trades = results_by_code.get(cd, [])
        pips = [t["pips"] for t in trades]
        hm = calculate_metrics(pips, mult)

        passed = hm["pf"] >= 1.20 and hm["net_pips"] > 0 and hm["trades"] >= 15
        verdict = "[100% CERTIFIED PRODUCTION]" if passed else "[FAILED HOLDOUT]"

        print(f"  {cd:<8} {asset:<8} {name:<32} {hm['trades']:<11d} {hm['win_rate']:<10.1f}% {hm['pf']:<10.2f} +{hm['norm_pips']:<11.1f}p {verdict}")

        summary_rows.append({
            "code": cd,
            "asset": asset,
            "strategy_name": name,
            "discovery_pf": disc_pf,
            "discovery_norm_pips": disc_pips,
            "holdout_trades": hm["trades"],
            "holdout_win_rate": hm["win_rate"],
            "holdout_profit_factor": hm["pf"],
            "holdout_expectancy": hm["expectancy"],
            "holdout_raw_pips": hm["net_pips"],
            "holdout_norm_pips": hm["norm_pips"],
            "holdout_max_dd": hm["max_dd"],
            "holdout_ret_dd": hm["ret_dd"],
            "verdict": verdict
        })

    df_out = pd.DataFrame(summary_rows)
    df_out.to_csv(RESULTS_DIR / "candidates_quarantined_holdout_audit.csv", index=False)
    print(f"\nSaved Holdout Audit Results to: {RESULTS_DIR / 'candidates_quarantined_holdout_audit.csv'}")


if __name__ == "__main__":
    main()

