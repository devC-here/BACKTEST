#!/usr/bin/env python3
"""
================================================================================
  EURGBP 5-STAR SPECIALIST FLEET AUDIT: FULL 157 WEEKS (3 YEARS)
================================================================================
Evaluating the combined portfolio of EURGBP's 5 Top Structural Champions:
  1. DISC46: Daily Pivot Wall Elite (M15 / H4 / Daily)
  2. DISC48: Daily Camarilla Floor Elite (M15 / H4 / Camarilla L3/H3)
  3. DISC47: Weekly Pivot Wall Elite (M30 / H4 / Weekly P/S1/R1)
  4. DISC42: Macro Structure H4/M30 Elite (M30 / H4 Donchian)
  5. DISC45: Continuous M5 Structural Rejection Scalper (M5)

Outputs:
  - Individual performance metrics
  - Multi-strategy combined equity curve, Max DD, Return/DD, and Profit Factor
  - Yearly breakdown (2021, 2022, 2024)
  - Detailed trade log saved to results/eurgbp_5star_fleet_trade_log.csv
================================================================================
"""

from __future__ import annotations
import sys
import os
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Any

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tick_engine.core.engine import BacktestEngine
from tick_engine.core.dataloader import get_week_dataset, load_manifest
from scratch.research_eurgbp_expansion_fleet import EURGBPWeeklyPivotWall, EURGBPCamarillaFloor
from strategies.eurgbp_fleet.disc46_daily_pivot_wall_elite.strategy import EURGBPDailyPivotWallElite
from strategies.eurgbp_fleet.disc42_macro_structure_h4_m30_elite.strategy import MacroStructureH4M30Elite
from strategies.eurgbp_fleet.disc45_continuous_m5_rejection_elite.strategy import ContinuousM5RejectionElite

PIP_SIZE = 0.0001
RAW_DIR = ROOT_DIR / "weekly_data" / "EURGBP"
RESULTS_DIR = ROOT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_week(week_num: int) -> Dict[str, Any]:
    engine = BacktestEngine(pip_size=PIP_SIZE, use_slippage=True)
    ticks, meta = get_week_dataset(week_num, raw_dir=RAW_DIR)

    models = [
        ("DISC46_Daily_Pivot_Wall", EURGBPDailyPivotWallElite()),
        ("DISC48_Camarilla_Floor", EURGBPCamarillaFloor()),
        ("DISC47_Weekly_Pivot_Wall", EURGBPWeeklyPivotWall()),
        ("DISC42_Macro_Structure", MacroStructureH4M30Elite(max_spread=2.0)),
        ("DISC45_M5_Rejection", ContinuousM5RejectionElite(max_spread=2.0)),
    ]

    week_trades = []
    for name, m in models:
        raw_trades = engine.run(ticks, m)
        for i in range(len(raw_trades)):
            row = raw_trades[i]
            week_trades.append({
                "strategy": name,
                "week": week_num,
                "entry_time": int(row["entry_time"]),
                "exit_time": int(row["exit_time"]),
                "direction": "BUY" if row["direction"] == 1 else "SELL",
                "entry_price": float(row["entry_price"]),
                "exit_price": float(row["exit_price"]),
                "pips": float(row["pips"]),
                "exit_reason": int(row["exit_reason"]),
            })

    return {"week": week_num, "trades": week_trades}


def compute_metrics(trades: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(trades)
    if n == 0:
        return {"trades": 0, "wr": 0.0, "pips": 0.0, "pf": 0.0, "max_dd": 0.0, "ret_dd": 0.0}
    pips = np.array([t["pips"] for t in trades])
    wins = np.sum(pips > 0)
    wr = wins / n * 100
    gp = float(np.sum(pips[pips > 0])) if wins > 0 else 0.0
    gl = float(abs(np.sum(pips[pips < 0]))) if (n - wins) > 0 else 0.0
    pf = gp / gl if gl > 0 else (99.0 if gp > 0 else 0.0)

    cum = np.cumsum(pips)
    dd = np.maximum.accumulate(cum) - cum
    max_dd = float(np.max(dd)) if len(dd) > 0 else 0.0
    net_pips = float(np.sum(pips))
    ret_dd = net_pips / max_dd if max_dd > 0 else 0.0

    return {
        "trades": n,
        "wr": round(wr, 1),
        "pips": round(net_pips, 1),
        "pf": round(pf, 2),
        "max_dd": round(max_dd, 1),
        "ret_dd": round(ret_dd, 2)
    }


def main():
    t0 = time.time()
    manifest = load_manifest(RAW_DIR)
    weeks = sorted(list(manifest.keys()))
    total = len(weeks)

    print("=" * 95)
    print(f"  EURGBP 5-STAR SPECIALIST FLEET AUDIT: 157 WEEKS (~65.89M TICKS)  ".center(95))
    print("=" * 95)

    all_trades: List[Dict[str, Any]] = []
    strat_trades: Dict[str, List[Dict[str, Any]]] = {
        "DISC46_Daily_Pivot_Wall": [],
        "DISC48_Camarilla_Floor": [],
        "DISC47_Weekly_Pivot_Wall": [],
        "DISC42_Macro_Structure": [],
        "DISC45_M5_Rejection": []
    }

    workers = 4
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(evaluate_week, w): w for w in weeks}
        done = 0
        for f in futures:
            res = f.result()
            done += 1
            all_trades.extend(res["trades"])
            for t in res["trades"]:
                strat_trades[t["strategy"]].append(t)
            if done % 25 == 0 or done == total:
                print(f"  -> Evaluated {done}/{total} weeks ({(done/total)*100:5.1f}%) in {time.time()-t0:.1f}s...")

    print("\n" + "=" * 95)
    print("   EURGBP 5 INDIVIDUAL CHAMPIONS SCORECARD   ".center(95))
    print("=" * 95)
    print(f"{'Strategy Model':<30} | {'Trades':>6} | {'Win Rate':>8} | {'Profit Factor':>13} | {'Net Pips':>9} | {'Max DD':>8} | {'Ret/DD':>7}")
    print("-" * 95)

    for name, t_list in strat_trades.items():
        t_list.sort(key=lambda x: x["entry_time"])
        s = compute_metrics(t_list)
        print(f"{name:<30} | {s['trades']:>6} | {s['wr']:>7.1f}% | {s['pf']:>13.2f} | {s['pips']:>+8.1f}p | {s['max_dd']:>7.1f}p | {s['ret_dd']:>6.2f}x")

    # Sort combined portfolio by entry time
    all_trades.sort(key=lambda x: x["entry_time"])
    comb_s = compute_metrics(all_trades)

    print("=" * 95)
    print(f"{'EURGBP 5-STAR COMBINED FLEET':<30} | {comb_s['trades']:>6} | {comb_s['wr']:>7.1f}% | {comb_s['pf']:>13.2f} | {comb_s['pips']:>+8.1f}p | {comb_s['max_dd']:>7.1f}p | {comb_s['ret_dd']:>6.2f}x")
    print("=" * 95)

    # Yearly breakdown
    df = pd.DataFrame(all_trades)
    df["dt"] = pd.to_datetime(df["entry_time"], unit="ms")
    df["year"] = df["dt"].dt.year

    print("\n" + "=" * 95)
    print("   EURGBP 5-STAR FLEET YEARLY BREAKDOWN   ".center(95))
    print("=" * 95)
    print(f"{'Year':<8} | {'Trades':>8} | {'Win Rate':>10} | {'Profit Factor':>14} | {'Net Return':>12} | {'Max DD':>10}")
    print("-" * 95)

    for yr in sorted(df["year"].unique()):
        yr_trades = df[df["year"] == yr].to_dict("records")
        ys = compute_metrics(yr_trades)
        print(f"{yr:<8} | {ys['trades']:>8} | {ys['wr']:>9.1f}% | {ys['pf']:>14.2f} | {ys['pips']:>+11.1f}p | {ys['max_dd']:>9.1f}p")

    print("=" * 95)

    # Save detailed CSV
    csv_path = RESULTS_DIR / "eurgbp_5star_fleet_trade_log.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[+] Successfully saved {len(df)} trades to: {csv_path}")
    print(f"Total audit completed in {time.time()-t0:.1f}s.")


if __name__ == "__main__":
    main()
