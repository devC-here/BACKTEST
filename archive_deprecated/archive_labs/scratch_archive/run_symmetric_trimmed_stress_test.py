#!/usr/bin/env python3
"""
================================================================================
  SYMMETRIC TRIMMED TRADE TEST: REMOVING 30% BEST AND 30% WORST TRADES
================================================================================
Tests the core structural edge of every strategy by eliminating the extreme tails:
  - Drops the TOP 30% best trades (removes lucky outlier windfall trades)
  - Drops the BOTTOM 30% worst trades (removes outlier flash crashes / black swans)
  - Analyzes the remaining MIDDLE 40% CORE TRADES:
      * Core Net Pips
      * Core Profit Factor
      * Core Win Rate
      * Core Expectancy (Pips/Trade)
      * Core Max Drawdown

Assets Analyzed:
  - EURUSD Fleet (14 Models over 262 weeks / 5 years)
  - EURGBP Fleet (6 Models over 157 weeks / 3 years)

Saves:
  - results/symmetric_trimmed_stress_test_results.csv
  - results/master_eurusd_14_models_trades.json
  - results/master_eurgbp_6_models_trades.json
================================================================================
"""

from __future__ import annotations
import sys
import os
import time
import json
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

# EURUSD Models
from strategies.eurusd_production_fleet import (
    MacroStructureH4M30Elite,
    ContinuousM5RejectionElite,
    RSIRegimeRangeShiftElite,
    IntradayPrecisionH1M5Elite,
    S66Bhanushali44MAElite,
    HighFrequencyM15ScalperElite,
    S10DualEMAPullbackElite,
    S14InsideBarM30Elite,
    NewYorkOpenOverlapBreakoutElite,
    ThursdayMacroTrendIgnitionElite,
    LondonMiddayMACDAccelerationElite,
    StochasticDeepPullbackElite,
    BollingerBandwidthExpansionElite,
    BollingerSupertrendHybridElite
)

# EURGBP Models
from strategies.eurgbp_fleet import (
    EURGBPDailyPivotWallElite,
    EURGBPCamarillaFloorElite,
    EURGBPWeeklyPivotWallElite
)

PIP_SIZE = 0.0001
RESULTS_DIR = ROOT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# Simulation Worker Functions
# ==============================================================================
def simulate_eurgbp_week(w: int) -> Dict[str, List[Dict[str, Any]]]:
    engine = BacktestEngine(pip_size=PIP_SIZE, use_slippage=True)
    ticks, meta = get_week_dataset(w, raw_dir="EURGBP")

    strats = [
        ("DISC46_Daily_Pivot", EURGBPDailyPivotWallElite()),
        ("DISC48_Camarilla", EURGBPCamarillaFloorElite()),
        ("DISC42_Macro_Structure", MacroStructureH4M30Elite(max_spread=2.0)),
        ("DISC45_M5_Rejection", ContinuousM5RejectionElite(max_spread=2.0)),
        ("DISC47_Weekly_Pivot", EURGBPWeeklyPivotWallElite()),
        ("DISC43_Intraday_Precision", IntradayPrecisionH1M5Elite(max_spread=2.0)),
    ]

    week_res = {}
    for name, s in strats:
        trades = engine.run(ticks, s)
        t_list = []
        for i in range(len(trades)):
            row = trades[i]
            t_list.append({
                "strategy": name,
                "asset": "EURGBP",
                "week": w,
                "pips": float(row["pips"]),
                "entry_time": int(row["entry_time"]),
                "exit_time": int(row["exit_time"]),
                "direction": int(row["direction"])
            })
        week_res[name] = t_list
    return week_res


def simulate_eurusd_week(w: int) -> Dict[str, List[Dict[str, Any]]]:
    engine = BacktestEngine(pip_size=PIP_SIZE, use_slippage=True)
    ticks, meta = get_week_dataset(w, raw_dir="EURUSD")

    strats = [
        ("DISC42_H4_M30", MacroStructureH4M30Elite()),
        ("DISC45_M5", ContinuousM5RejectionElite()),
        ("DISC40_RSI", RSIRegimeRangeShiftElite()),
        ("DISC43_H1_M5", IntradayPrecisionH1M5Elite()),
        ("S66_44MA", S66Bhanushali44MAElite()),
        ("HF_M15", HighFrequencyM15ScalperElite()),
        ("S10_EMA", S10DualEMAPullbackElite()),
        ("S14_IB", S14InsideBarM30Elite()),
        ("DISC03_NY", NewYorkOpenOverlapBreakoutElite()),
        ("DISC41_THURSDAY", ThursdayMacroTrendIgnitionElite()),
        ("DISC38_MACD", LondonMiddayMACDAccelerationElite()),
        ("DISC39_STOCH", StochasticDeepPullbackElite()),
        ("COMB12_BB", BollingerBandwidthExpansionElite()),
        ("COMB27_SUPERTREND", BollingerSupertrendHybridElite()),
    ]

    week_res = {}
    for name, s in strats:
        trades = engine.run(ticks, s)
        t_list = []
        for i in range(len(trades)):
            row = trades[i]
            t_list.append({
                "strategy": name,
                "asset": "EURUSD",
                "week": w,
                "pips": float(row["pips"]),
                "entry_time": int(row["entry_time"]),
                "exit_time": int(row["exit_time"]),
                "direction": int(row["direction"])
            })
        week_res[name] = t_list
    return week_res


# ==============================================================================
# Symmetric Trimmed Statistics
# ==============================================================================
def analyze_symmetric_trimmed_trades(name: str, asset: str, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(trades)
    if n == 0:
        return {}

    pips = np.array([t["pips"] for t in trades], dtype=np.float64)

    # 1. Baseline Full Distribution
    wins = pips[pips > 0]
    losses = pips[pips < 0]

    base_net = float(np.sum(pips))
    base_wr = (len(wins) / n) * 100.0
    base_gp = float(np.sum(wins)) if len(wins) > 0 else 0.0
    base_gl = float(abs(np.sum(losses))) if len(losses) > 0 else 0.0
    base_pf = base_gp / base_gl if base_gl > 0 else 99.0
    cum = np.cumsum(pips)
    dd = np.maximum.accumulate(cum) - cum
    base_max_dd = float(np.max(dd)) if len(dd) > 0 else 0.0
    base_ret_dd = base_net / base_max_dd if base_max_dd > 0 else 0.0
    base_exp = base_net / n

    # 2. Symmetric 30% Trimmed (Remove bottom 30% and top 30% -> Middle 40% Core)
    sorted_pips = np.sort(pips)
    k30 = int(round(n * 0.30))
    # Slice the middle 40%
    if k30 > 0 and (n - k30) > k30:
        core_pips = sorted_pips[k30 : n - k30]
    else:
        core_pips = sorted_pips

    n_core = len(core_pips)
    core_wins = core_pips[core_pips > 0]
    core_losses = core_pips[core_pips < 0]

    core_net = float(np.sum(core_pips))
    core_wr = (len(core_wins) / n_core) * 100.0 if n_core > 0 else 0.0
    core_gp = float(np.sum(core_wins)) if len(core_wins) > 0 else 0.0
    core_gl = float(abs(np.sum(core_losses))) if len(core_losses) > 0 else 0.0
    core_pf = core_gp / core_gl if core_gl > 0 else (99.0 if core_gp > 0 else 0.0)
    core_exp = core_net / n_core if n_core > 0 else 0.0

    cum_core = np.cumsum(core_pips)
    dd_core = np.maximum.accumulate(cum_core) - cum_core
    core_max_dd = float(np.max(dd_core)) if len(dd_core) > 0 else 0.0
    core_ret_dd = core_net / core_max_dd if core_max_dd > 0 else 0.0

    # 3. Symmetric 15% Trimmed (Remove bottom 15% and top 15% -> Middle 70% Core)
    k15 = int(round(n * 0.15))
    if k15 > 0 and (n - k15) > k15:
        mid70_pips = sorted_pips[k15 : n - k15]
    else:
        mid70_pips = sorted_pips

    n_mid70 = len(mid70_pips)
    mid70_wins = mid70_pips[mid70_pips > 0]
    mid70_losses = mid70_pips[mid70_pips < 0]
    mid70_net = float(np.sum(mid70_pips))
    mid70_gp = float(np.sum(mid70_wins)) if len(mid70_wins) > 0 else 0.0
    mid70_gl = float(abs(np.sum(mid70_losses))) if len(mid70_losses) > 0 else 0.0
    mid70_pf = mid70_gp / mid70_gl if mid70_gl > 0 else (99.0 if mid70_gp > 0 else 0.0)

    # Core Edge Health Diagnosis
    if core_net > 0 and core_pf >= 1.20:
        diagnosis = "STRONG CORE EDGE (Positive Middle 40%)"
    elif core_net > 0:
        diagnosis = "MODERATE CORE EDGE (Positive Middle 40%)"
    elif mid70_net > 0 and mid70_pf >= 1.10:
        diagnosis = "TREND RUNNER EDGE (Requires 70% Body)"
    else:
        diagnosis = "NEGATIVE CORE (Tail Dependant)"

    return {
        "strategy": name,
        "asset": asset,
        "base_trades": n,
        "base_wr": round(base_wr, 1),
        "base_net": round(base_net, 1),
        "base_pf": round(base_pf, 2),
        "base_max_dd": round(base_max_dd, 1),
        "base_ret_dd": round(base_ret_dd, 2),
        "base_exp": round(base_exp, 2),
        "core_trades": n_core,
        "core_wr": round(core_wr, 1),
        "core_net": round(core_net, 1),
        "core_pf": round(core_pf, 2),
        "core_exp": round(core_exp, 2),
        "core_max_dd": round(core_max_dd, 1),
        "core_ret_dd": round(core_ret_dd, 2),
        "mid70_net": round(mid70_net, 1),
        "mid70_pf": round(mid70_pf, 2),
        "diagnosis": diagnosis
    }


def main():
    t0 = time.time()
    print("=" * 135)
    print("   SYMMETRIC TRIMMED TRADE TEST: REMOVING 30% BEST AND 30% WORST TRADES   ".center(135))
    print("=" * 135)

    workers = 4

    # 1. Simulate EURGBP
    print("\n[*] Simulating EURGBP Strategies (157 Weeks)...")
    manifest_eg = load_manifest("EURGBP")
    weeks_eg = sorted(list(manifest_eg.keys()))
    eurgbp_trades = {
        "DISC46_Daily_Pivot": [],
        "DISC48_Camarilla": [],
        "DISC42_Macro_Structure": [],
        "DISC45_M5_Rejection": [],
        "DISC47_Weekly_Pivot": [],
        "DISC43_Intraday_Precision": [],
    }

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(simulate_eurgbp_week, w): w for w in weeks_eg}
        done = 0
        for f in futures:
            res = f.result()
            done += 1
            for name, tl in res.items():
                eurgbp_trades[name].extend(tl)
            if done % 40 == 0 or done == len(weeks_eg):
                print(f"  -> EURGBP: {done}/{len(weeks_eg)} weeks in {time.time()-t0:.1f}s...")

    # Save EURGBP raw trades
    eurgbp_json = RESULTS_DIR / "master_eurgbp_6_models_trades.json"
    with open(eurgbp_json, "w") as f:
        json.dump(eurgbp_trades, f)
    print(f"[+] Saved EURGBP trades to {eurgbp_json}")

    # 2. Simulate EURUSD
    print("\n[*] Simulating EURUSD Strategies (262 Weeks)...")
    manifest_eu = load_manifest("EURUSD")
    weeks_eu = sorted(list(manifest_eu.keys()))
    eurusd_trades = {
        "DISC42_H4_M30": [],
        "DISC45_M5": [],
        "DISC40_RSI": [],
        "DISC43_H1_M5": [],
        "S66_44MA": [],
        "HF_M15": [],
        "S10_EMA": [],
        "S14_IB": [],
        "DISC03_NY": [],
        "DISC41_THURSDAY": [],
        "DISC38_MACD": [],
        "DISC39_STOCH": [],
        "COMB12_BB": [],
        "COMB27_SUPERTREND": [],
    }

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(simulate_eurusd_week, w): w for w in weeks_eu}
        done = 0
        for f in futures:
            res = f.result()
            done += 1
            for name, tl in res.items():
                eurusd_trades[name].extend(tl)
            if done % 60 == 0 or done == len(weeks_eu):
                print(f"  -> EURUSD: {done}/{len(weeks_eu)} weeks in {time.time()-t0:.1f}s...")

    # Save EURUSD raw trades
    eurusd_json = RESULTS_DIR / "master_eurusd_14_models_trades.json"
    with open(eurusd_json, "w") as f:
        json.dump(eurusd_trades, f)
    print(f"[+] Saved EURUSD trades to {eurusd_json}")

    # 3. Analyze and Print Scorecards
    print("\n" + "=" * 145)
    print("   EURGBP FLEET: SYMMETRIC TRIMMED CORE DISTRIBUTION (MIDDLE 40% CORE)   ".center(145))
    print("=" * 145)
    print(f"{'Strategy Name':<26} | {'Base Net':>9} | {'Base PF':>7} | {'Base Ret/DD':>11} | {'Core Net (Middle 40%)':>22} | {'Core PF':>7} | {'Core WR':>7} | {'Diagnosis':<36}")
    print("-" * 145)

    eg_rows = []
    for name, tl in eurgbp_trades.items():
        row = analyze_symmetric_trimmed_trades(name, "EURGBP", tl)
        eg_rows.append(row)
        print(f"{name:<26} | {row['base_net']:>+8.1f}p | {row['base_pf']:>7.2f} | {row['base_ret_dd']:>10.2f}x | {row['core_net']:>+21.1f}p | {row['core_pf']:>7.2f} | {row['core_wr']:>6.1f}% | {row['diagnosis']}")

    print("\n" + "=" * 145)
    print("   EURUSD FLEET: SYMMETRIC TRIMMED CORE DISTRIBUTION (MIDDLE 40% CORE)   ".center(145))
    print("=" * 145)
    print(f"{'Strategy Name':<26} | {'Base Net':>9} | {'Base PF':>7} | {'Base Ret/DD':>11} | {'Core Net (Middle 40%)':>22} | {'Core PF':>7} | {'Core WR':>7} | {'Diagnosis':<36}")
    print("-" * 145)

    eu_rows = []
    for name, tl in eurusd_trades.items():
        row = analyze_symmetric_trimmed_trades(name, "EURUSD", tl)
        eu_rows.append(row)
        print(f"{name:<26} | {row['base_net']:>+8.1f}p | {row['base_pf']:>7.2f} | {row['base_ret_dd']:>10.2f}x | {row['core_net']:>+21.1f}p | {row['core_pf']:>7.2f} | {row['core_wr']:>6.1f}% | {row['diagnosis']}")

    # Save CSV
    df = pd.DataFrame(eg_rows + eu_rows)
    out_csv = RESULTS_DIR / "symmetric_trimmed_stress_test_results.csv"
    df.to_csv(out_csv, index=False)
    print("\n" + "=" * 145)
    print(f"[+] Successfully saved full symmetric audit results to: {out_csv}")
    print(f"Total audit completed in {time.time()-t0:.1f}s.")


if __name__ == "__main__":
    main()
