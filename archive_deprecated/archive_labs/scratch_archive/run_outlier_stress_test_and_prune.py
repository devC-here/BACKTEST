#!/usr/bin/env python3
"""
================================================================================
  REAL-LIFE VIABILITY AUDIT: TOP 30% WINNERS REMOVAL STRESS TEST
================================================================================
Evaluates all 14 EURUSD strategies and all 6 EURGBP strategies for true
Real-Life (RL) deployment viability.

Methodology:
  1. Full Baseline Multi-Year Metrics (Real Spreads + Slippage).
  2. Top 30% Winners Removal Test (Right-Tail Outlier Elimination):
     - Drops the top 30% best winning trades for each strategy.
     - Keeps 100% of all losses and the bottom 70% of winning trades.
     - Recalculates: Trimmed Net Pips, Trimmed Profit Factor, Trimmed Drawdown,
       and Profit Retention Ratio.
  3. Real-Life Viability Filter:
     - Tier 1: Real-Life Elite (Keep in Active Fleet):
       * Survives Top 30% Winners Truncation (Maintains positive or near-breakeven expectancy)
       * Baseline Return / Drawdown >= 4.0x
       * Meaningful annual yield (> 70 pips/year)
       * Profit Factor >= 1.25
     - Tier 2: Fragile / Outlier-Dependent / High Drag (Prune / Archive):
       * Collapses into severe negative loss when top winners are removed
       * Weak Return/DD (< 3.0x) or high capital drag
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

# EURUSD Strategies
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

# EURGBP Strategies
from strategies.eurgbp_fleet import (
    EURGBPDailyPivotWallElite,
    EURGBPCamarillaFloorElite,
    EURGBPWeeklyPivotWallElite
)

PIP_SIZE = 0.0001
RESULTS_DIR = ROOT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# 1. EURUSD 5-Year Simulation Runner
# ==============================================================================
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
# 2. EURGBP 3-Year Simulation Runner
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


# ==============================================================================
# 3. Robustness & Truncated Metrics Calculator
# ==============================================================================
def compute_outlier_stress_metrics(t_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(t_list)
    if n == 0:
        return {}

    all_pips = np.array([t["pips"] for t in t_list], dtype=np.float64)

    # 1. Baseline Full Performance
    wins = all_pips[all_pips > 0]
    losses = all_pips[all_pips < 0]

    base_trades = n
    base_wr = (len(wins) / n) * 100.0 if n > 0 else 0.0
    base_gp = float(np.sum(wins)) if len(wins) > 0 else 0.0
    base_gl = float(abs(np.sum(losses))) if len(losses) > 0 else 0.0
    base_net = float(np.sum(all_pips))
    base_pf = base_gp / base_gl if base_gl > 0 else 99.0

    cum = np.cumsum(all_pips)
    dd = np.maximum.accumulate(cum) - cum
    base_max_dd = float(np.max(dd)) if len(dd) > 0 else 0.0
    base_ret_dd = base_net / base_max_dd if base_max_dd > 0 else 0.0
    base_exp = base_net / n if n > 0 else 0.0

    # 2. Outlier Test: REMOVE TOP 30% OF WINNING TRADES
    if len(wins) > 0:
        sorted_wins = np.sort(wins)[::-1]  # descending
        n_drop = int(round(len(wins) * 0.30))
        # Keep remaining 70% of wins and 100% of losses
        kept_wins = sorted_wins[n_drop:]
        trimmed_pips = np.concatenate([kept_wins, losses])

        trim_trades = len(trimmed_pips)
        trim_wins = len(kept_wins)
        trim_wr = (trim_wins / trim_trades) * 100.0 if trim_trades > 0 else 0.0
        trim_gp = float(np.sum(kept_wins)) if len(kept_wins) > 0 else 0.0
        trim_gl = base_gl
        trim_net = float(np.sum(trimmed_pips))
        trim_pf = trim_gp / trim_gl if trim_gl > 0 else 0.0

        cum_trim = np.cumsum(trimmed_pips)
        dd_trim = np.maximum.accumulate(cum_trim) - cum_trim
        trim_max_dd = float(np.max(dd_trim)) if len(dd_trim) > 0 else 0.0
        trim_ret_dd = trim_net / trim_max_dd if trim_max_dd > 0 else 0.0
        trim_exp = trim_net / trim_trades if trim_trades > 0 else 0.0
        retention_pct = (trim_net / base_net * 100.0) if base_net > 0 else 0.0
    else:
        trim_net = -base_gl
        trim_pf = 0.0
        trim_max_dd = base_gl
        trim_ret_dd = -1.0
        trim_exp = 0.0
        retention_pct = 0.0
        trim_wr = 0.0

    # 3. Real-Life Viability Decision
    # Strict Live Deployment Filter:
    # Tier 1 Elite: High Ret/DD (>= 4.0x), strong base PF (>= 1.25), and maintains positive or resilient pips even without top 30% winners
    # Tier 2 Runner: Moderate Ret/DD (>= 2.5x), base PF >= 1.15
    # Tier 3 Exclude: Fails outlier test or low return/DD (< 2.5x) or microscopic pips
    is_elite = (base_ret_dd >= 4.0 and base_pf >= 1.25 and trim_pf >= 0.75 and base_net >= 350.0)
    is_viable = (base_ret_dd >= 2.5 and base_pf >= 1.15 and base_net >= 300.0)

    if is_elite:
        tier = "TIER 1: REAL-LIFE ELITE (MUST KEEP)"
        action = "KEEP IN LIVE FLEET"
    elif is_viable:
        tier = "TIER 2: VIABLE RUNNER (KEEP)"
        action = "KEEP AS SPECIALIST"
    else:
        tier = "TIER 3: HIGH DRAG / FRAGILE (PRUNE)"
        action = "ARCHIVE / QUARANTINE"

    return {
        "base_trades": base_trades,
        "base_wr": round(base_wr, 1),
        "base_net": round(base_net, 1),
        "base_pf": round(base_pf, 2),
        "base_max_dd": round(base_max_dd, 1),
        "base_ret_dd": round(base_ret_dd, 2),
        "base_exp": round(base_exp, 2),
        "trim_net": round(trim_net, 1),
        "trim_pf": round(trim_pf, 2),
        "trim_max_dd": round(trim_max_dd, 1),
        "trim_ret_dd": round(trim_ret_dd, 2),
        "retention_pct": round(retention_pct, 1),
        "tier": tier,
        "action": action
    }


def main():
    t0 = time.time()
    print("=" * 125)
    print("   REAL-LIFE VIABILITY AUDIT & TOP 30% WINNERS REMOVAL STRESS TEST   ".center(125))
    print("=" * 125)

    workers = 4

    # 1. Run EURGBP (157 weeks)
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

    # 2. Run EURUSD (262 weeks)
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

    # 3. Analyze and Print Scorecards
    print("\n" + "=" * 135)
    print("   EURGBP FLEET: OUTLIER STRESS TEST & REAL-LIFE VIABILITY SCORECARD   ".center(135))
    print("=" * 135)
    print(f"{'Strategy Name':<26} | {'Base Net':>9} | {'Base PF':>7} | {'Base Ret/DD':>11} | {'Trimmed Net (Top-30% Gone)':>27} | {'Trim PF':>7} | {'RL Verdict':<26}")
    print("-" * 135)

    eg_results = []
    for name, tl in eurgbp_trades.items():
        m = compute_outlier_stress_metrics(tl)
        m["strategy"] = name
        m["asset"] = "EURGBP"
        eg_results.append(m)
        print(f"{name:<26} | {m['base_net']:>+8.1f}p | {m['base_pf']:>7.2f} | {m['base_ret_dd']:>10.2f}x | {m['trim_net']:>+26.1f}p | {m['trim_pf']:>7.2f} | {m['tier']}")

    print("\n" + "=" * 135)
    print("   EURUSD FLEET: OUTLIER STRESS TEST & REAL-LIFE VIABILITY SCORECARD   ".center(135))
    print("=" * 135)
    print(f"{'Strategy Name':<26} | {'Base Net':>9} | {'Base PF':>7} | {'Base Ret/DD':>11} | {'Trimmed Net (Top-30% Gone)':>27} | {'Trim PF':>7} | {'RL Verdict':<26}")
    print("-" * 135)

    eu_results = []
    for name, tl in eurusd_trades.items():
        m = compute_outlier_stress_metrics(tl)
        m["strategy"] = name
        m["asset"] = "EURUSD"
        eu_results.append(m)
        print(f"{name:<26} | {m['base_net']:>+8.1f}p | {m['base_pf']:>7.2f} | {m['base_ret_dd']:>10.2f}x | {m['trim_net']:>+26.1f}p | {m['trim_pf']:>7.2f} | {m['tier']}")

    # Save summary table
    all_summary = eg_results + eu_results
    df_sum = pd.DataFrame(all_summary)
    summary_csv = RESULTS_DIR / "real_life_viability_stress_test_audit.csv"
    df_sum.to_csv(summary_csv, index=False)

    print("\n" + "=" * 135)
    print(f"[+] Successfully saved full audit results to: {summary_csv}")
    print(f"Total audit completed in {time.time()-t0:.1f}s.")


if __name__ == "__main__":
    main()
