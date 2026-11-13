#!/usr/bin/env python3
"""
================================================================================
  INDIVIDUAL STRATEGY BROKER-COST STRESS & RELIABILITY BATTERY
================================================================================
Conducts an exhaustive per-strategy reliability audit across all 101 strategies,
enforcing:
1. 5-Profile Broker Cost Stress Testing (Baseline, +50% Spread, 0.50p Slip, $6 Comm, Composite)
2. Outlier Fragility (Top 1, 3, 5 winners removed)
3. Statistical Confidence (95% CI, t-statistic, p-value)
4. In-Sample vs Quarantined Holdout Walk-Forward Stability
5. Standalone Dollar PnL and Expectancy under $300 (0.30%) fixed-risk sizing
================================================================================
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from tick_engine.core.config import PIP_SIZES, get_friction_pips
from tick_engine.reporting.run_manifest import compute_statistical_metrics

FLEET_MANIFEST = ROOT_DIR / "certified_production_fleet" / "CERTIFIED_FLEET_MANIFEST.json"
MANIFEST_DIR = ROOT_DIR / "results" / "certified_strategy_manifests"
OUTPUT_REPORT = ROOT_DIR / "results" / "individual_strategy_stress_and_reliability_report.json"

# Typical floating spreads per pair in Dukascopy tick data
TYPICAL_SPREADS_PIPS = {
    "EURUSD": 0.40,
    "USDJPY": 0.50,
    "USDCAD": 0.90,
    "AUDCAD": 1.20,
    "NZDCAD": 1.40,
    "AUDNZD": 1.50,
    "GBPCHF": 1.80,
    "GBPJPY": 1.40,
    "EURNZD": 2.20,
    "EURGBP": 0.60
}

# Pip values for 1 standard lot
PIP_VALUES = {
    "EURUSD": 10.00,
    "USDJPY": 6.70,
    "USDCAD": 7.25,
    "AUDCAD": 7.25,
    "NZDCAD": 7.25,
    "AUDNZD": 6.00,
    "EURNZD": 6.00,
    "EURGBP": 13.00,
    "GBPCHF": 12.25,
    "GBPJPY": 6.70
}

BASE_RISK_USD = 300.0  # 0.30% on $100k account


def audit_all_strategies():
    with open(FLEET_MANIFEST, "r", encoding="utf-8") as f:
        fleet_data = json.load(f)

    strategies = fleet_data["strategies"]
    print(f"Auditing {len(strategies)} individual strategies across all 10 reliability gates...\n")

    records = []

    for s in strategies:
        sym = s["symbol"]
        name = s["name"]
        cat = s.get("category", "")

        # 1. Locate manifest & trade log
        m_candidates = [
            MANIFEST_DIR / f"{sym}__{name}_run_manifest.json",
            MANIFEST_DIR / f"{sym.lower()}_{name}_run_manifest.json",
            MANIFEST_DIR / f"{name}_run_manifest.json"
        ]
        m_file = next((f for f in m_candidates if f.exists()), None)
        if not m_file:
            print(f"Warning: Manifest missing for {sym} {name}")
            continue

        with open(m_file, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        csv_candidates = [
            MANIFEST_DIR / f"{sym}__{name}_trades.csv",
            MANIFEST_DIR / f"{sym.lower()}_{name}_trades.csv",
            MANIFEST_DIR / f"{name}_trades.csv"
        ]
        csv_file = next((f for f in csv_candidates if f.exists()), None)
        if not csv_file:
            print(f"Warning: Trade CSV missing for {sym} {name}")
            continue

        df_trades = pd.read_csv(csv_file)
        n_trades = len(df_trades)
        if n_trades == 0:
            continue

        pips_arr = df_trades["pips"].values

        # Strategy parameters
        params = manifest.get("custom_metadata", {}).get("parameters", {})
        sl_pips = float(params.get("sl_pips", 20.0))
        tp_pips = float(params.get("tp_pips", sl_pips * 2.0))
        dollar_per_pip_sized = BASE_RISK_USD / max(sl_pips, 1.0)

        # Baseline metrics
        base_net = float(np.sum(pips_arr))
        base_wins = pips_arr[pips_arr > 0]
        base_loss = pips_arr[pips_arr < 0]
        base_pf = float(np.sum(base_wins) / abs(np.sum(base_loss))) if len(base_loss) > 0 and np.sum(base_loss) != 0 else 99.0
        base_exp = float(base_net / n_trades)
        base_sized_pnl = float(base_net * dollar_per_pip_sized)
        base_sized_exp = float(base_exp * dollar_per_pip_sized)

        # In-sample discovery metrics
        disc = manifest.get("custom_metadata", {}).get("discovery_in_sample_metrics", {})
        disc_net = float(disc.get("net_pips", 0.0))
        disc_pf = float(disc.get("profit_factor", 0.0))
        disc_trades = int(disc.get("total_trades", 0))

        # 2. Broker Cost Stress Battery
        # Typical spread and extra commission
        typical_spr = TYPICAL_SPREADS_PIPS.get(sym, 1.0)
        pip_val = PIP_VALUES.get(sym, 10.0)
        # Extra $2/lot commission in pips
        extra_comm_pips = 2.0 / (100_000 * PIP_SIZES.get(sym, 0.0001) * (10.0 / pip_val))

        # Profile A: Spread Shock (+50% spread) -> penalty = 0.5 * spread
        pips_spread_shock = pips_arr - (0.5 * typical_spr)
        net_spread_shock = float(np.sum(pips_spread_shock))
        pf_spread_shock = float(np.sum(pips_spread_shock[pips_spread_shock > 0]) / abs(np.sum(pips_spread_shock[pips_spread_shock < 0]))) if len(pips_spread_shock[pips_spread_shock < 0]) > 0 else 99.0

        # Profile B: Heavy Slippage Shock (0.50p slippage vs 0.20p base -> extra 0.30p)
        pips_slip_shock = pips_arr - 0.30
        net_slip_shock = float(np.sum(pips_slip_shock))
        pf_slip_shock = float(np.sum(pips_slip_shock[pips_slip_shock > 0]) / abs(np.sum(pips_slip_shock[pips_slip_shock < 0]))) if len(pips_slip_shock[pips_slip_shock < 0]) > 0 else 99.0

        # Profile C: High Commission Shock ($6.00/lot vs $4.00 base)
        pips_comm_shock = pips_arr - extra_comm_pips
        net_comm_shock = float(np.sum(pips_comm_shock))
        pf_comm_shock = float(np.sum(pips_comm_shock[pips_comm_shock > 0]) / abs(np.sum(pips_comm_shock[pips_comm_shock < 0]))) if len(pips_comm_shock[pips_comm_shock < 0]) > 0 else 99.0

        # Profile D: Severe Composite Stress (Spread +50% AND 0.50p slippage AND $6/lot commission)
        pips_composite = pips_arr - (0.5 * typical_spr + 0.30 + extra_comm_pips)
        net_composite = float(np.sum(pips_composite))
        pf_composite = float(np.sum(pips_composite[pips_composite > 0]) / abs(np.sum(pips_composite[pips_composite < 0]))) if len(pips_composite[pips_composite < 0]) > 0 else 99.0
        exp_composite = float(net_composite / n_trades)

        # 3. Outlier Fragility
        sorted_pips = np.sort(pips_arr)
        pips_ex1 = float(np.sum(sorted_pips[:-1])) if n_trades > 1 else 0.0
        pips_ex3 = float(np.sum(sorted_pips[:-3])) if n_trades > 3 else 0.0
        pips_ex5 = float(np.sum(sorted_pips[:-5])) if n_trades > 5 else 0.0

        is_outlier_dependent = bool(pips_ex3 <= 0.0) if n_trades >= 20 else bool(pips_ex1 <= 0.0)

        # 4. Statistical Confidence
        std_p = float(np.std(pips_arr, ddof=1)) if n_trades > 1 else 0.0
        std_err = std_p / np.sqrt(n_trades) if n_trades > 1 else 0.0
        ci_95 = [round(base_exp - 1.96 * std_err, 2), round(base_exp + 1.96 * std_err, 2)]
        t_stat = round(base_exp / std_err, 2) if std_err > 0 else 0.0
        import math
        p_val = round(math.erfc(abs(t_stat) / math.sqrt(2)), 4)

        # 5. Reliability Gates & Grading
        reasons = []
        if base_net <= 0:
            reasons.append("Holdout PnL <= 0")
        if base_pf < 1.15:
            reasons.append(f"Holdout PF {base_pf:.2f} < 1.15")
        if n_trades < 15:
            reasons.append(f"Holdout Trades {n_trades} < 15")
        if is_outlier_dependent:
            reasons.append(f"Outlier Fragile (ex-top3: {pips_ex3:.1f}p)")
        if disc_net <= 0:
            reasons.append(f"Discovery PnL {disc_net:.1f}p <= 0")
        if disc_pf < 1.15:
            reasons.append(f"Discovery PF {disc_pf:.2f} < 1.15")
        if net_composite <= 0 or pf_composite < 1.00:
            reasons.append(f"Fails Severe Stress (Net: {net_composite:.1f}p, PF: {pf_composite:.2f})")

        if len(reasons) == 0 and base_pf >= 1.30 and pf_composite >= 1.15 and pips_ex5 > 0:
            grade = "GRADE A: INSTITUTIONAL CHAMPION"
        elif len(reasons) == 0:
            grade = "GRADE B: PRODUCTION CERTIFIED"
        elif len(reasons) <= 2 and base_net > 0 and base_pf >= 1.10 and net_slip_shock > 0:
            grade = "GRADE C: PROVISIONAL / BORDERLINE"
        else:
            grade = "GRADE F: REJECTED / FRAGILE"

        records.append({
            "symbol": sym,
            "name": name,
            "category": cat,
            "grade": grade,
            "sl_pips": sl_pips,
            "tp_pips": tp_pips,
            "holdout_trades": n_trades,
            "holdout_net_pips": round(base_net, 1),
            "holdout_pf": round(base_pf, 2),
            "holdout_exp_pips": round(base_exp, 2),
            "sized_dollar_pnl": round(base_sized_pnl, 2),
            "sized_dollar_exp": round(base_sized_exp, 2),
            "discovery_net_pips": round(disc_net, 1),
            "discovery_pf": round(disc_pf, 2),
            "discovery_trades": disc_trades,
            "stress_spread_50_net": round(net_spread_shock, 1),
            "stress_spread_50_pf": round(pf_spread_shock, 2),
            "stress_slippage_050_net": round(net_slip_shock, 1),
            "stress_slippage_050_pf": round(pf_slip_shock, 2),
            "stress_commission_6_net": round(net_comm_shock, 1),
            "stress_commission_6_pf": round(pf_comm_shock, 2),
            "stress_composite_net": round(net_composite, 1),
            "stress_composite_pf": round(pf_composite, 2),
            "stress_composite_exp": round(exp_composite, 2),
            "pips_ex_top1": round(pips_ex1, 1),
            "pips_ex_top3": round(pips_ex3, 1),
            "pips_ex_top5": round(pips_ex5, 1),
            "is_outlier_dependent": is_outlier_dependent,
            "ci_95": ci_95,
            "t_statistic": t_stat,
            "p_value": p_val,
            "failure_reasons": "; ".join(reasons) if reasons else "NONE (All Reliability Gates Passed)"
        })

    df = pd.DataFrame(records)

    print("=" * 85)
    print("  EXECUTIVE SUMMARY: INDIVIDUAL STRATEGY AUDIT (101 STRATEGIES)")
    print("=" * 85)
    print(f"  GRADE A (Institutional Champion): {len(df[df['grade'] == 'GRADE A: INSTITUTIONAL CHAMPION'])}")
    print(f"  GRADE B (Production Certified):   {len(df[df['grade'] == 'GRADE B: PRODUCTION CERTIFIED'])}")
    print(f"  GRADE C (Provisional / Warn):     {len(df[df['grade'] == 'GRADE C: PROVISIONAL / BORDERLINE'])}")
    print(f"  GRADE F (Rejected / Fragile):     {len(df[df['grade'] == 'GRADE F: REJECTED / FRAGILE'])}")
    print("-" * 85)
    print(f"  Survives Severe Composite Stress: {len(df[df['stress_composite_net'] > 0])} / {len(df)}")
    print(f"  Outlier Resistant (Ex Top 3 > 0): {len(df[df['is_outlier_dependent'] == False])} / {len(df)}")
    print(f"  Dual-Profitable (Hold & Disc > 0):{len(df[(df['holdout_net_pips'] > 0) & (df['discovery_net_pips'] > 0)])} / {len(df)}")
    print("=" * 85)

    with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    print(f"\nSaved complete individual audit report to: {OUTPUT_REPORT}")
    return df


if __name__ == "__main__":
    audit_all_strategies()

