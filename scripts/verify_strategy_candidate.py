#!/usr/bin/env python3
"""
================================================================================
  INDEPENDENT VERIFICATION & RED-TEAM HARNESS FOR CANDIDATE STRATEGIES
================================================================================
Independent verification barrier:
1. Holdout Total Trades >= 15
2. Holdout Profit Factor >= 1.20
3. Holdout Outlier Resistance: Net Pips ex Top 3 trades > 0
4. Discovery Consistency: Discovery Net Pips > 0 and Discovery PF >= 1.15
5. Anti-Redundancy:
   - Pairwise correlation < 0.50 against all 63 fleet models
   - Trade signal overlap < 50% against all 63 fleet models
================================================================================
"""

from __future__ import annotations
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Tuple, List

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from scripts.certify_strategy import certify_strategy, load_strategy_module
from tick_engine.reporting.run_manifest import compute_code_hash

FLEET_MANIFEST = ROOT_DIR / "certified_production_fleet" / "CERTIFIED_FLEET_MANIFEST.json"
MANIFEST_DIR = ROOT_DIR / "results" / "certified_strategy_manifests"


def load_fleet_models():
    """Loads all 63 models and their trade data from the fleet manifest."""
    with open(FLEET_MANIFEST, "r", encoding="utf-8") as f:
        fleet = json.load(f)

    models = []
    for s in fleet.get("strategies", []):
        sym = s["symbol"]
        name = s["name"]
        mf = s.get("manifest_file")
        csv_file = None
        if mf:
            c = ROOT_DIR / Path(mf).with_name(Path(mf).name.replace("_run_manifest.json", "_trades.csv"))
            if c.exists():
                csv_file = c
        if not csv_file or not csv_file.exists():
            cands = list(MANIFEST_DIR.glob(f"*{sym}*{name}*trades.csv"))
            if not cands:
                cands = list(MANIFEST_DIR.glob(f"*{name}*trades.csv"))
            if cands:
                csv_file = cands[0]

        df_trades = pd.read_csv(csv_file) if csv_file and csv_file.exists() else pd.DataFrame()
        models.append({
            "symbol": sym,
            "name": name,
            "category": s.get("category", ""),
            "relative_path": s.get("relative_path", ""),
            "trades_df": df_trades
        })
    return models


def compute_redundancy_against_fleet(candidate_df: pd.DataFrame, candidate_sym: str, fleet_models: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Computes pairwise correlation and overlap against all fleet models.
    Returns max overlap, max correlation, and detailed comparisons.
    """
    if len(candidate_df) == 0:
        return {"max_overlap": 0.0, "max_correlation": 0.0, "high_redundancy": []}

    pip_col = "pnl_pips" if "pnl_pips" in candidate_df.columns else ("pips" if "pips" in candidate_df.columns else None)
    cand_times = candidate_df["entry_time"].values
    cand_dir = candidate_df["direction"].values if "direction" in candidate_df.columns else np.ones(len(candidate_df))

    cand_c = candidate_df.copy()
    cand_c["day"] = pd.to_datetime(cand_c["entry_time"], unit="ms", utc=True).dt.date
    cand_daily = cand_c.groupby("day")[pip_col].sum()

    tolerance_ms = 30 * 60 * 1000  # 30 minutes
    max_overlap = 0.0
    max_corr = -1.0
    comparisons = []
    high_redundancy = []

    for model in fleet_models:
        m_name = f"{model['symbol']}__{model['name']}"
        # Skip self-comparison if testing existing model
        if model["name"] == candidate_df.attrs.get("strategy_id", "") or m_name == candidate_df.attrs.get("strategy_key", ""):
            continue

        m_df = model["trades_df"]
        if len(m_df) == 0:
            continue

        m_pip_col = "pnl_pips" if "pnl_pips" in m_df.columns else ("pips" if "pips" in m_df.columns else None)
        m_times = m_df["entry_time"].values
        m_dir = m_df["direction"].values if "direction" in m_df.columns else np.ones(len(m_df))

        # Check overlap only if same symbol or across all symbols
        overlap_pct = 0.0
        if model["symbol"] == candidate_sym:
            matches = 0
            for idx, t1 in enumerate(cand_times):
                d1 = cand_dir[idx]
                diffs = np.abs(m_times - t1)
                match_idx = np.where((diffs <= tolerance_ms) & (m_dir == d1))[0]
                if len(match_idx) > 0:
                    matches += 1
            overlap_cand = (matches / len(cand_times)) * 100.0 if len(cand_times) > 0 else 0.0
            overlap_fleet = (matches / len(m_times)) * 100.0 if len(m_times) > 0 else 0.0
            overlap_pct = max(overlap_cand, overlap_fleet)

        # Correlation across common days
        m_c = m_df.copy()
        m_c["day"] = pd.to_datetime(m_c["entry_time"], unit="ms", utc=True).dt.date
        m_daily = m_c.groupby("day")[m_pip_col].sum()

        all_days = sorted(list(set(cand_daily.index).union(set(m_daily.index))))
        c_series = [cand_daily.get(d, 0.0) for d in all_days]
        m_series = [m_daily.get(d, 0.0) for d in all_days]

        corr = 0.0
        if len(all_days) > 5 and np.std(c_series) > 1e-6 and np.std(m_series) > 1e-6:
            r = np.corrcoef(c_series, m_series)[0, 1]
            corr = 0.0 if np.isnan(r) else float(r)

        if overlap_pct > max_overlap:
            max_overlap = overlap_pct
        if corr > max_corr:
            max_corr = corr

        comp_info = {
            "model": m_name,
            "symbol": model["symbol"],
            "overlap_pct": round(overlap_pct, 1),
            "correlation": round(corr, 3)
        }
        comparisons.append(comp_info)

        if overlap_pct >= 50.0 or corr >= 0.50:
            high_redundancy.append(comp_info)

    return {
        "max_overlap": round(max_overlap, 1),
        "max_correlation": round(max_corr, 3),
        "high_redundancy": high_redundancy,
        "all_comparisons": comparisons
    }


def verify_candidate_strategy(strategy_path: Path, symbol: str = None) -> Dict[str, Any]:
    """Runs full independent verification on both holdout and discovery."""
    strat_cls = load_strategy_module(strategy_path)
    tmp = strat_cls()
    sym = symbol or getattr(tmp, "symbol", None) or getattr(tmp, "asset", None)
    if not sym:
        for s in ["EURGBP", "AUDCAD", "NZDCAD", "GBPCHF", "EURUSD", "GBPUSD", "USDJPY", "AUDNZD", "EURNZD", "GBPJPY", "USDCAD"]:
            if s in strategy_path.stem.upper() or s in strategy_path.parent.name.upper():
                sym = s
                break
    if not sym:
        raise ValueError("Could not determine symbol for strategy")

    print(f"\n>>> INDEPENDENT VERIFIER: Commencing evaluation of {strategy_path.name} on {sym} <<<")
    
    # 1. Run Holdout (Weeks 211..300)
    print("\n--- Running Holdout Partition (Weeks 211..300) ---")
    holdout_manifest = certify_strategy(strategy_file=strategy_path, symbol=sym, mode="holdout")
    holdout_stats = holdout_manifest["statistical_metrics"]
    holdout_trades = holdout_manifest.get("trade_log", [])
    holdout_df = pd.DataFrame(holdout_trades)

    strategy_id = strategy_path.parent.name if strategy_path.name == "strategy.py" else strategy_path.stem
    holdout_df.attrs["strategy_id"] = strategy_id
    holdout_df.attrs["strategy_key"] = f"{sym}__{strategy_id}"

    # 2. Run Discovery (Weeks 1..210)
    print("\n--- Running Discovery Partition (Weeks 1..210) ---")
    disc_manifest = certify_strategy(strategy_file=strategy_path, symbol=sym, mode="discovery")
    disc_stats = disc_manifest["statistical_metrics"]

    # Calculate metrics
    h_trades = holdout_stats["total_trades"]
    h_pf = holdout_stats["profit_factor"]
    h_net_pips = holdout_stats["net_pips"]
    h_pips_ex3 = holdout_stats["outlier_sensitivity"]["pips_ex_top3"]

    d_trades = disc_stats["total_trades"]
    d_pf = disc_stats["profit_factor"]
    d_net_pips = disc_stats["net_pips"]

    # 3. Anti-Redundancy against all 63 fleet models
    fleet_models = load_fleet_models()
    redundancy = compute_redundancy_against_fleet(holdout_df, sym, fleet_models)

    # Verification Barriers:
    # 1. Holdout Trades >= 15
    # 2. Holdout PF >= 1.20
    # 3. Outlier resistance: Net Pips ex Top 3 trades > 0
    # 4. Discovery consistency: Net Pips > 0 and PF >= 1.15
    # 5. Anti-redundancy: Pairwise correlation < 0.50 and overlap < 50%
    gate_checks = {
        "gate_1_holdout_sample_size": {
            "required": ">= 15 trades",
            "actual": h_trades,
            "pass": bool(h_trades >= 15)
        },
        "gate_2_holdout_pf": {
            "required": ">= 1.20 PF",
            "actual": round(h_pf, 2),
            "pass": bool(h_pf >= 1.20)
        },
        "gate_3_outlier_resistance": {
            "required": "Net Pips ex Top 3 > 0.0",
            "actual": round(h_pips_ex3, 1),
            "pass": bool(h_pips_ex3 > 0.0)
        },
        "gate_4_discovery_consistency": {
            "required": "Discovery Net Pips > 0 and PF >= 1.15",
            "actual": f"Net Pips={d_net_pips:+.1f}, PF={d_pf:.2f}",
            "pass": bool(d_net_pips > 0.0 and d_pf >= 1.15)
        },
        "gate_5_anti_redundancy": {
            "required": "Overlap < 50% and Correlation < 0.50",
            "actual": f"Max Overlap={redundancy['max_overlap']}%, Max Corr={redundancy['max_correlation']}",
            "pass": bool(redundancy["max_overlap"] < 50.0 and redundancy["max_correlation"] < 0.50)
        }
    }

    all_passed = all(g["pass"] for g in gate_checks.values())
    defect_reasons = []
    for g_name, g_val in gate_checks.items():
        if not g_val["pass"]:
            defect_reasons.append(f"{g_name}: required {g_val['required']}, got {g_val['actual']}")

    result = {
        "strategy_file": str(strategy_path),
        "strategy_name": strat_cls.__name__,
        "symbol": sym,
        "certified": all_passed,
        "gate_checks": gate_checks,
        "defect_reasons": defect_reasons,
        "redundancy_audit": redundancy,
        "holdout_stats": holdout_stats,
        "discovery_stats": disc_stats,
        "code_sha256": compute_code_hash(strategy_path)
    }

    print("\n" + "=" * 80)
    print(f"  VERIFICATION VERDICT: {'CERTIFIED' if all_passed else 'REJECTED'}")
    print("=" * 80)
    for g_name, g_val in gate_checks.items():
        status_str = "PASS" if g_val["pass"] else "FAIL"
        print(f"  [{status_str}] {g_name:<30}: {g_val['actual']} (Req: {g_val['required']})")
    if defect_reasons:
        print("\n  Defect Reasons:")
        for d in defect_reasons:
            print(f"    - {d}")
    print("=" * 80 + "\n")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Independent Strategy Verification Harness")
    parser.add_argument("--strategy", type=str, required=True, help="Path to strategy python file")
    parser.add_argument("--symbol", type=str, default=None, help="Symbol ticker")
    args = parser.parse_args()

    res = verify_candidate_strategy(Path(args.strategy), args.symbol)
    if not res["certified"]:
        sys.exit(1)
