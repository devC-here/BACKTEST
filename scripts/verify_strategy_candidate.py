#!/usr/bin/env python3
"""
================================================================================
  INDEPENDENT VERIFICATION & RED-TEAM HARNESS FOR CANDIDATE STRATEGIES
================================================================================
Independent verification barriers:
0. Strict Truncation Causality: signals_truncated[:idx+1] == signals_full (0 violations)
1. Holdout Total Trades >= 15
2. Holdout Profit Factor >= 1.20 & Net Pips > 0
3. Holdout Outlier Resistance: Net Pips ex Top 3 trades > 0
4. Discovery Consistency: Discovery Net Pips > 0 and Discovery PF >= 1.15
5. Anti-Redundancy:
   - Pairwise correlation < 0.50 against active certified fleet models
   - Trade signal overlap < 50% against active certified fleet models
================================================================================
"""

from __future__ import annotations
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from scripts.certify_strategy import certify_strategy, load_strategy_module, _instantiate_strategy, WARMUP_PREV_WEEKS
from tick_engine.core.dataloader import get_week_dataset, get_complete_split_weeks
from tick_engine.core.config import PIP_SIZES
from tick_engine.reporting.run_manifest import compute_file_hash

FLEET_MANIFEST = ROOT_DIR / "certified_production_fleet" / "CERTIFIED_FLEET_MANIFEST.json"
MANIFEST_DIR = ROOT_DIR / "results" / "certified_strategy_manifests"


def verify_truncation_causality(
    strat_cls,
    symbol: str,
    pip_size: float,
    max_signals: int = 6,
    mode: str = "holdout",
) -> Dict[str, Any]:
    """
    Gold-standard lookahead causality test (Gate 0):
    For up to `max_signals` non-zero post-warmup signals emitted at tick index `idx`,
    truncates the tick stream to `ticks[:idx + 1]` (removing ALL future ticks) and verifies
    that the signal direction, SL, and TP at `idx` remain bit-for-bit identical.
    """
    split_name = "holdout_test_30" if mode == "holdout" else "discovery_70"
    weeks = get_complete_split_weeks(symbol, split_name)
    if not weeks:
        weeks = get_complete_split_weeks(symbol, "discovery_70")

    # Test evenly distributed sample weeks first, then fall back to remaining weeks if needed
    step = max(1, len(weeks) // 18)
    sample_weeks = weeks[::step]
    ordered_weeks = sample_weeks + [w for w in weeks if w not in sample_weeks]

    strat_inst = _instantiate_strategy(strat_cls, symbol, pip_size)
    raw_dir = ROOT_DIR / "data" / symbol

    tested = 0
    violations = 0
    violation_details = []

    for w in ordered_weeks:
        try:
            ticks, meta = get_week_dataset(w, raw_dir=raw_dir, warmup_prev_weeks=WARMUP_PREV_WEEKS)
        except Exception:
            continue
        w_cnt = int(meta.get("warmup_tick_count", 0))
        if len(ticks) <= w_cnt + 100:
            continue

        res_full = strat_inst.generate_signals(ticks)
        if isinstance(res_full, tuple):
            sig_full = res_full[0]
            sl_full = res_full[1] if len(res_full) > 1 and res_full[1] is not None else None
            tp_full = res_full[2] if len(res_full) > 2 and res_full[2] is not None else None
        else:
            sig_full, sl_full, tp_full = res_full, None, None

        nz = np.where(sig_full[w_cnt:] != 0)[0] + w_cnt
        if len(nz) == 0:
            continue

        for idx in nz[:2]:
            tested += 1
            res_tr = strat_inst.generate_signals(ticks[: idx + 1])
            if isinstance(res_tr, tuple):
                sig_tr = res_tr[0]
                sl_tr = res_tr[1] if len(res_tr) > 1 and res_tr[1] is not None else None
                tp_tr = res_tr[2] if len(res_tr) > 2 and res_tr[2] is not None else None
            else:
                sig_tr, sl_tr, tp_tr = res_tr, None, None

            sig_match = len(sig_tr) > idx and int(sig_tr[idx]) == int(sig_full[idx])
            sl_match = True
            tp_match = True
            if sl_full is not None and sl_tr is not None:
                v_f = float(sl_full[idx]) if np.ndim(sl_full) > 0 else float(sl_full)
                v_t = float(sl_tr[idx]) if np.ndim(sl_tr) > 0 else float(sl_tr)
                sl_match = abs(v_f - v_t) <= 1e-6
            if tp_full is not None and tp_tr is not None:
                v_f = float(tp_full[idx]) if np.ndim(tp_full) > 0 else float(tp_full)
                v_t = float(tp_tr[idx]) if np.ndim(tp_tr) > 0 else float(tp_tr)
                tp_match = abs(v_f - v_t) <= 1e-6

            if not (sig_match and sl_match and tp_match):
                violations += 1
                violation_details.append(
                    f"week={w}, idx={int(idx)}: full=({int(sig_full[idx])}) vs trunc=({int(sig_tr[idx]) if len(sig_tr) > idx else 'OOB'})"
                )
        if tested >= max_signals:
            break

    return {
        "tested_signals": tested,
        "violations": violations,
        "violation_details": violation_details,
        "pass": bool(tested >= 1 and violations == 0),
    }


def load_fleet_models(fleet_manifest_path: Path = FLEET_MANIFEST, manifest_dir: Path = MANIFEST_DIR):
    """Loads certified fleet models and their Holdout trade data from the fleet manifest."""
    if not fleet_manifest_path.exists():
        return []
    with open(fleet_manifest_path, "r", encoding="utf-8") as f:
        fleet = json.load(f)

    models = []
    entries = fleet if isinstance(fleet, list) else fleet.get("strategies", [])
    for s in entries:
        sym = s["symbol"]
        name = s.get("name") or s.get("strategy_id") or s.get("id")
        mf = s.get("manifest_file")
        csv_file = None
        if s.get("holdout_trades_csv") and (ROOT_DIR / s["holdout_trades_csv"]).exists():
            csv_file = ROOT_DIR / s["holdout_trades_csv"]
        elif mf:
            c_hold = ROOT_DIR / Path(mf).with_name(Path(mf).name.replace("_run_manifest.json", "_holdout_trades.csv"))
            c_run = ROOT_DIR / Path(mf).with_name(Path(mf).name.replace("_run_manifest.json", "_trades.csv"))
            if c_hold.exists():
                csv_file = c_hold
            elif c_run.exists():
                csv_file = c_run
        if (not csv_file or not csv_file.exists()) and name:
            cands = list(manifest_dir.glob(f"*{sym}*{name}*holdout_trades.csv")) or list(manifest_dir.glob(f"*{sym}*{name}*trades.csv"))
            if cands:
                csv_file = cands[0]

        df_trades = pd.read_csv(csv_file) if csv_file and csv_file.exists() else pd.DataFrame()
        models.append({
            "symbol": sym,
            "name": name,
            "category": s.get("category", ""),
            "relative_path": s.get("relative_path", ""),
            "trades_df": df_trades,
        })
    return models


def compute_redundancy_against_fleet(
    candidate_df: pd.DataFrame,
    candidate_sym: str,
    fleet_models: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Computes pairwise correlation and overlap against all fleet models.
    """
    if len(candidate_df) == 0:
        return {"max_overlap": 0.0, "max_correlation": 0.0, "high_redundancy": [], "all_comparisons": []}

    pip_col = "pnl_pips" if "pnl_pips" in candidate_df.columns else ("pips" if "pips" in candidate_df.columns else None)
    cand_times = candidate_df["entry_time"].values
    cand_dir = candidate_df["direction"].values if "direction" in candidate_df.columns else np.ones(len(candidate_df))

    cand_c = candidate_df.copy()
    cand_c["day"] = pd.to_datetime(cand_c["entry_time"], unit="ms", utc=True).dt.date
    cand_daily = cand_c.groupby("day")[pip_col].sum()

    tolerance_ms = 30 * 60 * 1000
    max_overlap = 0.0
    max_corr = 0.0
    comparisons = []
    high_redundancy = []

    for model in fleet_models:
        m_name = f"{model['symbol']}__{model['name']}"
        if model["name"] == candidate_df.attrs.get("strategy_id", "") or m_name == candidate_df.attrs.get("strategy_key", ""):
            continue

        m_df = model["trades_df"]
        if len(m_df) == 0:
            continue

        m_pip_col = "pnl_pips" if "pnl_pips" in m_df.columns else ("pips" if "pips" in m_df.columns else None)
        m_times = m_df["entry_time"].values
        m_dir = m_df["direction"].values if "direction" in m_df.columns else np.ones(len(m_df))

        overlap_pct = 0.0
        if model["symbol"] == candidate_sym:
            matches = 0
            for idx, t1 in enumerate(cand_times):
                d1 = cand_dir[idx]
                diffs = np.abs(m_times - t1)
                if np.any((diffs <= tolerance_ms) & (m_dir == d1)):
                    matches += 1
            overlap_cand = (matches / len(cand_times)) * 100.0 if len(cand_times) > 0 else 0.0
            overlap_fleet = (matches / len(m_times)) * 100.0 if len(m_times) > 0 else 0.0
            overlap_pct = max(overlap_cand, overlap_fleet)

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
            "correlation": round(corr, 3),
        }
        comparisons.append(comp_info)

        if overlap_pct >= 50.0 or corr >= 0.50:
            high_redundancy.append(comp_info)

    return {
        "max_overlap": round(max_overlap, 1),
        "max_correlation": round(max_corr, 3),
        "high_redundancy": high_redundancy,
        "all_comparisons": comparisons,
    }


def verify_candidate_strategy(
    strategy_path: Path,
    symbol: Optional[str] = None,
    fleet_models: Optional[List[Dict[str, Any]]] = None,
    output_dir: Path = MANIFEST_DIR,
    fleet_manifest_path: Path = FLEET_MANIFEST,
) -> Dict[str, Any]:
    """Runs full independent verification including Gate 0 Truncation Causality."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fleet_manifest_path = Path(fleet_manifest_path)
    strat_cls = load_strategy_module(strategy_path)
    tmp = strat_cls()
    sym = symbol or getattr(tmp, "symbol", None) or getattr(tmp, "asset", None)
    if not sym:
        for s in PIP_SIZES.keys():
            if s in strategy_path.stem.upper() or s in strategy_path.parent.name.upper():
                sym = s
                break
    if not sym:
        raise ValueError("Could not determine symbol for strategy")

    pip_size = PIP_SIZES.get(sym, 0.0001)
    print(f"\n>>> INDEPENDENT VERIFIER: Commencing evaluation of {strategy_path.name} on {sym} <<<")

    # 0. Strict Truncation Causality Gate
    print("\n--- Running Gate 0: Strict Truncation Causality Test ---")
    causality = verify_truncation_causality(strat_cls, sym, pip_size, max_signals=6, mode="holdout")
    print(f"  Causality Check: {causality['violations']} violations across {causality['tested_signals']} signals.")

    # 1. Run Holdout (Weeks 211..300)
    print("\n--- Running Holdout Partition (Weeks 211..300) ---")
    holdout_manifest = certify_strategy(strategy_file=strategy_path, symbol=sym, mode="holdout", output_dir=output_dir)
    holdout_stats = holdout_manifest["statistical_metrics"]
    holdout_trades = holdout_manifest.get("trade_log", [])
    holdout_df = pd.DataFrame(holdout_trades)

    strategy_id = strategy_path.parent.name if strategy_path.name == "strategy.py" else strategy_path.stem
    holdout_df.attrs["strategy_id"] = strategy_id
    holdout_df.attrs["strategy_key"] = f"{sym}__{strategy_id}"

    # 2. Run Discovery (Weeks 1..210)
    print("\n--- Running Discovery Partition (Weeks 1..210) ---")
    disc_manifest = certify_strategy(strategy_file=strategy_path, symbol=sym, mode="discovery", run_stress=False, output_dir=output_dir)
    disc_stats = disc_manifest["statistical_metrics"]

    h_trades = holdout_stats["total_trades"]
    h_pf = holdout_stats["profit_factor"]
    h_net_pips = holdout_stats["net_pips"]
    h_pips_ex3 = holdout_stats["outlier_sensitivity"]["pips_ex_top3"]

    d_trades = disc_stats["total_trades"]
    d_pf = disc_stats["profit_factor"]
    d_net_pips = disc_stats["net_pips"]

    # 3. Anti-Redundancy against active certified fleet
    models_to_check = (
        fleet_models
        if fleet_models is not None
        else load_fleet_models(fleet_manifest_path=fleet_manifest_path, manifest_dir=output_dir)
    )
    redundancy = compute_redundancy_against_fleet(holdout_df, sym, models_to_check)

    gate_checks = {
        "gate_0_truncation_causality": {
            "required": "0 lookahead violations (tested >= 1)",
            "actual": f"{causality['violations']} violations / {causality['tested_signals']} signals",
            "pass": causality["pass"],
        },
        "gate_1_holdout_sample_size": {
            "required": ">= 15 trades",
            "actual": h_trades,
            "pass": bool(h_trades >= 15),
        },
        "gate_2_holdout_pf": {
            "required": ">= 1.20 PF & Net Pips > 0",
            "actual": f"PF={round(h_pf, 2)}, Net={round(h_net_pips, 1):+.1f}p",
            "pass": bool(h_pf >= 1.20 and h_net_pips > 0.0),
        },
        "gate_3_outlier_resistance": {
            "required": "Net Pips ex Top 3 > 0.0",
            "actual": round(h_pips_ex3, 1),
            "pass": bool(h_pips_ex3 > 0.0),
        },
        "gate_4_discovery_consistency": {
            "required": "Discovery Net Pips > 0 and PF >= 1.15",
            "actual": f"Net Pips={d_net_pips:+.1f}, PF={d_pf:.2f} ({d_trades} tr)",
            "pass": bool(d_net_pips > 0.0 and d_pf >= 1.15),
        },
        "gate_5_anti_redundancy": {
            "required": "Overlap < 50% and Correlation < 0.50",
            "actual": f"Max Overlap={redundancy['max_overlap']}%, Max Corr={redundancy['max_correlation']}",
            "pass": bool(redundancy["max_overlap"] < 50.0 and redundancy["max_correlation"] < 0.50),
        },
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
        "truncation_causality_audit": causality,
        "redundancy_audit": redundancy,
        "holdout_stats": holdout_stats,
        "discovery_stats": disc_stats,
        "holdout_manifest": holdout_manifest,
        "code_sha256": compute_file_hash(strategy_path),
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
    parser.add_argument("--fleet-manifest", type=str, default=str(FLEET_MANIFEST), help="Path to fleet manifest JSON")
    parser.add_argument("--output-dir", type=str, default=str(MANIFEST_DIR), help="Directory for output manifests and trade CSVs")
    args = parser.parse_args()

    res = verify_candidate_strategy(
        Path(args.strategy),
        symbol=args.symbol,
        output_dir=Path(args.output_dir),
        fleet_manifest_path=Path(args.fleet_manifest),
    )
    if not res["certified"]:
        sys.exit(1)
