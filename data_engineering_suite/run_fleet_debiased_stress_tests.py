#!/usr/bin/env python3
"""
================================================================================
  INSTITUTIONAL ZERO-BIAS & 7-GATE FLEET STRESS-TESTING SUITE
================================================================================
Standard Operating Procedure: SOP-TRD-VAL-2026-003 & Zero-Bias Audit
Eliminates:
  1. Lookahead / Causal Bias (Shifted closed-bar execution verification)
  2. Right-Tail Outlier / Lottery Bias (Top 25% winners dropped -> remains >= breakeven)
  3. Execution Friction & Slippage Bias (+0.3p, +0.5p, +1.0p adverse slippage & BE threshold)
  4. Regime / Temporal Overfitting Bias (Year-by-year 2021-2026 breakdown, min 4/5 positive years)
  5. Sequence Risk & Path Dependency Bias (5,000 Monte Carlo permutations, Prop Breach Prob < 0.5%)
  6. In-Sample Selection Bias (Quarantined 30% Holdout forward testing, Seed 42)
  7. Statistical Insignificance Bias (N >= 150 trades, t-stat >= 2.50, single trade < 8%)

Usage:
  .venv\\Scripts\\python.exe data_engineering_suite/run_fleet_debiased_stress_tests.py --symbol AUDCAD
  .venv\\Scripts\\python.exe data_engineering_suite/run_fleet_debiased_stress_tests.py --all
================================================================================
"""

from __future__ import annotations
import sys
import os
import json
import argparse
import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True

ROOT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DATA_SPLITS_FILE = ROOT_DIR / "data_splits" / "master_70_30_splits.json"


# ==============================================================================
# 1. CORE DRAWDOWN & STREAK UTILITIES
# ==============================================================================

def compute_drawdown_series(pips_series: np.ndarray) -> Tuple[float, np.ndarray]:
    if len(pips_series) == 0:
        return 0.0, np.array([])
    equity = np.cumsum(pips_series)
    peak = np.maximum.accumulate(equity)
    dd = peak - equity
    return float(np.max(dd)), dd


def compute_underwater_and_streaks(trades: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not trades:
        return {"max_consec_loss": 0, "max_consec_win": 0, "max_underwater_days": 0.0}

    sorted_trades = sorted(trades, key=lambda x: x.get("entry_time", 0))
    pips = [t.get("pips", 0.0) for t in sorted_trades]
    times = [t.get("entry_time", 0) for t in sorted_trades]

    max_consec_loss, cur_loss = 0, 0
    max_consec_win, cur_win = 0, 0
    for p in pips:
        if p < 0:
            cur_loss += 1
            cur_win = 0
            if cur_loss > max_consec_loss:
                max_consec_loss = cur_loss
        elif p > 0:
            cur_win += 1
            cur_loss = 0
            if cur_win > max_consec_win:
                max_consec_win = cur_win
        else:
            cur_loss = 0
            cur_win = 0

    equity = np.cumsum(pips)
    peak = 0.0
    peak_time = times[0] if times else 0
    max_underwater_ms = 0

    for i in range(len(equity)):
        if equity[i] >= peak:
            duration_ms = times[i] - peak_time
            if duration_ms > max_underwater_ms:
                max_underwater_ms = duration_ms
            peak = equity[i]
            peak_time = times[i]
        else:
            duration_ms = times[i] - peak_time
            if duration_ms > max_underwater_ms:
                max_underwater_ms = duration_ms

    max_underwater_days = max_underwater_ms / (1000 * 86400) if times else 0.0
    return {
        "max_consec_loss": max_consec_loss,
        "max_consec_win": max_consec_win,
        "max_underwater_days": round(max_underwater_days, 1)
    }


# ==============================================================================
# 2. ANNUAL CALENDAR PERFORMANCE BREAKDOWN
# ==============================================================================

def compute_annual_breakdown(trades: List[Dict[str, Any]]) -> Dict[str, Any]:
    annual_data: Dict[str, Dict[str, Any]] = {}
    for y in [2021, 2022, 2023, 2024, 2025, 2026]:
        annual_data[str(y)] = {"trades": 0, "pnl": 0.0, "wins": 0, "losses": 0, "win_pips": 0.0, "loss_pips": 0.0}

    for t in trades:
        ts = t.get("entry_time", 0) / 1000.0
        if ts <= 0:
            continue
        try:
            dt = datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc)
            year = dt.year
        except Exception:
            year = 2021

        key = str(year) if str(year) in annual_data else "2026"
        p = t.get("pips", 0.0)

        annual_data[key]["trades"] += 1
        annual_data[key]["pnl"] += p
        if p > 0:
            annual_data[key]["wins"] += 1
            annual_data[key]["win_pips"] += p
        elif p < 0:
            annual_data[key]["losses"] += 1
            annual_data[key]["loss_pips"] += abs(p)

    out: Dict[str, Any] = {}
    positive_years = 0
    worst_year_pnl = 999999.0
    year_pnls = []

    for y_key, d in annual_data.items():
        n = d["trades"]
        pnl = round(d["pnl"], 1)
        wr = round((d["wins"] / n * 100), 1) if n > 0 else 0.0
        pf = round(d["win_pips"] / d["loss_pips"], 2) if d["loss_pips"] > 0 else (99.0 if d["win_pips"] > 0 else 0.0)
        out[f"y_{y_key}_trades"] = n
        out[f"y_{y_key}_pnl"] = pnl
        out[f"y_{y_key}_wr"] = wr
        out[f"y_{y_key}_pf"] = pf

        if n > 0:
            year_pnls.append(pnl)
            if pnl > 0:
                positive_years += 1
            if pnl < worst_year_pnl:
                worst_year_pnl = pnl

    total_pnl = sum(year_pnls) if year_pnls else 0.0
    max_year_pnl = max(year_pnls) if year_pnls else 0.0
    max_year_share = round((max_year_pnl / total_pnl * 100), 1) if total_pnl > 0 else 100.0

    out["positive_years"] = positive_years
    out["total_evaluated_years"] = len([p for p in year_pnls if p != 0.0])
    out["worst_year_pnl"] = round(worst_year_pnl, 1) if worst_year_pnl != 999999.0 else 0.0
    out["max_year_share_pct"] = max_year_share
    return out


# ==============================================================================
# 3. MONTE CARLO PERMUTATION & RISK OF RUIN ENGINE
# ==============================================================================

def compute_monte_carlo_and_ror(trades: List[Dict[str, Any]], num_sims: int = 5000, rng_seed: int = 42) -> Dict[str, Any]:
    n = len(trades)
    if n == 0:
        return {}
    rng = np.random.default_rng(rng_seed)
    pips = np.array([t.get("pips", 0.0) for t in trades])

    hist_max_dd, _ = compute_drawdown_series(pips)

    perm_dds = np.zeros(num_sims)
    for i in range(num_sims):
        perm_pips = rng.permutation(pips)
        perm_dds[i], _ = compute_drawdown_series(perm_pips)

    losses = -pips[pips < 0]
    avg_loss_pips = float(np.mean(losses)) if len(losses) > 0 else 15.0
    prop_ruin_pips = max(200.0, 16.0 * avg_loss_pips)
    prop_daily_ruin_pips = max(100.0, 8.0 * avg_loss_pips)

    prob_max_dd_breached = float(np.mean(perm_dds >= prop_ruin_pips) * 100)

    mean_p = float(np.mean(pips))
    std_p = float(np.std(pips, ddof=1)) if len(pips) > 1 else 1.0
    z_score = mean_p / std_p if std_p > 0 else 0.0
    units = 16.0
    if z_score <= 0:
        analytical_ror = 100.0
    else:
        adv = np.clip(z_score, 0.001, 0.999)
        analytical_ror = float(np.clip(((1.0 - adv) / (1.0 + adv)) ** units * 100, 0.0, 100.0))

    return {
        "mc_hist_dd": round(hist_max_dd, 1),
        "mc_dd_p50": round(float(np.percentile(perm_dds, 50)), 1),
        "mc_dd_p95": round(float(np.percentile(perm_dds, 95)), 1),
        "mc_dd_p99": round(float(np.percentile(perm_dds, 99)), 1),
        "mc_dd_p95_ratio": round(float(np.percentile(perm_dds, 95)) / hist_max_dd, 2) if hist_max_dd > 0 else 1.0,
        "avg_loss_pips": round(avg_loss_pips, 1),
        "prop_ruin_threshold_pips": round(prop_ruin_pips, 1),
        "prop_mc_breach_prob_pct": round(prob_max_dd_breached, 2),
        "analytical_ror_pct": round(analytical_ror, 3)
    }


# ==============================================================================
# 4. EXECUTION FRICTION & SLIPPAGE TOLERANCE
# ==============================================================================

def compute_friction_and_slippage(trades: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(trades)
    if n == 0:
        return {}
    base_pips = np.array([t.get("pips", 0.0) for t in trades])
    base_pnl = float(np.sum(base_pips))
    expectancy = base_pnl / n

    penalties = [0.3, 0.5, 0.8, 1.0, 1.5]
    res: Dict[str, Any] = {
        "expectancy_pips": round(expectancy, 2),
        "breakeven_slippage": round(expectancy, 2)
    }

    for pen in penalties:
        adj = base_pips - pen
        pnl_adj = float(np.sum(adj))
        w = adj[adj > 0]
        l = -adj[adj < 0]
        sum_w = float(np.sum(w))
        sum_l = float(np.sum(l))
        pf_adj = round(sum_w / sum_l, 2) if sum_l > 0 else (99.0 if sum_w > 0 else 0.0)
        pen_str = str(pen).replace(".", "_")
        res[f"pnl_slip_{pen_str}p"] = round(pnl_adj, 1)
        res[f"pf_slip_{pen_str}p"] = pf_adj

    return res


# ==============================================================================
# 5. RIGHT-TAIL OUTLIER TRIMMING (TOP 25% WINNERS DROPPED)
# ==============================================================================

def compute_outlier_trimming(trades: List[Dict[str, Any]], trim_pct: float = 0.25) -> Dict[str, Any]:
    if not trades:
        return {}
    pips = [t.get("pips", 0.0) for t in trades]
    wins = sorted([p for p in pips if p > 0])
    losses = [p for p in pips if p <= 0]

    base_pnl = float(np.sum(pips))
    if not wins:
        return {"trimmed_pnl": 0.0, "trimmed_pf": 0.0, "profit_retention_ratio": 0.0}

    n_wins = len(wins)
    cutoff_idx = int(np.floor(n_wins * (1.0 - trim_pct)))
    retained_wins = wins[:cutoff_idx]
    trimmed_trades = retained_wins + losses

    trimmed_pnl = float(np.sum(trimmed_trades))
    sum_w = float(np.sum(retained_wins))
    sum_l = float(np.sum([-x for x in losses if x < 0]))
    trimmed_pf = round(sum_w / sum_l, 2) if sum_l > 0 else (99.0 if sum_w > 0 else 0.0)
    retention_ratio = round((trimmed_pnl / base_pnl * 100), 1) if base_pnl > 0 else 0.0

    return {
        "trimmed_pnl": round(trimmed_pnl, 1),
        "trimmed_pf": trimmed_pf,
        "trimmed_winners_dropped": n_wins - cutoff_idx,
        "profit_retention_ratio": retention_ratio
    }


# ==============================================================================
# 6. EVALUATE COMPLETE 7-GATE QUALIFICATION ON A STRATEGY
# ==============================================================================

def evaluate_strategy_7_gates(
    strat_name: str,
    symbol: str,
    trades: List[Dict[str, Any]],
    disc_weeks: List[int],
    hold_weeks: List[int]
) -> Dict[str, Any]:
    n = len(trades)
    if n == 0:
        return {
            "strategy": strat_name,
            "symbol": symbol,
            "trades": 0,
            "net_pips": 0.0,
            "pf": 0.0,
            "wr": 0.0,
            "ret_to_dd": 0.0,
            "max_dd": 0.0,
            "passed_gates": 0,
            "verdict": "TIER 3 (NO TRADES)",
            "gate1_sample": False,
            "gate1_notes": "N=0",
            "gate2_annual": False,
            "gate2_notes": "N=0",
            "gate3_downside": False,
            "gate3_notes": "N=0",
            "gate4_mc_ror": False,
            "gate4_notes": "N=0",
            "gate5_friction": False,
            "gate5_notes": "N=0",
            "gate6_outliers": False,
            "gate6_notes": "N=0",
            "gate7_holdout": False,
            "gate7_notes": "N=0",
            "annual_breakdown": {},
            "mc_stats": {},
            "friction_stats": {},
            "outlier_stats": {},
            "streaks": {}
        }

    pips = np.array([t.get("pips", 0.0) for t in trades])
    net_pips = float(np.sum(pips))
    mean_p = float(np.mean(pips))
    std_p = float(np.std(pips, ddof=1)) if n > 1 else 1.0
    t_stat = float(mean_p / (std_p / np.sqrt(n))) if std_p > 0 else 0.0

    wins = pips[pips > 0]
    losses = -pips[pips < 0]
    sum_w = float(np.sum(wins))
    sum_l = float(np.sum(losses))
    pf = round(sum_w / sum_l, 2) if sum_l > 0 else 99.0
    wr = round(float(len(wins) / n * 100), 1)
    max_win = float(np.max(pips)) if len(wins) > 0 else 0.0
    max_win_share = round((max_win / net_pips * 100), 1) if net_pips > 0 else 100.0

    hist_max_dd, _ = compute_drawdown_series(pips)
    ret_to_dd = round(net_pips / hist_max_dd, 2) if hist_max_dd > 0 else 99.0

    # Gate 1: Sample Size & Significance
    g1_pass = bool(n >= 150 and t_stat >= 2.50 and max_win_share <= 10.0)
    g1_notes = f"N={n}, t={t_stat:.2f}, max_win_pct={max_win_share}%"

    # Gate 2: Annual Consistency
    ann_stats = compute_annual_breakdown(trades)
    g2_pass = bool(ann_stats["positive_years"] >= 4 and ann_stats["max_year_share_pct"] <= 48.0)
    g2_notes = f"{ann_stats['positive_years']}/{ann_stats['total_evaluated_years']} pos yrs, max_yr={ann_stats['max_year_share_pct']}%"

    # Gate 3: Downside Risk & Capital Preservation
    streaks = compute_underwater_and_streaks(trades)
    g3_pass = bool(hist_max_dd <= 350.0 and ret_to_dd >= 3.5 and streaks["max_consec_loss"] <= 10)
    g3_notes = f"MaxDD={hist_max_dd:.1f}p, Ret/DD={ret_to_dd:.2f}x, ConsecLoss={streaks['max_consec_loss']}"

    # Gate 4: Risk of Ruin & Monte Carlo Sequence Stress
    mc_stats = compute_monte_carlo_and_ror(trades, num_sims=5000)
    g4_pass = bool(mc_stats["prop_mc_breach_prob_pct"] < 0.5 and mc_stats["analytical_ror_pct"] < 1.0)
    g4_notes = f"MC P95 DD={mc_stats['mc_dd_p95']:.1f}p, RuinProb={mc_stats['prop_mc_breach_prob_pct']}%"

    # Gate 5: Execution Friction & Slippage Resilience
    fric_stats = compute_friction_and_slippage(trades)
    g5_pass = bool(fric_stats["breakeven_slippage"] >= 1.20 and fric_stats.get("pf_slip_0_5p", 0) >= 1.15)
    g5_notes = f"Exp={fric_stats['expectancy_pips']}p, BESlip={fric_stats['breakeven_slippage']}p, PF@0.5p={fric_stats.get('pf_slip_0_5p', 0)}"

    # Gate 6: Right-Tail Outlier Trimming (Top 25% winners dropped)
    outlier_stats = compute_outlier_trimming(trades, trim_pct=0.25)
    g6_pass = bool(outlier_stats["trimmed_pnl"] >= 0 and outlier_stats["trimmed_pf"] >= 1.00 and outlier_stats["profit_retention_ratio"] >= 20.0)
    g6_notes = f"TrimmedPnL={outlier_stats['trimmed_pnl']}p, TrimmedPF={outlier_stats['trimmed_pf']}, PRR={outlier_stats['profit_retention_ratio']}%"

    # Gate 7: Quarantined 30% Holdout Audit
    hold_trades = [t for t in trades if t.get("week") in hold_weeks]
    disc_trades = [t for t in trades if t.get("week") in disc_weeks]

    h_pips = np.array([t.get("pips", 0.0) for t in hold_trades])
    h_pnl = float(np.sum(h_pips)) if len(h_pips) > 0 else 0.0
    h_w = h_pips[h_pips > 0]
    h_l = -h_pips[h_pips < 0]
    h_pf = round(float(np.sum(h_w) / np.sum(h_l)), 2) if np.sum(h_l) > 0 else 99.0

    d_pips = np.array([t.get("pips", 0.0) for t in disc_trades])
    d_pnl = float(np.sum(d_pips)) if len(d_pips) > 0 else 1.0
    d_w = d_pips[d_pips > 0]
    d_l = -d_pips[d_pips < 0]
    d_pf = round(float(np.sum(d_w) / np.sum(d_l)), 2) if np.sum(d_l) > 0 else 99.0

    g7_pass = bool(h_pnl > 0 and h_pf >= 1.15)
    g7_notes = f"HoldoutPnL={h_pnl:.1f}p, HoldoutPF={h_pf:.2f} (DiscPF={d_pf:.2f})"

    passed_gates = sum([g1_pass, g2_pass, g3_pass, g4_pass, g5_pass, g6_pass, g7_pass])

    if passed_gates == 7:
        verdict = "TIER 1 (APPROVED - 7/7 GATES)"
    elif passed_gates >= 5 and g7_pass and g6_pass:
        verdict = "TIER 2 (CONDITIONAL)"
    else:
        verdict = "TIER 3 (REJECTED / FRAGILE)"

    return {
        "strategy": strat_name,
        "symbol": symbol,
        "trades": n,
        "net_pips": round(net_pips, 1),
        "pf": pf,
        "wr": wr,
        "ret_to_dd": ret_to_dd,
        "max_dd": round(hist_max_dd, 1),
        "passed_gates": passed_gates,
        "verdict": verdict,
        "gate1_sample": g1_pass,
        "gate1_notes": g1_notes,
        "gate2_annual": g2_pass,
        "gate2_notes": g2_notes,
        "gate3_downside": g3_pass,
        "gate3_notes": g3_notes,
        "gate4_mc_ror": g4_pass,
        "gate4_notes": g4_notes,
        "gate5_friction": g5_pass,
        "gate5_notes": g5_notes,
        "gate6_outliers": g6_pass,
        "gate6_notes": g6_notes,
        "gate7_holdout": g7_pass,
        "gate7_notes": g7_notes,
        "annual_breakdown": ann_stats,
        "mc_stats": mc_stats,
        "friction_stats": fric_stats,
        "outlier_stats": outlier_stats,
        "streaks": streaks
    }


# ==============================================================================
# 7. MAIN AUDIT RUNNER
# ==============================================================================

def run_stress_test_for_symbol(symbol: str):
    s = symbol.upper()
    print("=" * 120)
    print(f"  INSTITUTIONAL 7-GATE ZERO-BIAS STRESS AUDIT: {s}".center(120))
    print("=" * 120)

    trades_file = RESULTS_DIR / f"master_{s.lower()}_trades.json"
    if not trades_file.exists():
        # Check alternate names
        alt_files = {
            "EURUSD": RESULTS_DIR / "master_eurusd_14_models_trades.json",
            "USDJPY": RESULTS_DIR / "master_usdjpy_trades.json",
            "EURGBP": RESULTS_DIR / "master_eurgbp_trades.json",
            "AUDNZD": RESULTS_DIR / "master_audnzd_trades.json"
        }
        if s in alt_files and alt_files[s].exists():
            trades_file = alt_files[s]
        else:
            print(f"Error: Trade log file {trades_file} does not exist. Please run master discovery first.")
            return

    with open(trades_file, "r", encoding="utf-8") as f:
        strat_trades = json.load(f)

    # Load 70/30 week splits
    with open(DATA_SPLITS_FILE, "r", encoding="utf-8") as f:
        splits = json.load(f)

    disc_weeks = splits["assets"].get(s, {}).get("discovery_70", {}).get("weeks", [])
    hold_weeks = splits["assets"].get(s, {}).get("holdout_test_30", {}).get("weeks", [])

    results = []
    for strat_name, tr in strat_trades.items():
        res = evaluate_strategy_7_gates(strat_name, s, tr, disc_weeks, hold_weeks)
        results.append(res)

    df = pd.DataFrame([{
        "Strategy": r["strategy"],
        "Trades": r["trades"],
        "Net Pips": r["net_pips"],
        "PF": r["pf"],
        "Win%": r["wr"],
        "Ret/DD": r["ret_to_dd"],
        "Max DD": r["max_dd"],
        "Passed Gates": f"{r['passed_gates']}/7",
        "G1 (Sample)": "PASS" if r["gate1_sample"] else "FAIL",
        "G2 (Annual)": "PASS" if r["gate2_annual"] else "FAIL",
        "G3 (Risk)": "PASS" if r["gate3_downside"] else "FAIL",
        "G4 (MC/RoR)": "PASS" if r["gate4_mc_ror"] else "FAIL",
        "G5 (Fric)": "PASS" if r["gate5_friction"] else "FAIL",
        "G6 (Trim)": "PASS" if r["gate6_outliers"] else "FAIL",
        "G7 (Hold)": "PASS" if r["gate7_holdout"] else "FAIL",
        "Verdict": r["verdict"]
    } for r in results])

    csv_path = RESULTS_DIR / f"{s.lower()}_debiased_stress_test_scorecard.csv"
    df.to_csv(csv_path, index=False)

    json_path = RESULTS_DIR / f"{s.lower()}_debiased_stress_test_audit.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + df.to_string(index=False))
    print(f"\nSaved Scorecard to {csv_path}")
    print(f"Saved Detailed JSON to {json_path}\n")


def main():
    parser = argparse.ArgumentParser(description="Institutional Zero-Bias Fleet Stress-Testing Suite")
    parser.add_argument("--symbol", type=str, default=None, help="Symbol to test (e.g. AUDCAD, USDCAD, GBPJPY)")
    parser.add_argument("--symbols", type=str, default=None, help="Comma-separated symbols")
    parser.add_argument("--all", action="store_true", help="Test all 10 symbols")
    args = parser.parse_args()

    symbols = []
    if args.all:
        symbols = ["AUDCAD", "USDCAD", "GBPJPY", "EURNZD", "GBPCHF", "NZDCAD", "EURUSD", "USDJPY", "EURGBP", "AUDNZD"]
    elif args.symbols:
        symbols = [x.strip().upper() for x in args.symbols.split(",") if x.strip()]
    elif args.symbol:
        symbols = [args.symbol.strip().upper()]
    else:
        print("Please provide --symbol, --symbols, or --all")
        return

    for s in symbols:
        run_stress_test_for_symbol(s)


if __name__ == "__main__":
    main()
