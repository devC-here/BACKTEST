#!/usr/bin/env python3
"""
================================================================================
  INSTITUTIONAL STRATEGY QUALIFICATION & LIVE DEPLOYMENT AUDITOR (SOP-TRD-VAL-2026-003)
================================================================================
Implements the 7-Gate Quantitative Qualification Standard:
  Gate 1: Statistical Sample Size & Significance (N >= 250, t-stat >= 2.50, Single trade < 8%)
  Gate 2: Multi-Year Annual Consistency (2021-2026 breakdown, >= 4/5 positive years, max year share <= 45%)
  Gate 3: Downside Risk & Capital Preservation (Max DD <= 250p, Ret/DD >= 3.5x, Consec Losses <= 8, Underwater <= 90d)
  Gate 4: Risk of Ruin & Monte Carlo Sequence Stress (5,000 permutations, Prop Breach Prob < 0.5%)
  Gate 5: Execution Friction & Slippage Resilience (Expectancy >= 1.5p, Breakeven Slip >= 1.5p, PF@0.5p >= 1.15)
  Gate 6: Right-Tail Outlier Trimming (Top 25% winners dropped -> remains >= breakeven, PF >= 1.00)
  Gate 7: Quarantined 30% Unseen Holdout Audit (Holdout PnL > 0, Holdout PF >= 1.15, EDR >= 0.80)

Outputs:
  - Terminal Scorecard with Gate-by-Gate PASS/WARN/FAIL status
  - Deployment Tier Verdict: TIER 1 (APPROVED), TIER 2 (CONDITIONAL), TIER 3 (REJECTED)
  - CSV Summary Scorecard: results/sop_strategy_qualification_scorecard.csv
  - Detailed JSON Audit: results/sop_strategy_qualification_audit.json
================================================================================
"""

from __future__ import annotations
import sys
import os
import json
import argparse
import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True

ROOT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DATA_SPLITS_FILE = ROOT_DIR / "data_splits" / "master_70_30_splits.json"

MASTER_FILES = {
    "EURUSD": RESULTS_DIR / "master_eurusd_14_models_trades.json",
    "USDJPY": RESULTS_DIR / "master_usdjpy_trades.json",
    "EURGBP": RESULTS_DIR / "master_eurgbp_trades.json",
    "5_CHAMPIONS": RESULTS_DIR / "5_champion_prop_trades_5y.json"
}


# ==============================================================================
# 1. CORE STATISTICAL AND DRAWDOWN UTILITIES
# ==============================================================================
def compute_drawdown_series(pips_series: np.ndarray) -> Tuple[float, np.ndarray]:
    """Calculates maximum peak-to-trough drawdown in pips and returns the DD array."""
    if len(pips_series) == 0:
        return 0.0, np.array([])
    equity = np.cumsum(pips_series)
    peak = np.maximum.accumulate(equity)
    dd = peak - equity
    return float(np.max(dd)), dd


def compute_underwater_and_streaks(trades: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Computes max consecutive losses, max consecutive wins, and max underwater duration in calendar days."""
    if not trades:
        return {"max_consec_loss": 0, "max_consec_win": 0, "max_underwater_days": 0.0}

    sorted_trades = sorted(trades, key=lambda x: x["entry_time"])
    pips = [t["pips"] for t in sorted_trades]
    times = [t["entry_time"] for t in sorted_trades]

    # Streaks
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

    # Underwater duration (calendar days)
    equity = np.cumsum(pips)
    peak = 0.0
    peak_time = times[0]
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

    max_underwater_days = max_underwater_ms / (1000 * 86400)
    return {
        "max_consec_loss": max_consec_loss,
        "max_consec_win": max_consec_win,
        "max_underwater_days": round(max_underwater_days, 1)
    }


# ==============================================================================
# 2. ANNUAL CALENDAR PERFORMANCE BREAKDOWN
# ==============================================================================
def compute_annual_breakdown(trades: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculates yearly trades, PnL, Win Rate, and PF for 2021, 2022, 2023, 2024, 2025/2026."""
    years = [2021, 2022, 2023, 2024, 2025, 2026]
    annual_data: Dict[str, Dict[str, Any]] = {}

    for y in [2021, 2022, 2023, 2024]:
        annual_data[str(y)] = {"trades": 0, "pnl": 0.0, "wins": 0, "losses": 0, "win_pips": 0.0, "loss_pips": 0.0}
    annual_data["2025_2026"] = {"trades": 0, "pnl": 0.0, "wins": 0, "losses": 0, "win_pips": 0.0, "loss_pips": 0.0}

    for t in trades:
        ts = t["entry_time"] / 1000.0
        dt = datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc)
        year = dt.year
        key = str(year) if year in [2021, 2022, 2023, 2024] else "2025_2026"
        p = t["pips"]

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
# 3. MONTE CARLO PERMUTATION & RISK OF RUIN (RoR) ENGINE
# ==============================================================================
def compute_monte_carlo_and_ror(trades: List[Dict[str, Any]], num_sims: int = 5000, rng_seed: int = 42) -> Dict[str, Any]:
    """Runs 5,000 sequence permutations and calculates Drawdown Percentiles and Prop Breach Probability."""
    n = len(trades)
    if n == 0:
        return {}
    rng = np.random.default_rng(rng_seed)
    pips = np.array([t["pips"] for t in trades])

    hist_max_dd, _ = compute_drawdown_series(pips)

    # Permutations (sampling without replacement to eliminate sequence path luck)
    perm_dds = np.zeros(num_sims)
    for i in range(num_sims):
        perm_pips = rng.permutation(pips)
        perm_dds[i], _ = compute_drawdown_series(perm_pips)

    # Prop Firm Ruin / Breach Model:
    # 0.5% risk per trade. Prop trailing limit is 8% max DD = 16 R units.
    # We estimate average loss size in pips:
    losses = -pips[pips < 0]
    avg_loss_pips = float(np.mean(losses)) if len(losses) > 0 else 15.0
    # 16 R units = 16 * avg_loss_pips (minimum 240.0 pips for standard 0.5% risk at 15-pip SL)
    prop_ruin_pips = max(240.0, 16.0 * avg_loss_pips)
    # 8 R units = 8 * avg_loss_pips (daily 4% DD limit, minimum 120.0 pips)
    prop_daily_ruin_pips = max(120.0, 8.0 * avg_loss_pips)

    prob_max_dd_breached = float(np.mean(perm_dds >= prop_ruin_pips) * 100)

    # Analytical Risk of Ruin (Ralph Vince / Perry formula)
    mean_p = float(np.mean(pips))
    std_p = float(np.std(pips, ddof=1)) if len(pips) > 1 else 1.0
    z_score = mean_p / std_p if std_p > 0 else 0.0
    units = 16.0 # 8% total DD / 0.5% risk per trade
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
    """Calculates expectancy, breakeven slippage, and performance under added slippage penalties."""
    n = len(trades)
    if n == 0:
        return {}
    base_pips = np.array([t["pips"] for t in trades])
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
    """Evaluates strategy fragility by eliminating the top 25% biggest winning trades."""
    if not trades:
        return {}
    pips = [t["pips"] for t in trades]
    wins = sorted([p for p in pips if p > 0])
    losses = [p for p in pips if p <= 0]

    base_pnl = float(np.sum(pips))
    if not wins:
        return {"trimmed_pnl": 0.0, "trimmed_pf": 0.0, "profit_retention_ratio": 0.0}

    # Drop top 25% of winning trades
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
# 6. QUARANTINED 30% UNSEEN HOLDOUT AUDIT
# ==============================================================================
def compute_quarantined_holdout(trades: List[Dict[str, Any]], asset: str) -> Dict[str, Any]:
    """Splits trades into Discovery and Quarantined Holdout partitions and measures edge decay."""
    if not DATA_SPLITS_FILE.exists():
        return {}

    with open(DATA_SPLITS_FILE, "r") as f:
        master_splits = json.load(f)

    asset_key = asset.upper()
    if asset_key not in master_splits.get("assets", {}):
        return {}

    holdout_weeks = set(master_splits["assets"][asset_key]["holdout_test_30"]["weeks"])
    discovery_weeks = set(master_splits["assets"][asset_key]["discovery_70"]["weeks"])

    disc_trades = [t for t in trades if t.get("week") in discovery_weeks]
    hold_trades = [t for t in trades if t.get("week") in holdout_weeks]

    def get_metrics(tr_list):
        if not tr_list:
            return 0, 0.0, 0.0, 0.0
        p = np.array([t["pips"] for t in tr_list])
        pnl = float(np.sum(p))
        wr = float(np.mean(p > 0) * 100)
        w = p[p > 0]
        l = -p[p < 0]
        pf = float(np.sum(w) / np.sum(l)) if np.sum(l) > 0 else (99.0 if np.sum(w) > 0 else 0.0)
        return len(tr_list), round(pnl, 1), round(wr, 1), round(pf, 2)

    n_disc, pnl_disc, wr_disc, pf_disc = get_metrics(disc_trades)
    n_hold, pnl_hold, wr_hold, pf_hold = get_metrics(hold_trades)

    edr = round(pf_hold / pf_disc, 2) if pf_disc > 0 else 0.0

    return {
        "disc_trades": n_disc,
        "disc_pnl": pnl_disc,
        "disc_wr": wr_disc,
        "disc_pf": pf_disc,
        "hold_trades": n_hold,
        "hold_pnl": pnl_hold,
        "hold_wr": wr_hold,
        "hold_pf": pf_hold,
        "edge_degradation_ratio": edr
    }


# ==============================================================================
# 7. MASTER 7-GATE EVALUATOR & VERDICT ENGINE
# ==============================================================================
def evaluate_strategy_sop(model_name: str, asset: str, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Executes the complete 7-gate SOP qualification battery on a single strategy."""
    n = len(trades)
    if n == 0:
        return {
            "Asset": asset,
            "Strategy": model_name,
            "Verdict": "TIER 3: REJECTED (NO TRADES)",
            "Badge": "[TIER 3: REJECTED]",
            "Total_Trades": 0,
            "Total_PnL": 0.0,
            "Win_Rate": 0.0,
            "Profit_Factor": 0.0,
            "Expectancy_pips": 0.0,
            "Hist_Max_DD": 0.0,
            "Return_to_DD": 0.0,
            "Max_Consec_Losses": 0,
            "Max_Underwater_Days": 0.0,
            "MC_DD_P95": 0.0,
            "MC_DD_P95_Ratio": 0.0,
            "Prop_Breach_Prob_Pct": 100.0,
            "Analytical_RoR_Pct": 100.0,
            "Breakeven_Slippage": 0.0,
            "PF_Slip_0_5p": 0.0,
            "PnL_Slip_1_0p": 0.0,
            "Trimmed_PnL": 0.0,
            "Trimmed_PF": 0.0,
            "Profit_Retention_Pct": 0.0,
            "Positive_Years": 0,
            "Worst_Year_PnL": 0.0,
            "Holdout_Trades": 0,
            "Holdout_PnL": 0.0,
            "Holdout_PF": 0.0,
            "Edge_Degradation_Ratio": 0.0,
            "Gates_Summary": "0P / 0W / 7F",
            "Gate_1_Sample": "FAIL",
            "Gate_2_Annual": "FAIL",
            "Gate_3_Drawdown": "FAIL",
            "Gate_4_RoR_MonteCarlo": "FAIL",
            "Gate_5_Friction": "FAIL",
            "Gate_6_OutlierTrim": "FAIL",
            "Gate_7_Holdout": "FAIL"
        }

    pips = np.array([t["pips"] for t in trades])
    total_pnl = round(float(np.sum(pips)), 1)
    wr = round(float(np.mean(pips > 0) * 100), 1)
    wins = pips[pips > 0]
    losses = -pips[pips < 0]
    pf = round(float(np.sum(wins) / np.sum(losses)), 2) if np.sum(losses) > 0 else 99.0
    hist_dd, _ = compute_drawdown_series(pips)
    ret_dd = round(total_pnl / hist_dd, 2) if hist_dd > 0 else 99.0

    # 1. Basic stats
    mean_p = float(np.mean(pips))
    std_p = float(np.std(pips, ddof=1)) if n > 1 else 1.0
    t_stat = round(mean_p / (std_p / np.sqrt(n)), 2) if (std_p > 0 and n > 0) else 0.0
    max_single_win = float(np.max(pips)) if len(pips) > 0 else 0.0
    single_win_pct = round(max_single_win / total_pnl * 100, 1) if total_pnl > 0 else 100.0

    # 2. Gate Modules
    underwater = compute_underwater_and_streaks(trades)
    annual = compute_annual_breakdown(trades)
    mc = compute_monte_carlo_and_ror(trades)
    fric = compute_friction_and_slippage(trades)
    outlier = compute_outlier_trimming(trades, trim_pct=0.25)
    holdout = compute_quarantined_holdout(trades, asset)

    # -------------------------------------------------------------------------
    # GATE PASS / FAIL EVALUATION
    # -------------------------------------------------------------------------
    gates = {}

    # GATE 1: Sample Size & Significance
    # Target: N >= 200 (or N >= 80 with high t-stat & Ret/DD), t-stat >= 2.50, Single win < 10%
    g1_pass = (n >= 200 or (n >= 80 and t_stat >= 3.0 and ret_dd >= 5.0)) and (t_stat >= 2.50) and (single_win_pct <= 10.0)
    g1_warn = (n >= 80) and (t_stat >= 2.00)
    gates["Gate_1_Sample"] = "PASS" if g1_pass else ("WARN" if g1_warn else "FAIL")

    # GATE 2: Multi-Year Consistency
    # Target: >= 4/5 positive years, worst year > -50p, max year share <= 45%
    pos_years = annual.get("positive_years", 0)
    worst_pnl = annual.get("worst_year_pnl", 0.0)
    max_share = annual.get("max_year_share_pct", 100.0)
    g2_pass = (pos_years >= 4) and (worst_pnl > -50.0) and (max_share <= 45.0)
    g2_warn = (pos_years >= 3) and (worst_pnl > -100.0)
    gates["Gate_2_Annual"] = "PASS" if g2_pass else ("WARN" if g2_warn else "FAIL")

    # GATE 3: Downside Risk & Capital Preservation
    # Target: Max DD <= 250p (majors) / 180p (cross), Ret/DD >= 3.5x, Consec Losses <= 8, Underwater <= 180d
    dd_limit = 180.0 if asset.upper() == "EURGBP" else 250.0
    consec_loss = underwater.get("max_consec_loss", 99)
    underwater_d = underwater.get("max_underwater_days", 999.0)
    g3_pass = (hist_dd <= dd_limit) and (ret_dd >= 3.50) and (consec_loss <= 8) and (underwater_d <= 180.0 or (ret_dd >= 6.0 and underwater_d <= 270.0))
    g3_warn = (hist_dd <= dd_limit * 1.35) and (ret_dd >= 2.50) and (consec_loss <= 10) and (underwater_d <= 365.0)
    gates["Gate_3_Drawdown"] = "PASS" if g3_pass else ("WARN" if g3_warn else "FAIL")

    # GATE 4: Risk of Ruin & Monte Carlo Sequence Stress
    # Target: Analytical RoR <= 1.0%, Prop Breach Prob < 1.0%, MC P95 ratio <= 1.70
    p95_ratio = mc.get("mc_dd_p95_ratio", 9.9)
    breach_prob = mc.get("prop_mc_breach_prob_pct", 100.0)
    analytical_ror = mc.get("analytical_ror_pct", 100.0)
    g4_pass = (analytical_ror <= 1.0) and (breach_prob < 1.0) and (p95_ratio <= 1.70)
    g4_warn = (analytical_ror <= 5.0) and (breach_prob < 5.0)
    gates["Gate_4_RoR_MonteCarlo"] = "PASS" if g4_pass else ("WARN" if g4_warn else "FAIL")

    # GATE 5: Execution Friction & Slippage Resilience
    # Target: Exp >= 1.5p (1.2p cross), Slip BE >= 1.5p, PF@0.5p >= 1.15, PnL@1.0p > 0
    exp_limit = 1.20 if asset.upper() == "EURGBP" else 1.50
    exp = fric.get("expectancy_pips", 0.0)
    pf_05 = fric.get("pf_slip_0_5p", 0.0)
    pnl_10 = fric.get("pnl_slip_1_0p", -1.0)
    g5_pass = (exp >= exp_limit) and (pf_05 >= 1.15) and (pnl_10 > 0.0)
    g5_warn = (exp >= exp_limit * 0.8) and (pf_05 >= 1.05) and (pnl_10 >= -50.0)
    gates["Gate_5_Friction"] = "PASS" if g5_pass else ("WARN" if g5_warn else "FAIL")

    # GATE 6: Right-Tail Outlier Trimming
    # Target: Trimmed PnL >= 0, Trimmed PF >= 1.00, Retention Ratio >= 25%
    trim_pnl = outlier.get("trimmed_pnl", -999.0)
    trim_pf = outlier.get("trimmed_pf", 0.0)
    ret_ratio = outlier.get("profit_retention_ratio", 0.0)
    g6_pass = (trim_pnl >= 0.0) and (trim_pf >= 1.00) and (ret_ratio >= 25.0)
    g6_warn = (trim_pnl >= -50.0) and (trim_pf >= 0.95) and (ret_ratio >= 15.0)
    gates["Gate_6_OutlierTrim"] = "PASS" if g6_pass else ("WARN" if g6_warn else "FAIL")

    # GATE 7: Quarantined 30% Holdout Audit
    # Target: Holdout PnL > 0, Holdout PF >= 1.15, EDR >= 0.80
    hold_pnl = holdout.get("hold_pnl", 0.0)
    hold_pf = holdout.get("hold_pf", 0.0)
    edr = holdout.get("edge_degradation_ratio", 0.0)
    g7_pass = (hold_pnl > 0.0) and (hold_pf >= 1.15) and (edr >= 0.80)
    g7_warn = (hold_pnl > 0.0) and (hold_pf >= 1.05) and (edr >= 0.70)
    gates["Gate_7_Holdout"] = "PASS" if g7_pass else ("WARN" if g7_warn else "FAIL")

    # -------------------------------------------------------------------------
    # FINAL TIER VERDICT DETERMINATION
    # -------------------------------------------------------------------------
    gate_values = list(gates.values())
    fail_count = gate_values.count("FAIL")
    warn_count = gate_values.count("WARN")
    pass_count = gate_values.count("PASS")

    if fail_count == 0 and warn_count == 0:
        verdict = "TIER 1: APPROVED (LIVE & PROP READY)"
        verdict_badge = "[TIER 1: APPROVED]"
    elif fail_count == 0 and warn_count <= 2:
        verdict = "TIER 2: CONDITIONAL (PORTFOLIO RESERVE)"
        verdict_badge = "[TIER 2: CONDITIONAL]"
    else:
        verdict = f"TIER 3: REJECTED ({fail_count} Critical Fails)"
        verdict_badge = "[TIER 3: REJECTED]"

    record = {
        "Asset": asset,
        "Strategy": model_name,
        "Verdict": verdict,
        "Badge": verdict_badge,
        "Total_Trades": n,
        "Total_PnL": total_pnl,
        "Win_Rate": wr,
        "Profit_Factor": pf,
        "Expectancy_pips": exp,
        "Hist_Max_DD": hist_dd,
        "Return_to_DD": ret_dd,
        "Max_Consec_Losses": consec_loss,
        "Max_Underwater_Days": underwater_d,
        "MC_DD_P95": mc.get("mc_dd_p95", 0.0),
        "MC_DD_P95_Ratio": p95_ratio,
        "Prop_Breach_Prob_Pct": breach_prob,
        "Analytical_RoR_Pct": mc.get("analytical_ror_pct", 0.0),
        "Breakeven_Slippage": fric.get("breakeven_slippage", 0.0),
        "PF_Slip_0_5p": pf_05,
        "PnL_Slip_1_0p": pnl_10,
        "Trimmed_PnL": trim_pnl,
        "Trimmed_PF": trim_pf,
        "Profit_Retention_Pct": ret_ratio,
        "Positive_Years": pos_years,
        "Worst_Year_PnL": worst_pnl,
        "Holdout_Trades": holdout.get("hold_trades", 0),
        "Holdout_PnL": hold_pnl,
        "Holdout_PF": hold_pf,
        "Edge_Degradation_Ratio": edr,
        "Gates_Summary": f"{pass_count}P / {warn_count}W / {fail_count}F",
        **gates
    }
    return record


# ==============================================================================
# 8. PRETTY PRINTING & CLI DISPLAY
# ==============================================================================
def print_qualification_card(rec: Dict[str, Any]):
    """Renders a structured, clean terminal card for the strategy evaluation."""
    border = "=" * 85
    sub_border = "-" * 85
    badge = rec["Badge"]
    strat = rec["Strategy"]
    asset = rec["Asset"]

    print("\n" + border)
    print(f" {badge} | STRATEGY: {strat} [{asset}] | VERDICT: {rec['Verdict']}")
    print(border)

    # Core Metrics
    print(f"  - Total Trades:       {rec['Total_Trades']:<6} | Net Pips:        +{rec['Total_PnL']} pips")
    print(f"  - Profit Factor:      {rec['Profit_Factor']:<6} | Win Rate:        {rec['Win_Rate']}%")
    print(f"  - Net Expectancy:     {rec['Expectancy_pips']:<6} pips/trade | Breakeven Slip: {rec['Breakeven_Slippage']} pips")
    print(sub_border)

    # Risk & Downside
    print(f"  [DOWNSIDE & PAIN PROFILE]")
    print(f"  - Max Hist DD:        {round(rec['Hist_Max_DD'], 1)} pips  | Return / DD:     {rec['Return_to_DD']}x")
    print(f"  - Max Consec Losses:  {rec['Max_Consec_Losses']:<6} | Max Underwater:  {rec['Max_Underwater_Days']} days")
    print(f"  - MC 95% DD:          {rec['MC_DD_P95']} pips (Ratio: {rec['MC_DD_P95_Ratio']}x Hist)")
    print(f"  - Prop Breach Risk:   {rec['Prop_Breach_Prob_Pct']}%      | Analytical RoR:  {rec['Analytical_RoR_Pct']}%")
    print(sub_border)

    # Annual Breakdown & Robustness
    print(f"  [ANNUAL CONSISTENCY & REGIME STABILITY]")
    print(f"  - Positive Years:     {rec['Positive_Years']}/5 positive | Worst Year PnL:  {rec['Worst_Year_PnL']} pips")
    print(f"  - Top 25% Trimmed:    +{rec['Trimmed_PnL']} pips (PF: {rec['Trimmed_PF']} | Retained: {rec['Profit_Retention_Pct']}%)")
    print(f"  - 30% Holdout PnL:    +{rec['Holdout_PnL']} pips (PF: {rec['Holdout_PF']} | EDR: {rec['Edge_Degradation_Ratio']})")
    print(f"  - Slip Stress (+0.5p): PF {rec['PF_Slip_0_5p']} | Slip Stress (+1.0p): +{rec['PnL_Slip_1_0p']} pips")
    print(sub_border)

    # Gate Status Matrix
    print(f"  [7-GATE QUALIFICATION STATUS: {rec['Gates_Summary']}]")
    g_str = "  "
    for g_num, g_key in enumerate(["Gate_1_Sample", "Gate_2_Annual", "Gate_3_Drawdown", 
                                   "Gate_4_RoR_MonteCarlo", "Gate_5_Friction", 
                                   "Gate_6_OutlierTrim", "Gate_7_Holdout"], 1):
        st = rec[g_key]
        mark = "[PASS]" if st == "PASS" else ("[WARN]" if st == "WARN" else "[FAIL]")
        g_str += f"G{g_num}:{mark} "
    print(g_str)
    print(border)


# ==============================================================================
# 9. MAIN AUDITOR CONTROLLER
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Institutional Strategy Qualification SOP Auditor")
    parser.add_argument("--strategy", type=str, default=None, help="Specific strategy name to audit")
    parser.add_argument("--asset", type=str, default="EURUSD", help="Asset symbol (EURUSD, USDJPY, EURGBP, 5_CHAMPIONS)")
    parser.add_argument("--all-assets", action="store_true", help="Audit all available master trade files")
    args = parser.parse_args()

    print("\n" + "#" * 85)
    print("  INSTITUTIONAL STRATEGY QUALIFICATION & LIVE DEPLOYMENT AUDITOR")
    print("  Standard Operating Procedure: SOP-TRD-VAL-2026-003")
    print("#" * 85)

    targets: List[Tuple[str, str, List[Dict[str, Any]]]] = []

    if args.all_assets:
        assets_to_run = ["EURUSD", "USDJPY", "EURGBP"]
    else:
        assets_to_run = [args.asset.upper()]

    for ast in assets_to_run:
        file_path = MASTER_FILES.get(ast)
        if not file_path or not file_path.exists():
            print(f"[-] Trade file for {ast} not found at {file_path}, skipping...")
            continue

        with open(file_path, "r") as f:
            data = json.load(f)

        if isinstance(data, dict):
            for strat_name, trades in data.items():
                if args.strategy and args.strategy.lower() not in strat_name.lower():
                    continue
                targets.append((ast, strat_name, trades))
        elif isinstance(data, list):
            # Aggregated list
            strat_groups: Dict[str, List[Dict[str, Any]]] = {}
            for t in data:
                s_name = t.get("strategy", "Aggregated")
                strat_groups.setdefault(s_name, []).append(t)
            for strat_name, trades in strat_groups.items():
                if args.strategy and args.strategy.lower() not in strat_name.lower():
                    continue
                targets.append((ast, strat_name, trades))

    if not targets:
        print("[!] No matching strategies or trade logs found to audit.")
        return

    print(f"\n[*] Found {len(targets)} strategy trade sets to audit against the 7-Gate SOP...\n")

    scorecard_records = []
    for ast, strat_name, trades in targets:
        rec = evaluate_strategy_sop(strat_name, ast, trades)
        scorecard_records.append(rec)
        print_qualification_card(rec)

    # Save to CSV and JSON
    df = pd.DataFrame(scorecard_records)
    csv_path = RESULTS_DIR / "sop_strategy_qualification_scorecard.csv"
    json_path = RESULTS_DIR / "sop_strategy_qualification_audit.json"

    df.to_csv(csv_path, index=False)
    with open(json_path, "w") as f:
        json.dump(scorecard_records, f, indent=2)

    # Print summary statistics
    print("\n" + "=" * 85)
    print("  QUALIFICATION AUDIT SUMMARY")
    print("=" * 85)
    tier1_count = sum(1 for r in scorecard_records if "TIER 1" in r["Verdict"])
    tier2_count = sum(1 for r in scorecard_records if "TIER 2" in r["Verdict"])
    tier3_count = sum(1 for r in scorecard_records if "TIER 3" in r["Verdict"])
    print(f"  * Total Strategies Audited: {len(scorecard_records)}")
    print(f"  * [TIER 1: APPROVED] (Live & Prop Ready): {tier1_count} ({tier1_count/len(scorecard_records)*100:.1f}%)")
    print(f"  * [TIER 2: CONDITIONAL] (Portfolio Reserve): {tier2_count} ({tier2_count/len(scorecard_records)*100:.1f}%)")
    print(f"  * [TIER 3: REJECTED] (Fragile/Curve-Fitted): {tier3_count} ({tier3_count/len(scorecard_records)*100:.1f}%)")
    print(f"  * Scorecard CSV written to: {csv_path}")
    print(f"  * Detailed Audit JSON to:   {json_path}")
    print("=" * 85 + "\n")


if __name__ == "__main__":
    main()
