"""
Comprehensive Institutional Stress-Testing Battery for EURUSD and EURGBP Fleets
================================================================================
Tests:
1. Friction & Slippage Sensitivity Matrix (Breakeven Slippage Threshold)
2. Monte Carlo Sequence Reshuffling & Worst-Case Drawdown Risk (2,500 paths)
3. Random Execution Dropout (10%, 25%, 40% missed trades, 1,000 runs)
4. Pain Index: Max Consecutive Losses, Max Underwater Duration, Recovery Factor
5. Cross-Pair Multi-Strategy Portfolio Synergy (Correlation & Drawdown Reduction)
6. Regime Breakdown (2021-2022 vs 2023-2024 vs 2025-2026)
"""

import json
import datetime
import numpy as np
import pandas as pd
from pathlib import Path

RESULTS_DIR = Path("results")
MASTER_EURUSD_FILE = RESULTS_DIR / "master_eurusd_14_models_trades.json"
MASTER_EURGBP_FILE = RESULTS_DIR / "master_eurgbp_6_models_trades.json"

def compute_drawdown_series(pips_series):
    equity = np.cumsum(pips_series)
    peak = np.maximum.accumulate(equity)
    dd = peak - equity
    return max(dd) if len(dd) > 0 else 0.0

def compute_underwater_stats(trades):
    if not trades:
        return 0, 0, 0, 0.0
    sorted_trades = sorted(trades, key=lambda x: x["entry_time"])
    pips = [t["pips"] for t in sorted_trades]
    times = [t["entry_time"] for t in sorted_trades]
    
    # Max consecutive losses and wins
    max_consec_loss = 0
    cur_loss = 0
    max_consec_win = 0
    cur_win = 0
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
            # new peak
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
    return max_consec_loss, max_consec_win, round(max_underwater_days, 1)

def run_friction_stress(trades):
    n = len(trades)
    if n == 0:
        return {}
    base_pips = np.array([t["pips"] for t in trades])
    base_pnl = np.sum(base_pips)
    expectancy = base_pnl / n # pips per trade
    
    # Test slippage penalties: 0.3, 0.6, 1.0, 1.5, 2.0 pips
    penalties = [0.0, 0.3, 0.6, 1.0, 1.5, 2.0]
    res = {"trades": n, "base_pnl": round(base_pnl, 1), "expectancy": round(expectancy, 2), "breakeven_slip": round(expectancy, 2)}
    
    for pen in penalties:
        adj_pips = base_pips - pen
        adj_pnl = np.sum(adj_pips)
        wins = adj_pips[adj_pips > 0]
        losses = -adj_pips[adj_pips < 0]
        sum_w = np.sum(wins)
        sum_l = np.sum(losses)
        pf = round(sum_w / sum_l, 2) if sum_l > 0 else (99.0 if sum_w > 0 else 0.0)
        res[f"pnl_slip_{pen}p"] = round(adj_pnl, 1)
        res[f"pf_slip_{pen}p"] = pf
        
    return res

def run_monte_carlo_drawdown(trades, num_sims=2500, rng_seed=42):
    n = len(trades)
    if n == 0:
        return {}
    rng = np.random.default_rng(rng_seed)
    pips = np.array([t["pips"] for t in trades])
    
    # Historical DD
    hist_dd = compute_drawdown_series(pips)
    
    # Permutation (sequence reshuffling without replacement)
    perm_dds = np.zeros(num_sims)
    for i in range(num_sims):
        perm_pips = rng.permutation(pips)
        perm_dds[i] = compute_drawdown_series(perm_pips)
        
    return {
        "hist_max_dd": round(hist_dd, 1),
        "mc_dd_p50": round(float(np.percentile(perm_dds, 50)), 1),
        "mc_dd_p95": round(float(np.percentile(perm_dds, 95)), 1),
        "mc_dd_p99": round(float(np.percentile(perm_dds, 99)), 1),
        "prob_dd_gt_100p": round(float(np.mean(perm_dds > 100.0) * 100), 1),
        "prob_dd_gt_200p": round(float(np.mean(perm_dds > 200.0) * 100), 1),
        "prob_dd_gt_300p": round(float(np.mean(perm_dds > 300.0) * 100), 1),
    }

def run_dropout_stress(trades, drop_rates=[0.10, 0.25, 0.40], num_sims=1000, rng_seed=42):
    n = len(trades)
    if n == 0:
        return {}
    rng = np.random.default_rng(rng_seed)
    pips = np.array([t["pips"] for t in trades])
    
    res = {}
    for rate in drop_rates:
        keep_prob = 1.0 - rate
        sim_pnls = np.zeros(num_sims)
        for i in range(num_sims):
            mask = rng.random(n) < keep_prob
            sim_pnls[i] = np.sum(pips[mask])
            
        res[f"drop_{int(rate*100)}_pop"] = round(float(np.mean(sim_pnls > 0.0) * 100), 1)
        res[f"drop_{int(rate*100)}_p05"] = round(float(np.percentile(sim_pnls, 5)), 1)
        res[f"drop_{int(rate*100)}_p50"] = round(float(np.percentile(sim_pnls, 50)), 1)
    return res

def run_regime_breakdown(trades):
    if not trades:
        return {}
    regimes = {
        "2021_2022": (datetime.datetime(2021, 1, 1, tzinfo=datetime.timezone.utc).timestamp() * 1000,
                      datetime.datetime(2022, 12, 31, 23, 59, 59, tzinfo=datetime.timezone.utc).timestamp() * 1000),
        "2023_2024": (datetime.datetime(2023, 1, 1, tzinfo=datetime.timezone.utc).timestamp() * 1000,
                      datetime.datetime(2024, 12, 31, 23, 59, 59, tzinfo=datetime.timezone.utc).timestamp() * 1000),
        "2025_2026": (datetime.datetime(2025, 1, 1, tzinfo=datetime.timezone.utc).timestamp() * 1000,
                      datetime.datetime(2026, 12, 31, 23, 59, 59, tzinfo=datetime.timezone.utc).timestamp() * 1000),
    }
    res = {}
    for reg_name, (t_start, t_end) in regimes.items():
        reg_trades = [t for t in trades if t_start <= t["entry_time"] <= t_end]
        n = len(reg_trades)
        if n > 0:
            reg_pips = np.array([t["pips"] for t in reg_trades])
            pnl = round(float(np.sum(reg_pips)), 1)
            wr = round(float(np.mean(reg_pips > 0) * 100), 1)
            res[f"{reg_name}_trades"] = n
            res[f"{reg_name}_pnl"] = pnl
            res[f"{reg_name}_wr"] = wr
        else:
            res[f"{reg_name}_trades"] = 0
            res[f"{reg_name}_pnl"] = 0.0
            res[f"{reg_name}_wr"] = 0.0
    return res

def main():
    print("=" * 80)
    print("STARTING INSTITUTIONAL STRESS-TESTING BATTERY")
    print("=" * 80)
    
    with open(MASTER_EURUSD_FILE, "r") as f:
        eurusd_trades_dict = json.load(f)
    with open(MASTER_EURGBP_FILE, "r") as f:
        eurgbp_trades_dict = json.load(f)
        
    all_models = []
    for model_name, trades in eurusd_trades_dict.items():
        all_models.append(("EURUSD", model_name, trades))
    for model_name, trades in eurgbp_trades_dict.items():
        all_models.append(("EURGBP", model_name, trades))
        
    summary_rows = []
    
    for asset, model, trades in all_models:
        print(f"Processing Stress Tests for {asset} - {model} ({len(trades)} trades)...")
        # 1. Friction & Slippage
        fric = run_friction_stress(trades)
        # 2. Monte Carlo DD
        mc_dd = run_monte_carlo_drawdown(trades)
        # 3. Dropout
        dropout = run_dropout_stress(trades)
        # 4. Underwater & Losing streaks
        consec_loss, consec_win, underwater_days = compute_underwater_stats(trades)
        # 5. Regimes
        reg = run_regime_breakdown(trades)
        
        row = {
            "Asset": asset,
            "Strategy": model,
            "Total_Trades": fric.get("trades", 0),
            "Base_PnL": fric.get("base_pnl", 0.0),
            "Expectancy_pips": fric.get("expectancy", 0.0),
            "Breakeven_Slippage": fric.get("breakeven_slip", 0.0),
            "PnL_Slip_0.6p": fric.get("pnl_slip_0.6p", 0.0),
            "PF_Slip_0.6p": fric.get("pf_slip_0.6p", 0.0),
            "PnL_Slip_1.0p": fric.get("pnl_slip_1.0p", 0.0),
            "PF_Slip_1.0p": fric.get("pf_slip_1.0p", 0.0),
            "PnL_Slip_1.5p": fric.get("pnl_slip_1.5p", 0.0),
            "Hist_Max_DD": mc_dd.get("hist_max_dd", 0.0),
            "MC_DD_P50": mc_dd.get("mc_dd_p50", 0.0),
            "MC_DD_P95": mc_dd.get("mc_dd_p95", 0.0),
            "MC_DD_P99": mc_dd.get("mc_dd_p99", 0.0),
            "Prob_DD_gt_200p": mc_dd.get("prob_dd_gt_200p", 0.0),
            "Drop_10_PoP": dropout.get("drop_10_pop", 0.0),
            "Drop_25_PoP": dropout.get("drop_25_pop", 0.0),
            "Drop_40_PoP": dropout.get("drop_40_pop", 0.0),
            "Drop_25_P05": dropout.get("drop_25_p05", 0.0),
            "Max_Consec_Loss": consec_loss,
            "Max_Consec_Win": consec_win,
            "Max_Underwater_Days": underwater_days,
            "Regime_2021_2022_PnL": reg.get("2021_2022_pnl", 0.0),
            "Regime_2023_2024_PnL": reg.get("2023_2024_pnl", 0.0),
            "Regime_2025_2026_PnL": reg.get("2025_2026_pnl", 0.0),
        }
        summary_rows.append(row)
        
    df_results = pd.DataFrame(summary_rows)
    out_csv = RESULTS_DIR / "institutional_stress_test_battery_results.csv"
    df_results.to_csv(out_csv, index=False)
    print(f"\n[DONE] Saved stress test summary to: {out_csv}")
    
    # -------------------------------------------------------------
    # Cross-Pair Portfolio Synergy Test
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("RUNNING CROSS-PAIR MULTI-STRATEGY PORTFOLIO SYNERGY TEST")
    print("=" * 80)
    
    # Pick Top Champions
    eurusd_champs = ["DISC42_H4_M30", "DISC45_M5", "DISC40_RSI", "DISC43_H1_M5", "S66_44MA"]
    eurgbp_champs = ["DISC46_Daily_Pivot", "DISC48_Camarilla", "DISC42_Macro_Structure"]
    
    # Build a combined chronological trade list
    portfolio_trades = []
    for m in eurusd_champs:
        for t in eurusd_trades_dict.get(m, []):
            portfolio_trades.append({**t, "fleet": "EURUSD"})
    for m in eurgbp_champs:
        for t in eurgbp_trades_dict.get(m, []):
            portfolio_trades.append({**t, "fleet": "EURGBP"})
            
    portfolio_trades.sort(key=lambda x: x["entry_time"])
    
    port_pips = np.array([t["pips"] for t in portfolio_trades])
    port_total_pnl = round(float(np.sum(port_pips)), 1)
    port_max_dd = round(float(compute_drawdown_series(port_pips)), 1)
    
    # Sum of individual standalone max drawdowns
    indiv_dds = []
    for m in eurusd_champs:
        p = np.array([t["pips"] for t in eurusd_trades_dict[m]])
        indiv_dds.append(compute_drawdown_series(p))
    for m in eurgbp_champs:
        p = np.array([t["pips"] for t in eurgbp_trades_dict[m]])
        indiv_dds.append(compute_drawdown_series(p))
    sum_indiv_dd = round(float(sum(indiv_dds)), 1)
    
    dd_reduction_pct = round((1.0 - (port_max_dd / sum_indiv_dd)) * 100, 1)
    
    # Monte carlo on the combined portfolio
    port_mc = run_monte_carlo_drawdown(portfolio_trades, num_sims=2500)
    
    # Weekly returns correlation between fleets
    # Group trades by ISO calendar week (YYYY-WW)
    def assign_iso_week(ts_ms):
        dt = datetime.datetime.fromtimestamp(ts_ms / 1000, tz=datetime.timezone.utc)
        return f"{dt.isocalendar()[0]}-W{dt.isocalendar()[1]:02d}"
        
    fleet_eurusd_weekly = {}
    fleet_eurgbp_weekly = {}
    for t in portfolio_trades:
        w = assign_iso_week(t["entry_time"])
        if t["fleet"] == "EURUSD":
            fleet_eurusd_weekly[w] = fleet_eurusd_weekly.get(w, 0.0) + t["pips"]
        else:
            fleet_eurgbp_weekly[w] = fleet_eurgbp_weekly.get(w, 0.0) + t["pips"]
            
    all_iso_weeks = sorted(list(set(list(fleet_eurusd_weekly.keys()) + list(fleet_eurgbp_weekly.keys()))))
    series_eurusd = [fleet_eurusd_weekly.get(w, 0.0) for w in all_iso_weeks]
    series_eurgbp = [fleet_eurgbp_weekly.get(w, 0.0) for w in all_iso_weeks]
    
    corr = np.corrcoef(series_eurusd, series_eurgbp)[0, 1]
    
    portfolio_summary = {
        "portfolio_total_trades": len(portfolio_trades),
        "portfolio_total_pnl": port_total_pnl,
        "portfolio_max_dd": port_max_dd,
        "sum_individual_max_dds": sum_indiv_dd,
        "drawdown_diversification_benefit_pct": dd_reduction_pct,
        "weekly_return_cross_pair_correlation": round(float(corr), 3),
        "portfolio_mc_dd_p50": port_mc["mc_dd_p50"],
        "portfolio_mc_dd_p95": port_mc["mc_dd_p95"],
        "portfolio_mc_dd_p99": port_mc["mc_dd_p99"],
    }
    
    port_out = RESULTS_DIR / "combined_champions_portfolio_stress_test.json"
    with open(port_out, "w") as f:
        json.dump(portfolio_summary, f, indent=2)
        
    print(f"Portfolio Summary:\n{json.dumps(portfolio_summary, indent=2)}")
    print(f"[DONE] Portfolio summary saved to: {port_out}")

if __name__ == "__main__":
    main()
