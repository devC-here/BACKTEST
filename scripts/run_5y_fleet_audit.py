#!/usr/bin/env python3
"""
High-Speed Parallel 5-Year Production Fleet Tick Audit (262 Weeks / 123.22M Ticks).
Evaluates all 7 Champion Models adhering strictly to SOP-TRD-VAL-5Y-001:
- 3-Year In-Sample Training Split (Weeks 1 to 156)
- 2-Year Out-of-Sample Testing Split (Weeks 157 to 262)
- Consolidated Full 5-Year Multi-Model Fleet Scorecard
Uses multi-core parallel processing across available CPU cores.
"""

from __future__ import annotations
import sys
import os
import time
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd

# Prevent bytecode generation
sys.dont_write_bytecode = True

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tick_engine.core.engine import BacktestEngine, TRADE_DTYPE
from tick_engine.core.dataloader import get_week_dataset, load_manifest
from tick_engine.core.metrics import compute_metrics, compute_annual_metrics

# 7 Elite Production Champion Strategies
from strategies.high_frequency_m15_scalper_elite.strategy import HighFrequencyM15ScalperElite
from strategies.institutional_v12_elite.strategy import InstitutionalPrimeSweepV12
from strategies.s01_m15_displacement_retest_elite.strategy import S01M15DisplacementRetestElite
from strategies.s10_dual_ema_pullback_elite.strategy import S10DualEMAPullbackElite
from strategies.s14_inside_bar_m30_elite.strategy import S14InsideBarM30Elite
from strategies.s34_heikin_ashi_v2_elite.strategy import S34HeikinAshiElite
from strategies.s66_bhanushali_44ma_elite.strategy import S66Bhanushali44MAElite

STRATEGY_CONFIGS = [
    {"id": "HF_M15", "name": "HighFrequency M15 Scalper Elite", "cls": HighFrequencyM15ScalperElite},
    {"id": "V12_PRIME", "name": "Institutional Prime Sweep V12", "cls": InstitutionalPrimeSweepV12},
    {"id": "S01_M15", "name": "S01 M15 Displacement Retest Elite", "cls": S01M15DisplacementRetestElite},
    {"id": "S10_EMA", "name": "S10 Dual EMA Pullback Elite", "cls": S10DualEMAPullbackElite},
    {"id": "S14_IB", "name": "S14 Inside Bar M30 Elite", "cls": S14InsideBarM30Elite},
    {"id": "S34_HA", "name": "S34 Heikin Ashi Momentum Elite", "cls": S34HeikinAshiElite},
    {"id": "S66_44MA", "name": "S66 Bhanushali 44 MA Elite", "cls": S66Bhanushali44MAElite},
]

RAW_DIR = ROOT_DIR / "eurusd_weekly_data"
PROC_DIR = ROOT_DIR / "tick_engine" / "data" / "processed"
RESULTS_DIR = ROOT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_END_WEEK = 156  # 3-Year boundary (Weeks 1 to 156)


def evaluate_week_all_models(week_num: int) -> Dict[str, Any]:
    """
    Worker function executed across CPU processes.
    Loads ticks for week_num and evaluates all 7 champion models.
    """
    engine = BacktestEngine(pip_size=0.0001, use_slippage=True)
    ticks, meta = get_week_dataset(week_num, raw_dir=RAW_DIR, processed_dir=PROC_DIR, auto_convert=False)
    n_ticks = len(ticks)

    week_result = {
        "week": week_num,
        "start_date": meta.get("start_date", ""),
        "end_date": meta.get("end_date", ""),
        "ticks": n_ticks,
        "split": "train" if week_num <= TRAIN_END_WEEK else "test",
        "models": {}
    }

    for cfg in STRATEGY_CONFIGS:
        sid = cfg["id"]
        strat = cfg["cls"]()
        trades = engine.run(ticks, strat)
        m = compute_metrics(trades)

        week_result["models"][sid] = {
            "trades": trades,
            "metrics": m,
            "net_pips": m["net_pips"],
            "total_trades": m["total_trades"],
            "win_rate": m["win_rate_pct"],
            "profit_factor": m["profit_factor"],
            "max_dd": m["max_drawdown_pips"]
        }

    return week_result


def run_5y_fleet_audit():
    print("=" * 115)
    print("       HIGH-SPEED 5-YEAR PRODUCTION FLEET TICK AUDIT (SOP: 3Y TRAIN / 2Y TEST)       ".center(115))
    print("=" * 115)

    manifest = load_manifest(RAW_DIR)
    total_weeks = len(manifest)
    num_workers = min(8, os.cpu_count() or 4)

    print(f"[*] Total Weeks in Catalog: {total_weeks} (Week 1 to Week {total_weeks})")
    print(f"[*] In-Sample Training Split:   Weeks 1 to {TRAIN_END_WEEK} (3 Years: Oct 2021 - Sep 2024)")
    print(f"[*] Out-of-Sample Test Split:  Weeks {TRAIN_END_WEEK + 1} to {total_weeks} (2 Years: Sep 2024 - Oct 2026)")
    print(f"[*] Auditing {len(STRATEGY_CONFIGS)} Production Champion Models using {num_workers} Parallel Workers...")
    print("=" * 115)

    t_start = time.time()
    weeks_list = list(range(1, total_weeks + 1))

    # Parallel week execution
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(evaluate_week_all_models, weeks_list))

    total_sim_time = time.time() - t_start
    # Sort results by week
    results.sort(key=lambda x: x["week"])

    total_ticks = sum(r["ticks"] for r in results)
    print(f"\n[+] Evaluated {total_ticks:,} Ticks across {len(results)} Weeks in {total_sim_time:.1f}s ({total_ticks/max(total_sim_time, 0.001):,.0f} ticks/s)!")

    # Print sample weeks progress table
    print(f"\n{'Week':<5} | {'Date Range':<23} | {'Ticks':<9} | " + " | ".join([f"{cfg['id']:<8}" for cfg in STRATEGY_CONFIGS]))
    print("-" * 115)
    sample_weeks = [1, 2, 5, 52, 104, 155, 156, 157, 180, 208, 260, 261, 262]
    for r in results:
        w = r["week"]
        if w in sample_weeks or w % 50 == 0:
            date_str = f"{r['start_date']}->{r['end_date']}"
            pips_strs = [f"{r['models'][cfg['id']]['net_pips']:+6.1f}p" for cfg in STRATEGY_CONFIGS]
            split_tag = " [TRAIN END]" if w == TRAIN_END_WEEK else (" [TEST START]" if w == TRAIN_END_WEEK + 1 else "")
            print(f"W{w:<4} | {date_str:<23} | {r['ticks']:<9,d} | " + " | ".join(pips_strs) + split_tag)

    print("-" * 115)

    # Aggregate Metrics per Strategy: Train (Weeks 1-156), Test (Weeks 157-262), and Full 5-Year
    summary_rows = []
    weekly_matrix_rows = []

    print("\n" + "=" * 135)
    print("                         SOP-TRD-VAL-5Y-001 AUDIT SCORECARD: 3Y TRAIN vs 2Y TEST vs 5Y TOTAL                         ".center(135))
    print("=" * 135)

    for cfg in STRATEGY_CONFIGS:
        sid = cfg["id"]
        sname = cfg["name"]

        train_trades_list = []
        test_trades_list = []
        all_trades_list = []

        train_records = []
        test_records = []
        all_records = []

        for r in results:
            w = r["week"]
            m_data = r["models"][sid]
            t_arr = m_data["trades"]

            rec = {
                "week": w,
                "start_date": r["start_date"],
                "end_date": r["end_date"],
                "ticks": r["ticks"],
                "split": r["split"],
                "strategy_id": sid,
                "strategy_name": sname,
                "trades": m_data["total_trades"],
                "net_pips": m_data["net_pips"],
                "win_rate": m_data["win_rate"],
                "profit_factor": m_data["profit_factor"],
                "max_dd": m_data["max_dd"],
                "metrics": m_data["metrics"]
            }
            weekly_matrix_rows.append(rec)
            all_records.append(rec)

            if len(t_arr) > 0:
                all_trades_list.append(t_arr)

            if w <= TRAIN_END_WEEK:
                train_records.append(rec)
                if len(t_arr) > 0:
                    train_trades_list.append(t_arr)
            else:
                test_records.append(rec)
                if len(t_arr) > 0:
                    test_trades_list.append(t_arr)

        tr_all = np.concatenate(all_trades_list) if all_trades_list else np.empty(0, dtype=TRADE_DTYPE)
        tr_train = np.concatenate(train_trades_list) if train_trades_list else np.empty(0, dtype=TRADE_DTYPE)
        tr_test = np.concatenate(test_trades_list) if test_trades_list else np.empty(0, dtype=TRADE_DTYPE)

        m_5y = compute_metrics(tr_all)
        m_train = compute_metrics(tr_train)
        m_test = compute_metrics(tr_test)

        # Weekly stats
        def calc_weekly_stats(recs):
            pips = [rc["net_pips"] for rc in recs]
            w_win = sum(1 for p in pips if p > 0)
            w_tot = len(pips)
            w_pct = (w_win / w_tot * 100.0) if w_tot > 0 else 0.0
            avg_p = float(np.mean(pips)) if pips else 0.0
            std_p = float(np.std(pips)) if len(pips) > 1 else 0.0
            sharpe = (avg_p / std_p * np.sqrt(52)) if std_p > 0 else 0.0
            return w_pct, sharpe

        train_win_wk_pct, train_sharpe = calc_weekly_stats(train_records)
        test_win_wk_pct, test_sharpe = calc_weekly_stats(test_records)
        all_win_wk_pct, all_sharpe = calc_weekly_stats(all_records)

        # Gate 1 Pass/Fail (3Y Train: PF >= 1.35, Net > 300, DD <= 450, Trades >= 100)
        gate1_pass = (m_train["profit_factor"] >= 1.35 and 
                      m_train["net_pips"] > 300.0 and 
                      m_train["max_drawdown_pips"] <= 450.0 and 
                      m_train["total_trades"] >= 100)

        # Gate 2 Pass/Fail (2Y Test: Net > 150, PF >= 1.25, MaxDD <= 1.35*TrainDD)
        sharpe_ret = (test_sharpe / train_sharpe) if train_sharpe > 0 else 0.0
        gate2_pass = (m_test["net_pips"] > 150.0 and 
                      m_test["profit_factor"] >= 1.25 and 
                      m_test["max_drawdown_pips"] <= 1.35 * max(m_train["max_drawdown_pips"], 10.0))

        summary_rows.append({
            "strategy_id": sid,
            "strategy_name": sname,
            # 3Y Train
            "train_trades": m_train["total_trades"],
            "train_win_rate": m_train["win_rate_pct"],
            "train_net_pips": round(m_train["net_pips"], 1),
            "train_pf": round(m_train["profit_factor"], 2),
            "train_max_dd": round(m_train["max_drawdown_pips"], 1),
            "train_sharpe": round(train_sharpe, 2),
            "train_win_wk_pct": round(train_win_wk_pct, 1),
            "gate1_status": "PASS" if gate1_pass else "FAIL",
            # 2Y Test
            "test_trades": m_test["total_trades"],
            "test_win_rate": m_test["win_rate_pct"],
            "test_net_pips": round(m_test["net_pips"], 1),
            "test_pf": round(m_test["profit_factor"], 2),
            "test_max_dd": round(m_test["max_drawdown_pips"], 1),
            "test_sharpe": round(test_sharpe, 2),
            "test_win_wk_pct": round(test_win_wk_pct, 1),
            "sharpe_retention": round(sharpe_ret, 2),
            "gate2_status": "PASS" if gate2_pass else "FAIL",
            # 5Y Total
            "total_5y_trades": m_5y["total_trades"],
            "total_5y_win_rate": m_5y["win_rate_pct"],
            "total_5y_net_pips": round(m_5y["net_pips"], 1),
            "total_5y_pf": round(m_5y["profit_factor"], 2),
            "total_5y_max_dd": round(m_5y["max_drawdown_pips"], 1),
            "total_5y_sharpe": round(all_sharpe, 2),
            "final_qualification": "CERTIFIED DEPLOY" if (gate1_pass and gate2_pass) else ("RESEARCH APPROVED" if m_5y["net_pips"] > 0 else "REJECTED")
        })

    df_summary = pd.DataFrame(summary_rows)

    # Print comparison table
    print(f"{'STRATEGY':<32} | {'3Y TRAIN (W1-156)':<28} | {'2Y TEST (W157-262)':<28} | {'FULL 5-YEAR TOTAL':<25} | {'STATUS'}")
    print(f"{'':<32} | {'Net Pips':<9} {'PF':<5} {'MaxDD':<6} {'Gate1':<5} | {'Net Pips':<9} {'PF':<5} {'MaxDD':<6} {'Gate2':<5} | {'Net Pips':<9} {'PF':<5} {'MaxDD':<7} |")
    print("-" * 135)

    for _, r in df_summary.iterrows():
        train_s = f"{r['train_net_pips']:+7.1f}p {r['train_pf']:4.2f} {r['train_max_dd']:5.1f}p {r['gate1_status']:<5}"
        test_s = f"{r['test_net_pips']:+7.1f}p {r['test_pf']:4.2f} {r['test_max_dd']:5.1f}p {r['gate2_status']:<5}"
        total_s = f"{r['total_5y_net_pips']:+7.1f}p {r['total_5y_pf']:4.2f} {r['total_5y_max_dd']:5.1f}p"
        print(f"{r['strategy_name']:<32} | {train_s} | {test_s} | {total_s} | {r['final_qualification']}")

    # Export CSV files
    csv_summary_path = RESULTS_DIR / "production_fleet_5y_audit.csv"
    df_summary.to_csv(csv_summary_path, index=False)

    df_matrix = pd.DataFrame(weekly_matrix_rows)
    # Drop complex object column for CSV export
    if "metrics" in df_matrix.columns:
        df_matrix = df_matrix.drop(columns=["metrics"])
    csv_matrix_path = RESULTS_DIR / "production_fleet_weekly_matrix.csv"
    df_matrix.to_csv(csv_matrix_path, index=False)

    print("=" * 135)
    print(f"[+] Summary Audit CSV exported to : {csv_summary_path}")
    print(f"[+] Full 262-Week Matrix exported to : {csv_matrix_path}")
    print("=" * 135)


if __name__ == "__main__":
    run_5y_fleet_audit()
