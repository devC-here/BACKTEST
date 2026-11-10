#!/usr/bin/env python3
"""
================================================================================
  HIGH-SPEED 5-YEAR AUDIT: ALL 15 DISCOVERED CHAMPION STRATEGIES
  Evaluates all strategies found by our quantitative labs across 262 Weeks
  (~123.22 Million Authentic Dukascopy Ticks: Oct 2021 - Oct 2026).
  
  SOP Validation Protocol:
  - 3-Year In-Sample Split:  Weeks 1 to 156 (Oct 2021 - Sep 2024)
  - 2-Year Out-of-Sample Split: Weeks 157 to 262 (Sep 2024 - Oct 2026)
  - 5-Year Full Cumulative Cycle: Weeks 1 to 262
================================================================================
"""

from __future__ import annotations
import sys
import os
import time
import json
from pathlib import Path
from typing import Dict, List, Any
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tick_engine.core.engine import BacktestEngine, TRADE_DTYPE
from tick_engine.core.dataloader import get_week_dataset, load_manifest
from tick_engine.core.metrics import compute_metrics

# 15 Discovered Champion Strategies
from strategies.comb28_unified_squeeze_ensemble_elite.strategy import UnifiedSqueezeEnsembleElite
from strategies.comb12_bollinger_squeeze_elite.strategy import BollingerBandwidthExpansionElite
from strategies.disc23_keltner_squeeze_elite.strategy import KeltnerBandwidthExpansionElite
from strategies.disc03_ny_open_overlap_elite.strategy import NewYorkOpenOverlapBreakoutElite
from strategies.disc02_frankfurt_london_orb_elite.strategy import FrankfurtLondonPreMarketORBElite
from strategies.comb14_supertrend_vwap_elite.strategy import SupertrendVWAPPullbackElite
from strategies.comb17_psar_adx_elite.strategy import ParabolicSARADXElite
from strategies.comb08_asian_breakout_adx_elite.strategy import LondonAsianBreakoutADXElite
from strategies.comb26_asian_tight_coil_elite.strategy import AsianBreakoutTightCoilElite
from strategies.comb27_squeeze_supertrend_hybrid_elite.strategy import BollingerSupertrendHybridElite
from strategies.inst05_market_structure_shift_elite.strategy import MarketStructureShiftRetestElite
from strategies.disc31_london_lunch_trap_fade_elite.strategy import LondonLunchTrapFadeElite
from strategies.disc32_european_early_bird_elite.strategy import EuropeanEarlyBirdElite
from strategies.disc33_us_afternoon_trend_elite.strategy import USAfternoonTrendElite
from strategies.disc34_friday_weekly_reversal_elite.strategy import FridayWeeklyReversalElite

DISCOVERED_STRATEGY_CONFIGS = [
    # Core Volatility & Session Squeezes
    {"id": "COMB28", "name": "Unified Squeeze Ensemble Elite", "cls": UnifiedSqueezeEnsembleElite},
    {"id": "COMB12", "name": "Bollinger Squeeze Elite", "cls": BollingerBandwidthExpansionElite},
    {"id": "DISC23", "name": "Keltner Squeeze Elite", "cls": KeltnerBandwidthExpansionElite},
    {"id": "COMB27", "name": "Squeeze Supertrend Hybrid Elite", "cls": BollingerSupertrendHybridElite},
    
    # Session Breakouts & Liquidity Sweeps
    {"id": "DISC03", "name": "NY Open Overlap Breakout Elite", "cls": NewYorkOpenOverlapBreakoutElite},
    {"id": "DISC02", "name": "Frankfurt London ORB Elite", "cls": FrankfurtLondonPreMarketORBElite},
    {"id": "COMB08", "name": "Asian Breakout ADX Elite", "cls": LondonAsianBreakoutADXElite},
    {"id": "COMB26", "name": "Asian Tight Coil Breakout Elite", "cls": AsianBreakoutTightCoilElite},
    {"id": "INST05", "name": "Market Structure Shift Elite", "cls": MarketStructureShiftRetestElite},

    # Trend & Indicator Pullbacks
    {"id": "COMB14", "name": "Supertrend VWAP Pullback Elite", "cls": SupertrendVWAPPullbackElite},
    {"id": "COMB17", "name": "Parabolic SAR ADX Elite", "cls": ParabolicSARADXElite},

    # New In-Between Gap Champions
    {"id": "DISC31", "name": "London Lunch Trap Fade Elite", "cls": LondonLunchTrapFadeElite},
    {"id": "DISC32", "name": "European Early Bird Elite", "cls": EuropeanEarlyBirdElite},
    {"id": "DISC33", "name": "US Afternoon Trend Elite", "cls": USAfternoonTrendElite},
    {"id": "DISC34", "name": "Friday Weekly Reversal Elite", "cls": FridayWeeklyReversalElite},
]

RAW_DIR = ROOT_DIR / "eurusd_weekly_data"
PROC_DIR = ROOT_DIR / "tick_engine" / "data" / "processed"
RESULTS_DIR = ROOT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_END_WEEK = 156  # 3-Year boundary (Weeks 1 to 156)


def evaluate_week_discovered_fleet(week_num: int) -> Dict[str, Any]:
    """
    Worker function executed in parallel across CPU processes.
    Loads ticks for week_num and evaluates all 15 discovered models.
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

    for cfg in DISCOVERED_STRATEGY_CONFIGS:
        sid = cfg["id"]
        strat = cfg["cls"]()
        trades = engine.run(ticks, strat)
        m = compute_metrics(trades)

        week_result["models"][sid] = {
            "trades": trades,
            "net_pips": m["net_pips"],
            "total_trades": m["total_trades"],
            "win_rate": m["win_rate_pct"],
            "profit_factor": m["profit_factor"],
            "max_dd": m["max_drawdown_pips"]
        }

    return week_result


def run_5y_discovered_audit():
    print("=" * 130)
    print("      HIGH-SPEED 5-YEAR AUDIT: ALL 15 DISCOVERED CHAMPION MODELS (262 WEEKS / ~123.2M TICKS)      ".center(130))
    print("=" * 130)

    manifest = load_manifest(RAW_DIR)
    total_weeks = len(manifest)
    num_workers = min(8, os.cpu_count() or 4)

    print(f"[*] Total Weeks in Catalog: {total_weeks} (Week 1 to Week {total_weeks})")
    print(f"[*] In-Sample Training Split:   Weeks 1 to {TRAIN_END_WEEK} (3 Years: Oct 2021 - Sep 2024)")
    print(f"[*] Out-of-Sample Test Split:  Weeks {TRAIN_END_WEEK + 1} to {total_weeks} (2 Years: Sep 2024 - Oct 2026)")
    print(f"[*] Auditing {len(DISCOVERED_STRATEGY_CONFIGS)} Discovered Models using {num_workers} Parallel Workers...")
    print("=" * 130)

    t_start = time.time()
    weeks_list = list(range(1, total_weeks + 1))

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(evaluate_week_discovered_fleet, weeks_list))

    total_sim_time = time.time() - t_start
    results.sort(key=lambda x: x["week"])
    total_ticks = sum(r["ticks"] for r in results)

    print(f"\n[+] Evaluated {total_ticks:,} Ticks across {len(results)} Weeks in {total_sim_time:.1f}s ({total_ticks/max(total_sim_time, 0.001):,.0f} ticks/s)!\n")

    summary_rows = []
    
    print("-" * 130)
    print(f"{'ID':<8} | {'Strategy Name':<32} | {'3Y Train Pips':<13} | {'2Y Test Pips':<12} | {'5Y Total Pips':<13} | {'5Y Trd':<6} | {'5Y WR':<6} | {'5Y PF':<6} | {'5Y MaxDD':<8}")
    print("-" * 130)

    all_fleet_trades = []

    for cfg in DISCOVERED_STRATEGY_CONFIGS:
        sid = cfg["id"]
        sname = cfg["name"]

        train_trades_list = []
        test_trades_list = []
        all_trades_list = []

        for r in results:
            w = r["week"]
            t_arr = r["models"][sid]["trades"]
            if len(t_arr) > 0:
                all_trades_list.append(t_arr)
                all_fleet_trades.append(t_arr)
                if w <= TRAIN_END_WEEK:
                    train_trades_list.append(t_arr)
                else:
                    test_trades_list.append(t_arr)

        tr_all = np.concatenate(all_trades_list) if all_trades_list else np.empty(0, dtype=TRADE_DTYPE)
        tr_train = np.concatenate(train_trades_list) if train_trades_list else np.empty(0, dtype=TRADE_DTYPE)
        tr_test = np.concatenate(test_trades_list) if test_trades_list else np.empty(0, dtype=TRADE_DTYPE)

        m_5y = compute_metrics(tr_all)
        m_train = compute_metrics(tr_train)
        m_test = compute_metrics(tr_test)

        summary_rows.append({
            "id": sid,
            "name": sname,
            "train_pips": m_train["net_pips"],
            "train_wr": m_train["win_rate_pct"],
            "train_pf": m_train["profit_factor"],
            "train_trades": m_train["total_trades"],
            "train_dd": m_train["max_drawdown_pips"],
            "test_pips": m_test["net_pips"],
            "test_wr": m_test["win_rate_pct"],
            "test_pf": m_test["profit_factor"],
            "test_trades": m_test["total_trades"],
            "test_dd": m_test["max_drawdown_pips"],
            "total_5y_pips": m_5y["net_pips"],
            "total_5y_wr": m_5y["win_rate_pct"],
            "total_5y_pf": m_5y["profit_factor"],
            "total_5y_trades": m_5y["total_trades"],
            "total_5y_dd": m_5y["max_drawdown_pips"]
        })

        train_str = f"{m_train['net_pips']:+8.1f}p"
        test_str = f"{m_test['net_pips']:+7.1f}p"
        total_str = f"{m_5y['net_pips']:+8.1f}p"
        wr_str = f"{m_5y['win_rate_pct']:4.1f}%"
        pf_str = f"{m_5y['profit_factor']:4.2f}"
        dd_str = f"{m_5y['max_drawdown_pips']:6.1f}p"

        print(f"{sid:<8} | {sname:<32} | {train_str:<13} | {test_str:<12} | {total_str:<13} | {m_5y['total_trades']:<6} | {wr_str:<6} | {pf_str:<6} | {dd_str:<8}")

    print("-" * 130)

    # Consolidated 15-Strategy Fleet Ensemble Metrics
    if all_fleet_trades:
        combined_tr = np.concatenate(all_fleet_trades)
        combined_m = compute_metrics(combined_tr)
        
        # Sort combined trades by timestamp
        combined_tr_sorted = np.sort(combined_tr, order='entry_time')
        combined_cum = np.cumsum(combined_tr_sorted['pips'])
        combined_peak = np.maximum.accumulate(combined_cum)
        combined_max_dd = float(np.max(combined_peak - combined_cum))

        # Train vs Test splits for combined
        train_tr_list = []
        test_tr_list = []
        for r in results:
            w = r["week"]
            for cfg in DISCOVERED_STRATEGY_CONFIGS:
                t_arr = r["models"][cfg["id"]]["trades"]
                if len(t_arr) > 0:
                    if w <= TRAIN_END_WEEK:
                        train_tr_list.append(t_arr)
                    else:
                        test_tr_list.append(t_arr)

        tr_comb_train = np.concatenate(train_tr_list) if train_tr_list else np.empty(0, dtype=TRADE_DTYPE)
        tr_comb_test = np.concatenate(test_tr_list) if test_tr_list else np.empty(0, dtype=TRADE_DTYPE)
        m_comb_train = compute_metrics(tr_comb_train)
        m_comb_test = compute_metrics(tr_comb_test)

        print("\n" + "=" * 130)
        print("  CONSOLIDATED 15-MODEL DISCOVERED FLEET ENSEMBLE (5-YEAR / 262 WEEKS / ~123.2M TICKS)  ".center(130))
        print("=" * 130)
        print(f"Total Discovered Fleet Trades: {combined_m['total_trades']} trades ({combined_m['total_trades']/262:.1f} trd/week, {combined_m['total_trades']/1250:.2f} trd/day)")
        print(f"3-Year In-Sample Return:       {m_comb_train['net_pips']:+10.1f} pips | PF: {m_comb_train['profit_factor']:.2f} | WR: {m_comb_train['win_rate_pct']:.1f}% ({m_comb_train['total_trades']} trds)")
        print(f"2-Year Out-of-Sample Return:   {m_comb_test['net_pips']:+10.1f} pips | PF: {m_comb_test['profit_factor']:.2f} | WR: {m_comb_test['win_rate_pct']:.1f}% ({m_comb_test['total_trades']} trds)")
        print(f"5-Year Full Cumulative Return: {combined_m['net_pips']:+10.1f} pips | PF: {combined_m['profit_factor']:.2f} | WR: {combined_m['win_rate_pct']:.1f}%")
        print(f"5-Year Maximum Fleet Drawdown: {combined_max_dd:10.1f} pips")
        print("=" * 130 + "\n")

    # Export to JSON and CSV
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(RESULTS_DIR / "5y_discovered_fleet_summary.csv", index=False)
    
    with open(RESULTS_DIR / "5y_discovered_fleet_audit.json", "w") as f:
        json.dump(summary_rows, f, indent=2)
        
    print(f"[+] Full audit artifacts exported to:")
    print(f"    - {RESULTS_DIR / '5y_discovered_fleet_summary.csv'}")
    print(f"    - {RESULTS_DIR / '5y_discovered_fleet_audit.json'}")


if __name__ == "__main__":
    run_5y_discovered_audit()
