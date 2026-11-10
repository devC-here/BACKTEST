#!/usr/bin/env python3
"""
================================================================================
  CROSS-PAIR PERFORMANCE AUDIT: 14 ELITE PRODUCTION STRATEGIES ON EURGBP
================================================================================
Evaluates all 14 elite production models across all 157 weeks of EURGBP (~65.89M ticks)
spanning 3 calendar years (2021, 2022, 2024).

Objectives:
  1. Measure Out-of-Pair Generalization & Edge Transfer:
     Do strategies tuned on EURUSD hold positive expectancy on EURGBP?
  2. Pinpoint Cross-Pair Champions:
     Identify which models thrive on EURGBP's lower volatility and wider spread.
  3. Side-by-Side Comparison:
     Compare Win Rate, Profit Factor, Net Pips, and Max DD against EURUSD 5Y baselines.
  4. Multi-Strategy Fleet Portfolio Analysis:
     Evaluate combined equity curve, diversification benefit, and drawdown reduction.
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

# Active Elite Strategy Imports
from strategies.new_discoveries.disc42_macro_structure_h4_m30_elite.strategy import MacroStructureH4M30Elite
from strategies.new_discoveries.disc45_continuous_m5_rejection_elite.strategy import ContinuousM5RejectionElite
from strategies.new_discoveries.disc40_rsi_regime_range_shift_elite.strategy import RSIRegimeRangeShiftElite
from strategies.new_discoveries.disc43_intraday_precision_h1_m5_elite.strategy import IntradayPrecisionH1M5Elite
from strategies.s66_bhanushali_44ma_elite.strategy import S66Bhanushali44MAElite
from strategies.high_frequency_m15_scalper_elite.strategy import HighFrequencyM15ScalperElite
from strategies.s10_dual_ema_pullback_elite.strategy import S10DualEMAPullbackElite
from strategies.s14_inside_bar_m30_elite.strategy import S14InsideBarM30Elite
from strategies.disc03_ny_open_overlap_elite.strategy import NewYorkOpenOverlapBreakoutElite
from strategies.new_discoveries.disc41_thursday_macro_trend_ignition_elite.strategy import ThursdayMacroTrendIgnitionElite
from strategies.disc38_london_midday_macd_acceleration_elite.strategy import LondonMiddayMACDAccelerationElite
from strategies.new_discoveries.disc39_stochastic_deep_pullback_elite.strategy import StochasticDeepPullbackElite
from strategies.comb12_bollinger_squeeze_elite.strategy import BollingerBandwidthExpansionElite
from strategies.comb27_squeeze_supertrend_hybrid_elite.strategy import BollingerSupertrendHybridElite

PIP_SIZE = 0.0001
RAW_DIR = ROOT_DIR / "weekly_data" / "EURGBP"
RESULTS_DIR = ROOT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Reference 5-Year Baselines on EURUSD
EURUSD_5Y_BASELINES = {
    "DISC42": {"wr": 72.9, "pf": 3.97, "pips": 387.6, "dd": 20.4, "ret_dd": 19.00, "trades": 48},
    "DISC45": {"wr": 53.9, "pf": 1.57, "pips": 999.4, "dd": 60.9, "ret_dd": 16.41, "trades": 620},
    "DISC40": {"wr": 44.0, "pf": 1.31, "pips": 1112.8, "dd": 144.2, "ret_dd": 7.72, "trades": 611},
    "DISC43": {"wr": 52.0, "pf": 1.55, "pips": 519.8, "dd": 59.3, "ret_dd": 8.77, "trades": 321},
    "S66_44MA": {"wr": 40.7, "pf": 2.37, "pips": 1153.9, "dd": 221.0, "ret_dd": 5.22, "trades": 123},
    "HF_M15": {"wr": 35.7, "pf": 1.13, "pips": 1755.9, "dd": 336.5, "ret_dd": 5.22, "trades": 1903},
    "S10_EMA": {"wr": 33.0, "pf": 1.25, "pips": 803.6, "dd": 171.8, "ret_dd": 4.68, "trades": 176},
    "S14_IB": {"wr": 36.8, "pf": 1.46, "pips": 717.3, "dd": 159.0, "ret_dd": 4.51, "trades": 171},
    "DISC03": {"wr": 45.0, "pf": 1.29, "pips": 405.3, "dd": 140.3, "ret_dd": 2.89, "trades": 200},
    "DISC41": {"wr": 43.6, "pf": 1.30, "pips": 336.1, "dd": 130.8, "ret_dd": 2.57, "trades": 133},
    "DISC38": {"wr": 43.0, "pf": 1.28, "pips": 372.7, "dd": 147.7, "ret_dd": 2.52, "trades": 193},
    "DISC39": {"wr": 42.5, "pf": 1.22, "pips": 483.4, "dd": 202.2, "ret_dd": 2.39, "trades": 438},
    "COMB12": {"wr": 41.9, "pf": 1.18, "pips": 341.6, "dd": 162.6, "ret_dd": 2.10, "trades": 329},
    "COMB27": {"wr": 33.6, "pf": 1.14, "pips": 311.7, "dd": 171.7, "ret_dd": 1.82, "trades": 336},
}


def build_strategy_instances():
    """Builds instantiated strategies with EURGBP spread accommodation (max_spread=2.0)."""
    return [
        ("DISC42", MacroStructureH4M30Elite(max_spread=2.0)),
        ("DISC45", ContinuousM5RejectionElite(max_spread=2.0)),
        ("DISC40", RSIRegimeRangeShiftElite(max_spread_pips=2.0)),
        ("DISC43", IntradayPrecisionH1M5Elite(max_spread=2.0)),
        ("S66_44MA", S66Bhanushali44MAElite()),
        ("HF_M15", HighFrequencyM15ScalperElite()),
        ("S10_EMA", S10DualEMAPullbackElite()),
        ("S14_IB", S14InsideBarM30Elite()),
        ("DISC03", NewYorkOpenOverlapBreakoutElite()),
        ("DISC41", ThursdayMacroTrendIgnitionElite(max_spread_pips=2.0)),
        ("DISC38", LondonMiddayMACDAccelerationElite(max_spread_pips=2.0)),
        ("DISC39", StochasticDeepPullbackElite(max_spread_pips=2.0)),
        ("COMB12", BollingerBandwidthExpansionElite()),
        ("COMB27", BollingerSupertrendHybridElite()),
    ]


def evaluate_eurgbp_week(week_num: int) -> Dict[str, Any]:
    """Worker task: loads tick dataset for week_num and evaluates all 14 strategies."""
    engine = BacktestEngine(pip_size=PIP_SIZE, use_slippage=True)
    ticks, meta = get_week_dataset(week_num, raw_dir=RAW_DIR)

    strats = build_strategy_instances()
    week_trades = {}

    for strat_id, strat in strats:
        raw_trades = engine.run(ticks, strat)
        t_list = []
        for i in range(len(raw_trades)):
            row = raw_trades[i]
            t_list.append({
                "strategy": strat_id,
                "week": week_num,
                "start_date": meta.get("start_date", ""),
                "end_date": meta.get("end_date", ""),
                "entry_time": int(row["entry_time"]),
                "exit_time": int(row["exit_time"]),
                "direction": "BUY" if row["direction"] == 1 else "SELL",
                "entry_price": float(row["entry_price"]),
                "exit_price": float(row["exit_price"]),
                "pips": float(row["pips"]),
                "exit_reason": int(row["exit_reason"]),
            })
        week_trades[strat_id] = t_list

    return {
        "week": week_num,
        "ticks": len(ticks),
        "trades": week_trades
    }


def compute_metrics(trades: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Computes comprehensive trading metrics from a list of trades."""
    n = len(trades)
    if n == 0:
        return {
            "trades": 0, "wins": 0, "losses": 0, "wr": 0.0,
            "pips": 0.0, "gross_profit": 0.0, "gross_loss": 0.0,
            "pf": 0.0, "max_dd": 0.0, "ret_dd": 0.0,
            "expectancy": 0.0, "avg_win": 0.0, "avg_loss": 0.0
        }

    pips_arr = np.array([t["pips"] for t in trades], dtype=np.float64)
    wins = np.sum(pips_arr > 0)
    losses = np.sum(pips_arr < 0)
    wr = (wins / n) * 100.0

    gross_profit = float(np.sum(pips_arr[pips_arr > 0])) if wins > 0 else 0.0
    gross_loss = float(abs(np.sum(pips_arr[pips_arr < 0]))) if losses > 0 else 0.0
    pf = (gross_profit / gross_loss) if gross_loss > 0 else (99.0 if gross_profit > 0 else 0.0)

    # Compute Max Drawdown in pips
    cum_pips = np.cumsum(pips_arr)
    running_max = np.maximum.accumulate(cum_pips)
    dd_curve = running_max - cum_pips
    max_dd = float(np.max(dd_curve)) if len(dd_curve) > 0 else 0.0

    net_pips = float(np.sum(pips_arr))
    ret_dd = (net_pips / max_dd) if max_dd > 0 else (99.0 if net_pips > 0 else 0.0)

    avg_win = float(gross_profit / wins) if wins > 0 else 0.0
    avg_loss = float(gross_loss / losses) if losses > 0 else 0.0
    expectancy = float(np.mean(pips_arr))

    return {
        "trades": n,
        "wins": int(wins),
        "losses": int(losses),
        "wr": round(wr, 2),
        "pips": round(net_pips, 1),
        "gross_profit": round(gross_profit, 1),
        "gross_loss": round(gross_loss, 1),
        "pf": round(pf, 2),
        "max_dd": round(max_dd, 1),
        "ret_dd": round(ret_dd, 2),
        "expectancy": round(expectancy, 2),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2)
    }


def main():
    t_start = time.time()
    manifest = load_manifest(RAW_DIR)
    weeks = sorted(list(manifest.keys()))
    total_weeks = len(weeks)

    print("=" * 100)
    print(f"  CROSS-PAIR PERFORMANCE AUDIT: 14 STRATEGIES ON EURGBP ({total_weeks} WEEKS)  ".center(100))
    print("=" * 100)
    print(f"[*] Raw Directory: {RAW_DIR}")
    print(f"[*] Available Weeks: {len(weeks)} ({manifest[weeks[0]]['start_date']} to {manifest[weeks[-1]]['end_date']})")
    print(f"[*] Launching parallel execution across CPU workers...")

    all_strat_trades: Dict[str, List[Dict[str, Any]]] = {
        s_id: [] for s_id in EURUSD_5Y_BASELINES.keys()
    }
    all_combined_trades: List[Dict[str, Any]] = []

    # Parallel evaluation
    workers = min(os.cpu_count() or 4, 10)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(evaluate_eurgbp_week, w): w for w in weeks}
        done_count = 0
        for f in futures:
            res = f.result()
            done_count += 1
            w_num = res["week"]
            w_trades = res["trades"]
            for s_id, t_list in w_trades.items():
                all_strat_trades[s_id].extend(t_list)
                all_combined_trades.extend(t_list)

            if done_count % 20 == 0 or done_count == total_weeks:
                pct = (done_count / total_weeks) * 100
                print(f"  -> Processed {done_count}/{total_weeks} weeks ({pct:5.1f}%) in {time.time()-t_start:.1f}s...")

    print(f"\n[+] Backtest simulation complete across {total_weeks} weeks in {time.time()-t_start:.1f}s!")

    # Sort all combined trades temporally for accurate portfolio DD
    all_combined_trades.sort(key=lambda x: x["entry_time"])

    # Save detailed trade logs
    df_trades = pd.DataFrame(all_combined_trades)
    trade_log_csv = RESULTS_DIR / "eurgbp_cross_pair_audit_trades.csv"
    if not df_trades.empty:
        # Convert ms timestamps to readable UTC strings
        df_trades["entry_time_utc"] = pd.to_datetime(df_trades["entry_time"], unit="ms", utc=True).dt.strftime("%Y-%m-%d %H:%M:%S")
        df_trades["exit_time_utc"] = pd.to_datetime(df_trades["exit_time"], unit="ms", utc=True).dt.strftime("%Y-%m-%d %H:%M:%S")
        df_trades.to_csv(trade_log_csv, index=False)
        print(f"[+] Saved {len(df_trades):,} total trades to {trade_log_csv.name}.")

    # Evaluate each strategy
    strat_scorecard = []
    for s_id in EURUSD_5Y_BASELINES.keys():
        t_list = all_strat_trades[s_id]
        t_list.sort(key=lambda x: x["entry_time"])
        metrics = compute_metrics(t_list)
        base = EURUSD_5Y_BASELINES[s_id]

        strat_scorecard.append({
            "id": s_id,
            "trades": metrics["trades"],
            "wr": metrics["wr"],
            "pf": metrics["pf"],
            "pips": metrics["pips"],
            "max_dd": metrics["max_dd"],
            "ret_dd": metrics["ret_dd"],
            "expectancy": metrics["expectancy"],
            # Baseline comparisons
            "base_wr": base["wr"],
            "base_pf": base["pf"],
            "base_pips": base["pips"],
            "base_dd": base["dd"],
            "base_ret_dd": base["ret_dd"],
            "wr_delta": round(metrics["wr"] - base["wr"], 1),
            "pf_delta": round(metrics["pf"] - base["pf"], 2)
        })

    # Portfolio Metrics (Combined Fleet on EURGBP)
    portfolio_metrics = compute_metrics(all_combined_trades)

    # Print Summary Table
    print("\n" + "=" * 125)
    print("   EURGBP CROSS-PAIR AUDIT: INDIVIDUAL STRATEGY RESULTS VS EURUSD BASELINE   ".center(125))
    print("=" * 125)
    header = (
        f"{'Strategy':<10} | {'Trades':>6} | {'EURGBP WR':>9} | {'Base WR':>8} | {'WR Diff':>8} | "
        f"{'EURGBP PF':>9} | {'Base PF':>8} | {'Net Pips':>9} | {'Max DD':>8} | {'Ret/DD':>7} | {'Verdict':<12}"
    )
    print(header)
    print("-" * 125)

    profitable_fleet_trades = []
    profitable_count = 0

    for s in sorted(strat_scorecard, key=lambda x: x["pips"], reverse=True):
        if s["pips"] > 0 and s["pf"] >= 1.05:
            verdict = "PROFITABLE"
            profitable_count += 1
            profitable_fleet_trades.extend(all_strat_trades[s["id"]])
        elif s["pips"] > 0:
            verdict = "BREAKEVEN"
        else:
            verdict = "UNDERPERFORM"

        line = (
            f"{s['id']:<10} | {s['trades']:>6} | {s['wr']:>8.1f}% | {s['base_wr']:>7.1f}% | {s['wr_delta']:>+7.1f}% | "
            f"{s['pf']:>9.2f} | {s['base_pf']:>8.2f} | {s['pips']:>+8.1f}p | {s['max_dd']:>7.1f}p | {s['ret_dd']:>7.2f} | {verdict:<12}"
        )
        print(line)

    print("-" * 125)
    print(f"Total Combined Trades Across All 14 Models: {portfolio_metrics['trades']:,}")
    print(f"Combined Fleet Win Rate:                   {portfolio_metrics['wr']:.1f}%")
    print(f"Combined Fleet Profit Factor:              {portfolio_metrics['pf']:.2f}")
    print(f"Combined Fleet Net Return:                 {portfolio_metrics['pips']:+.1f} pips")
    print(f"Combined Fleet Max Drawdown:               {portfolio_metrics['max_dd']:.1f} pips")
    print(f"Combined Fleet Return/DD:                  {portfolio_metrics['ret_dd']:.2f}x")
    print("=" * 125)

    # Evaluate the Profitable Cross-Pair Subset
    profitable_fleet_trades.sort(key=lambda x: x["entry_time"])
    prof_metrics = compute_metrics(profitable_fleet_trades)
    print(f"\n[+] CURATED PROFITABLE FLEET ON EURGBP ({profitable_count} Models):")
    print(f"    * Trades:     {prof_metrics['trades']:,}")
    print(f"    * Win Rate:   {prof_metrics['wr']:.1f}%")
    print(f"    * PF:         {prof_metrics['pf']:.2f}")
    print(f"    * Net Pips:   {prof_metrics['pips']:+.1f} pips")
    print(f"    * Max DD:     {prof_metrics['max_dd']:.1f} pips")
    print(f"    * Ret / DD:   {prof_metrics['ret_dd']:.2f}x")

    # Save summary report as JSON for artifact generation
    summary_data = {
        "dataset": "EURGBP",
        "total_weeks": total_weeks,
        "date_range": f"{manifest[weeks[0]]['start_date']} to {manifest[weeks[-1]]['end_date']}",
        "strategies": strat_scorecard,
        "portfolio_combined": portfolio_metrics,
        "curated_profitable": prof_metrics,
        "execution_time_seconds": round(time.time() - t_start, 1)
    }
    json_path = RESULTS_DIR / "eurgbp_cross_pair_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    print(f"\n[+] Summary metadata written to {json_path.name}.")


if __name__ == "__main__":
    main()
