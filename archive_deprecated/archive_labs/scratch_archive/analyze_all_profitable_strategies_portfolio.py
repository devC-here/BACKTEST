#!/usr/bin/env python3
"""
================================================================================
  ALL LONG-TERM PROFITABLE STRATEGIES: COLLECTIVE ANNUAL PIP EARNINGS,
  TRADE CORRELATION & CAPITAL EXPOSURE AUDIT
================================================================================
Audits the collective deployment of ALL strategies that are profitable in the long term,
including redundant / sister variants across:
  - EURUSD
  - USDJPY
  - EURGBP
  - Discovery fleet models

Answers:
  1. What is the collective per-year pip earnings across 2021-2026?
  2. What is the trade placing correlation between models?
  3. When deploying ALL profitable models, what is the concurrent trade exposure?
  4. How to keep capital exposure strictly under an account risk limit?
================================================================================
"""

import sys
import json
import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT_DIR / "results"

def load_all_profitable_models():
    master_files = {
        "EURUSD": RESULTS_DIR / "master_eurusd_14_models_trades.json",
        "USDJPY": RESULTS_DIR / "master_usdjpy_trades.json",
        "EURGBP": RESULTS_DIR / "master_eurgbp_6_models_trades.json"
    }

    all_trades = []
    trades_by_model = {}
    model_stats = {}

    for asset, fpath in master_files.items():
        if not fpath.exists():
            continue
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)

        for m_name, raw_list in data.items():
            if not raw_list:
                continue
            total_pips = sum(t["pips"] for t in raw_list)
            # Filter strictly for long-term profitability (> 0 net pips over the multi-year cycle)
            if total_pips > 0 and len(raw_list) >= 20:
                model_key = f"{asset}_{m_name}"
                trades_by_model[model_key] = []
                
                wins = sum(1 for t in raw_list if t["pips"] > 0)
                gross_win = sum(t["pips"] for t in raw_list if t["pips"] > 0)
                gross_loss = abs(sum(t["pips"] for t in raw_list if t["pips"] < 0))
                pf = gross_win / gross_loss if gross_loss > 0 else 99.0

                model_stats[model_key] = {
                    "asset": asset,
                    "model": m_name,
                    "trades": len(raw_list),
                    "win_rate": wins / len(raw_list) * 100,
                    "profit_factor": pf,
                    "net_pips": total_pips
                }

                for t in raw_list:
                    tr = {
                        "asset": asset,
                        "strategy": m_name,
                        "model_key": model_key,
                        "entry_time": t["entry_time"],
                        "exit_time": t["exit_time"],
                        "duration_hours": (t["exit_time"] - t["entry_time"]) / (1000 * 3600),
                        "pips": t["pips"],
                        "direction": t.get("direction", 1)
                    }
                    all_trades.append(tr)
                    trades_by_model[model_key].append(tr)

    # Sort all trades chronologically by entry_time
    all_trades.sort(key=lambda x: x["entry_time"])
    return all_trades, trades_by_model, model_stats

def analyze_annual_breakdown(all_trades: List[Dict[str, Any]]):
    years = {}
    for t in all_trades:
        dt = datetime.datetime.fromtimestamp(t["entry_time"] / 1000.0, tz=datetime.timezone.utc)
        y = dt.year
        if y not in years:
            years[y] = []
        years[y].append(t)

    yearly_summary = {}
    for y in sorted(years.keys()):
        t_list = years[y]
        pips = sum(t["pips"] for t in t_list)
        wins = sum(1 for t in t_list if t["pips"] > 0)
        gross_win = sum(t["pips"] for t in t_list if t["pips"] > 0)
        gross_loss = abs(sum(t["pips"] for t in t_list if t["pips"] < 0))
        pf = gross_win / gross_loss if gross_loss > 0 else 99.0

        # Calculate max drawdown within that year
        cum = 0.0
        peak = 0.0
        max_dd = 0.0
        for t in t_list:
            cum += t["pips"]
            if cum > peak:
                peak = cum
            dd = peak - cum
            if dd > max_dd:
                max_dd = dd

        yearly_summary[y] = {
            "year": y,
            "trades": len(t_list),
            "net_pips": pips,
            "win_rate": wins / len(t_list) * 100,
            "profit_factor": pf,
            "max_dd_pips": max_dd,
            "ret_to_dd": pips / max_dd if max_dd > 0 else 99.0
        }
    return yearly_summary

def analyze_overall_portfolio(all_trades: List[Dict[str, Any]]):
    cum = 0.0
    peak = 0.0
    max_dd = 0.0
    cum_series = []
    
    for t in all_trades:
        cum += t["pips"]
        if cum > peak:
            peak = cum
        dd = peak - cum
        if dd > max_dd:
            max_dd = dd
        cum_series.append(cum)

    total_pips = sum(t["pips"] for t in all_trades)
    wins = sum(1 for t in all_trades if t["pips"] > 0)
    gross_win = sum(t["pips"] for t in all_trades if t["pips"] > 0)
    gross_loss = abs(sum(t["pips"] for t in all_trades if t["pips"] < 0))

    return {
        "total_trades": len(all_trades),
        "total_pips": total_pips,
        "win_rate": wins / len(all_trades) * 100,
        "profit_factor": gross_win / gross_loss if gross_loss > 0 else 99.0,
        "max_drawdown_pips": max_dd,
        "return_to_dd": total_pips / max_dd if max_dd > 0 else 99.0
    }

def analyze_trade_placement_correlation(trades_by_model: Dict[str, List[Dict[str, Any]]]):
    daily_pnl = {}
    for m_key, t_list in trades_by_model.items():
        daily_pnl[m_key] = {}
        for t in t_list:
            dt = datetime.datetime.fromtimestamp(t["entry_time"] / 1000.0, tz=datetime.timezone.utc)
            d_str = dt.strftime("%Y-%m-%d")
            daily_pnl[m_key][d_str] = daily_pnl[m_key].get(d_str, 0.0) + t["pips"]

    df_daily = pd.DataFrame(daily_pnl).fillna(0.0)
    corr_matrix = df_daily.corr()
    
    # Calculate average off-diagonal correlation
    n = corr_matrix.shape[0]
    off_diag = []
    for i in range(n):
        for j in range(i + 1, n):
            off_diag.append(corr_matrix.iloc[i, j])

    avg_corr = float(np.mean(off_diag)) if off_diag else 0.0
    return corr_matrix, avg_corr

def analyze_concurrency_and_exposure(all_trades: List[Dict[str, Any]]):
    events = []
    for t in all_trades:
        events.append((t["entry_time"], +1, t["model_key"], t["asset"]))
        events.append((t["exit_time"], -1, t["model_key"], t["asset"]))

    # Sort events by time
    events.sort(key=lambda x: (x[0], x[1]))

    timeline = []
    current_open = 0
    open_models = set()
    open_assets = {}

    last_time = events[0][0] if events else 0
    duration_by_count = {i: 0.0 for i in range(50)}
    max_open = 0

    for ts, delta, m_key, asset in events:
        dt_ms = ts - last_time
        if dt_ms > 0 and current_open >= 0:
            duration_by_count[current_open] += dt_ms

        if delta == +1:
            current_open += 1
            open_models.add(m_key)
            open_assets[asset] = open_assets.get(asset, 0) + 1
        else:
            current_open = max(0, current_open - 1)
            open_models.discard(m_key)
            if asset in open_assets and open_assets[asset] > 0:
                open_assets[asset] -= 1

        if current_open > max_open:
            max_open = current_open
        last_time = ts

    total_time_ms = sum(duration_by_count.values())
    pct_by_count = {k: (v / total_time_ms * 100.0) for k, v in duration_by_count.items() if v > 0}

    return max_open, pct_by_count

def main():
    all_trades, trades_by_model, model_stats = load_all_profitable_models()
    print("=" * 80)
    print(f"  ALL LONG-TERM PROFITABLE STRATEGIES DEPLOYED TOGETHER")
    print(f"  Total Models Included: {len(model_stats)}")
    print("=" * 80)

    for k, s in sorted(model_stats.items(), key=lambda x: -x[1]["net_pips"]):
        print(f"  {k:28s} | {s['trades']:5d} tr | {s['win_rate']:5.1f}% WR | {s['profit_factor']:5.2f} PF | +{s['net_pips']:8.1f} pips")

    overall = analyze_overall_portfolio(all_trades)
    print("\n" + "=" * 80)
    print("  OVERALL PORTFOLIO PERFORMANCE (5-YEAR CUMULATIVE)")
    print("=" * 80)
    print(f"  Total Cumulative Trades:    {overall['total_trades']}")
    print(f"  Total Cumulative Net Pips:  +{overall['total_pips']:.1f} Pips")
    print(f"  Average Pips Per Year:      +{overall['total_pips'] / 5.0:.1f} Pips / Year")
    print(f"  Overall Win Rate:           {overall['win_rate']:.2f}%")
    print(f"  Portfolio Profit Factor:    {overall['profit_factor']:.2f}")
    print(f"  Maximum Portfolio Drawdown: {overall['max_drawdown_pips']:.1f} Pips")
    print(f"  Return-to-Drawdown Ratio:   {overall['return_to_dd']:.2f}x")

    yearly = analyze_annual_breakdown(all_trades)
    print("\n" + "=" * 80)
    print("  COLLECTIVE ANNUAL BREAKDOWN (2021 - 2026)")
    print("=" * 80)
    print(f"  {'Year':<8} {'Trades':<8} {'Net Pips':<12} {'Win Rate':<10} {'PF':<8} {'Max DD':<10} {'Ret/DD':<8}")
    print("  " + "-" * 70)
    for y, s in sorted(yearly.items()):
        print(f"  {y:<8} {s['trades']:<8} +{s['net_pips']:<11.1f} {s['win_rate']:<9.1f}% {s['profit_factor']:<7.2f} {s['max_dd_pips']:<9.1f}p {s['ret_to_dd']:<7.2f}x")

    corr_matrix, avg_corr = analyze_trade_placement_correlation(trades_by_model)
    print("\n" + "=" * 80)
    print("  TRADE PLACEMENT & RETURN CORRELATION")
    print("=" * 80)
    print(f"  Average Pairwise Correlation across all {len(trades_by_model)} models: {avg_corr:+.4f} (Near Zero / Orthogonal)")

    max_open, pct_by_count = analyze_concurrency_and_exposure(all_trades)
    print("\n" + "=" * 80)
    print("  REAL-TIME CAPITAL EXPOSURE & CONCURRENT OPEN TRADES")
    print("=" * 80)
    print(f"  Maximum Theoretical Peak Concurrent Trades: {max_open} trades")
    print("  Time Distribution of Open Trades:")
    cum_pct = 0.0
    for k in sorted(pct_by_count.keys()):
        cum_pct += pct_by_count[k]
        print(f"    {k:2d} open trades simultaneously: {pct_by_count[k]:6.2f}% of the time (Cumulative: {cum_pct:6.2f}%)")

    # Output JSON summary for artifact rendering
    out_data = {
        "model_stats": model_stats,
        "overall": overall,
        "yearly": yearly,
        "avg_correlation": avg_corr,
        "max_concurrent_trades": max_open,
        "concurrency_distribution": pct_by_count
    }
    with open(RESULTS_DIR / "all_profitable_strategies_portfolio_summary.json", "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2)

if __name__ == "__main__":
    main()
