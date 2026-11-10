#!/usr/bin/env python3
"""
Edge Lab: Hypothesis Testing Experiment Runner.
Executes systematic iterations of the strategy across authentic tick data
and prints a progression scorecard showing edge evolution.
"""

from __future__ import annotations
import sys
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import pandas as pd

# Add repo root
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tick_engine.core.engine import BacktestEngine, TRADE_DTYPE
from tick_engine.core.dataloader import get_week_dataset, load_manifest
from tick_engine.core.metrics import compute_metrics
from edge_lab.london_breakout import LondonBreakoutStrategy


EXPERIMENT_MODES = [
    {
        "mode": "v0_naive",
        "title": "V0: Naive Breakout",
        "hypothesis": "Baseline: Enter Long/Short immediately on Asian High/Low touch with fixed 15p SL / 30p TP."
    },
    {
        "mode": "v1_range_filter",
        "title": "V1: Range Compression",
        "hypothesis": "Filter out exhausted days: Only trade when Asian range is between 8p and 25p."
    },
    {
        "mode": "v2_candle_close",
        "title": "V2: Candle Confirmation",
        "hypothesis": "Eliminate wick traps: Require confirmed M15 candle close outside the Asian boundary."
    },
    {
        "mode": "v3_trend_regime",
        "title": "V3: Macro Regime Bias",
        "hypothesis": "Trend alignment: Only buy if above Daily Open & 50 EMA; only sell if below."
    },
    {
        "mode": "v4_optimized",
        "title": "V4: Dynamic Risk & Exit",
        "hypothesis": "Structural SL at Asian Midpoint, 1.8R TP, force close at 17:00 UTC London close."
    },
    {
        "mode": "v5_sweep_fade",
        "title": "V5: Liquidity Sweep Fade",
        "hypothesis": "Fade the breakout: Sell Asian high sweeps that fail to hold; buy Asian low sweeps."
    },
    {
        "mode": "v6_sweep_trend",
        "title": "V6: Trend-Aligned Sweep Fade",
        "hypothesis": "Institutional alignment: Fade high sweeps in downtrends; fade low sweeps in uptrends."
    }
]


def run_experiments(weeks: List[int], raw_path: Path, proc_path: Path) -> pd.DataFrame:
    print("\n" + "=" * 90)
    print("      EDGE LAB: SYSTEMATIC STRATEGY ITERATION & EDGE DISCOVERY HARNESS      ".center(90))
    print(f"      Strategy Archetype: London Session Breakout (M15 / Tick Execution)      ".center(90))
    print(f"      Dataset: {len(weeks)} Weeks ({min(weeks)} to {max(weeks)}) | Spread Friction: 0.8 pips      ".center(90))
    print("=" * 90 + "\n")

    engine = BacktestEngine(pip_size=0.0001)

    results = []

    for exp in EXPERIMENT_MODES:
        mode_key = exp["mode"]
        title = exp["title"]
        hypo = exp["hypothesis"]

        print(f"[*] Testing {title}...")
        print(f"    Hypothesis: {hypo}")

        strat = LondonBreakoutStrategy(mode=mode_key)
        all_trades_list = []
        t0 = time.perf_counter()

        for w in weeks:
            ticks, _ = get_week_dataset(w, raw_dir=raw_path, processed_dir=proc_path, auto_convert=True)
            trades = engine.run(ticks, strat)
            if len(trades) > 0:
                all_trades_list.append(trades)

        elapsed = time.perf_counter() - t0

        if all_trades_list:
            all_trades = np.concatenate(all_trades_list)
        else:
            all_trades = np.empty(0, dtype=TRADE_DTYPE)

        m = compute_metrics(all_trades)

        results.append({
            "Version": title,
            "Mode": mode_key,
            "Trades": m["total_trades"],
            "WinRate%": m["win_rate_pct"],
            "GrossProfit": m["gross_profit_pips"],
            "GrossLoss": m["gross_loss_pips"],
            "NetPips": m["net_pips"],
            "MaxDD": m["max_drawdown_pips"],
            "ProfitFactor": m["profit_factor"],
            "AvgTrade": m["avg_trade_pips"],
            "ElapsedSec": elapsed,
            "Hypothesis": hypo
        })

        pf_disp = f"{m['profit_factor']:.2f}" if isinstance(m['profit_factor'], (int, float)) and m['profit_factor'] < 999 else "-"
        print(f"    -> Trades: {m['total_trades']} | Win Rate: {m['win_rate_pct']:.1f}% | Net Pips: {m['net_pips']:+.1f}p | PF: {pf_disp} | DD: {m['max_drawdown_pips']:.1f}p ({elapsed:.2f}s)\n")

    df_res = pd.DataFrame(results)

    # Print Progression Table
    print("=" * 95)
    print("                      EDGE PROGRESSION COMPARISON SCORECARD                      ".center(95))
    print("=" * 95)
    print(f"{'Version':<25} | {'Trades':<6} | {'Win Rate':<8} | {'Net Pips':<10} | {'Max DD':<8} | {'PF':<6} | {'Avg Trade':<9}")
    print("-" * 95)

    for _, r in df_res.iterrows():
        pf_val = r['ProfitFactor']
        pf_s = f"{pf_val:.2f}" if isinstance(pf_val, (int, float)) and pf_val < 999 else "-"
        print(f"{r['Version']:<25} | {r['Trades']:>6d} | {r['WinRate%']:>7.1f}% | {r['NetPips']:>+9.1f}p | {r['MaxDD']:>7.1f}p | {pf_s:>6} | {r['AvgTrade']:>+8.2f}p")

    print("=" * 95 + "\n")

    # Export CSV & Markdown
    csv_out = Path("edge_lab") / "experiment_results.csv"
    df_res.to_csv(csv_out, index=False)
    print(f"[+] Saved experiment results to {csv_out}")

    return df_res


def main():
    parser = argparse.ArgumentParser(description="Edge Lab Experiment Harness")
    parser.add_argument("--weeks", type=str, default="1-12", help="Weekly range to test (e.g. '1-4', '1-12', 'all')")
    args = parser.parse_args()

    raw_dir = ROOT_DIR / "eurusd_weekly_data"
    proc_dir = ROOT_DIR / "tick_engine" / "data" / "processed"

    if args.weeks.lower() == "all":
        manifest = load_manifest(raw_dir)
        weeks = sorted(list(manifest.keys())) if manifest else list(range(1, 54))
    elif "-" in args.weeks:
        s, e = args.weeks.split("-")
        weeks = list(range(int(s), int(e) + 1))
    else:
        weeks = [int(w) for w in args.weeks.split(",") if w.strip()]

    run_experiments(weeks, raw_dir, proc_dir)


if __name__ == "__main__":
    main()
