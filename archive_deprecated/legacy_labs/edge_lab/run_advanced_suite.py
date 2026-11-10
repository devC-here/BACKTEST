#!/usr/bin/env python3
"""
Edge Lab: Advanced Generation Experiment Runner (V6 to V11).
Benchmarks second-generation quantitative models across full annual tick datasets.
"""

from __future__ import annotations
import sys
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tick_engine.core.engine import BacktestEngine, TRADE_DTYPE
from tick_engine.core.dataloader import get_week_dataset, load_manifest
from tick_engine.core.metrics import compute_metrics
from edge_lab.london_breakout import LondonBreakoutStrategy
from edge_lab.advanced_strategy import AdvancedLiquidityStrategy


ADVANCED_EXPERIMENTS = [
    {
        "mode": "v6_baseline",
        "title": "V6: Gen-1 Baseline",
        "cls": "london_breakout",
        "key": "v6_sweep_trend",
        "hypothesis": "Gen-1 Benchmark: London High/Low sweep fade aligned with 50 EMA."
    },
    {
        "mode": "v7_pinbar",
        "title": "V7: Pin-Bar Rejection",
        "cls": "advanced",
        "key": "v7_pinbar_sweep",
        "hypothesis": "Candle Quality: Require rejection shadow >= 35% of total candle range."
    },
    {
        "mode": "v8_dual_session",
        "title": "V8: Dual Session (LDN+NY)",
        "cls": "advanced",
        "key": "v8_dual_session",
        "hypothesis": "Opportunity Expansion: Trade both London Open (07-12) AND NY Open (12-16) sweeps."
    },
    {
        "mode": "v9_asymmetric",
        "title": "V9: Asymmetric Trend Bias",
        "cls": "advanced",
        "key": "v9_short_bias",
        "hypothesis": "Macro Asymmetry: Favor downtrend shorts; require strict 50/200 EMA for longs."
    },
    {
        "mode": "v10_atr_scaling",
        "title": "V10: Volatility ATR Scaling",
        "cls": "advanced",
        "key": "v10_atr_bracket",
        "hypothesis": "Dynamic Sizing: Scale SL to 1.2x ATR(14) with 2.2R dynamic take profit."
    },
    {
        "mode": "v11_hybrid_alpha",
        "title": "V11: The Hybrid Alpha Model",
        "cls": "advanced",
        "key": "v11_hybrid_alpha",
        "hypothesis": "Unified Alpha: Pin-bar filter + Dual-session + Asymmetric trend + ATR scaling."
    }
]


def run_suite(weeks: List[int], raw_path: Path, proc_path: Path) -> pd.DataFrame:
    print("\n" + "=" * 95)
    print("      EDGE LAB GEN-2: ADVANCED STATISTICAL EDGE OPTIMIZATION SUITE      ".center(95))
    print(f"      Dataset: {len(weeks)} Weeks ({min(weeks)} to {max(weeks)}) | Raw Dukascopy Tick Execution      ".center(95))
    print(f"      Spread Friction: 0.8 pips | Accounting: Strictly Pip-Denominated      ".center(95))
    print("=" * 95 + "\n")

    engine = BacktestEngine(pip_size=0.0001)
    results = []

    for exp in ADVANCED_EXPERIMENTS:
        title = exp["title"]
        hypo = exp["hypothesis"]
        cls_type = exp["cls"]
        key = exp["key"]

        print(f"[*] Testing {title}...")
        print(f"    Hypothesis: {hypo}")

        if cls_type == "london_breakout":
            strat = LondonBreakoutStrategy(mode=key)
        else:
            strat = AdvancedLiquidityStrategy(mode=key)

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

        pf_val = m["profit_factor"]
        pf_disp = f"{pf_val:.2f}" if isinstance(pf_val, (int, float)) and pf_val < 999 else "-"
        print(f"    -> Trades: {m['total_trades']:3d} | Win Rate: {m['win_rate_pct']:>5.1f}% | Net Pips: {m['net_pips']:>+7.1f}p | PF: {pf_disp:>5} | Max DD: {m['max_drawdown_pips']:>6.1f}p ({elapsed:.2f}s)\n")

        results.append({
            "Version": title,
            "Trades": m["total_trades"],
            "WinRate%": m["win_rate_pct"],
            "NetPips": m["net_pips"],
            "GrossProfit": m["gross_profit_pips"],
            "GrossLoss": m["gross_loss_pips"],
            "ProfitFactor": m["profit_factor"],
            "MaxDD": m["max_drawdown_pips"],
            "AvgTrade": m["avg_trade_pips"],
            "WinLossRatio": m["win_loss_ratio"],
            "LongTrades": m["long_trades"],
            "ShortTrades": m["short_trades"],
            "LongNetPips": m["long_net_pips"],
            "ShortNetPips": m["short_net_pips"],
            "ElapsedSec": elapsed,
            "Hypothesis": hypo
        })

    df_res = pd.DataFrame(results)

    # Print Summary Scorecard
    print("=" * 110)
    print("                         GEN-2 ADVANCED EDGE SCORECARD COMPARISON                         ".center(110))
    print("=" * 110)
    print(f"{'Model Version':<28} | {'Trades':<6} | {'Win Rate':<8} | {'Net Pips':<10} | {'PF':<6} | {'Max DD':<8} | {'Avg Trade':<9} | {'Win/Loss':<8}")
    print("-" * 110)

    for _, r in df_res.iterrows():
        pf_v = r['ProfitFactor']
        pf_s = f"{pf_v:.2f}" if isinstance(pf_v, (int, float)) and pf_v < 999 else "-"
        print(f"{r['Version']:<28} | {r['Trades']:>6d} | {r['WinRate%']:>7.1f}% | {r['NetPips']:>+9.1f}p | {pf_s:>6} | {r['MaxDD']:>7.1f}p | {r['AvgTrade']:>+8.2f}p | {r['WinLossRatio']:>7.2f}")

    print("=" * 110 + "\n")

    csv_out = Path("edge_lab") / "advanced_results.csv"
    df_res.to_csv(csv_out, index=False)
    print(f"[+] Exported detailed results matrix to {csv_out}")

    return df_res


def main():
    parser = argparse.ArgumentParser(description="Edge Lab Advanced Suite")
    parser.add_argument("--weeks", type=str, default="all", help="Weeks range to test (e.g. '1-12', 'all')")
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

    run_suite(weeks, raw_dir, proc_dir)


if __name__ == "__main__":
    main()
