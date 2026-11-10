"""
Full 53-Week Annual Simulation of Renko SMA + OBV Pullback Strategy.
Tests across 18,026,436 authentic Dukascopy ticks with real floating spread.
"""

import sys
from pathlib import Path
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from renko_lab.explore_renko_edge import AdvancedRenkoStrategy
from tick_engine.core.engine import BacktestEngine
from tick_engine.core.dataloader import get_week_dataset
from tick_engine.core.metrics import compute_metrics, print_scorecard

def run_annual_renko_benchmark():
    engine = BacktestEngine(pip_size=0.0001)

    print("Loading all 53 weeks of tick data for Annual Renko Benchmark...")
    weeks_data = [get_week_dataset(w)[0] for w in range(1, 54)]
    total_ticks = sum(len(t) for t in weeks_data)
    print(f"Loaded {len(weeks_data)} weeks ({total_ticks:,} ticks).\n")

    grid = [
        {"brick_size_pips": 7.0, "target_rr": 1.8, "label": "Renko 7p Brick (RR 1.8)"},
        {"brick_size_pips": 8.0, "target_rr": 2.0, "label": "Renko 8p Brick (RR 2.0)"},
        {"brick_size_pips": 8.0, "target_rr": 2.2, "label": "Renko 8p Brick (RR 2.2)"},
        {"brick_size_pips": 9.0, "target_rr": 2.0, "label": "Renko 9p Brick (RR 2.0)"},
        {"brick_size_pips": 10.0, "target_rr": 2.0, "label": "Renko 10p Brick (RR 2.0)"},
    ]

    print("=" * 86)
    print("           ANNUAL 53-WEEK RENKO SMA-10 + OBV STRATEGY BENCHMARK           ")
    print("=" * 86)
    print(f"{'Configuration':30s} | {'Trades':6s} | {'Win Rate':8s} | {'Net Pips':9s} | {'PF':4s} | {'Max DD':7s} | {'Avg':6s}")
    print("-" * 86)

    best_trades = None
    best_label = ""
    best_pips = -9999.0

    for cfg in grid:
        label = cfg.pop("label")
        strat = AdvancedRenkoStrategy(
            mode="pullback_bounce",
            session_filter=True,
            **cfg
        )

        all_trades = []
        for ticks in weeks_data:
            trades = engine.run(ticks, strat)
            if len(trades) > 0:
                all_trades.append(trades)

        if len(all_trades) > 0:
            trades_arr = np.concatenate(all_trades)
            m = compute_metrics(trades_arr)
            print(f"{label:30s} | {len(trades_arr):6d} | {m['win_rate_pct']:7.1f}% | {m['net_pips']:+8.1f}p | {m['profit_factor']:4.2f} | {m['max_drawdown_pips']:6.1f}p | {m['avg_trade_pips']:+5.2f}p")

            if m['net_pips'] > best_pips:
                best_pips = m['net_pips']
                best_trades = trades_arr
                best_label = label
        else:
            print(f"{label:30s} | 0 trades")

    print("=" * 86)

    if best_trades is not None:
        m_best = compute_metrics(best_trades)
        print("\n")
        print_scorecard(m_best, title=f"CHAMPION RENKO CONFIGURATION ({best_label})")

if __name__ == "__main__":
    run_annual_renko_benchmark()
