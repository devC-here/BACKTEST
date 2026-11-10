"""
Annual 53-Week Benchmark of Champion Multi-Timeframe Renko Strategies.
Evaluates across 18,026,436 authentic Dukascopy ticks with tick spread.
"""

import sys
from pathlib import Path
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from renko_lab.mtf_renko_strategy import MultiTimeframeRenkoStrategy
from tick_engine.core.engine import BacktestEngine
from tick_engine.core.dataloader import get_week_dataset
from tick_engine.core.metrics import compute_metrics, print_scorecard

def run_annual_mtf():
    engine = BacktestEngine(pip_size=0.0001)

    print("Loading 53 weeks of tick data for Annual Multi-Timeframe Benchmark...")
    weeks_data = [get_week_dataset(w)[0] for w in range(1, 54)]
    print("Ticks loaded. Evaluating champion timeframe models...\n")

    champions = [
        {"base": "15min", "macro": "none", "brick": 7.0,  "label": "A. M15-Sampled Renko (7p Brick, RR 1.8)"},
        {"base": "tick",  "macro": "1h",   "brick": 8.0,  "label": "B. Tick Renko (8p) + H1 Macro Filter (RR 1.8)"},
        {"base": "tick",  "macro": "4h",   "brick": 15.0, "label": "C. H4-Major Swing Renko (15p) + H4 Filter (RR 1.8)"},
    ]

    print("=" * 92)
    print("          ANNUAL 53-WEEK MULTI-TIMEFRAME RENKO CHAMPIONSHIP (18M TICKS)          ")
    print("=" * 92)
    print(f"{'Timeframe Configuration':46s} | {'Trades':6s} | {'Win Rate':8s} | {'Net Pips':9s} | {'PF':4s} | {'Max DD':7s} | {'Avg':6s}")
    print("-" * 92)

    for cfg in champions:
        label = cfg["label"]
        strat = MultiTimeframeRenkoStrategy(
            base_timeframe=cfg["base"],
            macro_timeframe=cfg["macro"],
            brick_size_pips=cfg["brick"],
            target_rr=1.8
        )

        all_trades = []
        for ticks in weeks_data:
            trades = engine.run(ticks, strat)
            if len(trades) > 0:
                all_trades.append(trades)

        if len(all_trades) > 0:
            trades_arr = np.concatenate(all_trades)
            m = compute_metrics(trades_arr)
            print(f"{label:46s} | {len(trades_arr):6d} | {m['win_rate_pct']:7.1f}% | {m['net_pips']:+8.1f}p | {m['profit_factor']:4.2f} | {m['max_drawdown_pips']:6.1f}p | {m['avg_trade_pips']:+5.2f}p")
        else:
            print(f"{label:46s} | 0 trades")

    print("=" * 92)

if __name__ == "__main__":
    run_annual_mtf()
