"""
Renko Strategy Test Runner.
Backtests the user's Renko SMA-10 + OBV strategy across multiple brick sizes
with tick-by-tick spread awareness.
"""

import sys
from pathlib import Path
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from renko_lab.renko_strategy import RenkoSMAOBVStrategy
from tick_engine.core.engine import BacktestEngine
from tick_engine.core.dataloader import get_week_dataset
from tick_engine.core.metrics import compute_metrics, print_scorecard

def run_renko_experiments(weeks_to_test=list(range(1, 5))):
    engine = BacktestEngine(pip_size=0.0001)

    print(f"Loading {len(weeks_to_test)} weeks of tick data for Renko evaluation...")
    weeks_data = [get_week_dataset(w)[0] for w in weeks_to_test]
    total_ticks = sum(len(t) for t in weeks_data)
    print(f"Loaded {total_ticks:,} ticks.\n")

    brick_sizes = [3.0, 5.0, 8.0, 10.0, 15.0]

    print("=" * 85)
    print("      RENKO 10-SMA + OBV STRATEGY BENCHMARK (SAMPLE EXPLORATION)      ")
    print("=" * 85)
    print(f"{'Brick Size':15s} | {'Trades':6s} | {'Win Rate':8s} | {'Net Pips':9s} | {'PF':4s} | {'Max DD':7s} | {'Avg':6s}")
    print("-" * 85)

    for bs in brick_sizes:
        # Default: 2 bricks SL, 3 bricks TP (1.5R target)
        strat = RenkoSMAOBVStrategy(
            brick_size_pips=bs,
            sma_period=10,
            sl_bricks=2.0,
            tp_bricks=3.0
        )

        all_trades = []
        for ticks in weeks_data:
            trades = engine.run(ticks, strat)
            if len(trades) > 0:
                all_trades.append(trades)

        if len(all_trades) > 0:
            trades_arr = np.concatenate(all_trades)
            m = compute_metrics(trades_arr)
            lbl = f"Renko {bs:.1f}p Brick"
            print(f"{lbl:15s} | {len(trades_arr):6d} | {m['win_rate_pct']:7.1f}% | {m['net_pips']:+8.1f}p | {m['profit_factor']:4.2f} | {m['max_drawdown_pips']:6.1f}p | {m['avg_trade_pips']:+5.2f}p")
        else:
            print(f"Renko {bs:.1f}p Brick | 0 trades")

    print("=" * 85)

if __name__ == "__main__":
    run_renko_experiments(weeks_to_test=[1, 2, 3, 4])
