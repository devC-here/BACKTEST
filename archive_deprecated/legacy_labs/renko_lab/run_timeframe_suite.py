"""
Timeframe Matrix Runner for Renko SMA + OBV Strategy.
Tests varying base timeframes and higher-timeframe macro filters.
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

def run_timeframe_experiments(weeks_list=list(range(1, 13))):
    engine = BacktestEngine(pip_size=0.0001)

    print(f"Loading {len(weeks_list)} weeks of tick data for Timeframe Matrix...")
    weeks_data = [get_week_dataset(w)[0] for w in weeks_list]
    total_ticks = sum(len(t) for t in weeks_data)
    print(f"Loaded {total_ticks:,} ticks.\n")

    experiments = [
        # 1. Base Sampling Timeframe variations (with 7p brick)
        {"base": "tick",  "macro": "none",  "brick": 7.0, "label": "1. Pure Tick Renko (7p)"},
        {"base": "1min",  "macro": "none",  "brick": 7.0, "label": "2. M1-Sampled Renko (7p)"},
        {"base": "5min",  "macro": "none",  "brick": 7.0, "label": "3. M5-Sampled Renko (7p)"},
        {"base": "15min", "macro": "none",  "brick": 7.0, "label": "4. M15-Sampled Renko (7p)"},

        # 2. Higher Timeframe Macro Trend Filtering
        {"base": "tick",  "macro": "15min", "brick": 7.0, "label": "5. Tick Renko + M15 Macro Filter"},
        {"base": "tick",  "macro": "1h",    "brick": 7.0, "label": "6. Tick Renko + H1 Macro Filter"},
        {"base": "tick",  "macro": "4h",    "brick": 7.0, "label": "7. Tick Renko + H4 Macro Filter"},

        # 3. Timeframe-Calibrated Volatility Bricks
        {"base": "tick",  "macro": "1h",    "brick": 5.0,  "label": "8. M5-Scalp Brick (5p) + H1 Filter"},
        {"base": "tick",  "macro": "1h",    "brick": 8.0,  "label": "9. M15-Intraday Brick (8p) + H1 Filter"},
        {"base": "tick",  "macro": "1h",    "brick": 12.0, "label": "10. H1-Swing Brick (12p) + H1 Filter"},
        {"base": "tick",  "macro": "4h",    "brick": 15.0, "label": "11. H4-Major Brick (15p) + H4 Filter"},
    ]

    print("=" * 92)
    print("                RENKO TIMEFRAME VARIATION BENCHMARK (12 WEEKS)                ")
    print("=" * 92)
    print(f"{'Timeframe Configuration':42s} | {'Trades':6s} | {'Win Rate':8s} | {'Net Pips':9s} | {'PF':4s} | {'Max DD':7s} | {'Avg':6s}")
    print("-" * 92)

    for exp in experiments:
        label = exp["label"]
        strat = MultiTimeframeRenkoStrategy(
            base_timeframe=exp["base"],
            macro_timeframe=exp["macro"],
            brick_size_pips=exp["brick"],
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
            print(f"{label:42s} | {len(trades_arr):6d} | {m['win_rate_pct']:7.1f}% | {m['net_pips']:+8.1f}p | {m['profit_factor']:4.2f} | {m['max_drawdown_pips']:6.1f}p | {m['avg_trade_pips']:+5.2f}p")
        else:
            print(f"{label:42s} | 0 trades")

    print("=" * 92)

if __name__ == "__main__":
    run_timeframe_experiments()
