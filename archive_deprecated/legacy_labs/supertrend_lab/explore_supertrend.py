"""
Supertrend Exploration Benchmark.
Explores:
1. Pure Reversal (Hold until opposite Supertrend + Strong Candle)
2. Supertrend Flip Exit
3. Dynamic Supertrend Trailing Stop
4. Multiple timeframes (M5, M15, M30, H1)
5. Supertrend multipliers (2.0, 2.5, 3.0) and ATR periods (7, 10, 14)
"""

import sys
from pathlib import Path
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from supertrend_lab.supertrend_strategy import SupertrendStrongCandleStrategy
from tick_engine.core.engine import BacktestEngine
from tick_engine.core.dataloader import get_week_dataset
from tick_engine.core.metrics import compute_metrics, print_scorecard

def run_exploration():
    engine = BacktestEngine(pip_size=0.0001)

    print("Loading 15 weeks of tick data for Supertrend Exploration...")
    weeks = [get_week_dataset(w)[0] for w in range(1, 16)]
    total_ticks = sum(len(w) for w in weeks)
    print(f"Loaded 15 weeks ({total_ticks:,} ticks).\n")

    experiments = [
        # 1. Pure Reversal on M5 (User's Exact Specification)
        {"timeframe": "5min",  "atr_period": 10, "multiplier": 3.0, "exit_mode": "pure_reversal",  "label": "1. M5 ST(10, 3.0) Pure Reversal"},
        {"timeframe": "5min",  "atr_period": 10, "multiplier": 2.0, "exit_mode": "pure_reversal",  "label": "2. M5 ST(10, 2.0) Pure Reversal"},
        {"timeframe": "5min",  "atr_period": 7,  "multiplier": 3.0, "exit_mode": "pure_reversal",  "label": "3. M5 ST(7, 3.0) Pure Reversal"},

        # 2. Higher Timeframe Pure Reversal (M15, M30, H1)
        {"timeframe": "15min", "atr_period": 10, "multiplier": 3.0, "exit_mode": "pure_reversal",  "label": "4. M15 ST(10, 3.0) Pure Reversal"},
        {"timeframe": "15min", "atr_period": 10, "multiplier": 2.0, "exit_mode": "pure_reversal",  "label": "5. M15 ST(10, 2.0) Pure Reversal"},
        {"timeframe": "30min", "atr_period": 10, "multiplier": 3.0, "exit_mode": "pure_reversal",  "label": "6. M30 ST(10, 3.0) Pure Reversal"},
        {"timeframe": "1h",    "atr_period": 10, "multiplier": 3.0, "exit_mode": "pure_reversal",  "label": "7. H1 ST(10, 3.0) Pure Reversal"},

        # 3. Supertrend Flip Exit (Exits as soon as Supertrend flips direction)
        {"timeframe": "5min",  "atr_period": 10, "multiplier": 3.0, "exit_mode": "st_flip",        "label": "8. M5 ST(10, 3.0) Exit on ST Flip"},
        {"timeframe": "15min", "atr_period": 10, "multiplier": 3.0, "exit_mode": "st_flip",        "label": "9. M15 ST(10, 3.0) Exit on ST Flip"},
        {"timeframe": "15min", "atr_period": 10, "multiplier": 2.0, "exit_mode": "st_flip",        "label": "10. M15 ST(10, 2.0) Exit on ST Flip"},

        # 4. Supertrend Line Trailing SL
        {"timeframe": "15min", "atr_period": 10, "multiplier": 2.5, "exit_mode": "st_trailing_sl", "label": "11. M15 ST(10, 2.5) Trailing SL"},
        {"timeframe": "30min", "atr_period": 10, "multiplier": 2.5, "exit_mode": "st_trailing_sl", "label": "12. M30 ST(10, 2.5) Trailing SL"},
    ]

    print("=" * 96)
    print("                     SUPERTREND + STRONG CANDLE EXPLORATION (15 WEEKS)                    ")
    print("=" * 96)
    print(f"{'Strategy Configuration':46s} | {'Trades':6s} | {'Win Rate':8s} | {'Net Pips':9s} | {'PF':4s} | {'Max DD':7s} | {'Avg':6s}")
    print("-" * 96)

    for exp in experiments:
        label = exp.pop("label")
        strat = SupertrendStrongCandleStrategy(**exp)

        trades_list = []
        for t in weeks:
            tr = engine.run(t, strat)
            if len(tr) > 0:
                trades_list.append(tr)

        if len(trades_list) > 0:
            all_tr = np.concatenate(trades_list)
            m = compute_metrics(all_tr)
            print(f"{label:46s} | {len(all_tr):6d} | {m['win_rate_pct']:7.1f}% | {m['net_pips']:+8.1f}p | {m['profit_factor']:4.2f} | {m['max_drawdown_pips']:6.1f}p | {m['avg_trade_pips']:+5.2f}p")
        else:
            print(f"{label:46s} | 0 trades")

    print("=" * 96)

if __name__ == "__main__":
    run_exploration()
