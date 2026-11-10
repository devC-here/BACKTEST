import sys
from pathlib import Path
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from edge_lab.session_suite import MultiSessionStrategy
from tick_engine.core.engine import BacktestEngine
from tick_engine.core.dataloader import get_week_dataset
from tick_engine.core.metrics import compute_metrics, print_scorecard

def main():
    engine = BacktestEngine(pip_size=0.0001)

    print("Loading 53 weeks of tick data...")
    weeks_data = [get_week_dataset(w)[0] for w in range(1, 54)]
    total_ticks = sum(len(t) for t in weeks_data)
    print(f"Loaded {len(weeks_data)} weeks ({total_ticks:,} ticks). Running Multi-Session Suite...\n")

    modes = [
        ("london_only", "Model A: London Morning Specialist (Asian Sweeps)"),
        ("new_york_only", "Model B: New York Open Specialist (London Sweeps)"),
        ("dual_session", "Model C: Unified Dual-Session Portfolio (Both Sessions)"),
    ]

    results = {}

    for mode_id, mode_label in modes:
        strat = MultiSessionStrategy(session_mode=mode_id)
        all_trades = []
        for ticks in weeks_data:
            trades = engine.run(ticks, strat)
            if len(trades) > 0:
                all_trades.append(trades)

        if len(all_trades) > 0:
            trades_arr = np.concatenate(all_trades)
            # Sort chronologically by entry_time
            trades_arr = trades_arr[np.argsort(trades_arr['entry_time'])]
            m = compute_metrics(trades_arr)
            results[mode_id] = (m, trades_arr, mode_label)
        else:
            results[mode_id] = (None, None, mode_label)

    print("=" * 86)
    print("                  MULTI-SESSION SPECIALIST ANNUAL BENCHMARK (53 WEEKS)                 ")
    print("=" * 86)
    print(f"{'Session Model':44s} | {'Trades':6s} | {'Win Rate':8s} | {'Net Pips':9s} | {'PF':4s} | {'Max DD':7s} | {'Avg':6s}")
    print("-" * 86)

    for mode_id, (m, trades_arr, mode_label) in results.items():
        if m is not None:
            print(f"{mode_label:44s} | {len(trades_arr):6d} | {m['win_rate_pct']:7.1f}% | {m['net_pips']:+8.1f}p | {m['profit_factor']:4.2f} | {m['max_drawdown_pips']:6.1f}p | {m['avg_trade_pips']:+5.2f}p")

    print("=" * 86)
    print("\n")

    # Detailed report for the Dual-Session Portfolio
    m_combo, trades_combo, label_combo = results["dual_session"]
    if m_combo:
        print_scorecard(m_combo, title="DUAL SESSION UNIFIED PORTFOLIO (1 FULL YEAR)")

if __name__ == "__main__":
    main()
