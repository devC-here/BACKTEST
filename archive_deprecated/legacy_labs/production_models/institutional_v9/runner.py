"""
Standalone Execution Runner for Institutional Asymmetric Sweep (Model V9).
Allows running backtests, 10x leverage accounting, or live dilution stress tests at any time.

Usage:
  python runner.py                     # Runs full 53-week backtest + scorecard
  python runner.py --leverage 10       # Runs 10x leverage portfolio simulation
  python runner.py --dilution          # Runs 25% live execution dilution test
  python runner.py --week 1            # Runs on single week (Week 1)
"""

import sys
import argparse
from pathlib import Path
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from production_models.institutional_v9.strategy import InstitutionalAsymmetricSweepV9
from tick_engine.core.engine import BacktestEngine
from tick_engine.core.dataloader import get_week_dataset
from tick_engine.core.metrics import compute_metrics, print_scorecard

def run_simulation(args):
    strat = InstitutionalAsymmetricSweepV9.from_config()
    engine = BacktestEngine(pip_size=0.0001)

    if args.week:
        weeks = [args.week]
        print(f"Running Institutional V9 on Week {args.week}...")
    else:
        weeks = list(range(1, 54))
        print("Running Institutional V9 across all 53 weeks (1 full year)...")

    all_trades = []
    for w in weeks:
        ticks, _ = get_week_dataset(w)
        trades = engine.run(ticks, strat)
        if len(trades) > 0:
            all_trades.append(trades)

    if len(all_trades) == 0:
        print("No trades generated.")
        return

    trades_arr = np.concatenate(all_trades)
    trades_arr = trades_arr[np.argsort(trades_arr['entry_time'])]

    m = compute_metrics(trades_arr)
    print_scorecard(m, title="INSTITUTIONAL V9 AUDITED BACKTEST SCORECARD")

    # 10x Leverage Simulation
    if args.leverage:
        initial_cap = float(args.capital)
        lev = float(args.leverage)
        balances = [initial_cap]
        for t in trades_arr:
            pips = t['pips']
            notional = balances[-1] * lev
            pnl = notional * (pips * 0.0001)
            balances.append(balances[-1] + pnl)

        balances = np.array(balances)
        peaks = np.maximum.accumulate(balances)
        dds = (peaks - balances) / peaks * 100.0
        max_dd_pct = float(np.max(dds))
        net_prof = balances[-1] - initial_cap
        net_pct = (balances[-1] - initial_cap) / initial_cap * 100.0

        print("\n" + "=" * 65)
        print(f"       PORTFOLIO CAPITAL SIMULATION ({lev:.0f}x LEVERAGE)       ")
        print("=" * 65)
        print(f"  Initial Capital    : ${initial_cap:,.2f}")
        print(f"  Final Capital      : ${balances[-1]:,.2f}")
        print(f"  Net Profit ($ / %) : +${net_prof:,.2f} ({net_pct:+.2f}%)")
        print(f"  Max Drawdown (%)   : -{max_dd_pct:.2f}% (Limit: <10.0%)")
        print("=" * 65)

    # 25% Live Execution Dilution Test
    if args.dilution:
        diluted = trades_arr.copy()
        for i in range(len(diluted)):
            if diluted[i]['pips'] > 0:
                diluted[i]['pips'] *= 0.75  # 25% haircut on winners
            else:
                diluted[i]['pips'] -= 0.5   # 0.5 pip slippage on losers

        m_dil = compute_metrics(diluted)
        print("\n" + "=" * 65)
        print("       LIVE EXECUTION DILUTION STRESS TEST (25% HAIRCUT)      ")
        print("=" * 65)
        print(f"  Diluted Net Pips   : {m_dil['net_pips']:+.1f} pips")
        print(f"  Diluted PF         : {m_dil['profit_factor']:.2f}")
        print(f"  Diluted Max DD     : {m_dil['max_drawdown_pips']:.1f} pips ({m_dil['max_drawdown_pips'] * 0.1:.2f}% at 10x)")
        print(f"  Pass <10% DD Limit : {'YES (PASS)' if m_dil['max_drawdown_pips'] * 0.1 < 10.0 and m_dil['net_pips'] > 0 else 'NO (FAIL)'}")
        print("=" * 65)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Institutional V9 Strategy")
    parser.add_argument("--week", type=int, default=None, help="Specific week to backtest (1-53)")
    parser.add_argument("--leverage", type=float, default=10.0, help="Leverage multiplier (e.g. 10.0)")
    parser.add_argument("--capital", type=float, default=10000.0, help="Initial capital in USD")
    parser.add_argument("--dilution", action="store_true", help="Run 25 percent live dilution stress test")
    parser.set_defaults(dilution=True)
    args = parser.parse_args()
    run_simulation(args)
