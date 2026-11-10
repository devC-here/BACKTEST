import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from edge_lab.session_suite import MultiSessionStrategy
from tick_engine.core.engine import BacktestEngine
from tick_engine.core.dataloader import get_week_dataset
from tick_engine.core.metrics import compute_metrics

def simulate_account_growth(
    trades: np.ndarray,
    initial_capital: float = 10000.0,
    leverage: float = 10.0,
    compound: bool = True
):
    """
    Simulates portfolio equity curve under FX leverage.
    - initial_capital: e.g. $10,000
    - leverage: 10x (notional = leverage * balance)
    - 1 pip on EURUSD = 0.0001
    - Dollar PnL = Notional * (pips * 0.0001) = (leverage * Balance) * pips * 0.0001
    """
    n = len(trades)
    balances = np.zeros(n + 1, dtype=np.float64)
    balances[0] = initial_capital
    trade_pnls = np.zeros(n, dtype=np.float64)
    trade_pcts = np.zeros(n, dtype=np.float64)

    cur_bal = initial_capital
    for i in range(n):
        pips = trades[i]['pips']
        # Sizing: 10x leverage
        if compound:
            notional = cur_bal * leverage
        else:
            notional = initial_capital * leverage
        
        # PnL in USD
        pnl = notional * (pips * 0.0001)
        cur_bal += pnl
        balances[i + 1] = cur_bal
        trade_pnls[i] = pnl
        trade_pcts[i] = pnl / (balances[i]) * 100.0

    # Drawdown calculation
    peaks = np.maximum.accumulate(balances)
    drawdowns = (peaks - balances) / peaks * 100.0
    dollar_dds = peaks - balances
    max_dd_pct = float(np.max(drawdowns))
    max_dd_dollar = float(np.max(dollar_dds))

    total_return_dollar = cur_bal - initial_capital
    total_return_pct = (cur_bal - initial_capital) / initial_capital * 100.0

    # Sharpe ratio (annualized based on trades)
    returns = trade_pcts / 100.0
    if len(returns) > 1 and np.std(returns) > 0:
        # Approximate ~191 trades in a 252-day trading year
        trades_per_year = len(returns)
        sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(trades_per_year)
    else:
        sharpe = 0.0

    return {
        "initial_capital": initial_capital,
        "final_capital": cur_bal,
        "net_profit_dollar": total_return_dollar,
        "return_pct": total_return_pct,
        "max_drawdown_dollar": max_dd_dollar,
        "max_drawdown_pct": max_dd_pct,
        "sharpe_ratio": sharpe,
        "profit_factor": float(np.sum(trade_pnls[trade_pnls > 0]) / abs(np.sum(trade_pnls[trade_pnls < 0]))) if np.sum(trade_pnls < 0) != 0 else 0.0,
        "balances": balances,
        "trade_pnls": trade_pnls
    }

def main():
    engine = BacktestEngine(pip_size=0.0001)
    weeks_data = [get_week_dataset(w)[0] for w in range(1, 54)]

    models = [
        ("london_only", "Model A: London Morning Specialist (Asian Sweeps)"),
        ("new_york_only", "Model B: New York Open Specialist (London Sweeps)"),
        ("dual_session", "Model C: Unified Dual-Session Portfolio (Both Sessions)"),
    ]

    initial_cap = 10000.0
    leverage = 10.0

    print("=" * 85)
    print(f"      CAPITAL GROWTH & 10X LEVERAGE PERFORMANCE AUDIT (1 FULL YEAR)      ")
    print(f"      Initial Balance: ${initial_cap:,.2f}  |  Leverage: {leverage:.0f}x (1.0 Lot per $10k)      ")
    print("=" * 85)

    for mode_id, label in models:
        strat = MultiSessionStrategy(session_mode=mode_id)
        all_trades = []
        for ticks in weeks_data:
            trades = engine.run(ticks, strat)
            if len(trades) > 0:
                all_trades.append(trades)

        trades_arr = np.concatenate(all_trades)
        trades_arr = trades_arr[np.argsort(trades_arr['entry_time'])]

        # 1. Compounding 10x leverage
        res_comp = simulate_account_growth(trades_arr, initial_cap, leverage, compound=True)
        # 2. Fixed 10x leverage (no compounding)
        res_fixed = simulate_account_growth(trades_arr, initial_cap, leverage, compound=False)

        print(f"\n>>> {label}")
        print(f"    Trades: {len(trades_arr)} | Total Pips: {np.sum(trades_arr['pips']):+.1f} pips")
        print(f"    [Compounding 10x Leverage]:")
        print(f"      * Final Balance     : ${res_comp['final_capital']:,.2f}")
        print(f"      * Net Profit ($ / %): +${res_comp['net_profit_dollar']:,.2f} ({res_comp['return_pct']:+.2f}%)")
        print(f"      * Max Drawdown ($/%): -${res_comp['max_drawdown_dollar']:,.2f} ({res_comp['max_drawdown_pct']:.2f}%)")
        print(f"      * Return / Max DD   : {res_comp['return_pct'] / res_comp['max_drawdown_pct']:.2f}x")
        print(f"      * Sharpe Ratio      : {res_comp['sharpe_ratio']:.2f}")
        print(f"      * Profit Factor     : {res_comp['profit_factor']:.2f}")

        print(f"    [Fixed 10x Leverage - No Compounding ($10/pip)]: ")
        print(f"      * Final Balance     : ${res_fixed['final_capital']:,.2f}")
        print(f"      * Net Profit ($ / %): +${res_fixed['net_profit_dollar']:,.2f} ({res_fixed['return_pct']:+.2f}%)")
        print(f"      * Max Drawdown ($/%): -${res_fixed['max_drawdown_dollar']:,.2f} ({res_fixed['max_drawdown_pct']:.2f}%)")

    print("\n" + "=" * 85)

if __name__ == "__main__":
    main()
