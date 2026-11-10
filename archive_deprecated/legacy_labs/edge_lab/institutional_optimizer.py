import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tick_engine.core.engine import BacktestEngine
from tick_engine.core.dataloader import get_week_dataset
from tick_engine.core.metrics import compute_metrics
from edge_lab.session_suite import MultiSessionStrategy

def evaluate_with_live_dilution(trades: np.ndarray, dilution_pct: float = 0.25):
    """
    Applies a 25% institutional execution penalty:
    - Reduces all winning pips by dilution_pct (e.g. slippage on exit, partial fills)
    - Adds an extra 0.5 pip adverse execution penalty to all losing trades
    """
    if len(trades) == 0:
        return None

    diluted_trades = trades.copy()
    for i in range(len(diluted_trades)):
        p = diluted_trades[i]['pips']
        if p > 0:
            diluted_trades[i]['pips'] = p * (1.0 - dilution_pct)
        else:
            diluted_trades[i]['pips'] = p - 0.5  # Extra 0.5p slippage on stop-outs

    m_raw = compute_metrics(trades)
    m_diluted = compute_metrics(diluted_trades)

    # 10x leverage accounting on $10,000 initial capital
    # 1 pip = $10.00
    pips = diluted_trades['pips']
    balances = [10000.0]
    for p in pips:
        balances.append(balances[-1] + p * 10.0)
    balances = np.array(balances)
    peaks = np.maximum.accumulate(balances)
    dd_pct = np.max((peaks - balances) / peaks) * 100.0
    net_return_pct = (balances[-1] - 10000.0) / 10000.0 * 100.0

    # Sharpe ratio
    returns = (pips * 10.0) / 10000.0
    sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(len(returns)) if np.std(returns) > 0 else 0.0

    return {
        "trades": len(trades),
        "raw_net_pips": m_raw['net_pips'],
        "raw_pf": m_raw['profit_factor'],
        "raw_dd_pips": m_raw['max_drawdown_pips'],
        "diluted_net_pips": m_diluted['net_pips'],
        "diluted_pf": m_diluted['profit_factor'],
        "diluted_dd_pips": m_diluted['max_drawdown_pips'],
        "diluted_dd_pct": dd_pct,
        "diluted_return_pct": net_return_pct,
        "diluted_sharpe": sharpe,
        "diluted_avg_pips": m_diluted['avg_trade_pips']
    }

def main():
    engine = BacktestEngine(pip_size=0.0001)
    weeks_data = [get_week_dataset(w)[0] for w in range(1, 54)]

    # Test various target RR and session configurations
    grid = [
        {"session_mode": "london_only", "target_rr_london": 1.8},
        {"session_mode": "london_only", "target_rr_london": 2.0},
        {"session_mode": "london_only", "target_rr_london": 2.2},
        {"session_mode": "london_only", "target_rr_london": 2.5},
        {"session_mode": "dual_session", "target_rr_london": 2.2, "target_rr_ny": 1.8},
        {"session_mode": "dual_session", "target_rr_london": 2.5, "target_rr_ny": 2.2},
    ]

    print("=" * 95)
    print("   INSTITUTIONAL AUDIT UNDER 25% LIVE EXECUTION DILUTION & SLIPPAGE (53 WEEKS)   ")
    print("=" * 95)
    print(f"{'Configuration':32s} | {'Trades':6s} | {'Diluted Net':11s} | {'Diluted PF':10s} | {'Diluted DD %':12s} | {'Sharpe':6s} | {'Pass <10% DD?':13s}")
    print("-" * 95)

    for cfg in grid:
        strat = MultiSessionStrategy(**cfg)
        all_trades = []
        for ticks in weeks_data:
            trades = engine.run(ticks, strat)
            if len(trades) > 0:
                all_trades.append(trades)

        trades_arr = np.concatenate(all_trades)
        trades_arr = trades_arr[np.argsort(trades_arr['entry_time'])]
        res = evaluate_with_live_dilution(trades_arr, dilution_pct=0.25)

        lbl = f"{cfg['session_mode']} (RR_L={cfg['target_rr_london']})"
        passed = "YES (PASS)" if res['diluted_dd_pct'] < 10.0 and res['diluted_net_pips'] > 0 else "NO (FAIL)"
        print(f"{lbl:32s} | {res['trades']:6d} | {res['diluted_net_pips']:+9.1f}p | {res['diluted_pf']:9.2f}  | {res['diluted_dd_pct']:10.2f}% | {res['diluted_sharpe']:5.2f}  | {passed}")

    print("=" * 95)

if __name__ == "__main__":
    main()
