import sys
from pathlib import Path
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tick_engine.core.engine import BacktestEngine, TRADE_DTYPE
from tick_engine.core.dataloader import get_week_dataset
from tick_engine.core.metrics import compute_metrics
from edge_lab.advanced_strategy import AdvancedLiquidityStrategy

strat = AdvancedLiquidityStrategy(mode="v9_short_bias")
engine = BacktestEngine(pip_size=0.0001)

all_trades_list = []
for w in range(1, 54):
    ticks, _ = get_week_dataset(w)
    trades = engine.run(ticks, strat)
    if len(trades) > 0:
        all_trades_list.append(trades)

all_trades = np.concatenate(all_trades_list)
m_raw = compute_metrics(all_trades)

print("=" * 70)
print("     V9 SPREAD & BROKER FRICTION STRESS TEST (1 FULL YEAR)     ")
print("=" * 70)
print(f"Total Trades: {len(all_trades)}")
print(f"1. Real Floating Dukascopy Spread (Paid on every trade):")
print(f"   -> Net Pips: {m_raw['net_pips']:+.1f} pips | Profit Factor: {m_raw['profit_factor']:.2f}")

# Stress Test: Add +0.5 pip extra markup per trade
extra_05 = all_trades.copy()
extra_05['pips'] -= 0.5
m_05 = compute_metrics(extra_05)
print(f"2. Real Spread + 0.5 Pip Extra Broker Markup (Total ~1.0 pip spread):")
print(f"   -> Net Pips: {m_05['net_pips']:+.1f} pips | Profit Factor: {m_05['profit_factor']:.2f}")

# Stress Test: Add +1.0 pip extra markup per trade
extra_10 = all_trades.copy()
extra_10['pips'] -= 1.0
m_10 = compute_metrics(extra_10)
print(f"3. Real Spread + 1.0 Pip Heavy Markup / Slippage (Total ~1.5 pip spread):")
print(f"   -> Net Pips: {m_10['net_pips']:+.1f} pips | Profit Factor: {m_10['profit_factor']:.2f}")
print("=" * 70)
