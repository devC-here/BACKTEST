import json
from pathlib import Path
import pandas as pd
import numpy as np

manifest_dir = Path("results/certified_strategy_manifests")

# Let's inspect the certified strategies
csv_files = sorted(list(manifest_dir.glob("*_trades.csv")))
print(f"Total trade logs found: {len(csv_files)}")

all_trades = []
strat_pnl = {}

for f in csv_files:
    if "strategy_01" in f.name or "strategy_14" in f.name:
        continue # Skip unprofitable slots
    df = pd.read_csv(f)
    if len(df) == 0:
        continue
    df['strategy'] = f.stem.replace("_trades", "")
    all_trades.append(df)
    strat_pnl[df['strategy'].iloc[0]] = {
        "trades": len(df),
        "net_pips": round(float(df['pips'].sum()), 1),
        "pf": round(float(df[df['pips'] > 0]['pips'].sum() / abs(df[df['pips'] < 0]['pips'].sum())), 2) if len(df[df['pips'] < 0]) > 0 else 99.0,
        "wr": round(float(len(df[df['pips'] > 0]) / len(df) * 100), 1)
    }

print("\n--- INDIVIDUAL CERTIFIED STRATEGY HOLDOUT AUDIT ---")
for k, v in sorted(strat_pnl.items(), key=lambda x: x[1]['net_pips'], reverse=True):
    print(f"{k:<35} | Trades: {v['trades']:>3} | Net: {v['net_pips']:>+6.1f}p | PF: {v['pf']:>4.2f} | WR: {v['wr']:>4.1f}%")

if all_trades:
    comb = pd.concat(all_trades, ignore_index=True)
    comb['entry_time'] = comb['entry_time'].astype(np.int64)
    comb = comb.sort_values('entry_time').reset_index(drop=True)
    
    total_trades = len(comb)
    net_pips = float(comb['pips'].sum())
    wins = comb[comb['pips'] > 0]['pips'].sum()
    losses = abs(comb[comb['pips'] < 0]['pips'].sum())
    pf = round(float(wins / losses), 2)
    wr = round(float(len(comb[comb['pips'] > 0]) / total_trades * 100), 1)
    
    # Running drawdown
    cum_pips = comb['pips'].cumsum().values
    peak = np.maximum.accumulate(cum_pips)
    dd = peak - cum_pips
    max_dd = round(float(np.max(dd)), 1)
    
    print("\n--- COMBINED CERTIFIED HOLDOUT PORTFOLIO (14 INVARIANT STRATEGIES) ---")
    print(f"Total Portfolio Trades:   {total_trades}")
    print(f"Total Holdout Net Pips:   {net_pips:+.1f} pips")
    print(f"Portfolio Profit Factor:  {pf:.2f}")
    print(f"Portfolio Win Rate:       {wr:.1f}%")
    print(f"Portfolio Max Drawdown:   {max_dd:.1f} pips")
    print(f"Return / Drawdown Ratio:  {round(net_pips / max_dd, 2) if max_dd > 0 else 0.0}")

