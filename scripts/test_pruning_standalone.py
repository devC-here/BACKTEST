#!/usr/bin/env python3
import json
import pandas as pd
import numpy as np

report = json.load(open('results/individual_strategy_independent_audit.json'))
df = pd.DataFrame(report)

# Non-negotiable standalone criteria:
# 1. holdout_trades >= 15
# 2. holdout_pf >= 1.15
# 3. pips_ex_top3 > 0.0
# 4. discovery_net_pips > 0.0

pass_mask = (
    (df['holdout_trades'] >= 15) &
    (df['holdout_pf'] >= 1.15) &
    (df['pips_ex_top3'] > 0.0) &
    (df['discovery_net_pips'] > 0.0)
)

passing = df[pass_mask]
failing = df[~pass_mask]

print(f"Total strategies: {len(df)}")
print(f"Passing strict standalone criteria: {len(passing)}")
print(f"Rejected (bad/fragile/undersampled): {len(failing)}")

print("\nRejected models summary:")
for _, r in failing.iterrows():
    flags_str = "; ".join(r['flags'])
    print(f"  - [{r['symbol']}] {r['name']}: {r['holdout_net_pips']}p, {r['holdout_trades']} tr, PF {r['holdout_pf']}, ExTop3 {r['pips_ex_top3']}p, Disc {r['discovery_net_pips']}p (PF {r['discovery_pf']})")
    print(f"    Defects: {flags_str}")

print("\n" + "=" * 70)
print("PORTFOLIO IMPACT OF REMOVING REJECTED STRATEGIES:")
print("=" * 70)
orig_pips = df['holdout_net_pips'].sum()
orig_trades = df['holdout_trades'].sum()
clean_pips = passing['holdout_net_pips'].sum()
clean_trades = passing['holdout_trades'].sum()
print(f"Original 78 Fleet:  {orig_pips:.1f} net pips | {orig_trades} trades | Avg Exp: {orig_pips/orig_trades:.2f}p | Mean PF: {df['holdout_pf'].mean():.2f}")
print(f"Purified {len(passing)} Fleet:  {clean_pips:.1f} net pips | {clean_trades} trades | Avg Exp: {clean_pips/clean_trades:.2f}p | Mean PF: {passing['holdout_pf'].mean():.2f}")
print(f"Pip retention: {clean_pips/orig_pips*100:.1f}%")
print(f"Trades eliminated: {orig_trades - clean_trades} (low edge / noise eliminated)")

