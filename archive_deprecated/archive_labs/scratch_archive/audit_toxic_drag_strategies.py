#!/usr/bin/env python3
"""
================================================================================
  AUDIT OF LOW PIP GAIN & HIGH DRAWDOWN DRAG STRATEGIES
================================================================================
Audits all 20 previously categorized "profitable" strategies and flags the ones
that suffer from the "Toxic Drag" syndrome:
  - Low 5-Year Net Pips (< 200 - 300 pips over 5 years)
  - Microscopic Expectancy (< +1.0 pip per trade)
  - Inverted / Disastrous Return-to-Drawdown Ratio (< 1.5 : 1)
  - Paying more in spread friction than keeping in net profit
================================================================================
"""

import sys
import pandas as pd
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

# Roster of all 20 strategies
data = [
    # id, name, trades_5y, win_rate, pf, net_pips, max_dd
    ("DISC34", "Friday Weekly Reversal Elite", 80, 41.2, 1.08, 35.2, 149.5),
    ("COMB28", "Unified Squeeze Ensemble Elite", 530, 36.8, 1.02, 60.6, 343.9),
    ("DISC31", "London Lunch Trap Fade Elite", 300, 40.0, 1.05, 80.4, 329.3),
    ("S01_M15", "M15 Displacement Retest Elite", 425, 27.5, 1.05, 165.0, 382.2),
    ("COMB26", "Asian Tight Coil Breakout Elite", 463, 32.4, 1.09, 264.9, 306.6),
    ("S34_HA", "Heikin Ashi Momentum Elite", 235, 30.2, 1.12, 505.3, 328.1),
    ("INST05", "Market Structure Shift Elite", 734, 40.2, 1.09, 422.6, 385.7),
    ("COMB27", "Squeeze Supertrend Hybrid Elite", 336, 33.6, 1.14, 311.7, 171.7),
    ("COMB12", "Bollinger Squeeze Elite", 329, 41.9, 1.18, 341.6, 162.6),
    ("DISC41", "Thursday Macro Trend Ignition", 133, 43.6, 1.30, 336.1, 130.8),
    ("DISC38", "London Midday MACD Acceleration", 193, 43.0, 1.28, 372.7, 147.7),
    ("DISC03", "NY Open Overlap Breakout Elite", 200, 45.0, 1.29, 405.3, 140.3),
    ("DISC42", "Macro Structure H4 Level + M30 Trigger", 48, 72.9, 3.97, 387.6, 20.4),
    ("DISC43", "Intraday Precision H1 Level + M5 Trigger", 292, 48.3, 1.47, 437.2, 68.7),
    ("DISC39", "Stochastic Deep Pullback in Trend", 438, 42.5, 1.22, 483.4, 202.2),
    ("S14_IB", "Inside Bar Compression Elite", 171, 36.8, 1.46, 717.3, 159.0),
    ("S10_EMA", "Dual EMA Macro Pullback", 176, 33.0, 1.25, 803.6, 171.8),
    ("DISC40", "RSI Bull/Bear Regime Range Shift", 611, 44.0, 1.31, 1112.8, 144.2),
    ("S66_44MA", "Bhanushali 44 MA Elite", 123, 40.7, 2.37, 1153.9, 221.0),
    ("HF_M15", "HighFrequency M15 Scalper Elite", 1903, 35.7, 1.13, 1755.9, 336.5)
]

records = []
for s_id, name, trades, wr, pf, net_p, max_dd in data:
    annual_pips = net_p / 5.0
    monthly_pips = annual_pips / 12.0
    expectancy = net_p / trades
    ret_dd = net_p / max_dd
    spread_paid = trades * 0.60  # ~0.6 pips friction per trade

    # Classification
    if net_p < 200 or ret_dd < 0.60:
        verdict = "TOXIC DRAG (Must Archive)"
    elif net_p < 450 and ret_dd < 1.20:
        verdict = "MARGINAL / UNSTABLE"
    elif ret_dd >= 2.0:
        verdict = "GENUINE HIGH-ALPHA"
    else:
        verdict = "MODERATE"

    records.append({
        'id': s_id,
        'name': name,
        'trades': trades,
        'net_5y': net_p,
        'annual_pips': round(annual_pips, 1),
        'monthly_pips': round(monthly_pips, 2),
        'expectancy': round(expectancy, 2),
        'max_dd': max_dd,
        'ret_dd': round(ret_dd, 2),
        'spread_paid': round(spread_paid, 1),
        'verdict': verdict
    })

df = pd.DataFrame(records)
# Sort by Return/DD ascending (worst at the top)
df.sort_values(by=['ret_dd', 'net_5y'], ascending=[True, True], inplace=True)

print("=" * 135)
print("               EXPOSING LOW PIP GAIN & HIGH DRAWDOWN 'TOXIC DRAG' STRATEGIES               ".center(135))
print("=" * 135)

headers = ["ID", "Strategy Name", "5Y Pips", "Pips/Yr", "Pips/Mo", "Exp/Trd", "Max DD", "Ret/DD", "Spread Paid", "Verdict"]
print(f"{headers[0]:<10} | {headers[1]:<36} | {headers[2]:<8} | {headers[3]:<7} | {headers[4]:<7} | {headers[5]:<7} | {headers[6]:<7} | {headers[7]:<6} | {headers[8]:<11} | {headers[9]}")
print("-" * 135)

for _, r in df.iterrows():
    p_yr = f"+{r['annual_pips']:.1f}p"
    p_mo = f"+{r['monthly_pips']:.2f}p"
    net_str = f"+{r['net_5y']:.1f}p"
    exp_str = f"+{r['expectancy']:.2f}p"
    dd_str = f"{r['max_dd']:.1f}p"
    spr_str = f"{r['spread_paid']:.1f}p"
    print(f"{r['id']:<10} | {r['name']:<36} | {net_str:>8} | {p_yr:>7} | {p_mo:>7} | {exp_str:>7} | {dd_str:>7} | {r['ret_dd']:>6.2f} | {spr_str:>11} | {r['verdict']}")

print("=" * 135)
