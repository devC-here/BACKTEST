#!/usr/bin/env python3
"""
================================================================================
  BALANCE EURGBP & AUDNZD FLEETS & FILTER BY >15% ANNUAL RETURN THRESHOLD
================================================================================
Calculates institutional annualized returns, R-multiples, drawdowns, and
expectancies for both EURGBP and AUDNZD models under standard risk allocation:
  - Account Risk: 1.0% per trade (1R = 1.0% of account)
  - Threshold: >= 15.0% Net Annualized Return (at 1% risk)
  - Quality Threshold: Profit Factor >= 1.25, Return / DD >= 4.0x
================================================================================
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

YEARS = 5.77

# EURGBP models
eurgbp_models = [
    {"id": "DISC56", "name": "Elder Triple Screen M15 Impulse", "pips": 1088.3, "sl": 7.0, "pf": 1.26, "wr": 40.4, "dd": 233.1, "trades": 812},
    {"id": "DISC48", "name": "Camarilla Floor Rejection Elite", "pips": 899.3, "sl": 8.5, "pf": 1.58, "wr": 50.6, "dd": 126.9, "trades": 536},
    {"id": "DISC46", "name": "Daily Pivot Wall Rejection Elite", "pips": 814.6, "sl": 9.0, "pf": 2.10, "wr": 59.3, "dd": 56.5, "trades": 195},
    {"id": "DISC74", "name": "Daily Pivot Trend Pullback Elite", "pips": 744.5, "sl": 8.0, "pf": 1.52, "wr": 44.5, "dd": 117.9, "trades": 328},
    {"id": "DISC58", "name": "H1 Keltner Squeeze Explosion", "pips": 726.2, "sl": 8.0, "pf": 1.57, "wr": 44.7, "dd": 94.4, "trades": 293},
    {"id": "DISC65", "name": "Session VWAP Band Fade Elite", "pips": 635.2, "sl": 8.0, "pf": 1.23, "wr": 40.4, "dd": 381.9, "trades": 557},
    {"id": "DISC60", "name": "Asian Compression Breakout Elite", "pips": 536.0, "sl": 8.0, "pf": 1.25, "wr": 39.7, "dd": 120.3, "trades": 429},
    {"id": "DISC51", "name": "Macro Asymmetric Runner Elite", "pips": 535.1, "sl": 8.0, "pf": 2.31, "wr": 49.5, "dd": 40.7, "trades": 103},
    {"id": "DISC49", "name": "Intraday Macro Retest Elite", "pips": 498.6, "sl": 8.0, "pf": 2.50, "wr": 59.2, "dd": 27.6, "trades": 103},
    {"id": "DISC42", "name": "Macro Structure Level Elite", "pips": 481.9, "sl": 10.0, "pf": 2.45, "wr": 62.5, "dd": 93.4, "trades": 77},
    {"id": "DISC47", "name": "Weekly Pivot Wall Rejection Elite", "pips": 418.3, "sl": 12.0, "pf": 1.26, "wr": 43.7, "dd": 172.9, "trades": 312},
    {"id": "DISC45", "name": "Continuous M5 Structural Rejection", "pips": 386.6, "sl": 6.0, "pf": 1.21, "wr": 47.6, "dd": 83.5, "trades": 577},
    {"id": "DISC76", "name": "VWAP 1-SD Trend Rebound Elite", "pips": 378.1, "sl": 8.0, "pf": 1.40, "wr": 43.9, "dd": 93.4, "trades": 196},
    {"id": "DISC52", "name": "Fib Golden Pocket Pin Defense", "pips": 334.0, "sl": 8.0, "pf": 1.36, "wr": 41.4, "dd": 75.3, "trades": 203},
    {"id": "DISC73", "name": "Donchian Trend EMA Pullback Elite", "pips": 250.2, "sl": 8.0, "pf": 1.07, "wr": 36.7, "dd": 177.1, "trades": 695},
    {"id": "DISC81", "name": "BB Squeeze Breakout Ignition Elite", "pips": 143.6, "sl": 8.0, "pf": 1.18, "wr": 39.1, "dd": 120.0, "trades": 156},
    {"id": "DISC43", "name": "Intraday Precision Level Elite", "pips": 132.2, "sl": 6.0, "pf": 1.14, "wr": 41.7, "dd": 94.7, "trades": 174},
]

# AUDNZD models
audnzd_models = [
    {"id": "DISC115", "name": "Fib Golden Pocket LB48 Elite", "pips": 2955.6, "sl": 14.0, "pf": 1.73, "wr": 51.4, "dd": 128.2, "trades": 549},
    {"id": "DISC116", "name": "Fib Golden Pocket Precision Elite", "pips": 2461.1, "sl": 12.0, "pf": 1.65, "wr": 49.6, "dd": 120.4, "trades": 560},
    {"id": "DISC117", "name": "Fib Golden Pocket High WR Elite", "pips": 2404.8, "sl": 12.0, "pf": 1.70, "wr": 54.1, "dd": 120.4, "trades": 560},
    {"id": "DISC118", "name": "Fib Golden Pocket LB36 Elite", "pips": 2310.5, "sl": 14.0, "pf": 1.48, "wr": 46.9, "dd": 256.1, "trades": 599},
    {"id": "DISC119", "name": "Fib Golden Pocket LB36 Precision", "pips": 1926.1, "sl": 12.0, "pf": 1.44, "wr": 46.0, "dd": 186.4, "trades": 607},
    {"id": "DISC120", "name": "Session VWAP 2.5-SD Band Fade", "pips": 1788.7, "sl": 12.0, "pf": 1.26, "wr": 40.5, "dd": 320.0, "trades": 933},
    {"id": "DISC111", "name": "H4 Macro Structure Retest Elite", "pips": 1498.3, "sl": 14.0, "pf": 4.46, "wr": 71.0, "dd": 87.4, "trades": 107},
    {"id": "DISC114", "name": "H4 Macro Wide Runner Elite", "pips": 1347.0, "sl": 15.0, "pf": 3.52, "wr": 66.4, "dd": 93.5, "trades": 107},
    {"id": "DISC112", "name": "H4 Macro Precision Retest Elite", "pips": 1171.6, "sl": 12.0, "pf": 3.64, "wr": 67.3, "dd": 69.7, "trades": 107},
    {"id": "DISC113", "name": "H1 Macro Intraday Retest Elite", "pips": 458.1, "sl": 12.0, "pf": 2.02, "wr": 55.4, "dd": 37.8, "trades": 83},
]


def analyze_asset(asset_name, models):
    print(f"\n{'='*120}")
    print(f"  ASSET: {asset_name} - SCORECARD & ANNUALIZED RETURN % (At 1.0% Risk / Trade)")
    print(f"{'='*120}")
    print(f"{'ID':<8} | {'Strategy Name':<34} | {'Trades':<6} | {'5.77y Pips':<10} | {'Ann Pips':<8} | {'SL':<4} | {'Ann R':<7} | {'Ann Ret%':<8} | {'PF':<5} | {'WR%':<5} | {'Max DD%':<7} | {'Ret/DD':<6} | {'Status'}")
    print("-" * 120)
    
    qualified = []
    rejected = []
    
    for m in models:
        ann_pips = m["pips"] / YEARS
        ann_r = ann_pips / m["sl"]
        ann_ret_pct = ann_r * 1.0 # 1% risk
        max_dd_pct = (m["dd"] / m["sl"]) * 1.0
        ret_to_dd = m["pips"] / m["dd"] if m["dd"] > 0 else 0.0
        
        # Threshold: Annual Return >= 15.0%, PF >= 1.25, Ret/DD >= 4.0x
        meets_threshold = (ann_ret_pct >= 15.0 and m["pf"] >= 1.25 and ret_to_dd >= 4.0)
        status = "QUALIFIED" if meets_threshold else "EXCLUDED"
        
        row = {
            **m,
            "ann_pips": round(ann_pips, 1),
            "ann_r": round(ann_r, 2),
            "ann_ret_pct": round(ann_ret_pct, 1),
            "max_dd_pct": round(max_dd_pct, 1),
            "ret_to_dd": round(ret_to_dd, 2),
            "meets_threshold": meets_threshold
        }
        
        if meets_threshold:
            qualified.append(row)
        else:
            rejected.append(row)
            
        print(f"{m['id']:<8} | {m['name']:<34} | {m['trades']:<6} | {m['pips']:<+10.1f} | {ann_pips:<+8.1f} | {m['sl']:<4.1f} | {ann_r:<+7.2f} | {ann_ret_pct:<+7.1f}% | {m['pf']:<5.2f} | {m['wr']:<5.1f} | {max_dd_pct:<6.1f}% | {ret_to_dd:<6.2f} | {status}")
        
    print("-" * 120)
    print(f"Total Qualified Models (>= 15% Annual Return, PF >= 1.25, Ret/DD >= 4.0x): {len(qualified)}")
    return qualified, rejected


if __name__ == "__main__":
    eurgbp_qual, eurgbp_rej = analyze_asset("EURGBP", eurgbp_models)
    audnzd_qual, audnzd_rej = analyze_asset("AUDNZD", audnzd_models)

