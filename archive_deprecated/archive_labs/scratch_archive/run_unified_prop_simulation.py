#!/usr/bin/env python3
"""
================================================================================
  UNIFIED MULTI-ASSET PROP CHALLENGE SIMULATION RUNNER
================================================================================
Combines multi-asset trades across EURUSD, USDJPY, and EURGBP from the master trade
databases and runs 1,000 Monte Carlo trials simulating authentic prop firm challenges:
  - Account Size: $100,000
  - Phase 1 Target: +8.0% ($8,000) in 30 trading days
  - Phase 2 Target: +5.0% ($5,000) in 60 trading days
  - Daily Drawdown Limit: 5.0% ($5,000)
  - Max Drawdown Limit: 10.0% ($10,000)
  - Risk per Trade: 0.50% ($500)
  - Daily Loss Circuit Breaker: 2.0% ($2,000)
================================================================================
"""

from __future__ import annotations
import sys
import os
import json
import time
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scratch.simulate_multi_asset_prop_challenge import run_monte_carlo_prop_simulation

RESULTS_DIR = ROOT_DIR / "results"


def load_master_trades() -> pd.DataFrame:
    records = []
    
    # 1. USDJPY Trades
    uj_path = RESULTS_DIR / "master_usdjpy_trades.json"
    if uj_path.exists():
        with open(uj_path, "r") as f:
            data = json.load(f)
            for strat_name, t_list in data.items():
                for t in t_list:
                    # SL default for USDJPY
                    sl = 12.0
                    if "Pivot" in strat_name:
                        sl = 9.0
                    records.append({
                        "timestamp": int(t["entry_time"]),
                        "pips": float(t["pips"]),
                        "sl_pips": sl,
                        "asset": "USDJPY",
                        "strategy": strat_name
                    })

    # 2. EURUSD Trades
    eu_path = RESULTS_DIR / "master_eurusd_14_models_trades.json"
    if eu_path.exists():
        with open(eu_path, "r") as f:
            data = json.load(f)
            for strat_name, t_list in data.items():
                for t in t_list:
                    sl = 9.0
                    records.append({
                        "timestamp": int(t["entry_time"]),
                        "pips": float(t["pips"]),
                        "sl_pips": sl,
                        "asset": "EURUSD",
                        "strategy": strat_name
                    })

    # 3. EURGBP Trades
    eg_path = RESULTS_DIR / "master_eurgbp_trades.json"
    if eg_path.exists():
        with open(eg_path, "r") as f:
            data = json.load(f)
            for strat_name, t_list in data.items():
                for t in t_list:
                    sl = 8.0
                    records.append({
                        "timestamp": int(t["entry_time"]),
                        "pips": float(t["pips"]),
                        "sl_pips": sl,
                        "asset": "EURGBP",
                        "strategy": strat_name
                    })

    df = pd.DataFrame(records)
    print(f"[*] Loaded {len(df)} total trades across {df['asset'].nunique()} assets and {df['strategy'].nunique()} strategies.")
    return df


def execute_prop_suite():
    t_start = time.time()
    print("=" * 80)
    print("  LAUNCHING UNIFIED MULTI-ASSET PROP CHALLENGE SIMULATION SUITE")
    print("=" * 80)

    trades_df = load_master_trades()
    if trades_df.empty:
        print("[!] No trades available.")
        return

    # Filter top non-overlapping champions to simulate realistic prop portfolio
    champions = [
        "DISC46_Daily_Pivot",
        "DISC48_Camarilla",
        "DISC42_H4_M30",
        "DISC45_M5",
        "DISC40_RSI",
        "DISC43_H1_M5",
        "DISC38_MACD"
    ]
    champ_df = trades_df[trades_df['strategy'].isin(champions)].copy()
    print(f"[*] Focused champion sub-fleet: {len(champ_df)} trades across {champ_df['strategy'].nunique()} models.")

    # Run simulations under different risk allocations: 0.35%, 0.50%, 0.75%
    for risk in [0.0035, 0.0050, 0.0075]:
        print(f"\n--- Testing Risk Allocation: {risk*100:.2f}% per trade ---")
        metrics = run_monte_carlo_prop_simulation(
            trades_df=champ_df,
            num_simulations=1000,
            risk_pct=risk,
            initial_balance=100000.0,
            daily_dd_limit=0.05,
            max_dd_limit=0.10,
            p1_target_pct=0.08,
            p2_target_pct=0.05,
            circuit_breaker_pct=0.02
        )

        print(f"  [+] Phase 1 Pass Rate:       {metrics['p1_pass_rate']}%")
        print(f"  [+] Full 2-Phase Pass Rate:  {metrics['full_pass_rate']}%")
        print(f"  [+] Daily DD Breach Rate:    {metrics['daily_dd_breach_rate']}%")
        print(f"  [+] Max DD Breach Rate:      {metrics['max_dd_breach_rate']}%")
        print(f"  [+] Avg P1 Days to Pass:     {metrics['avg_p1_days_to_pass']} days")
        print(f"  [+] Avg P2 Days to Pass:     {metrics['avg_p2_days_to_pass']} days")

    print("\n" + "=" * 80)
    print(f"[*] Prop simulation suite complete in {time.time() - t_start:.2f} seconds.")


if __name__ == "__main__":
    execute_prop_suite()
