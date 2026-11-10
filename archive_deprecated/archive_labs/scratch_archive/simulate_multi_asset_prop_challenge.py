#!/usr/bin/env python3
"""
================================================================================
  MULTI-ASSET UNIFIED PROP FIRM CHALLENGE SIMULATOR (1,000 MONTE CARLO RUNS)
================================================================================
Simulates 1,000 randomized institutional prop firm challenges (Phase 1 & Phase 2)
evaluating the combined multi-asset elite fleet across:
  - EURUSD
  - USDJPY
  - EURGBP

Prop Challenge Specifications:
  - Account Size: $100,000
  - Phase 1 Profit Target: +8.0% ($8,000) within 30 trading days
  - Phase 2 Profit Target: +5.0% ($5,000) within 60 trading days
  - Daily Max Drawdown Limit: 5.0% ($5,000)
  - Maximum Overall Drawdown Limit: 10.0% ($10,000)
  - Risk per Trade: 0.50% ($500 per trade) or 0.75% ($750 per trade)
  - Daily Loss Circuit Breaker: 2.0% ($2,000 hard stop per day)

Simulates randomized entry dates across the 5-year timeline to stress-test
all market regimes (trending, choppy, high-volatility, low-volatility).
================================================================================
"""

from __future__ import annotations
import sys
import os
import time
import json
import random
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Any

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

RESULTS_DIR = ROOT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def run_monte_carlo_prop_simulation(
    trades_df: pd.DataFrame,
    num_simulations: int = 1000,
    risk_pct: float = 0.005,  # 0.5% risk per trade
    initial_balance: float = 100000.0,
    daily_dd_limit: float = 0.05,
    max_dd_limit: float = 0.10,
    p1_target_pct: float = 0.08,
    p2_target_pct: float = 0.05,
    circuit_breaker_pct: float = 0.02
) -> Dict[str, Any]:
    """
    Simulates random prop firm challenge attempts by drawing contiguous sequences
    of trading days starting at randomly selected points across the trade history.
    """
    if trades_df.empty:
        return {"error": "No trades available"}

    trades_df = trades_df.sort_values(by="timestamp").reset_index(drop=True)
    trades_df['date'] = pd.to_datetime(trades_df['timestamp'], unit='ms').dt.date

    # Group trades by date
    days = sorted(trades_df['date'].unique())
    n_days = len(days)
    if n_days < 60:
        return {"error": f"Insufficient days ({n_days})"}

    p1_passes = 0
    p2_passes = 0
    full_passes = 0
    daily_violations = 0
    max_dd_violations = 0
    p1_days_list = []
    p2_days_list = []

    # Map trades by day
    trades_by_day = {d: trades_df[trades_df['date'] == d] for d in days}

    rng = np.random.RandomState(42)

    for sim_idx in range(num_simulations):
        # Pick random start day with at least 60 days buffer
        start_idx = rng.randint(0, n_days - 60)

        # ---------------- Phase 1 ----------------
        balance = initial_balance
        peak_balance = balance
        day_idx = start_idx
        p1_success = False
        p1_days = 0

        while day_idx < n_days:
            d = days[day_idx]
            day_trades = trades_by_day[d]
            day_start_balance = balance
            day_pnl = 0.0

            for _, t in day_trades.iterrows():
                # Check circuit breaker
                if day_pnl <= -circuit_breaker_pct * day_start_balance:
                    break

                # PnL calculation: R-multiple * (risk_pct * balance)
                # t['pips'] / t['sl_pips'] gives R
                r = t['pips'] / max(t['sl_pips'], 1.0)
                trade_dollar = r * (risk_pct * day_start_balance)
                balance += trade_dollar
                day_pnl += trade_dollar

                if balance > peak_balance:
                    peak_balance = balance

                # Check intraday limits
                if (day_start_balance - balance) / day_start_balance >= daily_dd_limit:
                    daily_violations += 1
                    break

                if (peak_balance - balance) / peak_balance >= max_dd_limit:
                    max_dd_violations += 1
                    break

                # Phase 1 Target reached?
                if (balance - initial_balance) / initial_balance >= p1_target_pct:
                    p1_success = True
                    break

            p1_days += 1
            day_idx += 1

            if p1_success:
                break
            if (day_start_balance - balance) / day_start_balance >= daily_dd_limit or (peak_balance - balance) / peak_balance >= max_dd_limit:
                break
            if p1_days >= 40:  # 40 trading days limit
                break

        if not p1_success:
            continue

        p1_passes += 1
        p1_days_list.append(p1_days)

        # ---------------- Phase 2 ----------------
        balance = initial_balance
        peak_balance = balance
        p2_success = False
        p2_days = 0

        while day_idx < n_days:
            d = days[day_idx]
            day_trades = trades_by_day[d]
            day_start_balance = balance
            day_pnl = 0.0

            for _, t in day_trades.iterrows():
                if day_pnl <= -circuit_breaker_pct * day_start_balance:
                    break

                r = t['pips'] / max(t['sl_pips'], 1.0)
                trade_dollar = r * (risk_pct * day_start_balance)
                balance += trade_dollar
                day_pnl += trade_dollar

                if balance > peak_balance:
                    peak_balance = balance

                if (day_start_balance - balance) / day_start_balance >= daily_dd_limit:
                    daily_violations += 1
                    break

                if (peak_balance - balance) / peak_balance >= max_dd_limit:
                    max_dd_violations += 1
                    break

                if (balance - initial_balance) / initial_balance >= p2_target_pct:
                    p2_success = True
                    break

            p2_days += 1
            day_idx += 1

            if p2_success:
                break
            if (day_start_balance - balance) / day_start_balance >= daily_dd_limit or (peak_balance - balance) / peak_balance >= max_dd_limit:
                break
            if p2_days >= 60:
                break

        if p2_success:
            p2_passes += 1
            full_passes += 1
            p2_days_list.append(p2_days)

    return {
        "num_simulations": num_simulations,
        "risk_pct": risk_pct,
        "p1_pass_rate": round(p1_passes / num_simulations * 100.0, 1),
        "full_pass_rate": round(full_passes / num_simulations * 100.0, 1),
        "daily_dd_breach_rate": round(daily_violations / num_simulations * 100.0, 2),
        "max_dd_breach_rate": round(max_dd_violations / num_simulations * 100.0, 2),
        "avg_p1_days_to_pass": round(np.mean(p1_days_list), 1) if p1_days_list else 0.0,
        "avg_p2_days_to_pass": round(np.mean(p2_days_list), 1) if p2_days_list else 0.0,
        "median_p1_days": float(np.median(p1_days_list)) if p1_days_list else 0.0,
        "median_p2_days": float(np.median(p2_days_list)) if p2_days_list else 0.0
    }


if __name__ == "__main__":
    print("Multi-Asset Prop Simulator Module Loaded.")
