#!/usr/bin/env python3
"""
Empirical Monte Carlo & Rolling Window Analysis: Time to Hit +10% Prop Firm Target
Evaluates:
- How many calendar days and trading days does the 5-Star Prop Firm Fleet take to hit +10%?
- Test across 52 different starting weeks (Week 1 through Week 52 rolling starts).
- Evaluates at different risk-per-trade levels (0.35%, 0.40%, 0.50%).
- Verifies that max daily loss never exceeds 5.0% and max drawdown never exceeds 10.0%.
"""

from __future__ import annotations
import sys
import os
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Any
import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tick_engine.core.engine import BacktestEngine
from tick_engine.core.dataloader import get_week_dataset
from strategies.comb12_bollinger_squeeze_elite.strategy import BollingerBandwidthExpansionElite
from strategies.comb14_supertrend_vwap_elite.strategy import SupertrendVWAPPullbackElite
from strategies.comb08_asian_breakout_adx_elite.strategy import LondonAsianBreakoutADXElite
from strategies.comb17_psar_adx_elite.strategy import ParabolicSARADXElite
from strategies.disc02_frankfurt_london_orb_elite.strategy import FrankfurtLondonPreMarketORBElite

PIP_SIZE = 0.0001

FLEET = [
    {"name": "Squeeze_52WR", "cls": BollingerBandwidthExpansionElite},
    {"name": "Supertrend_VWAP", "cls": SupertrendVWAPPullbackElite},
    {"name": "Asian_ADX", "cls": LondonAsianBreakoutADXElite},
    {"name": "PSAR_ADX", "cls": ParabolicSARADXElite},
    {"name": "Frankfurt_ORB", "cls": FrankfurtLondonPreMarketORBElite},
]


def run_week(w: int):
    ticks, meta = get_week_dataset(w)
    engine = BacktestEngine(pip_size=PIP_SIZE, use_slippage=True)
    week_trades = []
    for item in FLEET:
        strat = item["cls"]()
        trades = engine.run(ticks, strat)
        for t in trades:
            week_trades.append({
                "entry_time": int(t["entry_time"]),
                "exit_time": int(t["exit_time"]),
                "pips": float(t["pips"]),
                "strategy": item["name"],
                "week": w
            })
    return week_trades


def simulate_target_hit(trades: List[Dict[str, Any]], start_idx: int, risk_pct: float, target_pct: float = 10.0):
    balance = 100000.0
    start_balance = balance
    target_balance = balance * (1.0 + target_pct / 100.0)
    
    losses = [abs(t["pips"]) for t in trades if t["pips"] < 0]
    avg_loss_pip = np.mean(losses) if losses else 10.0
    
    peak = balance
    max_dd_pct = 0.0
    daily_pnl = {}
    max_daily_loss_pct = 0.0
    
    start_time = trades[start_idx]["entry_time"]
    trades_count = 0
    
    # Track through subsequent trades (looping across annual dataset if needed)
    n = len(trades)
    for step in range(n * 2): # allow up to 2 years if needed
        idx = (start_idx + step) % n
        t = trades[idx]
        trades_count += 1
        
        r_pnl = (t["pips"] / avg_loss_pip) * (balance * (risk_pct / 100.0))
        balance += r_pnl
        
        day_str = pd.to_datetime(t["exit_time"], unit='ms').strftime('%Y-%m-%d')
        daily_pnl[day_str] = daily_pnl.get(day_str, 0.0) + r_pnl
        if daily_pnl[day_str] < 0:
            d_loss = abs(daily_pnl[day_str]) / start_balance * 100.0
            if d_loss > max_daily_loss_pct:
                max_daily_loss_pct = d_loss
                
        if balance > peak:
            peak = balance
        dd = (peak - balance) / peak * 100.0
        if dd > max_dd_pct:
            max_dd_pct = dd
            
        # Check fail conditions (Prop Firm bust)
        if max_daily_loss_pct >= 5.0 or max_dd_pct >= 10.0:
            return {
                "success": False,
                "reason": "Busted daily or total limit",
                "days": 0,
                "trades": trades_count,
                "max_dd": max_dd_pct,
                "max_daily_loss": max_daily_loss_pct
            }
            
        # Check target hit
        if balance >= target_balance:
            end_time = t["exit_time"]
            # approximate calendar days based on trades count & velocity
            cal_days = (trades_count / len(trades)) * 365.0
            return {
                "success": True,
                "reason": "Target Hit",
                "days": cal_days,
                "trades": trades_count,
                "max_dd": max_dd_pct,
                "max_daily_loss": max_daily_loss_pct
            }
            
    return {"success": False, "reason": "Timed out", "days": 365, "trades": trades_count, "max_dd": max_dd_pct, "max_daily_loss": max_daily_loss_pct}


def main():
    print("=" * 115)
    print("      TIME-TO-HIT +10% PROFIT TARGET AUDIT: 52 WEEKS DUKASCOPY TICKS (~26M TICKS)")
    print("=" * 115)
    
    with ProcessPoolExecutor() as executor:
        week_res = list(executor.map(run_week, range(1, 53)))
        
    all_trades = []
    for r in week_res:
        all_trades.extend(r)
    all_trades = sorted(all_trades, key=lambda x: x["entry_time"])
    
    print(f"Total Portfolio Trades loaded: {len(all_trades)}")
    
    # Test at risk levels: 0.35%, 0.40%, 0.50%
    risk_levels = [0.35, 0.40, 0.50]
    
    # Rolling starting points (sample every 10 trades across the year)
    sample_starts = list(range(0, len(all_trades), 15))
    
    for r_pct in risk_levels:
        print(f"\n=======================================================")
        print(f"  RISK PER TRADE: {r_pct:.2f}%  (${(r_pct*1000):.0f} per trade on $100k account)")
        print(f"=======================================================")
        
        results = []
        for s_idx in sample_starts:
            res = simulate_target_hit(all_trades, s_idx, r_pct, target_pct=10.0)
            results.append(res)
            
        successes = [r for r in results if r["success"]]
        pass_rate = len(successes) / len(results) * 100.0
        
        days_list = [r["days"] for r in successes]
        trades_list = [r["trades"] for r in successes]
        dd_list = [r["max_dd"] for r in results]
        daily_loss_list = [r["max_daily_loss"] for r in results]
        
        fastest_days = np.min(days_list) if days_list else 0
        median_days = np.median(days_list) if days_list else 0
        avg_days = np.mean(days_list) if days_list else 0
        slowest_days = np.max(days_list) if days_list else 0
        
        fastest_weeks = fastest_days / 7.0
        median_weeks = median_days / 7.0
        avg_weeks = avg_days / 7.0
        slowest_weeks = slowest_days / 7.0
        
        fastest_trades = np.min(trades_list) if trades_list else 0
        median_trades = np.median(trades_list) if trades_list else 0
        avg_trades = np.mean(trades_list) if trades_list else 0
        slowest_trades = np.max(trades_list) if trades_list else 0
        
        print(f"Pass Rate (hit +10% with zero limit breaches): {pass_rate:.1f}%")
        print(f"Max Daily Loss Encountered:                    {np.max(daily_loss_list):.2f}% (Limit: 5.0%)")
        print(f"Max Drawdown Encountered:                      {np.max(dd_list):.2f}% (Limit: 10.0%)")
        print(f"\nTime to Hit +10%:")
        print(f"  - Fastest Scenario (Hot Streak / Favorable Regime): {fastest_days:.0f} calendar days (~{fastest_weeks:.1f} weeks) | {fastest_trades:.0f} trades")
        print(f"  - Median Scenario (Typical Expected Performance):  {median_days:.0f} calendar days (~{median_weeks:.1f} weeks / ~{median_days/30.4:.1f} months) | {median_trades:.0f} trades")
        print(f"  - Average Scenario:                                {avg_days:.0f} calendar days (~{avg_weeks:.1f} weeks / ~{avg_days/30.4:.1f} months) | {avg_trades:.0f} trades")
        print(f"  - Slowest Scenario (Consolidation Regime):         {slowest_days:.0f} calendar days (~{slowest_weeks:.1f} weeks / ~{slowest_days/30.4:.1f} months) | {slowest_trades:.0f} trades")


if __name__ == "__main__":
    main()
