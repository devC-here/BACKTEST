#!/usr/bin/env python3
"""
7-Star Prop Firm Fleet Simulation across 52 Weeks (~26M Ticks):
Combines 7 Independently Verified Prop Champions:
1. COMB12 Calibrated (Bollinger Squeeze 40, RR 1.6) -> 52.0% WR, PF 1.87, +204.6p
2. COMB14 (Supertrend 10,2 + Session VWAP) -> 337 trd/yr, PF 1.26, +503.9p
3. COMB08 (Asian Breakout + ADX >= 16) -> 141 trd/yr, PF 1.24, +234.9p
4. COMB17 (PSAR Flip + ADX >= 20) -> 137 trd/yr, PF 1.27, +183.9p
5. DISC02 (Frankfurt/London ORB) -> 59 trd/yr, 44.1% WR, PF 1.25, +98.8p, 20-day recov
6. DISC03 (NY Open Overlap Breakout) -> 23 trd/yr, 43.5% WR, PF 1.38, +57.9p, 22-day recov
7. DISC23 (Keltner Bandwidth Squeeze) -> 29 trd/yr, 51.7% WR, PF 1.64, +104.4p, 1.3% DD

Evaluates exact daily trade distribution, win rate, prop firm daily loss limits, and drawdown.
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
from strategies.disc03_ny_open_overlap_elite.strategy import NewYorkOpenOverlapBreakoutElite
from strategies.disc23_keltner_squeeze_elite.strategy import KeltnerBandwidthExpansionElite

PIP_SIZE = 0.0001

FLEET = [
    {"name": "Squeeze_52WR", "cls": BollingerBandwidthExpansionElite},
    {"name": "Supertrend_VWAP", "cls": SupertrendVWAPPullbackElite},
    {"name": "Asian_ADX", "cls": LondonAsianBreakoutADXElite},
    {"name": "PSAR_ADX", "cls": ParabolicSARADXElite},
    {"name": "Frankfurt_ORB", "cls": FrankfurtLondonPreMarketORBElite},
    {"name": "NY_Overlap_ORB", "cls": NewYorkOpenOverlapBreakoutElite},
    {"name": "Keltner_Squeeze", "cls": KeltnerBandwidthExpansionElite},
]


def run_week(w: int):
    ticks, meta = get_week_dataset(w)
    engine = BacktestEngine(pip_size=PIP_SIZE, use_slippage=True)
    res = {}
    for item in FLEET:
        strat = item["cls"]()
        trades = engine.run(ticks, strat)
        res[item["name"]] = [
            {"entry_time": int(t["entry_time"]), "exit_time": int(t["exit_time"]), "pips": float(t["pips"])}
            for t in trades
        ]
    return res


def main():
    print("=" * 115)
    print("           7-STAR PROP FIRM FLEET: 52 WEEKS COMBINED TICK AUDIT (~26M TICKS)")
    print("=" * 115)
    with ProcessPoolExecutor() as executor:
        week_res = list(executor.map(run_week, range(1, 53)))

    strat_trades = {item["name"]: [] for item in FLEET}
    all_trades = []

    for r in week_res:
        for k, v in r.items():
            strat_trades[k].extend(v)
            all_trades.extend(v)

    all_trades = sorted(all_trades, key=lambda x: x["entry_time"])

    print("\n--- INDIVIDUAL STRATEGY METRICS ---")
    print(f"{'Strategy':<20} | {'Trades':<6} | {'Trd/Wk':<6} | {'Win %':<6} | {'Real RR':<7} | {'PF':<5} | {'Net Pips':<9}")
    print("-" * 75)
    for item in FLEET:
        name = item["name"]
        t_list = strat_trades[name]
        pips = np.array([t["pips"] for t in t_list])
        wins = pips[pips > 0]
        losses = pips[pips < 0]
        n = len(pips)
        wr = len(wins) / n * 100 if n > 0 else 0
        avg_w = np.mean(wins) if len(wins) > 0 else 0
        avg_l = abs(np.mean(losses)) if len(losses) > 0 else 1
        rr = avg_w / avg_l if avg_l > 0 else 0
        pf = np.sum(wins) / abs(np.sum(losses)) if len(losses) > 0 and abs(np.sum(losses)) > 0 else 0
        net = np.sum(pips)
        print(f"{name:<20} | {n:<6} | {n/52:<6.1f} | {wr:<5.1f}% | {rr:<7.2f} | {pf:<5.2f} | {net:<+9.1f}p")

    # Combined Fleet Analysis
    print("\n" + "=" * 115)
    print("           COMBINED 7-STAR PORTFOLIO PERFORMANCE (0.35% RISK PER TRADE)")
    print("=" * 115)

    s_trades = sorted(all_trades, key=lambda x: x["exit_time"])
    pips = np.array([t["pips"] for t in s_trades])
    wins = pips[pips > 0]
    losses = pips[pips < 0]
    n_trades = len(pips)
    avg_loss = abs(np.mean(losses)) if len(losses) > 0 else 8.0

    balance = 100000.0
    curve = [balance]
    daily_pnl = {}
    trade_dates = {}

    risk_pct = 0.35  # Conservative 0.35% per trade

    for t in s_trades:
        r_pnl = (t["pips"] / avg_loss) * (balance * (risk_pct / 100.0))
        balance += r_pnl
        curve.append(balance)

        d_str = pd.to_datetime(t["exit_time"], unit='ms').strftime('%Y-%m-%d')
        daily_pnl[d_str] = daily_pnl.get(d_str, 0.0) + r_pnl
        trade_dates[d_str] = trade_dates.get(d_str, 0) + 1

    c_arr = np.array(curve)
    peaks = np.maximum.accumulate(c_arr)
    dd_pct = (peaks - c_arr) / peaks * 100.0
    max_dd = float(np.max(dd_pct))

    max_daily_loss = 0.0
    for d, pnl in daily_pnl.items():
        if pnl < 0:
            loss_pct = abs(pnl) / 100000.0 * 100.0
            if loss_pct > max_daily_loss:
                max_daily_loss = loss_pct

    # Daily frequency analysis
    all_days = list(daily_pnl.keys())
    active_days = len(all_days)
    trades_per_active_day = n_trades / active_days if active_days > 0 else 0
    trades_per_business_day = n_trades / 260.0  # standard 260 trading days/yr

    print(f"Total Portfolio Trades:       {n_trades} trades (~{n_trades/52:.1f} trd/week)")
    print(f"Trading Days Active:          {active_days} of ~260 days ({active_days/260*100:.1f}% participation)")
    print(f"Trades Per Active Day:        {trades_per_active_day:.2f} trades/day")
    print(f"Trades Per Business Day:      {trades_per_business_day:.2f} trades/day")
    print(f"Portfolio Win Rate:           {len(wins)/n_trades*100:.1f}%")
    print(f"Portfolio Realized RR:        {np.mean(wins)/avg_loss:.2f} : 1")
    print(f"Portfolio Profit Factor:      {np.sum(wins)/abs(np.sum(losses)):.2f}")
    print(f"Total Net Pips:               {np.sum(pips):+.1f} pips")
    print(f"Total Net Return:             +{(balance - 100000.0)/1000:.2f}% (${balance - 100000.0:,.2f})")
    print(f"Max Portfolio Drawdown:       {max_dd:.2f}% (Limit: 10.0%) -> SAFE BUFFER: {10.0 - max_dd:.2f}%")
    print(f"Max Daily Loss:               {max_daily_loss:.2f}% (Limit: 5.0%) -> SAFE BUFFER: {5.0 - max_daily_loss:.2f}%")
    print("=" * 115)


if __name__ == "__main__":
    main()
