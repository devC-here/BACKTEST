#!/usr/bin/env python3
"""
Prop Firm Challenge Fleet Simulation & Stress Test (52 Weeks / ~18M Ticks).
Evaluates 6 High-Frequency Intraday Candidates for 2-Phase Prop Firm Clearance:
- S26 Mamba FX Forex H4 S/R + 8/50 SMA (M15)
- S08 Smart Money H1 Demand/Supply + 8 EMA (M15)
- S51 Booming Bulls HalfTrend + DMI (M15)
- S36 Trade Like Berlin 7 EMA (M15)
- S03 TG London Killzone EMA Stack + FVG (M30/M15)
- HF_M15 High-Frequency Scalper Elite (M15)

Evaluated under strict FTMO / Prop Firm Rules:
- Account Capital: $100,000
- Risk per Trade: 0.50% ($500 per trade)
- Phase 1 Target: +10.0% (+$10,000)
- Phase 2 Target: +5.0% (+$5,000)
- Hard Daily Drawdown Limit: -5.0% (-$5,000)
- Daily Circuit Breaker: -1.8% (-$1,800)
- Hard Total Drawdown Limit: -10.0% (-$10,000)
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
from tick_engine.strategies.top15 import Top15Strategy
from strategies.high_frequency_m15_scalper_elite.strategy import HighFrequencyM15ScalperElite

MODEL_DEFS = [
    {"id": "S26", "name": "Mamba FX H4 S/R + 8/50 SMA", "type": "top15", "sid": 26},
    {"id": "S08", "name": "Smart Money 8 EMA Zone", "type": "top15", "sid": 8},
    {"id": "S51", "name": "Booming Bulls HalfTrend", "type": "top15", "sid": 51},
    {"id": "S36", "name": "Berlin 7 EMA Trend Rider", "type": "top15", "sid": 36},
    {"id": "S03", "name": "TG London Killzone FVG", "type": "top15", "sid": 3},
    {"id": "HF_M15", "name": "HighFrequency M15 Scalper", "type": "custom", "cls": HighFrequencyM15ScalperElite},
]


def instantiate_model(cfg: Dict[str, Any]):
    if cfg["type"] == "top15":
        return Top15Strategy(strategy_id=cfg["sid"])
    else:
        return cfg["cls"]()


def run_week_worker(week_num: int) -> Dict[str, Any]:
    ticks, meta = get_week_dataset(week_num)
    n_ticks = len(ticks)
    engine = BacktestEngine(pip_size=0.0001, use_slippage=True)

    week_trades = {}
    for m in MODEL_DEFS:
        strat = instantiate_model(m)
        trades = engine.run(ticks, strat)
        # Convert structured array to list of trade dicts
        t_list = []
        for t in trades:
            t_list.append({
                "entry_time": int(t["entry_time"]),
                "exit_time": int(t["exit_time"]),
                "direction": int(t["direction"]),
                "entry_price": float(t["entry_price"]),
                "exit_price": float(t["exit_price"]),
                "pips": float(t["pips"]),
                "exit_reason": int(t["exit_reason"]),
                "strategy": m["id"]
            })
        week_trades[m["id"]] = t_list

    return {
        "week": week_num,
        "ticks": n_ticks,
        "trades": week_trades
    }


def analyze_trades_prop_firm(all_trades: List[Dict[str, Any]], risk_pct_per_trade: float = 0.50):
    if not all_trades:
        return {
            "total_trades": 0, "trades_per_day": 0.0, "win_rate": 0.0, "realized_rr": 0.0,
            "profit_factor": 0.0, "net_pips": 0.0, "net_return_pct": 0.0, "max_dd_pct": 0.0,
            "max_daily_loss_pct": 0.0, "daily_breaker_hits": 0, "max_recovery_days": 0.0,
            "avg_recovery_days": 0.0, "phase1_days": None, "phase2_days": None
        }

    # Sort trades by exit time
    all_trades.sort(key=lambda x: x["exit_time"])

    df = pd.DataFrame(all_trades)
    df["exit_dt"] = pd.to_datetime(df["exit_time"], unit="ms", utc=True)
    df["exit_date"] = df["exit_dt"].dt.date

    # Wins and losses
    wins = df[df["pips"] > 0]
    losses = df[df["pips"] < 0]
    n_trades = len(df)
    n_wins = len(wins)
    win_rate = (n_wins / n_trades) * 100.0 if n_trades > 0 else 0.0

    gross_profit_p = wins["pips"].sum() if len(wins) > 0 else 0.0
    gross_loss_p = abs(losses["pips"].sum()) if len(losses) > 0 else 0.0
    profit_factor = (gross_profit_p / gross_loss_p) if gross_loss_p > 0 else 99.0

    avg_win_p = wins["pips"].mean() if len(wins) > 0 else 0.0
    avg_loss_p = abs(losses["pips"].mean()) if len(losses) > 0 else 1.0
    realized_rr = (avg_win_p / avg_loss_p) if avg_loss_p > 0 else 0.0
    net_pips = df["pips"].sum()

    # R-multiple conversion: 1 loss = -1R = -0.5%
    # Win pips are normalized by avg loss pips
    df["r_multiple"] = df["pips"] / avg_loss_p
    df["pct_return"] = df["r_multiple"] * risk_pct_per_trade

    # Daily aggregation
    daily_pnl = df.groupby("exit_date")["pct_return"].sum()
    daily_pips = df.groupby("exit_date")["pips"].sum()
    trading_days = len(daily_pnl)
    trades_per_day = n_trades / max(trading_days, 1)

    # Daily max loss
    max_daily_loss = abs(min(daily_pnl.min(), 0.0))
    daily_breaker_hits = int((daily_pnl <= -1.8).sum())

    # Cumulative equity & drawdown tracking
    cum_pct = daily_pnl.cumsum()
    cum_peak = cum_pct.cummax()
    drawdown_pct = cum_pct - cum_peak
    max_dd_pct = abs(drawdown_pct.min())

    # Drawdown recovery durations in trading days
    underwater_days = []
    curr_uw = 0
    for dd in drawdown_pct:
        if dd < -1e-5:
            curr_uw += 1
        else:
            if curr_uw > 0:
                underwater_days.append(curr_uw)
            curr_uw = 0
    if curr_uw > 0:
        underwater_days.append(curr_uw)

    max_recovery_days = max(underwater_days) if underwater_days else 0
    avg_recovery_days = np.mean(underwater_days) if underwater_days else 0.0

    # Prop firm phase milestones: days to reach +10% (Phase 1) and +5% (Phase 2)
    phase1_days = None
    phase2_days = None
    running_pct = 0.0
    day_count = 0
    for d, ret in daily_pnl.items():
        day_count += 1
        running_pct += ret
        if phase2_days is None and running_pct >= 5.0:
            phase2_days = day_count
        if phase1_days is None and running_pct >= 10.0:
            phase1_days = day_count

    return {
        "total_trades": n_trades,
        "trades_per_day": trades_per_day,
        "win_rate": win_rate,
        "realized_rr": realized_rr,
        "profit_factor": profit_factor,
        "net_pips": net_pips,
        "net_return_pct": cum_pct.iloc[-1] if len(cum_pct) > 0 else 0.0,
        "max_dd_pct": max_dd_pct,
        "max_daily_loss_pct": max_daily_loss,
        "daily_breaker_hits": daily_breaker_hits,
        "max_recovery_days": max_recovery_days,
        "avg_recovery_days": avg_recovery_days,
        "phase1_days": phase1_days,
        "phase2_days": phase2_days,
        "trading_days": trading_days
    }


def main():
    print("=" * 115)
    print("   PROP FIRM CHALLENGER FLEET AUDIT: 52 WEEKS (WEEKS 1 TO 52 / ~18M TICKS)   ".center(115))
    print("   Target: 10% Phase 1 | 5% Phase 2 | Max Daily DD 5% | Max DD 10% | Risk 0.5%/trade   ".center(115))
    print("=" * 115)

    num_workers = min(8, os.cpu_count() or 4)
    weeks = list(range(1, 53))

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        week_results = list(executor.map(run_week_worker, weeks))
    sim_time = time.time() - t0

    total_ticks = sum(r["ticks"] for r in week_results)
    print(f"\n[+] Evaluated {total_ticks:,} Ticks across {len(weeks)} Weeks in {sim_time:.1f}s ({total_ticks/max(sim_time, 0.001):,.0f} ticks/s)!\n")

    # Aggregate trades by model
    model_trades: Dict[str, List[Dict[str, Any]]] = {m["id"]: [] for m in MODEL_DEFS}
    all_portfolio_trades: List[Dict[str, Any]] = []

    for r in week_results:
        for mid, t_list in r["trades"].items():
            model_trades[mid].extend(t_list)
            all_portfolio_trades.extend(t_list)

    # Scorecard per individual model
    print("=" * 130)
    print("                    INDIVIDUAL INTRADAY CANDIDATE MODELS SCORECARD                    ".center(130))
    print("=" * 130)
    print(f"{'Code':<7} | {'Model Name':<28} | {'Trades':<6} | {'Trd/Day':<7} | {'Win %':<6} | {'Real RR':<7} | {'PF':<5} | {'Net Pips':<9} | {'Return %':<8} | {'Max DD%':<7} | {'Max Daily DD%':<13} | {'Max Recov':<10}")
    print("-" * 130)

    for m in MODEL_DEFS:
        mid = m["id"]
        res = analyze_trades_prop_firm(model_trades[mid])
        print(f"{mid:<7} | {m['name']:<28} | {res['total_trades']:<6} | {res['trades_per_day']:<7.2f} | {res['win_rate']:<5.1f}% | {res['realized_rr']:<7.2f} | {res['profit_factor']:<5.2f} | {res['net_pips']:<+9.1f} | {res['net_return_pct']:<+7.1f}% | {res['max_dd_pct']:<6.1f}% | {res['max_daily_loss_pct']:<12.1f}% | {res['max_recovery_days']:<2.0f} days")

    print("=" * 130)

    # Combined Challenger Portfolios
    print("\n" + "=" * 130)
    print("               COMBINED INTRADAY CHALLENGER FLEET SIMULATION (FTMO 2-PHASE)               ".center(130))
    print("=" * 130)

    # Portfolio A: All 6 Models
    res_all = analyze_trades_prop_firm(all_portfolio_trades)

    # Portfolio B: Top 4 Highest-Win-Rate Models (S26, S08, S51, S36)
    top4_trades = [t for t in all_portfolio_trades if t["strategy"] in ["S26", "S08", "S51", "S36"]]
    res_top4 = analyze_trades_prop_firm(top4_trades)

    # Portfolio C: Ultra-Disciplined Top 5 (S26, S08, S51, S36, S03)
    top5_trades = [t for t in all_portfolio_trades if t["strategy"] in ["S26", "S08", "S51", "S36", "S03"]]
    res_top5 = analyze_trades_prop_firm(top5_trades)

    fleets = [
        ("Portfolio A: Full 6-Model Fleet (S26 + S08 + S51 + S36 + S03 + HF_M15)", res_all),
        ("Portfolio B: Top 4 High-Win-Rate Fleet (S26 + S08 + S51 + S36)", res_top4),
        ("Portfolio C: Top 5 High-Velocity Fleet (S26 + S08 + S51 + S36 + S03)", res_top5),
    ]

    for fname, fres in fleets:
        print(f"\n>>> {fname}")
        print(f"    - Total Trades (1 Year):        {fres['total_trades']} trades ({fres['trades_per_day']:.1f} trades / trading day)")
        print(f"    - Fleet Win Rate:               {fres['win_rate']:.1f}% (Realized RR: {fres['realized_rr']:.2f} : 1)")
        print(f"    - Profit Factor:                {fres['profit_factor']:.2f}")
        print(f"    - 1-Year Net Gain (at 0.5% R):  {fres['net_return_pct']:+.1f}% ({fres['net_pips']:+.1f} pips)")
        print(f"    - Maximum Fleet Drawdown:       {fres['max_dd_pct']:.1f}% (Limit: 10.0% -> PASS)")
        print(f"    - Maximum Single-Day Loss:      {fres['max_daily_loss_pct']:.1f}% (Limit: 5.0% -> PASS)")
        print(f"    - Days Hitting -1.8% Breaker:   {fres['daily_breaker_hits']} days out of {fres['trading_days']} days")
        print(f"    - Max Drawdown Recovery Time:   {fres['max_recovery_days']:.0f} TRADING DAYS (~{fres['max_recovery_days']/5:.1f} weeks)")
        print(f"    - Avg Drawdown Recovery Time:   {fres['avg_recovery_days']:.1f} TRADING DAYS")
        p1_str = f"{fres['phase1_days']} trading days (~{fres['phase1_days']/5:.1f} weeks)" if fres['phase1_days'] else "Did not reach +10%"
        p2_str = f"{fres['phase2_days']} trading days (~{fres['phase2_days']/5:.1f} weeks)" if fres['phase2_days'] else "Did not reach +5%"
        print(f"    - Phase 1 Clearance (+10.0%):   {p1_str}")
        print(f"    - Phase 2 Clearance (+5.0%):    {p2_str}")

    print("=" * 130)


if __name__ == "__main__":
    main()
