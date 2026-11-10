#!/usr/bin/env python3
"""
================================================================================
  TWO-STAGE PROP FRAMEWORK: CHALLENGE CLEARING VS. FUNDED ACCOUNT GROWTH
================================================================================
Evaluates two distinct strategy combos engineered for two distinct mandates:

STAGE 1: THE CHALLENGE CLEARING COMBO (Speedrun & High Pass Rate)
  Mandate: Clear Phase 1 (+8%) and Phase 2 (+5%) in minimal calendar time
  Risk Profile: Dynamic Alpha-Weighted (0.60% - 0.90%) with Milestone Tapering
  Active Models:
    - DISC42: Macro Structure H4 Level + M30 Trigger (72.9% WR, PF 3.97)
    - DISC43: Intraday Precision H1 Level + M5 Trigger (48.3% WR, PF 1.47)
    - DISC40: RSI Regime Range Shift (44.0% WR, PF 1.31)
    - DISC41: Thursday Macro Trend Ignition (43.6% WR, PF 1.30)
    - DISC38: London Midday MACD Acceleration (43.0% WR, PF 1.28)

STAGE 2: THE FUNDED ACCOUNT GROWTH COMBO (Consistent Monthly Cash Flow)
  Mandate: Produce steady monthly payouts (2% - 5%/mo) with minimal drawdown (< 4%)
  Risk Profile: Conservative Risk Parity (0.25% - 0.35% per trade)
  Active Models: All-Weather 7-Model Multi-Session Fleet
    - DISC42, DISC43, DISC40, DISC41, DISC38, DISC39, HF_M15
================================================================================
"""

from __future__ import annotations
import sys
import os
import time
import json
import random
import bisect
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True

ROOT_DIR = Path(__file__).resolve().parent.parent
MASTER_FILE = ROOT_DIR / "results" / "master_7_prop_trades_5y.json"


def load_trades():
    with open(MASTER_FILE, 'r') as f:
        trades = json.load(f)
    trades.sort(key=lambda x: x['entry_time'])
    return trades


# ==============================================================================
# 1. STAGE 1: CHALLENGE CLEARING SIMULATION
# ==============================================================================
def simulate_challenge_clearing(trades: List[Dict[str, Any]], n_trials: int = 500):
    print("=" * 105)
    print("   STAGE 1: EVALUATING CHALLENGE CLEARING COMBOS (500 MONTE CARLO RANDOM STARTS)   ".center(105))
    print("=" * 105)

    t_min = trades[0]['entry_time']
    t_max = trades[-1]['entry_time']
    cutoff_ms = 140 * 24 * 3600 * 1000
    random.seed(42)
    random_start_times = sorted([random.randint(t_min, t_max - cutoff_ms) for _ in range(n_trials)])

    # Combos to test for clearing:
    # Combo A: Precision Duo (DISC42 + DISC43 @ 0.90% speedrun)
    # Combo B: 4-Star Core (DISC42, 43, 40, 41 @ 0.70%)
    # Combo C: Alpha-Weighted Top 5 + Milestone Taper (Unbreakable Clearing Fleet)
    clearing_combos = [
        {
            'name': 'Clearing Combo A: Precision Duo Speedrun (DISC42 + DISC43 @ 0.90%)',
            'strats': ['DISC42_H4_M30', 'DISC43_H1_M5'],
            'risk_fn': lambda s, gain, p: 0.90,
            'd_stop': 2.0,
            'max_loss': 999
        },
        {
            'name': 'Clearing Combo B: Elite 4-Star Precision Core (0.70% Flat)',
            'strats': ['DISC42_H4_M30', 'DISC43_H1_M5', 'DISC40_RSI', 'DISC41_THURSDAY'],
            'risk_fn': lambda s, gain, p: 0.70,
            'd_stop': 2.0,
            'max_loss': 999
        },
        {
            'name': 'Clearing Combo C: High-Certainty 5-Fleet + Milestone Taper (Optimal)',
            'strats': ['DISC42_H4_M30', 'DISC43_H1_M5', 'DISC40_RSI', 'DISC41_THURSDAY', 'DISC38_MACD'],
            'risk_fn': lambda s, gain, p: (
                (1.00 if s == 'DISC42_H4_M30' else (0.80 if s == 'DISC43_H1_M5' else 0.55)) *
                (0.35 if (p == 1 and gain >= 6.8) or (p == 2 and gain >= 4.0) else
                 (0.60 if (p == 1 and gain >= 5.2) or (p == 2 and gain >= 3.0) else 1.0))
            ),
            'd_stop': 2.0,
            'max_loss': 2
        }
    ]

    for cfg in clearing_combos:
        allowed = set(cfg['strats'])
        filt_trades = [t for t in trades if t['strategy'] in allowed]
        filt_times = [t['entry_time'] for t in filt_trades]
        n_filt = len(filt_trades)

        # Precompute strat avg losses
        strat_losses = {}
        for s in cfg['strats']:
            sl = [abs(t['pips']) for t in filt_trades if t['strategy'] == s and t['pips'] < 0]
            strat_losses[s] = float(np.mean(sl)) if sl else 8.5

        p1_passes = 0
        funded_passes = 0
        p1_days_list = []
        funded_days_list = []
        peak_dds = []
        worst_days = []

        for st in random_start_times:
            idx = bisect.bisect_left(filt_times, st)
            if idx >= n_filt - 10:
                continue

            # Phase 1
            bal = 100000.0
            peak = bal
            curr_d = None
            d_start = bal
            d_loss = 0.0
            d_loss_cnt = 0
            p1_days = 0
            p1_pass = False
            max_dd = 0.0
            max_daily_dd = 0.0

            k = idx
            while k < n_filt:
                tr = filt_trades[k]
                t_day = pd.to_datetime(tr['entry_time'], unit='ms').date()
                if t_day != curr_d:
                    curr_d = t_day
                    d_start = bal
                    d_loss = 0.0
                    d_loss_cnt = 0
                    p1_days += 1

                if (d_loss / d_start) * 100.0 >= cfg['d_stop'] or d_loss_cnt >= cfg['max_loss']:
                    k += 1
                    continue

                gain = (bal - 100000.0) / 100000.0 * 100.0
                r_pct = cfg['risk_fn'](tr['strategy'], gain, 1)
                r_mult = tr['pips'] / strat_losses.get(tr['strategy'], 8.5)
                pnl = r_mult * (bal * (r_pct / 100.0))
                bal += pnl

                if pnl < 0:
                    d_loss += abs(pnl)
                    d_loss_cnt += 1

                d_dd = (d_loss / d_start) * 100.0
                if d_dd > max_daily_dd: max_daily_dd = d_dd
                if d_loss >= 5000.0: break

                if bal > peak: peak = bal
                dd = (peak - bal) / peak * 100.0
                if dd > max_dd: max_dd = dd
                if dd >= 10.0: break

                if bal >= 108000.0:
                    p1_pass = True
                    break
                k += 1

            if p1_pass:
                p1_passes += 1
                p1_days_list.append(p1_days)

                # Phase 2
                bal2 = 100000.0
                peak2 = bal2
                curr_d2 = None
                d_start2 = bal2
                d_l2 = 0.0
                d_loss_cnt2 = 0
                p2_days = 0
                p2_pass = False

                k2 = k + 1
                while k2 < n_filt:
                    tr2 = filt_trades[k2]
                    t_day2 = pd.to_datetime(tr2['entry_time'], unit='ms').date()
                    if t_day2 != curr_d2:
                        curr_d2 = t_day2
                        d_start2 = bal2
                        d_l2 = 0.0
                        d_loss_cnt2 = 0
                        p2_days += 1

                    if (d_l2 / d_start2) * 100.0 >= cfg['d_stop'] or d_loss_cnt2 >= cfg['max_loss']:
                        k2 += 1
                        continue

                    gain2 = (bal2 - 100000.0) / 100000.0 * 100.0
                    r_pct2 = cfg['risk_fn'](tr2['strategy'], gain2, 2)
                    r_mult2 = tr2['pips'] / strat_losses.get(tr2['strategy'], 8.5)
                    pnl2 = r_mult2 * (bal2 * (r_pct2 / 100.0))
                    bal2 += pnl2

                    if pnl2 < 0:
                        d_l2 += abs(pnl2)
                        d_loss_cnt2 += 1

                    d_dd2 = (d_l2 / d_start2) * 100.0
                    if d_dd2 > max_daily_dd: max_daily_dd = d_dd2
                    if d_l2 >= 5000.0: break

                    if bal2 > peak2: peak2 = bal2
                    dd2 = (peak2 - bal2) / peak2 * 100.0
                    if dd2 > max_dd: max_dd = dd2
                    if dd2 >= 10.0: break

                    if bal2 >= 105000.0:
                        p2_pass = True
                        break
                    k2 += 1

                if p2_pass:
                    funded_passes += 1
                    funded_days_list.append(p1_days + p2_days)

            peak_dds.append(max_dd)
            worst_days.append(max_daily_dd)

        p1_rate = p1_passes / n_trials * 100.0
        fund_rate = funded_passes / n_trials * 100.0
        med_p1 = int(np.median(p1_days_list)) if p1_days_list else 0
        med_fund = int(np.median(funded_days_list)) if funded_days_list else 0

        print(f"\n[-] {cfg['name']}:")
        print(f"    - Phase 1 Pass Rate : {p1_rate:.1f}% ({p1_passes} / {n_trials}) | Median Days: {med_p1} days (~{med_p1/5:.1f} weeks)")
        print(f"    - Fully Funded Rate : {fund_rate:.1f}% ({funded_passes} / {n_trials}) | Median Days: {med_fund} days (~{med_fund/5:.1f} weeks)")
        print(f"    - Worst Day Recorded: {max(worst_days):.2f}% (Limit 5.0%) | Peak Max DD: {max(peak_dds):.2f}% (Limit 10.0%)")


# ==============================================================================
# 2. STAGE 2: FUNDED ACCOUNT GROWTH & COMPOUNDING SIMULATION
# ==============================================================================
def simulate_funded_account_growth(trades: List[Dict[str, Any]]):
    print("\n" + "=" * 105)
    print("   STAGE 2: EVALUATING FUNDED ACCOUNT GROWTH & COMPOUNDING (FULL 5 YEARS / 60 MONTHS)   ".center(105))
    print("=" * 105)

    # Growth Mandate:
    # Initial Capital: $100,000 (Funded Account)
    # Objective: Generate steady monthly payouts with Max DD strictly < 4.5% (zero danger to account)
    # Risk Profile: Conservative Risk Parity (0.25% to 0.35% per trade)
    growth_fleets = [
        {
            'name': 'Growth Fleet 1: The All-Weather 7-Model Multi-Session Fleet',
            'desc': 'All 7 models, balanced risk (0.25% base risk, 0.40% on DISC42/43, 0.15% on HF_M15)',
            'strats': ['DISC42_H4_M30', 'DISC43_H1_M5', 'DISC40_RSI', 'DISC41_THURSDAY', 'DISC38_MACD', 'DISC39_STOCH', 'HF_M15'],
            'weights': {
                'DISC42_H4_M30': 0.45,
                'DISC43_H1_M5': 0.40,
                'DISC40_RSI': 0.30,
                'DISC41_THURSDAY': 0.25,
                'DISC38_MACD': 0.25,
                'DISC39_STOCH': 0.20,
                'HF_M15': 0.12
            }
        },
        {
            'name': 'Growth Fleet 2: The High-Sharpe 5-Model Core (Zero Scalper Drag)',
            'desc': 'Excludes HF_M15 and DISC39, focusing strictly on top 5 models with 0.35% risk',
            'strats': ['DISC42_H4_M30', 'DISC43_H1_M5', 'DISC40_RSI', 'DISC41_THURSDAY', 'DISC38_MACD'],
            'weights': {
                'DISC42_H4_M30': 0.50,
                'DISC43_H1_M5': 0.45,
                'DISC40_RSI': 0.35,
                'DISC41_THURSDAY': 0.30,
                'DISC38_MACD': 0.30
            }
        }
    ]

    for gf in growth_fleets:
        allowed = set(gf['strats'])
        f_trades = [t for t in trades if t['strategy'] in allowed]
        strat_losses = {}
        for s in gf['strats']:
            sl = [abs(t['pips']) for t in f_trades if t['strategy'] == s and t['pips'] < 0]
            strat_losses[s] = float(np.mean(sl)) if sl else 8.5

        INITIAL_CAPITAL = 100000.0
        bal = INITIAL_CAPITAL
        peak = bal
        max_dd_pct = 0.0
        daily_max_dd_pct = 0.0

        # Group trades by calendar month to analyze payout consistency
        monthly_pnls = {}
        curr_day = None
        d_start = bal
        d_loss = 0.0

        for tr in f_trades:
            dt = pd.to_datetime(tr['entry_time'], unit='ms')
            m_key = dt.strftime('%Y-%m')
            t_day = dt.date()

            if t_day != curr_day:
                curr_day = t_day
                d_start = bal
                d_loss = 0.0

            # Daily circuit breaker: 1.5% max daily risk
            if (d_loss / d_start) * 100.0 >= 1.5:
                continue

            r_pct = gf['weights'].get(tr['strategy'], 0.25)
            r_mult = tr['pips'] / strat_losses.get(tr['strategy'], 8.5)
            pnl = r_mult * (bal * (r_pct / 100.0))
            bal += pnl

            if pnl < 0:
                d_loss += abs(pnl)

            d_dd = (d_loss / d_start) * 100.0
            if d_dd > daily_max_dd_pct:
                daily_max_dd_pct = d_dd

            if bal > peak:
                peak = bal
            dd = (peak - bal) / peak * 100.0
            if dd > max_dd_pct:
                max_dd_pct = dd

            if m_key not in monthly_pnls:
                monthly_pnls[m_key] = 0.0
            monthly_pnls[m_key] += pnl

        m_values = list(monthly_pnls.values())
        winning_months = [v for v in m_values if v > 0]
        win_month_rate = len(winning_months) / len(m_values) * 100.0 if m_values else 0
        total_pnl = bal - INITIAL_CAPITAL
        total_return_pct = total_pnl / INITIAL_CAPITAL * 100.0
        avg_monthly_payout = float(np.mean(m_values))
        avg_monthly_pct = avg_monthly_payout / INITIAL_CAPITAL * 100.0
        annualized_return_pct = total_return_pct / 5.0

        print(f"\n[-] {gf['name'].upper()}:")
        print(f"    - Active Strategy Models    : {len(gf['strats'])} models ({', '.join(gf['strats'])})")
        print(f"    - Total 5-Year Net Profit   : ${total_pnl:,.2f} (+{total_return_pct:.2f}% total / +{annualized_return_pct:.2f}% per year)")
        print(f"    - Peak Total Drawdown       : {max_dd_pct:.2f}% (Safety margin vs 10% limit: {10.0 - max_dd_pct:.2f}%)")
        print(f"    - Peak Daily Drawdown       : {daily_max_dd_pct:.2f}% (Safety margin vs 5% daily limit: {5.0 - daily_max_dd_pct:.2f}%)")
        print(f"    - Total Months Evaluated    : {len(m_values)} consecutive months (2021 to 2026)")
        print(f"    - Winning Months Ratio      : {win_month_rate:.1f}% ({len(winning_months)} / {len(m_values)} profitable months)")
        print(f"    - Average Monthly Return    : +{avg_monthly_pct:.2f}% (${avg_monthly_payout:,.2f} per month on $100k account)")
        print(f"    - Worst Monthly Drawdown    : {min(m_values) / INITIAL_CAPITAL * 100.0:.2f}% (${min(m_values):,.2f})")
        print(f"    - Best Month                : +{max(m_values) / INITIAL_CAPITAL * 100.0:.2f}% (${max(m_values):,.2f})")
        print(f"    - Return / Drawdown Ratio   : {total_return_pct / max_dd_pct:.2f} : 1")


def main():
    trades = load_trades()
    simulate_challenge_clearing(trades, n_trials=500)
    simulate_funded_account_growth(trades)


if __name__ == '__main__':
    main()
