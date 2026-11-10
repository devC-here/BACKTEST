#!/usr/bin/env python3
"""
================================================================================
  COMPREHENSIVE PROP CHALLENGE COMBO STRESS TEST (MONTE CARLO COMPARISON)
================================================================================
Compares 11 candidate combinations & risk architectures across 500 IDENTICAL
random calendar start timestamps spanning the full 5-year tick library (2021-2026).

Prop Rules (FTMO / FundedNext standard):
  - Initial Capital: $100,000
  - Phase 1 Target: +8.0% ($8,000)
  - Phase 2 Target: +5.0% ($5,000)
  - Max Daily Loss Limit: 5.0% ($5,000)
  - Max Total DD Limit: 10.0% ($10,000)
  - Default Daily Loss Circuit Breaker: 2.0% ($2,000 hard stop per day)
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
from typing import Dict, List, Any, Callable

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True

ROOT_DIR = Path(__file__).resolve().parent.parent
MASTER_FILE = ROOT_DIR / "results" / "master_7_prop_trades_5y.json"


def load_trades():
    print(f"[+] Loading trade pool from {MASTER_FILE}...")
    with open(MASTER_FILE, 'r') as f:
        trades = json.load(f)
    trades.sort(key=lambda x: x['entry_time'])
    return trades


def run_single_prop_trial(
    trades: List[Dict[str, Any]],
    trade_times: List[int],
    start_time: int,
    risk_func: Callable[[str, float, float, int], float],
    daily_stop_pct: float = 2.0,
    max_daily_losses: int = 999,
    use_dd_shield: bool = False,
    strat_avg_losses: Dict[str, float] = None
) -> Dict[str, Any]:
    INITIAL_CAPITAL = 100000.0
    P1_TARGET = 8000.0       # +8.0%
    P2_TARGET = 5000.0       # +5.0%
    MAX_DAILY_LOSS = 5000.0  # 5.0%
    MAX_TOTAL_DD = 10000.0   # 10.0%

    start_idx = bisect.bisect_left(trade_times, start_time)
    n_trades = len(trades)
    if start_idx >= n_trades - 10:
        return {'p1_passed': False, 'p2_passed': False, 'funded': False, 'fail_reason': 'Near end of dataset'}

    trial_start_date = pd.to_datetime(trades[start_idx]['entry_time'], unit='ms').strftime('%Y-%m-%d')

    if strat_avg_losses is None:
        strat_avg_losses = {}

    # ==================== PHASE 1 ====================
    bal = INITIAL_CAPITAL
    peak = bal
    current_day = None
    daily_start_bal = bal
    day_loss = 0.0
    day_loss_count = 0
    p1_days = 0
    p1_trades = 0
    max_dd_p1 = 0.0
    max_daily_dd_p1 = 0.0
    p1_passed = False
    fail_reason = None

    k = start_idx
    while k < n_trades:
        tr = trades[k]
        t_day = pd.to_datetime(tr['entry_time'], unit='ms').date()

        if t_day != current_day:
            current_day = t_day
            daily_start_bal = bal
            day_loss = 0.0
            day_loss_count = 0
            p1_days += 1

        # Daily loss circuit breaker & consecutive loss breaker
        if (day_loss / daily_start_bal) * 100.0 >= daily_stop_pct or day_loss_count >= max_daily_losses:
            k += 1
            continue

        pnl_pct_gain = (bal - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100.0
        tot_dd_from_peak = (peak - bal) / peak * 100.0

        # Calculate dynamic risk %
        base_risk = risk_func(tr['strategy'], pnl_pct_gain, tot_dd_from_peak, 1)

        # Drawdown shield: if in > 3% DD, reduce risk by 40%
        if use_dd_shield and tot_dd_from_peak >= 3.0:
            base_risk *= 0.60

        avg_loss = strat_avg_losses.get(tr['strategy'], 8.5)
        r_mult = tr['pips'] / avg_loss
        pnl = r_mult * (bal * (base_risk / 100.0))
        bal += pnl
        p1_trades += 1

        if pnl < 0:
            day_loss += abs(pnl)
            day_loss_count += 1

        day_dd_pct = (day_loss / daily_start_bal) * 100.0
        if day_dd_pct > max_daily_dd_p1:
            max_daily_dd_p1 = day_dd_pct
        if day_loss >= MAX_DAILY_LOSS:
            fail_reason = "Daily DD Breached in P1"
            break

        if bal > peak:
            peak = bal
        tot_dd_pct = (peak - bal) / peak * 100.0
        if tot_dd_pct > max_dd_p1:
            max_dd_p1 = tot_dd_pct
        if tot_dd_pct >= 10.0:
            fail_reason = "Total DD Breached in P1"
            break

        if bal >= INITIAL_CAPITAL + P1_TARGET:
            p1_passed = True
            break
        k += 1

    # ==================== PHASE 2 ====================
    p2_passed = False
    p2_days = 0
    p2_trades = 0
    max_dd_p2 = 0.0
    max_daily_dd_p2 = 0.0

    if p1_passed:
        bal2 = INITIAL_CAPITAL
        peak2 = bal2
        curr_d2 = None
        day_start2 = bal2
        day_l2 = 0.0
        day_loss_count2 = 0
        k2 = k + 1

        while k2 < n_trades:
            tr2 = trades[k2]
            t_day2 = pd.to_datetime(tr2['entry_time'], unit='ms').date()

            if t_day2 != curr_d2:
                curr_d2 = t_day2
                day_start2 = bal2
                day_l2 = 0.0
                day_loss_count2 = 0
                p2_days += 1

            if (day_l2 / day_start2) * 100.0 >= daily_stop_pct or day_loss_count2 >= max_daily_losses:
                k2 += 1
                continue

            pnl_pct_gain2 = (bal2 - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100.0
            tot_dd_from_peak2 = (peak2 - bal2) / peak2 * 100.0

            base_risk2 = risk_func(tr2['strategy'], pnl_pct_gain2, tot_dd_from_peak2, 2)
            if use_dd_shield and tot_dd_from_peak2 >= 3.0:
                base_risk2 *= 0.60

            avg_loss2 = strat_avg_losses.get(tr2['strategy'], 8.5)
            r_mult2 = tr2['pips'] / avg_loss2
            pnl2 = r_mult2 * (bal2 * (base_risk2 / 100.0))
            bal2 += pnl2
            p2_trades += 1

            if pnl2 < 0:
                day_l2 += abs(pnl2)
                day_loss_count2 += 1

            day_dd_pct2 = (day_l2 / day_start2) * 100.0
            if day_dd_pct2 > max_daily_dd_p2:
                max_daily_dd_p2 = day_dd_pct2
            if day_l2 >= MAX_DAILY_LOSS:
                fail_reason = "Daily DD Breached in P2"
                break

            if bal2 > peak2:
                peak2 = bal2
            tot_dd_pct2 = (peak2 - bal2) / peak2 * 100.0
            if tot_dd_pct2 > max_dd_p2:
                max_dd_p2 = tot_dd_pct2
            if tot_dd_pct2 >= 10.0:
                fail_reason = "Total DD Breached in P2"
                break

            if bal2 >= INITIAL_CAPITAL + P2_TARGET:
                p2_passed = True
                break
            k2 += 1

    return {
        'start_time': start_time,
        'start_date': trial_start_date,
        'p1_passed': p1_passed,
        'p2_passed': p2_passed,
        'funded': p1_passed and p2_passed,
        'fail_reason': fail_reason,
        'p1_days': p1_days,
        'p1_trades': p1_trades,
        'p1_max_dd': max_dd_p1,
        'p1_daily_dd': max_daily_dd_p1,
        'total_trading_days': p1_days + p2_days,
        'total_trades': p1_trades + p2_trades,
        'max_dd_overall': max(max_dd_p1, max_dd_p2)
    }


def evaluate_combo(
    name: str,
    all_trades: List[Dict[str, Any]],
    random_start_times: List[int],
    allowed_strats: List[str],
    risk_func: Callable[[str, float, float, int], float],
    daily_stop_pct: float = 2.0,
    max_daily_losses: int = 999,
    use_dd_shield: bool = False
) -> Dict[str, Any]:
    strat_set = set(allowed_strats)
    filtered = [t for t in all_trades if t['strategy'] in strat_set]
    trade_times = [t['entry_time'] for t in filtered]

    strat_avg_losses = {}
    for s in allowed_strats:
        s_losses = [abs(t['pips']) for t in filtered if t['strategy'] == s and t['pips'] < 0]
        strat_avg_losses[s] = float(np.mean(s_losses)) if s_losses else 8.5

    trials = [
        run_single_prop_trial(
            filtered, trade_times, st, risk_func, daily_stop_pct, max_daily_losses, use_dd_shield, strat_avg_losses
        )
        for st in random_start_times
    ]

    n_trials = len(trials)
    p1_pass = [t for t in trials if t['p1_passed']]
    funded = [t for t in trials if t['funded']]
    daily_breaches = [t for t in trials if t['fail_reason'] and "Daily DD" in t['fail_reason']]
    total_breaches = [t for t in trials if t['fail_reason'] and "Total DD" in t['fail_reason']]

    p1_rate = len(p1_pass) / n_trials * 100.0
    funded_rate = len(funded) / n_trials * 100.0

    p1_days = [t['p1_days'] for t in p1_pass] if p1_pass else [0]
    total_days = [t['total_trading_days'] for t in funded] if funded else [0]
    all_dds = [t['max_dd_overall'] for t in trials]
    worst_days = [t['p1_daily_dd'] for t in trials]

    return {
        'name': name,
        'strategies': len(allowed_strats),
        'trades_pool': len(filtered),
        'p1_pass_rate': p1_rate,
        'funded_rate': funded_rate,
        'daily_breaches': len(daily_breaches),
        'total_breaches': len(total_breaches),
        'median_p1_days': int(np.median(p1_days)),
        'fastest_p1_days': min(p1_days),
        'median_funded_days': int(np.median(total_days)),
        'fastest_funded_days': min(total_days),
        'avg_max_dd': float(np.mean(all_dds)),
        'peak_max_dd': float(max(all_dds)),
        'avg_worst_day': float(np.mean(worst_days)),
        'peak_worst_day': float(max(worst_days))
    }


def main():
    print("=" * 115)
    print("   MONTE CARLO PROP COMBO BENCHMARK: 11 COMBOS TESTED ON 500 EXACT UNSEEN DATES   ".center(115))
    print("=" * 115)

    trades = load_trades()
    n_trades = len(trades)
    t_min = trades[0]['entry_time']
    t_max = trades[-1]['entry_time']
    print(f"Master Trade Pool: {n_trades} trades across 5 years ({pd.to_datetime(t_min, unit='ms').date()} to {pd.to_datetime(t_max, unit='ms').date()})\n")

    # Uniformly generate 500 random start timestamps up to 140 days before data end
    cutoff_ms = 140 * 24 * 3600 * 1000
    random.seed(42)
    N_TRIALS = 500
    random_start_times = sorted([random.randint(t_min, t_max - cutoff_ms) for _ in range(N_TRIALS)])

    # Setup 11 Combos
    combos = []

    # 1. Baseline 5-Champion Conservative (0.50% flat)
    c1_strats = ["DISC42_H4_M30", "DISC43_H1_M5", "DISC40_RSI", "DISC38_MACD", "HF_M15"]
    combos.append((
        "1. Baseline 5-Champion (0.50% flat)",
        c1_strats,
        lambda s, gain, dd, phase: 0.50,
        2.0, 999, False
    ))

    # 2. Baseline 5-Champion Speedrun (0.70% flat)
    combos.append((
        "2. Baseline 5-Champion (0.70% flat)",
        c1_strats,
        lambda s, gain, dd, phase: 0.70,
        2.0, 999, False
    ))

    # 3. Pure New Discoveries 6-Pack (0.50% flat, NO HF_M15)
    c3_strats = ["DISC42_H4_M30", "DISC43_H1_M5", "DISC40_RSI", "DISC41_THURSDAY", "DISC38_MACD", "DISC39_STOCH"]
    combos.append((
        "3. Pure New Discoveries 6-Pack (0.50% flat, NO HF_M15)",
        c3_strats,
        lambda s, gain, dd, phase: 0.50,
        2.0, 999, False
    ))

    # 4. Pure New Discoveries 6-Pack Speedrun (0.70% flat, NO HF_M15)
    combos.append((
        "4. Pure New Discoveries 6-Pack (0.70% flat, NO HF_M15)",
        c3_strats,
        lambda s, gain, dd, phase: 0.70,
        2.0, 999, False
    ))

    # 5. Elite Precision 4-Star Core (0.70% flat: DISC42, 43, 40, 41)
    c5_strats = ["DISC42_H4_M30", "DISC43_H1_M5", "DISC40_RSI", "DISC41_THURSDAY"]
    combos.append((
        "5. Elite Precision 4-Star Core (0.70% flat)",
        c5_strats,
        lambda s, gain, dd, phase: 0.70,
        2.0, 999, False
    ))

    # 6. Sniper Trio (0.75% flat: DISC42, 43, 40)
    c6_strats = ["DISC42_H4_M30", "DISC43_H1_M5", "DISC40_RSI"]
    combos.append((
        "6. Sniper Trio (0.75% flat: DISC42, 43, 40)",
        c6_strats,
        lambda s, gain, dd, phase: 0.75,
        2.0, 999, False
    ))

    # 7. Asymmetric Alpha-Weighted Allocation (All 7 Models)
    def alpha_weighted_risk(s: str, gain: float, dd: float, phase: int) -> float:
        weights = {
            'DISC42_H4_M30': 1.00,
            'DISC43_H1_M5': 0.80,
            'DISC40_RSI': 0.55,
            'DISC41_THURSDAY': 0.45,
            'DISC38_MACD': 0.40,
            'DISC39_STOCH': 0.35,
            'HF_M15': 0.20
        }
        return weights.get(s, 0.50)

    combos.append((
        "7. Alpha-Weighted 7-Model Fleet",
        ["DISC42_H4_M30", "DISC43_H1_M5", "DISC40_RSI", "DISC41_THURSDAY", "DISC38_MACD", "DISC39_STOCH", "HF_M15"],
        alpha_weighted_risk,
        2.0, 999, False
    ))

    # 8. 4-Star Core + Milestone Tapering Shield
    def milestone_taper_risk(s: str, gain: float, dd: float, phase: int) -> float:
        if phase == 1:
            if gain >= 7.0:
                return 0.20
            elif gain >= 5.5:
                return 0.35
            else:
                return 0.65
        else:
            if gain >= 4.2:
                return 0.15
            elif gain >= 3.2:
                return 0.25
            else:
                return 0.50

    combos.append((
        "8. 4-Star Core + Milestone Tapering Shield",
        c5_strats,
        milestone_taper_risk,
        2.0, 999, False
    ))

    # 9. Pure Discoveries 6-Pack + Milestone Tapering Shield
    def pack6_milestone_risk(s: str, gain: float, dd: float, phase: int) -> float:
        if phase == 1:
            if gain >= 7.0:
                return 0.20
            elif gain >= 5.5:
                return 0.35
            else:
                return 0.65
        else:
            if gain >= 4.2:
                return 0.15
            elif gain >= 3.2:
                return 0.25
            else:
                return 0.50

    combos.append((
        "9. Discoveries 6-Pack + Milestone Tapering Shield",
        c3_strats,
        pack6_milestone_risk,
        2.0, 999, False
    ))

    # 10. UNBREAKABLE CHAMPION (Weighted + Taper + 2-Loss Halt + DD Shield on Top 5)
    c10_strats = ["DISC42_H4_M30", "DISC43_H1_M5", "DISC40_RSI", "DISC41_THURSDAY", "DISC38_MACD"]
    def unbreakable_risk(s: str, gain: float, dd: float, phase: int) -> float:
        weights = {
            'DISC42_H4_M30': 1.10,
            'DISC43_H1_M5': 0.85,
            'DISC40_RSI': 0.60,
            'DISC41_THURSDAY': 0.50,
            'DISC38_MACD': 0.45
        }
        w = weights.get(s, 0.50)
        if phase == 1:
            if gain >= 6.8:
                w *= 0.30
            elif gain >= 5.2:
                w *= 0.60
        else:
            if gain >= 4.0:
                w *= 0.30
            elif gain >= 3.0:
                w *= 0.60
        return w

    combos.append((
        "10. UNBREAKABLE CHAMPION (Top 5 + Weighted + Taper + Breakers)",
        c10_strats,
        unbreakable_risk,
        2.0, 2, True
    ))

    # 11. High-Precision Duo (DISC42 + DISC43 at 0.90% flat)
    c11_strats = ["DISC42_H4_M30", "DISC43_H1_M5"]
    combos.append((
        "11. High-Precision Duo (DISC42 + DISC43 @ 0.90% flat)",
        c11_strats,
        lambda s, gain, dd, phase: 0.90,
        2.0, 999, False
    ))

    results = []
    print(f"Simulating {len(combos)} distinct combos across {N_TRIALS} identical random start dates...\n")
    for name, strats, r_func, d_stop, m_loss, dd_shield in combos:
        t0 = time.time()
        res = evaluate_combo(name, trades, random_start_times, strats, r_func, d_stop, m_loss, dd_shield)
        dt = time.time() - t0
        results.append(res)
        print(f"[*] Evaluated in {dt:.1f}s: {name:<62} | P1: {res['p1_pass_rate']:>5.1f}% | Funded: {res['funded_rate']:>5.1f}% | Max DD: {res['peak_max_dd']:>5.2f}%")

    print("\n" + "=" * 135)
    print("                      MONTE CARLO PROP CHALLENGE LEADERBOARD (500 IDENTICAL UNSEEN START DATES)                      ".center(135))
    print("=" * 135)
    headers = ["Combo Configuration", "P1 Pass", "FUNDED Rate", "Worst Day", "Peak DD", "P1 Days (Med)", "Funded (Med)"]
    print(f"{headers[0]:<64} | {headers[1]:<8} | {headers[2]:<11} | {headers[3]:<10} | {headers[4]:<10} | {headers[5]:<13} | {headers[6]:<14}")
    print("-" * 135)

    # Sort by Funded Rate descending
    results.sort(key=lambda x: -x['funded_rate'])
    for r in results:
        print(f"{r['name']:<64} | {r['p1_pass_rate']:>6.1f}% | {r['funded_rate']:>9.1f}% | {r['peak_worst_day']:>8.2f}% | {r['peak_max_dd']:>8.2f}% | {r['median_p1_days']:>11} d | {r['median_funded_days']:>12} d")

    print("=" * 135)

    # Save to results
    out_json = ROOT_DIR / "results" / "prop_challenge_combos_monte_carlo.json"
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n[+] Saved detailed benchmark results to {out_json}")


if __name__ == '__main__':
    main()
