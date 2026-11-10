#!/usr/bin/env python3
"""
================================================================================
  RANDOM UNSEEN PROP FIRM CHALLENGE STRESS TEST (500 MONTE CARLO TRIALS)
================================================================================
Tests the 5-Champion True Alpha Fleet:
  1. DISC42: Macro Structure H4 Level + M30 Trigger Elite (72.9% WR, PF 3.97)
  2. DISC40: RSI Regime Range Shift Elite (44.0% WR, PF 1.31, +1,112.8p)
  3. DISC43: Intraday Precision H1 Level + M5 Trigger Elite (48.3% WR, PF 1.47)
  4. DISC38: London Midday MACD Acceleration Elite (43.0% WR, PF 1.28)
  5. HF_M15: HighFrequency M15 Scalper Elite (35.7% WR, PF 1.13, +1,755.9p)

Simulates 500 randomized prop challenges starting at random unseen dates
across the full 5-year tick library (262 weeks / 123.22M ticks):
  - Initial Capital: $100,000
  - Phase 1 Target: +8.0% ($8,000)
  - Phase 2 Target: +5.0% ($5,000)
  - Max Daily Loss Limit: 5.0% ($5,000)
  - Max Total Drawdown Limit: 10.0% ($10,000)
  - Daily Loss Circuit Breaker: 2.0% ($2,000 hard stop per day)
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

from tick_engine.core.engine import BacktestEngine
from tick_engine.core.dataloader import get_week_dataset

# 5 Champions
from strategies.high_frequency_m15_scalper_elite.strategy import HighFrequencyM15ScalperElite
from strategies.new_discoveries.disc40_rsi_regime_range_shift_elite.strategy import RSIRegimeRangeShiftElite
from strategies.new_discoveries.disc42_macro_structure_h4_m30_elite.strategy import MacroStructureH4M30Elite
from strategies.new_discoveries.disc43_intraday_precision_h1_m5_elite.strategy import IntradayPrecisionH1M5Elite
from strategies.disc38_london_midday_macd_acceleration_elite.strategy import LondonMiddayMACDAccelerationElite

PIP_SIZE = 0.0001
RESULTS_DIR = ROOT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_FILE = RESULTS_DIR / "5_champion_prop_trades_5y.json"


def _simulate_week_5_champions(w: int):
    ticks, _ = get_week_dataset(w)
    engine = BacktestEngine(pip_size=PIP_SIZE, use_slippage=True)

    strats = [
        ("HF_M15", HighFrequencyM15ScalperElite()),
        ("DISC40_RSI", RSIRegimeRangeShiftElite()),
        ("DISC42_H4_M30", MacroStructureH4M30Elite()),
        ("DISC43_H1_M5", IntradayPrecisionH1M5Elite()),
        ("DISC38_MACD", LondonMiddayMACDAccelerationElite())
    ]

    trades = []
    for s_id, s in strats:
        tr = engine.run(ticks, s)
        for t in tr:
            trades.append({
                'week': w,
                'entry_time': int(t['entry_time']),
                'exit_time': int(t['exit_time']),
                'pips': float(t['pips']),
                'strategy': s_id
            })

    return trades


def load_or_generate_trades() -> List[Dict[str, Any]]:
    if CACHE_FILE.exists():
        print(f"[+] Loading cached 5-Champion trade pool from {CACHE_FILE}...")
        with open(CACHE_FILE, 'r') as f:
            trades = json.load(f)
        return trades

    print("[*] Simulating 5 Champion models across 262 weeks (~123M ticks)...")
    weeks = list(range(1, 263))
    t0 = time.time()
    num_workers = min(4, os.cpu_count() or 4)

    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        week_res = list(ex.map(_simulate_week_5_champions, weeks))

    dt = time.time() - t0
    all_trades = [t for wr in week_res for t in wr]
    all_trades.sort(key=lambda x: x['entry_time'])
    print(f"[+] 262 weeks simulated in {dt:.1f}s. Extracted {len(all_trades)} trades.")

    with open(CACHE_FILE, 'w') as f:
        json.dump(all_trades, f)
    print(f"[+] Saved trade cache to {CACHE_FILE}.")
    return all_trades


def run_random_prop_trial(
    trades: List[Dict[str, Any]],
    start_idx: int,
    risk_pct: float,
    daily_stop_pct: float = 2.0,
    avg_loss: float = 8.5
) -> Dict[str, Any]:
    INITIAL_CAPITAL = 100000.0
    P1_TARGET = 8000.0       # +8.0%
    P2_TARGET = 5000.0       # +5.0%
    MAX_DAILY_LOSS = 5000.0  # 5.0%
    MAX_TOTAL_DD = 10000.0   # 10.0%

    n_total = len(trades)
    trial_start_date = pd.to_datetime(trades[start_idx]['entry_time'], unit='ms').strftime('%Y-%m-%d')

    # ==================== PHASE 1 ====================
    bal = INITIAL_CAPITAL
    peak = bal
    current_day = None
    daily_start_bal = bal
    day_loss = 0.0
    p1_days = 0
    p1_trades = 0
    max_dd_p1 = 0.0
    max_daily_dd_p1 = 0.0
    p1_passed = False
    fail_reason = None

    k = start_idx
    while k < n_total:
        tr = trades[k]
        t_day = pd.to_datetime(tr['entry_time'], unit='ms').date()

        if t_day != current_day:
            current_day = t_day
            daily_start_bal = bal
            day_loss = 0.0
            p1_days += 1

        # Daily loss circuit breaker: halt remaining trades today if cap reached
        if (day_loss / daily_start_bal) * 100.0 >= daily_stop_pct:
            k += 1
            continue

        r_mult = tr['pips'] / avg_loss
        pnl = r_mult * (bal * (risk_pct / 100.0))
        bal += pnl
        p1_trades += 1
        if pnl < 0:
            day_loss += abs(pnl)

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
        k2 = k + 1

        while k2 < n_total:
            tr2 = trades[k2]
            t_day2 = pd.to_datetime(tr2['entry_time'], unit='ms').date()

            if t_day2 != curr_d2:
                curr_d2 = t_day2
                day_start2 = bal2
                day_l2 = 0.0
                p2_days += 1

            if (day_l2 / day_start2) * 100.0 >= daily_stop_pct:
                k2 += 1
                continue

            r_mult2 = tr2['pips'] / avg_loss
            pnl2 = r_mult2 * (bal2 * (risk_pct / 100.0))
            bal2 += pnl2
            p2_trades += 1
            if pnl2 < 0:
                day_l2 += abs(pnl2)

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
        'start_idx': start_idx,
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


def main():
    print("=" * 115)
    print("   MONTE CARLO RANDOM UNSEEN PROP CHALLENGE STRESS TEST (500 RANDOM START DATES)   ".center(115))
    print("=" * 115)

    trades = load_or_generate_trades()
    losses = [abs(t['pips']) for t in trades if t['pips'] < 0]
    avg_loss = float(np.mean(losses)) if losses else 8.5

    n_trades = len(trades)
    print(f"Total 5-Champion Trade Pool: {n_trades} trades | Avg Loss: {avg_loss:.2f} pips | 5Y Net: +{sum(t['pips'] for t in trades):.1f}p\n")

    # Set deterministic random seed for perfect reproducibility
    random.seed(42)
    N_TRIALS = 500
    valid_start_indices = [random.randint(0, n_trades - 120) for _ in range(N_TRIALS)]

    for risk_pct, label in [(0.50, "Conservative (0.50% Risk / $500)"), (0.70, "Accelerated Speedrun (0.70% Risk / $700)")]:
        trials = [run_random_prop_trial(trades, s_idx, risk_pct=risk_pct, avg_loss=avg_loss) for s_idx in valid_start_indices]

        p1_successes = [t for t in trials if t['p1_passed']]
        funded_successes = [t for t in trials if t['funded']]
        daily_fails = [t for t in trials if t['fail_reason'] and "Daily DD" in t['fail_reason']]
        total_fails = [t for t in trials if t['fail_reason'] and "Total DD" in t['fail_reason']]

        p1_rate = len(p1_successes) / N_TRIALS * 100
        funded_rate = len(funded_successes) / N_TRIALS * 100

        p1_days = [t['p1_days'] for t in p1_successes]
        p1_trds = [t['p1_trades'] for t in p1_successes]
        full_days = [t['total_trading_days'] for t in funded_successes]
        all_dds = [t['max_dd_overall'] for t in trials]
        daily_dds = [t['p1_daily_dd'] for t in trials]

        print("=" * 115)
        print(f"CONFIG: {label.upper()} -- TARGET: +8% (P1) / +5% (P2) -- HARD LIMITS: 5% DAILY / 10% TOTAL")
        print("=" * 115)
        print(f"Random Challenge Trials Tested     : {N_TRIALS} random start points spanning 2021 - 2026")
        print(f"Phase 1 Pass Rate                  : {p1_rate:.1f}% ({len(p1_successes)} / {N_TRIALS})")
        print(f"Full 2-Phase Pass Rate (FUNDED!)   : {funded_rate:.1f}% ({len(funded_successes)} / {N_TRIALS})")
        print(f"Daily Drawdown Breaches            : {len(daily_fails)} ({len(daily_fails)/N_TRIALS*100:.1f}%)")
        print(f"Total Drawdown Breaches            : {len(total_fails)} ({len(total_fails)/N_TRIALS*100:.1f}%)")
        print("-" * 115)
        print(f"Fastest Phase 1 Clear              : {min(p1_days)} trading days (~{min(p1_days)/5:.1f} weeks) [{min(p1_trds)} trades]")
        print(f"Median Phase 1 Clear               : {int(np.median(p1_days))} trading days (~{np.median(p1_days)/5:.1f} weeks) [{int(np.median(p1_trds))} trades]")
        print(f"Average Phase 1 Clear              : {np.mean(p1_days):.1f} trading days (~{np.mean(p1_days)/5:.1f} weeks) [{int(np.mean(p1_trds))} trades]")
        print(f"Median Time to Fully Funded        : {int(np.median(full_days))} trading days (~{np.median(full_days)/5:.1f} weeks)")
        print(f"Fastest Time to Fully Funded       : {min(full_days)} trading days (~{min(full_days)/5:.1f} weeks)")
        print(f"Average Maximum Drawdown           : {np.mean(all_dds):.2f}% (Peak Worst DD Encountered: {max(all_dds):.2f}%)")
        print(f"Average Worst Day Drawdown         : {np.mean(daily_dds):.2f}% (Peak Worst Day Encountered: {max(daily_dds):.2f}%)")
        print("-" * 115)

        # Print 8 Concrete Random Walkthroughs
        print(f"\nSAMPLE OF 8 RANDOM UNSEEN CHALLENGE WALKTHROUGHS ({label}):")
        print(f"{'Trial':<7} | {'Random Start Date':<18} | {'P1 Days':<9} | {'P1 Trades':<10} | {'Max DD':<8} | {'Worst Day':<10} | {'Outcome'}")
        print("-" * 90)
        sample_indices = [15, 42, 88, 120, 210, 305, 412, 489]
        for idx in sample_indices:
            t = trials[idx]
            outcome = "PASSED & FUNDED!" if t['funded'] else ("P1 PASSED (P2 stopped)" if t['p1_passed'] else f"FAILED ({t['fail_reason']})")
            print(f"#{idx:<6} | {t['start_date']:<18} | {t['p1_days']:<9} | {t['p1_trades']:<10} | {t['p1_max_dd']:<6.2f}% | {t['p1_daily_dd']:<8.2f}% | {outcome}")
        print()

    print("=" * 115)


if __name__ == '__main__':
    main()
