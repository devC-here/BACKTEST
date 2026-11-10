#!/usr/bin/env python3
"""
================================================================================
  FAST-TRACK PROP CHALLENGE SPEEDRUN AUDIT (REAL 5-YEAR TICK DATA)
================================================================================
Tests high-velocity speedrun configurations to clear funding firm challenges
in minimal calendar time (under 10 to 20 trading days / 2 to 3 weeks) without
ever breaching the 5% daily loss limit or 10% maximum total drawdown.

Models included:
  1. HF_M15_Scalper_Elite (High Velocity M15 FVG & Sweep)
  2. DISC44_High_Frequency_M5_Scalper (Calibrated M5 FVG & Retest)
  3. DISC40_RSI_Regime_Range_Shift (M15 Cash Flow)
  4. DISC43_Intraday_Precision_H1_M5 (High Precision Sniper)
  5. DISC38_London_Midday_MACD (Lunch Momentum Breakout)
  6. S14_Inside_Bar_M30 (VCP Compression Breakout)
  7. S66_Bhanushali_44MA (High Asymmetry Runner, 3.45R)

Sizing Experiments:
  Config 1: 0.50% base risk + 1.5% daily stop
  Config 2: 0.75% speedrun risk + 2.0% daily stop
  Config 3: Dynamic Prop Scaling (0.80% risk when equity > 0, scaling down to 0.40% in drawdown)
================================================================================
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

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tick_engine.core.engine import BacktestEngine, TRADE_DTYPE
from tick_engine.core.dataloader import get_week_dataset

from strategies.high_frequency_m15_scalper_elite.strategy import HighFrequencyM15ScalperElite
from strategies.new_discoveries.disc44_high_frequency_m5_scalper_elite.strategy import HighFrequencyM5ScalperElite
from strategies.new_discoveries.disc40_rsi_regime_range_shift_elite.strategy import RSIRegimeRangeShiftElite
from strategies.new_discoveries.disc43_intraday_precision_h1_m5_elite.strategy import IntradayPrecisionH1M5Elite
from strategies.disc38_london_midday_macd_acceleration_elite.strategy import LondonMiddayMACDAccelerationElite
from strategies.s14_inside_bar_m30_elite.strategy import S14InsideBarM30Elite
from strategies.s66_bhanushali_44ma_elite.strategy import S66Bhanushali44MAElite

PIP_SIZE = 0.0001


def _simulate_week(w: int):
    ticks, _ = get_week_dataset(w)
    engine = BacktestEngine(pip_size=PIP_SIZE, use_slippage=True)

    strats = [
        HighFrequencyM15ScalperElite(),
        HighFrequencyM5ScalperElite(),
        RSIRegimeRangeShiftElite(),
        IntradayPrecisionH1M5Elite(),
        LondonMiddayMACDAccelerationElite(),
        S14InsideBarM30Elite(),
        S66Bhanushali44MAElite()
    ]

    week_trades = []
    for s in strats:
        tr = engine.run(ticks, s)
        for t in tr:
            week_trades.append({
                'entry_time': int(t['entry_time']),
                'exit_time': int(t['exit_time']),
                'pips': float(t['pips']),
                'strategy': s.name
            })

    return sorted(week_trades, key=lambda x: x['entry_time'])


def run_speedrun_simulation(all_trades: List[Dict[str, Any]], risk_pct: float, daily_stop_pct: float, dynamic_scaling: bool = False):
    INITIAL_CAPITAL = 100000.0
    MAX_DAILY_DD_PCT = 5.0
    MAX_TOTAL_DD_PCT = 10.0
    TARGET_P1_PCT = 8.0
    TARGET_P2_PCT = 5.0

    losses = [abs(t['pips']) for t in all_trades if t['pips'] < 0]
    avg_loss = float(np.mean(losses)) if losses else 8.5

    step = 5
    start_indices = list(range(0, len(all_trades) - 80, step))

    p1_passed = 0
    p1_failed_daily = 0
    p1_failed_total = 0
    p1_trading_days_to_pass = []
    p1_trades_to_pass = []
    p1_max_dds = []
    p1_max_daily_dds = []

    full_two_phase_passed = 0
    full_trading_days = []

    for s_idx in start_indices:
        # Phase 1
        bal = INITIAL_CAPITAL
        peak = bal
        current_day = None
        daily_start_bal = bal
        current_day_loss = 0.0
        p1_success = False
        trading_days_count = 0
        p1_trades_count = 0

        k = s_idx
        while k < len(all_trades):
            tr = all_trades[k]
            trade_day = pd.to_datetime(tr['entry_time'], unit='ms').date()

            if trade_day != current_day:
                current_day = trade_day
                daily_start_bal = bal
                current_day_loss = 0.0
                trading_days_count += 1

            # Daily Circuit Breaker: If already reached daily stop limit, skip remaining trades today
            if (current_day_loss / daily_start_bal) * 100.0 >= daily_stop_pct:
                k += 1
                continue

            # Sizing
            if dynamic_scaling:
                # If in drawdown, scale down risk to protect account
                dd_from_peak = (peak - bal) / peak * 100.0
                curr_risk = risk_pct * 0.50 if dd_from_peak >= 3.0 else risk_pct
            else:
                curr_risk = risk_pct

            risk_usd = bal * (curr_risk / 100.0)
            r_mult = tr['pips'] / avg_loss
            pnl = r_mult * risk_usd

            bal += pnl
            p1_trades_count += 1
            if pnl < 0:
                current_day_loss += abs(pnl)

            # Check Hard Rules
            day_dd_pct = (current_day_loss / daily_start_bal) * 100.0
            if day_dd_pct >= MAX_DAILY_DD_PCT:
                p1_failed_daily += 1
                break

            if bal > peak:
                peak = bal
            total_dd_pct = (peak - bal) / peak * 100.0
            if total_dd_pct >= MAX_TOTAL_DD_PCT:
                p1_failed_total += 1
                break

            # Target Check
            if bal >= INITIAL_CAPITAL * (1.0 + TARGET_P1_PCT / 100.0):
                p1_success = True
                p1_passed += 1
                p1_trading_days_to_pass.append(trading_days_count)
                p1_trades_to_pass.append(p1_trades_count)
                p1_max_dds.append(total_dd_pct)
                p1_max_daily_dds.append(day_dd_pct)
                break

            k += 1

        # Phase 2
        if p1_success:
            bal_p2 = INITIAL_CAPITAL
            peak_p2 = bal_p2
            current_day_p2 = None
            daily_start_bal_p2 = bal_p2
            current_day_loss_p2 = 0.0
            p2_days_count = 0
            p2_success = False

            k2 = k + 1
            while k2 < len(all_trades):
                tr = all_trades[k2]
                trade_day = pd.to_datetime(tr['entry_time'], unit='ms').date()

                if trade_day != current_day_p2:
                    current_day_p2 = trade_day
                    daily_start_bal_p2 = bal_p2
                    current_day_loss_p2 = 0.0
                    p2_days_count += 1

                if (current_day_loss_p2 / daily_start_bal_p2) * 100.0 >= daily_stop_pct:
                    k2 += 1
                    continue

                if dynamic_scaling:
                    dd_p2 = (peak_p2 - bal_p2) / peak_p2 * 100.0
                    curr_risk = risk_pct * 0.50 if dd_p2 >= 3.0 else risk_pct
                else:
                    curr_risk = risk_pct

                risk_usd = bal_p2 * (curr_risk / 100.0)
                r_mult = tr['pips'] / avg_loss
                pnl = r_mult * risk_usd

                bal_p2 += pnl
                if pnl < 0:
                    current_day_loss_p2 += abs(pnl)

                day_dd_pct = (current_day_loss_p2 / daily_start_bal_p2) * 100.0
                if day_dd_pct >= MAX_DAILY_DD_PCT:
                    break

                if bal_p2 > peak_p2:
                    peak_p2 = bal_p2
                total_dd_pct = (peak_p2 - bal_p2) / peak_p2 * 100.0
                if total_dd_pct >= MAX_TOTAL_DD_PCT:
                    break

                if bal_p2 >= INITIAL_CAPITAL * (1.0 + TARGET_P2_PCT / 100.0):
                    p2_success = True
                    full_two_phase_passed += 1
                    full_trading_days.append(trading_days_count + p2_days_count)
                    break

                k2 += 1

    n_challenges = len(start_indices)
    return {
        'risk_pct': risk_pct,
        'daily_stop': daily_stop_pct,
        'dynamic': dynamic_scaling,
        'evaluated': n_challenges,
        'p1_passed': p1_passed,
        'p1_pass_rate': p1_passed / n_challenges * 100,
        'daily_fails': p1_failed_daily,
        'total_dd_fails': p1_failed_total,
        'funded_count': full_two_phase_passed,
        'funded_rate': full_two_phase_passed / n_challenges * 100,
        'fastest_p1_days': np.min(p1_trading_days_to_pass) if p1_trading_days_to_pass else 0,
        'median_p1_days': np.median(p1_trading_days_to_pass) if p1_trading_days_to_pass else 0,
        'average_p1_days': np.mean(p1_trading_days_to_pass) if p1_trading_days_to_pass else 0,
        'median_full_days': np.median(full_trading_days) if full_trading_days else 0,
        'avg_dd': np.mean(p1_max_dds) if p1_max_dds else 0,
        'avg_daily_dd': np.mean(p1_max_daily_dds) if p1_max_daily_dds else 0
    }


def main():
    print("=" * 115)
    print("     PROP CHALLENGE SPEEDRUN AUDIT: MINIMIZING PASS TIME (5 YEARS / 262 WEEKS)     ".center(115))
    print("=" * 115)

    weeks = list(range(1, 263))
    t0 = time.time()
    num_workers = min(4, os.cpu_count() or 4)

    print(f"Simulating 7 High-Velocity Prop Models across all 262 weeks (workers={num_workers})...")
    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        week_results = list(ex.map(_simulate_week, weeks))

    dt = time.time() - t0
    print(f"[+] All 262 weeks simulated in {dt:.1f}s!\n")

    all_trades = [t for wr in week_results for t in wr]
    all_trades.sort(key=lambda x: x['entry_time'])

    print(f"Combined Trade Pool: {len(all_trades)} trades across 5 years (~{len(all_trades)/262:.1f} trades/week / ~{len(all_trades)/1300:.1f} trades/day)\n")

    experiments = [
        # (risk_pct, daily_stop_pct, dynamic_scaling, label)
        (0.50, 2.0, False, "1. Balanced Fleet (0.50% Risk, 2% Daily Cap)"),
        (0.70, 2.0, False, "2. Accelerated Speedrun (0.70% Risk, 2% Daily Cap)"),
        (0.85, 2.0, True,  "3. Dynamic Speedrun (0.85% Base Risk + Defensive DD Sizing)"),
        (1.00, 2.0, True,  "4. Ultra Speedrun (1.00% Base Risk + Defensive DD Sizing)")
    ]

    print("-" * 115)
    print(f"{'Strategy Profile':<36} | {'P1 Pass %':<10} | {'Funded %':<9} | {'Fastest P1':<12} | {'Median P1':<11} | {'Median Both':<12} | {'Max Daily DD'}")
    print("-" * 115)

    for risk, d_stop, dyn, label in experiments:
        res = run_speedrun_simulation(all_trades, risk, d_stop, dyn)
        print(f"{label:<36} | {res['p1_pass_rate']:<9.1f}% | {res['funded_rate']:<8.1f}% | {res['fastest_p1_days']:<4.0f} trading d | {res['median_p1_days']:<4.0f} trading d | {res['median_full_days']:<4.0f} trading d | {res['avg_daily_dd']:<5.2f}%")

    print("-" * 115)


if __name__ == '__main__':
    main()
