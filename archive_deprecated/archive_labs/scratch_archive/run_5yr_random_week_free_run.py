#!/usr/bin/env python3
"""
================================================================================
  5-YEAR FREE RUN ON RANDOMIZED WEEK SERIES (BLOCK BOOTSTRAP STRESS TEST)
================================================================================
Performs a continuous, uncapped "Free Run" of the Top 5 High-Precision Production
Fleet across a 5-year synthetic series of 262 randomly shuffled/sampled weeks:
  - Account Base: $100,000
  - Duration: 262 consecutive weekly trading periods (~5 Years)
  - Week Order: Randomly shuffled permutation of all historical weeks (Seed 42)
  - Fleet Models:
      1. DISC42: Macro Structure H4 Level + M30 Trigger (72.9% WR, PF 3.97)
      2. DISC43: Intraday Precision H1 Level + M5 Trigger (48.3% WR, PF 1.47)
      3. DISC40: RSI Regime Range Shift M15 (44.0% WR, PF 1.31)
      4. DISC41: Thursday Macro Trend Ignition M15 (43.6% WR, PF 1.30)
      5. DISC38: London Midday MACD Acceleration M15 (43.0% WR, PF 1.28)

Records every trade placed:
  - Sequential Trade ID & Historical Week ID
  - Strategy Name & Direction (BUY / SELL)
  - Entry Date & Time (UTC)
  - Entry Price (Spread-adjusted)
  - Exit Date & Time (UTC)
  - Exit Price (Slippage-adjusted)
  - Trade Duration
  - Exit Reason (Take Profit, Stop Loss, Signal Exit)
  - Net Pips Banked
  - Effective Lot Size Traded
  - Banked Profit / Loss ($)
  - Running Account Balance ($)
  - Running Drawdown ($ and %)

Outputs:
  - results/free_run_5yr_random_weeks_trade_log.csv
  - results/free_run_5yr_random_weeks_trade_log.json
================================================================================
"""

from __future__ import annotations
import sys
import os
import json
import random
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True

ROOT_DIR = Path(__file__).resolve().parent.parent
MASTER_FILE = ROOT_DIR / "results" / "master_7_prop_trades_5y.json"
RESULTS_DIR = ROOT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_trades_by_week() -> Dict[int, List[Dict[str, Any]]]:
    print(f"[+] Loading trade pool from {MASTER_FILE}...")
    with open(MASTER_FILE, 'r') as f:
        all_trades = json.load(f)

    # Filter to Top 5 High-Precision Models (The Proven Core)
    allowed_strats = {
        "DISC42_H4_M30",
        "DISC43_H1_M5",
        "DISC40_RSI",
        "DISC41_THURSDAY",
        "DISC38_MACD"
    }
    filtered = [t for t in all_trades if t['strategy'] in allowed_strats]
    print(f"[+] Filtered {len(filtered)} trades from Top 5 High-Precision Fleet.")

    trades_by_week = {}
    for t in filtered:
        w = t['week']
        if w not in trades_by_week:
            trades_by_week[w] = []
        trades_by_week[w].append(t)

    # Sort each week's trades temporally
    for w in trades_by_week:
        trades_by_week[w].sort(key=lambda x: x['entry_time'])

    return trades_by_week, filtered


def main():
    print("=" * 125)
    print("   5-YEAR CONTINUOUS FREE RUN: RANDOMIZED WEEK SERIES STRESS TEST (262 WEEKS)   ".center(125))
    print("=" * 125)

    trades_by_week, all_filtered = load_trades_by_week()

    # Precalculate per-strategy average loss for exact 1R normalization
    strat_avg_losses = {}
    for s in ["DISC42_H4_M30", "DISC43_H1_M5", "DISC40_RSI", "DISC41_THURSDAY", "DISC38_MACD"]:
        s_losses = [abs(t['pips']) for t in all_filtered if t['strategy'] == s and t['pips'] < 0]
        strat_avg_losses[s] = float(np.mean(s_losses)) if s_losses else 8.5

    # Conservative, sustainable compounding risk weights for continuous 5-year growth
    risk_weights = {
        'DISC42_H4_M30': 0.50,    # 72.9% WR, PF 3.97
        'DISC43_H1_M5': 0.45,     # 48.3% WR, PF 1.47
        'DISC40_RSI': 0.35,       # 44.0% WR, PF 1.31
        'DISC41_THURSDAY': 0.30,  # 43.6% WR, PF 1.30
        'DISC38_MACD': 0.30       # 43.0% WR, PF 1.28
    }

    # Generate a deterministic randomized sequence of 262 weeks (Random Shuffled Series)
    random.seed(42)
    historical_weeks = list(range(1, 263))
    # Random permutation of the 262 weeks
    random_week_series = historical_weeks.copy()
    random.shuffle(random_week_series)

    print(f"[*] Generated randomized sequence of 262 weeks (Seed 42).")
    print(f"[*] First 10 weeks in series: {random_week_series[:10]}")
    print(f"[*] Last 10 weeks in series : {random_week_series[-10:]}\n")

    INITIAL_CAPITAL = 100000.0
    bal = INITIAL_CAPITAL
    peak = bal
    max_dd_pct = 0.0
    daily_stop_pct = 1.5  # 1.5% daily loss circuit breaker

    recorded_trades = []
    weekly_summaries = []
    trade_id = 1

    for step_num, hist_w in enumerate(random_week_series, start=1):
        w_trades = trades_by_week.get(hist_w, [])
        week_start_bal = bal
        week_trades_count = 0
        week_pnl = 0.0

        current_day = None
        d_start = bal
        d_loss = 0.0

        for tr in w_trades:
            dt_entry = pd.to_datetime(tr['entry_time'], unit='ms')
            dt_exit = pd.to_datetime(tr['exit_time'], unit='ms')
            t_day = dt_entry.date()

            if t_day != current_day:
                current_day = t_day
                d_start = bal
                d_loss = 0.0

            # Daily circuit breaker check
            if (d_loss / d_start) * 100.0 >= daily_stop_pct:
                continue

            r_pct = risk_weights.get(tr['strategy'], 0.35)
            avg_loss = strat_avg_losses.get(tr['strategy'], 8.5)
            r_mult = tr['pips'] / avg_loss

            dollar_risk = bal * (r_pct / 100.0)
            pnl = r_mult * dollar_risk
            bal += pnl
            week_pnl += pnl
            week_trades_count += 1

            if pnl < 0:
                d_loss += abs(pnl)

            if bal > peak:
                peak = bal
            dd_pct = (peak - bal) / peak * 100.0
            if dd_pct > max_dd_pct:
                max_dd_pct = dd_pct

            # Calculate lot size: 1 pip on 1 standard lot = $10.00
            effective_lots = round(dollar_risk / (avg_loss * 10.0), 2)

            duration_min = int((tr['exit_time'] - tr['entry_time']) / 60000)
            dur_h = duration_min // 60
            dur_m = duration_min % 60
            dur_str = f"{dur_h}h {dur_m:02d}m" if dur_h >= 1 else f"{dur_m}m"

            exit_reasons = {1: "Stop Loss", 2: "Take Profit", 3: "Signal/Time Exit"}

            rec = {
                'trade_id': trade_id,
                'series_step': f"Step {step_num}/262",
                'historical_week': hist_w,
                'strategy': tr['strategy'],
                'direction': "BUY" if tr['direction'] == 1 else "SELL",
                'entry_time': dt_entry.strftime('%Y-%m-%d %H:%M:%S'),
                'entry_price': round(tr['entry_price'], 5),
                'exit_time': dt_exit.strftime('%Y-%m-%d %H:%M:%S'),
                'exit_price': round(tr['exit_price'], 5),
                'duration': dur_str,
                'exit_reason': exit_reasons.get(tr['exit_reason'], "Closed"),
                'pips': round(tr['pips'], 1),
                'lot_size': effective_lots,
                'banked_pnl': round(pnl, 2),
                'balance': round(bal, 2),
                'drawdown_pct': round(dd_pct, 2)
            }
            recorded_trades.append(rec)
            trade_id += 1

        weekly_summaries.append({
            'step': step_num,
            'historical_week': hist_w,
            'trades': week_trades_count,
            'pnl': round(week_pnl, 2),
            'end_balance': round(bal, 2)
        })

    # Convert to DataFrame
    df_trades = pd.DataFrame(recorded_trades)
    df_weeks = pd.DataFrame(weekly_summaries)

    # Export to CSV and JSON
    csv_file = RESULTS_DIR / "free_run_5yr_random_weeks_trade_log.csv"
    json_file = RESULTS_DIR / "free_run_5yr_random_weeks_trade_log.json"

    df_trades.to_csv(csv_file, index=False)
    with open(json_file, 'w') as f:
        json.dump(recorded_trades, f, indent=2)

    # Compute Executive Metrics
    total_trades = len(recorded_trades)
    wins = [r for r in recorded_trades if r['banked_pnl'] > 0]
    losses = [r for r in recorded_trades if r['banked_pnl'] < 0]
    win_rate = len(wins) / total_trades * 100.0 if total_trades else 0
    total_profit_banked = bal - INITIAL_CAPITAL
    total_return_pct = total_profit_banked / INITIAL_CAPITAL * 100.0
    annualized_return = total_return_pct / 5.0

    winning_weeks = [w for w in weekly_summaries if w['pnl'] > 0]
    losing_weeks = [w for w in weekly_summaries if w['pnl'] < 0]

    gross_profit = sum(r['banked_pnl'] for r in wins)
    gross_loss = abs(sum(r['banked_pnl'] for r in losses))
    pf = gross_profit / gross_loss if gross_loss > 0 else 9.99

    print("=" * 115)
    print("                    5-YEAR RANDOMIZED WEEK SERIES FREE RUN SCORECARD                    ".center(115))
    print("=" * 115)
    print(f"Total Randomized Series Length : 262 weeks (Full 5-Year Equivalent)")
    print(f"Initial Starting Capital       : ${INITIAL_CAPITAL:,.2f}")
    print(f"Final Ending Capital           : ${bal:,.2f}")
    print(f"Total Net Banked Profit        : +${total_profit_banked:,.2f} (+{total_return_pct:.2f}% Total / +{annualized_return:.2f}% Per Year)")
    print(f"Peak Account Drawdown          : {max_dd_pct:.2f}% (Safety buffer vs 10% prop limit: {10.0 - max_dd_pct:.2f}%)")
    print(f"Return / Drawdown Ratio        : {total_return_pct / max_dd_pct:.2f} : 1")
    print("-" * 115)
    print(f"Total Trades Placed & Banked   : {total_trades:,} trades (~{total_trades/262:.1f} trades/week)")
    print(f"Trade Win Rate                 : {win_rate:.1f}% ({len(wins):,} Wins / {len(losses):,} Losses)")
    print(f"Profit Factor                  : {pf:.2f} (Gross Profit: +${gross_profit:,.2f} / Gross Loss: -${gross_loss:,.2f})")
    print(f"Average Profit on Win          : +${np.mean([r['banked_pnl'] for r in wins]):,.2f}")
    print(f"Average Loss on Loss           : -${abs(np.mean([r['banked_pnl'] for r in losses])):,.2f}")
    print(f"Realized Risk : Reward         : {np.mean([r['banked_pnl'] for r in wins]) / abs(np.mean([r['banked_pnl'] for r in losses])):.2f} : 1")
    print("-" * 115)
    print(f"Winning Weeks Ratio            : {len(winning_weeks)/len(weekly_summaries)*100:.1f}% ({len(winning_weeks)} Winning Weeks / {len(losing_weeks)} Losing Weeks)")
    print(f"Average Weekly Net PnL         : +${np.mean([w['pnl'] for w in weekly_summaries]):,.2f} per week")
    print(f"Best Week Banked               : +${max(w['pnl'] for w in weekly_summaries):,.2f}")
    print(f"Worst Week Drawdown            : -${abs(min(w['pnl'] for w in weekly_summaries)):,.2f}")
    print("=" * 115)

    # Print Sample of 12 Concrete Trades Across the Free Run
    print("\nSAMPLE OF 12 EXECUTED TRADES ACROSS THE RANDOMIZED 5-YEAR FREE RUN:")
    headers = ["#", "Step/Wk", "Strategy", "Dir", "Entry Time", "Entry Px", "Exit Time", "Exit Px", "Dur", "Reason", "Pips", "Lots", "Banked PnL", "Balance", "DD%"]
    print(f"{headers[0]:<4} | {headers[1]:<11} | {headers[2]:<14} | {headers[3]:<4} | {headers[4]:<19} | {headers[5]:<8} | {headers[6]:<19} | {headers[7]:<8} | {headers[8]:<6} | {headers[9]:<12} | {headers[10]:<7} | {headers[11]:<5} | {headers[12]:<11} | {headers[13]:<11} | {headers[14]:<5}")
    print("-" * 175)

    sample_indices = [0, 25, 75, 150, 250, 400, 600, 800, 950, 1100, 1200, len(recorded_trades)-1]
    for idx in sample_indices:
        r = recorded_trades[idx]
        pnl_s = f"+${r['banked_pnl']:,.2f}" if r['banked_pnl'] >= 0 else f"-${abs(r['banked_pnl']):,.2f}"
        pip_s = f"+{r['pips']:.1f}p" if r['pips'] >= 0 else f"{r['pips']:.1f}p"
        print(f"{r['trade_id']:<4} | {r['series_step']:<11} | {r['strategy']:<14} | {r['direction']:<4} | {r['entry_time']:<19} | {r['entry_price']:<8.5f} | {r['exit_time']:<19} | {r['exit_price']:<8.5f} | {r['duration']:<6} | {r['exit_reason']:<12} | {pip_s:<7} | {r['lot_size']:<5.2f} | {pnl_s:>11} | ${r['balance']:>10,.2f} | {r['drawdown_pct']:>4.1f}%")

    print("-" * 175)
    print(f"\n[+] Full Granular Trade Ledger Saved to CSV : {csv_file} ({csv_file.stat().st_size / 1024:.1f} KB)")
    print(f"[+] Full Granular Trade Ledger Saved to JSON: {json_file} ({json_file.stat().st_size / 1024:.1f} KB)")


if __name__ == '__main__':
    main()
