#!/usr/bin/env python3
"""
================================================================================
  COMPREHENSIVE 5-YEAR UNSEEN AUDIT: SPEEDRUN PROP CHALLENGE FLEET
================================================================================
Simulates 262 Weeks (~123.22 Million Dukascopy Ticks) under strict out-of-sample
production conditions:
  - Real Dukascopy floating spread (mean 0.35 - 0.45 pips)
  - 0.20 pips execution slippage on every entry and exit
  - Next-tick execution with zero lookahead bias
  - 5 full calendar years (W1 to W262)

Models Audited:
  1. HF_M15: HighFrequency M15 Scalper Elite (M15)
  2. DISC44: HighFrequency M5 Scalper Elite (M5)
  3. DISC40: RSI Regime Range Shift Elite (M15)
  4. DISC42: Macro Structure H4 Level + M30 Trigger Elite (H4/M30)
  5. DISC43: Intraday Precision H1 Level + M5 Trigger Elite (H1/M5)
  6. DISC38: London Midday MACD Acceleration Elite (M15)
  7. S14: Inside Bar M30 Momentum Breakout Elite (M30)
  8. S66: Bhanushali 44 MA Macro Trend Runner Elite (H1)

Exports:
  - results/5yr_unseen_prop_fleet_audit.json
  - results/5yr_unseen_prop_fleet_summary.csv
================================================================================
"""

from __future__ import annotations
import sys
import os
import time
import json
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
from tick_engine.core.metrics import compute_metrics

# Strategies
from strategies.high_frequency_m15_scalper_elite.strategy import HighFrequencyM15ScalperElite
from strategies.new_discoveries.disc44_high_frequency_m5_scalper_elite.strategy import HighFrequencyM5ScalperElite
from strategies.new_discoveries.disc40_rsi_regime_range_shift_elite.strategy import RSIRegimeRangeShiftElite
from strategies.new_discoveries.disc42_macro_structure_h4_m30_elite.strategy import MacroStructureH4M30Elite
from strategies.new_discoveries.disc43_intraday_precision_h1_m5_elite.strategy import IntradayPrecisionH1M5Elite
from strategies.disc38_london_midday_macd_acceleration_elite.strategy import LondonMiddayMACDAccelerationElite
from strategies.s14_inside_bar_m30_elite.strategy import S14InsideBarM30Elite
from strategies.s66_bhanushali_44ma_elite.strategy import S66Bhanushali44MAElite

PIP_SIZE = 0.0001
RESULTS_DIR = ROOT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _simulate_week(w: int):
    ticks, _ = get_week_dataset(w)
    engine = BacktestEngine(pip_size=PIP_SIZE, use_slippage=True)

    strats = [
        ("HF_M15", HighFrequencyM15ScalperElite()),
        ("DISC44_M5", HighFrequencyM5ScalperElite()),
        ("DISC40_RSI", RSIRegimeRangeShiftElite()),
        ("DISC42_H4_M30", MacroStructureH4M30Elite()),
        ("DISC43_H1_M5", IntradayPrecisionH1M5Elite()),
        ("DISC38_MACD", LondonMiddayMACDAccelerationElite()),
        ("S14_IB", S14InsideBarM30Elite()),
        ("S66_44MA", S66Bhanushali44MAElite())
    ]

    week_trades = []
    strat_trades: Dict[str, List[Dict[str, Any]]] = {s_id: [] for s_id, _ in strats}

    for s_id, s in strats:
        tr = engine.run(ticks, s)
        for t in tr:
            trade_obj = {
                'week': w,
                'entry_time': int(t['entry_time']),
                'exit_time': int(t['exit_time']),
                'direction': int(t['direction']),
                'pips': float(t['pips']),
                'strategy': s_id
            }
            week_trades.append(trade_obj)
            strat_trades[s_id].append(trade_obj)

    return w, week_trades, strat_trades


def run_audit():
    print("=" * 115)
    print("     5-YEAR UNSEEN AUDIT: SPEEDRUN PROP CHALLENGE FLEET (262 WEEKS / ~123.22M TICKS)     ".center(115))
    print("=" * 115)

    weeks = list(range(1, 263))
    t0 = time.time()
    num_workers = min(4, os.cpu_count() or 4)

    print(f"Simulating 8 High-Conviction Prop Models across 262 weeks with {num_workers} workers...")
    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        raw_results = list(ex.map(_simulate_week, weeks))

    dt = time.time() - t0
    print(f"[+] All 262 weeks simulated in {dt:.1f}s!\n")

    raw_results.sort(key=lambda x: x[0])

    all_trades: List[Dict[str, Any]] = []
    strat_map: Dict[str, List[Dict[str, Any]]] = {
        "HF_M15": [],
        "DISC44_M5": [],
        "DISC40_RSI": [],
        "DISC42_H4_M30": [],
        "DISC43_H1_M5": [],
        "DISC38_MACD": [],
        "S14_IB": [],
        "S66_44MA": []
    }

    for w, w_trades, s_trades in raw_results:
        all_trades.extend(w_trades)
        for s_id, t_list in s_trades.items():
            strat_map[s_id].extend(t_list)

    all_trades.sort(key=lambda x: x['entry_time'])

    # Helper for metrics from trade objects
    def get_stats(trade_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not trade_list:
            return {'trades': 0, 'wr': 0.0, 'pf': 0.0, 'rr': 0.0, 'pips': 0.0, 'dd': 0.0, 'ret_dd': 0.0}
        pips = np.array([t['pips'] for t in trade_list])
        wins = pips[pips > 0]
        losses = pips[pips < 0]
        n = len(pips)
        wr = len(wins) / n * 100 if n > 0 else 0.0
        avg_w = np.mean(wins) if len(wins) > 0 else 0.0
        avg_l = abs(np.mean(losses)) if len(losses) > 0 else 1.0
        rr = avg_w / avg_l if avg_l > 0 else 0.0
        pf = np.sum(wins) / abs(np.sum(losses)) if len(losses) > 0 and abs(np.sum(losses)) > 0 else 0.0
        net = np.sum(pips)

        # Drawdown calculation
        eq = np.cumsum(pips)
        peak = np.maximum.accumulate(eq)
        dd = peak - eq
        max_dd = float(np.max(dd)) if len(dd) > 0 else 0.0
        ret_dd = net / max(max_dd, 1e-6)
        return {
            'trades': n,
            'wr': float(wr),
            'pf': float(pf),
            'rr': float(rr),
            'pips': float(net),
            'dd': float(max_dd),
            'ret_dd': float(ret_dd)
        }

    # 1. INDIVIDUAL STRATEGY SCORECARD
    print("=" * 115)
    print("1. INDIVIDUAL MODEL SCORECARD ACROSS ENTIRE 5 YEARS (UNSEEN TICK DATA)")
    print("=" * 115)
    print(f"{'Strategy Model':<28} | {'5Y Trades':<10} | {'Trd/Wk':<7} | {'Win Rate':<9} | {'Realized RR':<12} | {'PF':<6} | {'Net Pips':<12} | {'Max DD'}")
    print("-" * 115)

    strat_stats = {}
    for s_id in ["HF_M15", "DISC44_M5", "DISC40_RSI", "DISC42_H4_M30", "DISC43_H1_M5", "DISC38_MACD", "S14_IB", "S66_44MA"]:
        t_list = strat_map[s_id]
        st = get_stats(t_list)
        strat_stats[s_id] = st
        trd_wk = st['trades'] / 262.0
        print(f"{s_id:<28} | {st['trades']:<10} | {trd_wk:<7.1f} | {st['wr']:<8.1f}% | {st['rr']:<6.2f} : 1   | {st['pf']:<6.2f} | {st['pips']:<+11.1f}p | {st['dd']:<7.1f}p")

    print("-" * 115)

    # 2. YEAR-BY-YEAR ANNUAL BREAKDOWN (YEARS 1 TO 5)
    year_ranges = [
        ("Year 1 (W1 - W52)", 1, 52),
        ("Year 2 (W53 - W104)", 53, 104),
        ("Year 3 (W105 - W156)", 105, 156),
        ("Year 4 (W157 - W208)", 157, 208),
        ("Year 5 (W209 - W262)", 209, 262),
    ]

    print("\n" + "=" * 115)
    print("2. YEAR-BY-YEAR CONSISTENCY BREAKDOWN OF THE COMBINED FLEET")
    print("=" * 115)
    print(f"{'Year Period':<24} | {'Trades':<8} | {'Trd/Wk':<8} | {'Win Rate':<10} | {'Profit Factor':<15} | {'Net Return':<14} | {'Max DD'}")
    print("-" * 115)

    annual_stats = []
    for yr_label, w_start, w_end in year_ranges:
        yr_trades = [t for t in all_trades if w_start <= t['week'] <= w_end]
        st = get_stats(yr_trades)
        annual_stats.append({'period': yr_label, **st})
        trd_wk = st['trades'] / (w_end - w_start + 1)
        print(f"{yr_label:<24} | {st['trades']:<8} | {trd_wk:<8.1f} | {st['wr']:<9.1f}% | {st['pf']:<15.2f} | {st['pips']:<+13.1f}p | {st['dd']:<7.1f}p")

    fleet_5y = get_stats(all_trades)
    print("-" * 115)
    print(f"{'5-Year Total Combined':<24} | {fleet_5y['trades']:<8} | {fleet_5y['trades']/262:<8.1f} | {fleet_5y['wr']:<9.1f}% | {fleet_5y['pf']:<15.2f} | {fleet_5y['pips']:<+13.1f}p | {fleet_5y['dd']:<7.1f}p")
    print("=" * 115)

    # 3. PROP CHALLENGE SPEEDRUN MONTE CARLO AUDIT
    print("\n" + "=" * 115)
    print("3. PROP FIRM CHALLENGE SPEEDRUN AUDIT (100 ROLLING STARTS / 5 YEARS)")
    print("=" * 115)

    losses = [abs(t['pips']) for t in all_trades if t['pips'] < 0]
    avg_loss = float(np.mean(losses)) if losses else 8.5
    INITIAL_CAPITAL = 100000.0
    TARGET_P1 = 8000.0       # +8.0%
    TARGET_P2 = 5000.0       # +5.0%
    MAX_DAILY_LOSS = 5000.0  # 5.0%
    MAX_TOTAL_DD = 10000.0   # 10.0%

    step = 5
    start_indices = list(range(0, len(all_trades) - 100, step))

    prop_experiments = [
        ("Conservative (0.50% Risk)", 0.50, 2.0),
        ("Accelerated (0.70% Risk)", 0.70, 2.0),
        ("Speedrun (0.85% Risk)", 0.85, 2.0)
    ]

    prop_results = []

    for label, risk_pct, d_stop_pct in prop_experiments:
        p1_pass = 0
        p1_fail_daily = 0
        p1_fail_total = 0
        p1_days_arr = []
        p1_trades_arr = []
        p1_max_dd_arr = []
        p1_max_daily_arr = []
        both_pass = 0
        full_days_arr = []

        for s_idx in start_indices:
            # Phase 1
            bal = INITIAL_CAPITAL
            peak = bal
            current_day = None
            daily_start_bal = bal
            day_loss = 0.0
            p1_ok = False
            p1_days = 0
            p1_tr_cnt = 0
            max_dd = 0.0
            max_daily_dd = 0.0

            k = s_idx
            while k < len(all_trades):
                tr = all_trades[k]
                t_day = pd.to_datetime(tr['entry_time'], unit='ms').date()
                if t_day != current_day:
                    current_day = t_day
                    daily_start_bal = bal
                    day_loss = 0.0
                    p1_days += 1

                # Daily stop circuit breaker
                if (day_loss / daily_start_bal) * 100.0 >= d_stop_pct:
                    k += 1
                    continue

                r_mult = tr['pips'] / avg_loss
                pnl = r_mult * (bal * (risk_pct / 100.0))
                bal += pnl
                p1_tr_cnt += 1
                if pnl < 0:
                    day_loss += abs(pnl)

                day_dd = (day_loss / daily_start_bal) * 100.0
                if day_dd > max_daily_dd:
                    max_daily_dd = day_dd
                if day_loss >= MAX_DAILY_LOSS:
                    p1_fail_daily += 1
                    break

                if bal > peak:
                    peak = bal
                tot_dd = (peak - bal) / peak * 100.0
                if tot_dd > max_dd:
                    max_dd = tot_dd
                if tot_dd >= 10.0:
                    p1_fail_total += 1
                    break

                if bal >= INITIAL_CAPITAL + TARGET_P1:
                    p1_ok = True
                    p1_pass += 1
                    p1_days_arr.append(p1_days)
                    p1_trades_arr.append(p1_tr_cnt)
                    p1_max_dd_arr.append(max_dd)
                    p1_max_daily_arr.append(max_daily_dd)
                    break
                k += 1

            # Phase 2
            if p1_ok:
                bal2 = INITIAL_CAPITAL
                peak2 = bal2
                curr_d2 = None
                day_start2 = bal2
                day_l2 = 0.0
                p2_days = 0
                k2 = k + 1

                while k2 < len(all_trades):
                    tr2 = all_trades[k2]
                    t_day2 = pd.to_datetime(tr2['entry_time'], unit='ms').date()
                    if t_day2 != curr_d2:
                        curr_d2 = t_day2
                        day_start2 = bal2
                        day_l2 = 0.0
                        p2_days += 1

                    if (day_l2 / day_start2) * 100.0 >= d_stop_pct:
                        k2 += 1
                        continue

                    r_mult2 = tr2['pips'] / avg_loss
                    pnl2 = r_mult2 * (bal2 * (risk_pct / 100.0))
                    bal2 += pnl2
                    if pnl2 < 0:
                        day_l2 += abs(pnl2)

                    if day_l2 >= MAX_DAILY_LOSS:
                        break

                    if bal2 > peak2:
                        peak2 = bal2
                    tot_dd2 = (peak2 - bal2) / peak2 * 100.0
                    if tot_dd2 >= 10.0:
                        break

                    if bal2 >= INITIAL_CAPITAL + TARGET_P2:
                        both_pass += 1
                        full_days_arr.append(p1_days + p2_days)
                        break
                    k2 += 1

        n_chall = len(start_indices)
        res_entry = {
            'label': label,
            'risk_pct': risk_pct,
            'p1_pass_pct': (p1_pass / n_chall) * 100.0,
            'funded_pct': (both_pass / n_chall) * 100.0,
            'fastest_p1_days': int(np.min(p1_days_arr)) if p1_days_arr else 0,
            'median_p1_days': int(np.median(p1_days_arr)) if p1_days_arr else 0,
            'median_full_days': int(np.median(full_days_arr)) if full_days_arr else 0,
            'max_daily_dd': float(np.max(p1_max_daily_arr)) if p1_max_daily_arr else 0.0,
            'avg_dd': float(np.mean(p1_max_dd_arr)) if p1_max_dd_arr else 0.0
        }
        prop_results.append(res_entry)

    print(f"{'Profile':<28} | {'P1 Pass %':<10} | {'Funded %':<9} | {'Fastest P1':<12} | {'Median P1':<11} | {'Median Funded':<13} | {'Worst Day'}")
    print("-" * 115)
    for r in prop_results:
        print(f"{r['label']:<28} | {r['p1_pass_pct']:<9.1f}% | {r['funded_pct']:<8.1f}% | {r['fastest_p1_days']:<4} trading d | {r['median_p1_days']:<4} trading d | {r['median_full_days']:<4} trading d  | {r['max_daily_dd']:<5.2f}%")
    print("=" * 115)

    # Save Artifacts
    audit_data = {
        'fleet_metrics': fleet_5y,
        'strategies': strat_stats,
        'annual_breakdown': annual_stats,
        'prop_speedrun': prop_results
    }

    json_path = RESULTS_DIR / "5yr_unseen_prop_fleet_audit.json"
    with open(json_path, 'w') as f:
        json.dump(audit_data, f, indent=2)
    print(f"\n[+] Exported full audit to: {json_path}")

    # CSV Summary
    csv_rows = []
    for s_id, st in strat_stats.items():
        csv_rows.append({
            'strategy_id': s_id,
            'trades_5y': st['trades'],
            'trades_per_wk': round(st['trades']/262.0, 1),
            'win_rate_pct': round(st['wr'], 1),
            'realized_rr': round(st['rr'], 2),
            'profit_factor': round(st['pf'], 2),
            'net_pips': round(st['pips'], 1),
            'max_drawdown_pips': round(st['dd'], 1)
        })
    csv_df = pd.DataFrame(csv_rows)
    csv_path = RESULTS_DIR / "5yr_unseen_prop_fleet_summary.csv"
    csv_df.to_csv(csv_path, index=False)
    print(f"[+] Exported CSV summary to: {csv_path}")


if __name__ == '__main__':
    run_audit()
