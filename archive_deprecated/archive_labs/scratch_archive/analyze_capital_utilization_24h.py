#!/usr/bin/env python3
"""
================================================================================
  24-HOUR CAPITAL UTILIZATION & TIME-IN-MARKET ANALYTICS
  Audits the 7-Star All-Weather Discovered Fleet across 5 Years (262 Weeks):
    1. DISC03: NY Open Overlap Breakout Elite
    2. INST05: Market Structure Shift Retest Elite
    3. DISC36: Frankfurt Ultra-Coil London ORB Elite
    4. COMB12: Bollinger Squeeze Elite
    5. COMB27: Squeeze Supertrend Hybrid Elite
    6. DISC37: European Midday Ultra-Coil NY Breakout Elite
    7. DISC31: London Lunch Trap Fade Elite
================================================================================
"""

from __future__ import annotations
import sys
import os
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tick_engine.core.engine import BacktestEngine, TRADE_DTYPE
from tick_engine.core.dataloader import get_week_dataset
from tick_engine.core.metrics import compute_metrics

from strategies.disc03_ny_open_overlap_elite.strategy import NewYorkOpenOverlapBreakoutElite
from strategies.inst05_market_structure_shift_elite.strategy import MarketStructureShiftRetestElite
from strategies.disc36_frankfurt_ultra_coil_orb_elite.strategy import FrankfurtUltraCoilLondonORBElite
from strategies.comb12_bollinger_squeeze_elite.strategy import BollingerBandwidthExpansionElite
from strategies.comb27_squeeze_supertrend_hybrid_elite.strategy import BollingerSupertrendHybridElite
from strategies.disc37_european_midday_ultracoil_breakout_elite.strategy import EuropeanMiddayUltraCoilBreakoutElite
from strategies.disc31_london_lunch_trap_fade_elite.strategy import LondonLunchTrapFadeElite

STRATEGIES = [
    ("DISC03", NewYorkOpenOverlapBreakoutElite),
    ("INST05", MarketStructureShiftRetestElite),
    ("DISC36", FrankfurtUltraCoilLondonORBElite),
    ("COMB12", BollingerBandwidthExpansionElite),
    ("COMB27", BollingerSupertrendHybridElite),
    ("DISC37", EuropeanMiddayUltraCoilBreakoutElite),
    ("DISC31", LondonLunchTrapFadeElite),
]


def eval_week(w):
    ticks, meta = get_week_dataset(w)
    engine = BacktestEngine(pip_size=0.0001, use_slippage=True)
    trades_by_strat = {}
    for sid, cls in STRATEGIES:
        strat = cls()
        tr = engine.run(ticks, strat)
        trades_by_strat[sid] = tr

    # Week duration info
    t_min = ticks['timestamp'][0]
    t_max = ticks['timestamp'][-1]
    return w, t_min, t_max, trades_by_strat


def analyze_capital_utilization():
    print("=" * 90)
    print("     24-HOUR CAPITAL UTILIZATION & MARKET EXPOSURE ANALYSIS (5-YEAR TICK AUDIT)     ")
    print("=" * 90)

    t0 = time.time()
    weeks = list(range(1, 263))
    num_workers = min(8, os.cpu_count() or 4)

    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        results = list(ex.map(eval_week, weeks))

    results.sort(key=lambda x: x[0])
    dt = time.time() - t0
    print(f"[*] Evaluated 262 Weeks of Tick Data across 7 Strategies in {dt:.1f}s\n")

    # Aggregate all trades with strategy tags
    all_trade_records = []
    total_market_open_ms = 0

    for w, t_min, t_max, t_dict in results:
        week_span_ms = t_max - t_min
        total_market_open_ms += week_span_ms
        for sid, tr_arr in t_dict.items():
            for t in tr_arr:
                all_trade_records.append({
                    'week': w,
                    'strategy': sid,
                    'entry_time': t['entry_time'],
                    'exit_time': t['exit_time'],
                    'duration_hours': (t['exit_time'] - t['entry_time']) / (1000 * 3600),
                    'pips': t['pips'],
                    'direction': t['direction'],
                    'exit_reason': t['exit_reason']
                })

    df_trades = pd.DataFrame(all_trade_records)
    total_trades = len(df_trades)
    print(f"[+] Total Fleet Trades Analyzed: {total_trades:,}")

    # 1. TRADE DURATION METRICS
    durations = df_trades['duration_hours']
    avg_dur = durations.mean()
    median_dur = durations.median()
    p90_dur = durations.quantile(0.90)
    max_dur = durations.max()
    min_dur = durations.min()

    print("\n" + "-" * 90)
    print("1. TRADE DURATION & HOLDING TIME PROFILE")
    print("-" * 90)
    print(f"Average Trade Duration:     {avg_dur:.2f} hours ({avg_dur * 60:.1f} minutes)")
    print(f"Median Trade Duration:      {median_dur:.2f} hours ({median_dur * 60:.1f} minutes)")
    print(f"90th Percentile Duration:   {p90_dur:.2f} hours ({p90_dur * 60:.1f} minutes)")
    print(f"Minimum Duration:           {min_dur * 60:.1f} minutes")
    print(f"Maximum Duration:           {max_dur:.2f} hours")

    # Duration breakdown per strategy
    print("\nDuration Breakdown per Strategy:")
    for sid, group in df_trades.groupby('strategy'):
        print(f"  {sid:<8}: Avg {group['duration_hours'].mean():5.2f}h | Median {group['duration_hours'].median():5.2f}h | Total Trades: {len(group):4d} | Net Pips: {group['pips'].sum():+7.1f}p")

    # 2. TIME-IN-MARKET & PORTFOLIO EXPOSURE (UNION OF TIME INTERVALS)
    # Merge overlapping intervals to find exact time at least 1 position is in the market
    intervals = df_trades[['entry_time', 'exit_time']].sort_values('entry_time').values
    merged_intervals = []
    for start, end in intervals:
        if not merged_intervals:
            merged_intervals.append([start, end])
        else:
            prev_start, prev_end = merged_intervals[-1]
            if start <= prev_end:
                merged_intervals[-1][1] = max(prev_end, end)
            else:
                merged_intervals.append([start, end])

    total_active_ms = sum(end - start for start, end in merged_intervals)
    total_active_hours = total_active_ms / (1000 * 3600)
    total_market_hours = total_market_open_ms / (1000 * 3600)
    total_calendar_hours = 262 * 7 * 24

    exposure_pct_market = (total_active_ms / total_market_open_ms) * 100.0
    exposure_pct_calendar = (total_active_hours / total_calendar_hours) * 100.0

    print("\n" + "-" * 90)
    print("2. OVERALL CAPITAL EXPOSURE & TIME-IN-MARKET")
    print("-" * 90)
    print(f"Total Market Hours Available (5 Years): {total_market_hours:,.1f} hours (~120 hrs/week)")
    print(f"Total Active Market Exposure Time:       {total_active_hours:,.1f} hours ({total_active_hours/262:.1f} hrs/week, {total_active_hours/(262*5):.2f} hrs/day)")
    print(f"Time-in-Market Exposure Rate:            {exposure_pct_market:.2f}% of trading hours")
    print(f"Total Calendar Exposure Rate:            {exposure_pct_calendar:.2f}% (24/7 basis)")
    print(f"Idle Cash Time (Zero Risk Exposure):     {100.0 - exposure_pct_market:.2f}% of trading hours")

    # 3. CONCURRENT POSITION / MARGIN UTILIZATION DISTRIBUTION
    # Sample every 15 minutes across the 262 weeks to evaluate concurrent active positions
    # Build timeline of events
    events = []
    for idx, row in df_trades.iterrows():
        events.append((row['entry_time'], 1))
        events.append((row['exit_time'], -1))

    events.sort(key=lambda x: (x[0], x[1]))

    # Calculate time spent in each concurrency state
    concurrency_durations = {}  # state -> total_ms
    current_pos = 0
    last_time = None

    for ev_time, ev_delta in events:
        if last_time is not None and ev_time > last_time:
            delta_ms = ev_time - last_time
            concurrency_durations[current_pos] = concurrency_durations.get(current_pos, 0) + delta_ms
        current_pos += ev_delta
        last_time = ev_time

    # Add the zero-position idle market time
    time_with_at_least_one = sum(concurrency_durations.get(k, 0) for k in concurrency_durations if k > 0)
    idle_market_ms = max(0, total_market_open_ms - time_with_at_least_one)
    concurrency_durations[0] = idle_market_ms

    print("\n" + "-" * 90)
    print("3. CONCURRENT POSITION & LEVERAGE CONCENTRATION")
    print("-" * 90)
    for pos_count in sorted(concurrency_durations.keys()):
        ms_spent = concurrency_durations[pos_count]
        hrs_spent = ms_spent / (1000 * 3600)
        pct = (ms_spent / total_market_open_ms) * 100.0
        label = "Idle (0 Positions / 100% Cash)" if pos_count == 0 else f"{pos_count} Concurrent Position(s)"
        print(f"  {label:<32}: {hrs_spent:8.1f} hrs | {pct:6.2f}% of market time")

    # 4. HOURLY 24-HOUR CAPITAL UTILIZATION PROFILE (00:00 to 23:00 UTC)
    # Discretize open position minutes into hourly bins
    hourly_active_ms = np.zeros(24, dtype=np.float64)
    total_hours_in_sample = 262 * 5  # ~1,310 trading days

    # For each trade, distribute its time into hourly buckets
    for idx, row in df_trades.iterrows():
        t_entry = pd.to_datetime(row['entry_time'], unit='ms')
        t_exit = pd.to_datetime(row['exit_time'], unit='ms')
        # Generate 15-min points
        sample_pts = pd.date_range(start=t_entry, end=t_exit, freq='15min')
        for pt in sample_pts:
            hourly_active_ms[pt.hour] += 15 * 60 * 1000

    # Hourly capacity: total possible market time for that hour over 262 weeks
    # Each weekday has 1 hour per day, 5 days per week * 262 weeks = 1310 hours
    total_hour_avail_ms = 262 * 5 * 3600 * 1000

    print("\n" + "-" * 90)
    print("4. 24-HOUR UTC CAPITAL UTILIZATION HEATMAP")
    print("-" * 90)
    print(f"{'Hour (UTC)':<12} | {'Session Regime':<35} | {'Active Exposure %':<18} | {'Avg Open Trades':<15}")
    print("-" * 90)

    session_names = {
        0: "Asian Session (Tokyo Early)",
        1: "Asian Session (Tokyo Active)",
        2: "Asian Session (Tokyo Active)",
        3: "Asian Session (Pre-China)",
        4: "Asian Session (Late Asian)",
        5: "Frankfurt Early Bird Open",
        6: "Frankfurt Pre-London Accumulation",
        7: "London Cash Open (Initial Surge)",
        8: "London Peak Liquidity Window",
        9: "London Morning Expansion",
        10: "London Midday Transition",
        11: "London Lunch Stagnation",
        12: "European Midday / US Pre-Market",
        13: "NY Cash Open / London-NY Overlap",
        14: "London-NY Peak Overlap Liquidity",
        15: "Pre-London Fix Institutional Flow",
        16: "London Cash Close / US Afternoon",
        17: "US Afternoon Trend Continuation",
        18: "US Afternoon Trend Continuation",
        19: "Late US Session",
        20: "US Cash Close (4 PM EST)",
        21: "Post-Close / Global Rollover Spread",
        22: "Sydney Open (Early Asian)",
        23: "Wellington / Sydney Active"
    }

    hourly_pcts = []
    for h in range(24):
        act_ms = hourly_active_ms[h]
        # Active exposure is percentage of that hour across the 5 years where at least 1 trade is active
        # Notice multiple trades can be open, so avg open trades = act_ms / total_hour_avail_ms
        avg_open = act_ms / total_hour_avail_ms
        # Cap exposure % at 100%
        util_pct = min(avg_open * 100.0, 100.0)
        hourly_pcts.append((h, util_pct, avg_open))
        bar = "#" * int(round(util_pct / 4))
        print(f"{h:02d}:00 - {h+1:02d}:00  | {session_names[h]:<35} | {util_pct:5.1f}% {bar:<10} | {avg_open:4.2f} positions")

    print("-" * 90)

    # 5. CAPITAL EFFICIENCY RATIOS
    net_total_pips = df_trades['pips'].sum()
    pip_per_exposure_hour = net_total_pips / max(total_active_hours, 1e-6)
    pip_per_market_hour = net_total_pips / max(total_market_hours, 1e-6)

    print("\n" + "-" * 90)
    print("5. CAPITAL EFFICIENCY & RETURN ON EXPOSURE (ROE)")
    print("-" * 90)
    print(f"5-Year Cumulative Net Return:   {net_total_pips:+.1f} pips")
    print(f"Return per Active Exposure Hr:  {pip_per_exposure_hour:+.2f} pips / active hour")
    print(f"Return per Market Hour:         {pip_per_market_hour:+.2f} pips / trading hour")
    print(f"Average Profit per Trade:       {net_total_pips / total_trades:+.2f} pips / trade")
    print("=" * 90 + "\n")


if __name__ == "__main__":
    analyze_capital_utilization()
