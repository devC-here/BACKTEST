#!/usr/bin/env python3
"""
================================================================================
  DATA ENGINEERING SUITE: DEEP DATA SCAN & MICROSTRUCTURE EXPLOITATION RUNNER
================================================================================
Orchestrates a comprehensive multi-year empirical exploration:
  1. 24-Hour Microstructure Diurnal Smile (Spread, Volume, Volatility, Hurst)
  2. Information Sampling Density (Time vs Tick vs Range Bars)
  3. Audits New Microstructure Alpha Opportunities (OFI Absorption & Velocity Ignition)
================================================================================
"""

from __future__ import annotations
import sys
import os
import time
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tick_engine.core.engine import BacktestEngine, TRADE_DTYPE
from tick_engine.core.dataloader import get_week_dataset, load_manifest
from tick_engine.core.metrics import compute_metrics

from data_engineering_suite.analytics import analyze_week_microstructure
from data_engineering_suite.opportunities import OFIAbsorptionFadeStrategy, TickVelocityIgnitionStrategy

RESULTS_DIR = ROOT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _worker_scan_week(w: int):
    ticks, _ = get_week_dataset(w)
    # 1. Microstructure analytics
    micro_stats = analyze_week_microstructure(ticks, week_num=w)

    # 2. Backtest the two new microstructure alpha models
    engine = BacktestEngine(pip_size=0.0001, use_slippage=True)
    ofi_strat = OFIAbsorptionFadeStrategy(target_rr=1.65)
    vel_strat = TickVelocityIgnitionStrategy(target_rr=1.65)

    ofi_trades = engine.run(ticks, ofi_strat)
    vel_trades = engine.run(ticks, vel_strat)

    return w, micro_stats, ofi_trades, vel_trades


def run_deep_data_scan():
    print("=" * 105)
    print("     DATA ENGINEERING SUITE: 52-WEEK DEEP MICROSTRUCTURE SCAN & ALPHA DISCOVERY     ".center(105))
    print("=" * 105)

    t0 = time.time()
    weeks = list(range(1, 53))
    num_workers = min(8, os.cpu_count() or 4)

    print(f"[*] Dispatching deep microstructure analysis across {len(weeks)} weeks (~26M ticks)...")
    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        results = list(ex.map(_worker_scan_week, weeks))

    results.sort(key=lambda x: x[0])
    dt = time.time() - t0
    print(f"[+] Scan completed in {dt:.1f}s!\n")

    # Aggregate Microstructure Stats
    all_micro = [r[1] for r in results if r[1]]
    total_ticks = sum(m['ticks'] for m in all_micro)
    total_m15 = sum(m['m15_bars_count'] for m in all_micro)
    total_range = sum(m['range_bars_10p_count'] for m in all_micro)
    total_tick_bars = sum(m['tick_bars_1000t_count'] for m in all_micro)

    # Hourly Diurnal Smile Aggregation
    hourly_spread = np.zeros(24)
    hourly_ticks = np.zeros(24)
    hourly_range = np.zeros(24)
    hourly_p_vol = np.zeros(24)
    hourly_hurst = np.zeros(24)
    count_weeks = len(all_micro)

    for m in all_micro:
        for h in range(24):
            prof = m['hourly_profiles'][h]
            hourly_spread[h] += prof['spread_pips']
            hourly_ticks[h] += prof['tick_count']
            hourly_range[h] += prof['range_pips']
            hourly_p_vol[h] += prof['parkinson_vol']
            hourly_hurst[h] += prof['hurst']

    hourly_spread /= count_weeks
    hourly_ticks /= count_weeks
    hourly_range /= count_weeks
    hourly_p_vol /= count_weeks
    hourly_hurst /= count_weeks

    # 1. PRINT 24-HOUR MICROSTRUCTURE SMILE
    print("-" * 105)
    print("1. 24-HOUR EUR/USD MICROSTRUCTURE DIURNAL SMILE (52 WEEKS)")
    print("-" * 105)
    print(f"{'Hour (UTC)':<12} | {'Session Regime':<35} | {'Spread (p)':<10} | {'M15 Ticks':<10} | {'M15 Range':<10} | {'Hurst (H)':<9}")
    print("-" * 105)

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

    for h in range(24):
        h_tag = " [TREND PERSIST]" if hourly_hurst[h] > 0.52 else (" [MEAN REVERT]" if hourly_hurst[h] < 0.48 else "")
        print(f"{h:02d}:00 - {h+1:02d}:00  | {session_names[h]:<35} | {hourly_spread[h]:6.2f} p    | {hourly_ticks[h]:6.0f} t    | {hourly_range[h]:5.1f} p   | {hourly_hurst[h]:.3f}{h_tag}")

    print("-" * 105)

    # 2. INFORMATION SAMPLING DENSITY COMPARISON
    print("\n" + "-" * 105)
    print("2. INFORMATION SAMPLING DENSITY: TIME BARS vs RANGE BARS vs TICK BARS")
    print("-" * 105)
    print(f"Total Authentic Ticks Analyzed:       {total_ticks:,} ticks")
    print(f"Standard 15-Minute Time Bars (M15):   {total_m15:,} bars (constant clock, variable information)")
    print(f"10-Pip Range Bars (Pure Volatility):   {total_range:,} bars ({total_m15/max(total_range,1):.2f}x compression ratio)")
    print(f"1,000-Tick Information Bars:          {total_tick_bars:,} bars ({total_m15/max(total_tick_bars,1):.2f}x information synchronization)")
    print(f"Mean Spread across 52 Weeks:          {np.mean(hourly_spread):.2f} pips (Min: {np.min(hourly_spread):.2f}p, Max: {np.max(hourly_spread):.2f}p)")

    # 3. MICROSTRUCTURE ALPHA AUDIT RESULTS
    ofi_list = [t for r in results for t in [r[2]] if len(t) > 0]
    vel_list = [t for r in results for t in [r[3]] if len(t) > 0]

    tr_ofi = np.concatenate(ofi_list) if ofi_list else np.empty(0, dtype=TRADE_DTYPE)
    tr_vel = np.concatenate(vel_list) if vel_list else np.empty(0, dtype=TRADE_DTYPE)

    m_ofi = compute_metrics(tr_ofi)
    m_vel = compute_metrics(tr_vel)

    print("\n" + "=" * 105)
    print("3. EXPLOITING THE DATA: 2 NEW MICROSTRUCTURE ALPHA OPPORTUNITIES")
    print("=" * 105)
    print(f"A. OFI_Absorption_Fade_Elite (CVD Delta Divergence at Session Extremes):")
    print(f"   Trades:       {m_ofi['total_trades']} ({m_ofi['total_trades']/52:.1f}/wk, {m_ofi['total_trades']/250:.2f}/day)")
    print(f"   Win Rate:     {m_ofi['win_rate_pct']:.1f}%")
    print(f"   Profit Factor:{m_ofi['profit_factor']:.2f}")
    print(f"   Net Pips:     {m_ofi['net_pips']:+.1f} pips")
    print(f"   Max Drawdown: {m_ofi['max_drawdown_pips']:.1f} pips")
    print("-" * 105)
    print(f"B. Tick_Velocity_Ignition_Elite (Hyper-Frequency Order Arrival Momentum):")
    print(f"   Trades:       {m_vel['total_trades']} ({m_vel['total_trades']/52:.1f}/wk, {m_vel['total_trades']/250:.2f}/day)")
    print(f"   Win Rate:     {m_vel['win_rate_pct']:.1f}%")
    print(f"   Profit Factor:{m_vel['profit_factor']:.2f}")
    print(f"   Net Pips:     {m_vel['net_pips']:+.1f} pips")
    print(f"   Max Drawdown: {m_vel['max_drawdown_pips']:.1f} pips")
    print("=" * 105 + "\n")

    # Export scan summary to JSON
    export_payload = {
        'total_ticks': total_ticks,
        'weeks_scanned': len(weeks),
        'hourly_diurnal_smile': {
            h: {
                'regime': session_names[h],
                'spread_pips': round(float(hourly_spread[h]), 2),
                'ticks_per_15m': round(float(hourly_ticks[h]), 0),
                'range_pips_15m': round(float(hourly_range[h]), 1),
                'parkinson_vol': round(float(hourly_p_vol[h]), 5),
                'hurst_exponent': round(float(hourly_hurst[h]), 3)
            } for h in range(24)
        },
        'sampling_density': {
            'm15_bars': total_m15,
            'range_10p_bars': total_range,
            'tick_1000t_bars': total_tick_bars
        },
        'ofi_absorption_metrics': {
            'trades': m_ofi['total_trades'],
            'win_rate': m_ofi['win_rate_pct'],
            'profit_factor': m_ofi['profit_factor'],
            'net_pips': m_ofi['net_pips'],
            'max_dd': m_ofi['max_drawdown_pips']
        },
        'tick_velocity_ignition_metrics': {
            'trades': m_vel['total_trades'],
            'win_rate': m_vel['win_rate_pct'],
            'profit_factor': m_vel['profit_factor'],
            'net_pips': m_vel['net_pips'],
            'max_dd': m_vel['max_drawdown_pips']
        }
    }

    with open(RESULTS_DIR / "deep_data_scan_results.json", "w") as f:
        json.dump(export_payload, f, indent=2)

    print(f"[+] Detailed scan payload saved to: {RESULTS_DIR / 'deep_data_scan_results.json'}")


if __name__ == "__main__":
    run_deep_data_scan()
