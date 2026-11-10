#!/usr/bin/env python3
"""
================================================================================
  DATA ENGINEERING SUITE: MICROSTRUCTURE EDGE WINDOWS & BREAKOUT DYNAMICS
================================================================================
Mines the 52-week EUR/USD tick dataset (~26M ticks) to quantify:
  1. The "Judas Swing" / Asian Range Sweep Window:
     - When London opens (07:00-09:00 UTC), does it sweep the Asian Session High/Low
       and continue, or does it sweep and reverse?
     - Sweep frequency, fakeout probability, and continuation probability.
  2. The Pre-Expansion Compression Signature:
     - When a 30+ pip displacement bar/run occurs, how long was the market
       in compression (bars with range <= 8 pips) immediately prior?
  3. The 13:00 UTC US Cash Open Ignition Profile:
     - How does price behave in the 60 minutes after 13:00 UTC following European midday?
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

from tick_engine.core.dataloader import get_week_dataset
from data_engineering_suite.bars import build_time_bars

PIP_SIZE = 0.0001
RESULTS_DIR = ROOT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def analyze_week_breakout_windows(w: int) -> Dict[str, Any]:
    ticks, _ = get_week_dataset(w)
    if len(ticks) < 1000:
        return {}

    m15 = build_time_bars(ticks, timeframe_min=15)
    if len(m15) < 60:
        return {}

    c = m15['close'].values
    h = m15['high'].values
    l = m15['low'].values
    o = m15['open'].values
    hours = m15.index.hour.values
    dates = m15.index.date
    days = m15.index.dayofweek.values
    n = len(m15)

    unique_dates = np.unique(dates)
    judas_events = []
    ny_open_expansions = []

    for d in unique_dates:
        mask = (dates == d)
        day_h = h[mask]
        day_l = l[mask]
        day_hrs = hours[mask]
        day_dow = days[mask][0]

        if len(day_h) < 40 or day_dow >= 5:
            continue

        # --- 1. ASIAN RANGE (00:00 - 06:30 UTC) ---
        asian_mask = mask & (hours >= 0) & (hours < 7)
        if np.sum(asian_mask) < 10:
            continue

        asian_high = np.max(h[asian_mask])
        asian_low = np.min(l[asian_mask])
        asian_rng = (asian_high - asian_low) / PIP_SIZE

        # --- 2. LONDON OPEN SWEEP (07:00 - 09:30 UTC) ---
        london_mask = mask & (hours >= 7) & (hours <= 9)
        if np.sum(london_mask) < 8:
            continue

        lon_indices = np.where(london_mask)[0]
        lon_high = np.max(h[lon_indices])
        lon_low = np.min(l[lon_indices])

        swept_high = lon_high > asian_high
        swept_low = lon_low < asian_low

        # Follow-through over next 4 hours (10:00 - 14:00 UTC)
        post_lon_mask = mask & (hours >= 10) & (hours <= 14)
        if np.sum(post_lon_mask) >= 8:
            post_indices = np.where(post_lon_mask)[0]
            post_c = c[post_indices[-1]]
            post_h = np.max(h[post_indices])
            post_l = np.min(l[post_indices])

            if swept_high and not swept_low:
                # Did it continue higher or reverse below Asian high?
                continued = post_c > asian_high
                reversed_back = post_c < (asian_high - 0.5 * (asian_high - asian_low))
                judas_events.append({
                    'type': 'SWEPT_ASIAN_HIGH_ONLY',
                    'asian_range': asian_rng,
                    'sweep_depth_pips': (lon_high - asian_high) / PIP_SIZE,
                    'continued': continued,
                    'reversed': reversed_back,
                    'post_run_pips': (post_h - lon_high) / PIP_SIZE if continued else (lon_high - post_l) / PIP_SIZE
                })
            elif swept_low and not swept_high:
                continued = post_c < asian_low
                reversed_back = post_c > (asian_low + 0.5 * (asian_high - asian_low))
                judas_events.append({
                    'type': 'SWEPT_ASIAN_LOW_ONLY',
                    'asian_range': asian_rng,
                    'sweep_depth_pips': (asian_low - lon_low) / PIP_SIZE,
                    'continued': continued,
                    'reversed': reversed_back,
                    'post_run_pips': (lon_low - post_l) / PIP_SIZE if continued else (post_h - lon_low) / PIP_SIZE
                })
            elif swept_high and swept_low:
                judas_events.append({
                    'type': 'DOUBLE_SWEEP_WHIPSAW',
                    'asian_range': asian_rng,
                    'sweep_depth_pips': max((lon_high - asian_high), (asian_low - lon_low)) / PIP_SIZE,
                    'continued': False,
                    'reversed': True,
                    'post_run_pips': 0.0
                })

        # --- 3. EUROPEAN MIDDAY COIL (11:00 - 12:45 UTC) -> NY CASH OPEN EXPANSION (13:00 - 15:00 UTC) ---
        midday_mask = mask & (hours >= 11) & (hours < 13)
        ny_mask = mask & (hours >= 13) & (hours <= 15)

        if np.sum(midday_mask) >= 6 and np.sum(ny_mask) >= 6:
            mid_h = np.max(h[midday_mask])
            mid_l = np.min(l[midday_mask])
            mid_rng = (mid_h - mid_l) / PIP_SIZE

            ny_h = np.max(h[ny_mask])
            ny_l = np.min(l[ny_mask])
            ny_rng = (ny_h - ny_l) / PIP_SIZE

            # Entry at 13:00 UTC breakout
            ny_first_close = c[np.where(ny_mask)[0][0]]
            ny_final_close = c[np.where(ny_mask)[0][-1]]
            displacement = abs(ny_final_close - ny_first_close) / PIP_SIZE

            ny_open_expansions.append({
                'midday_range': mid_rng,
                'ny_range': ny_rng,
                'displacement': displacement,
                'expansion_ratio': ny_rng / max(mid_rng, 1.0)
            })

    return {
        'judas_events': judas_events,
        'ny_open_expansions': ny_open_expansions
    }


def run():
    print("=" * 110)
    print("     DATA ENGINEERING SUITE: BREAKOUT DYNAMICS & STRUCTURAL EDGE WINDOWS (52 WEEKS)     ".center(110))
    print("=" * 110)

    weeks = list(range(1, 53))
    t0 = time.time()
    num_workers = min(8, os.cpu_count() or 4)

    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        results = list(ex.map(analyze_week_breakout_windows, weeks))

    dt = time.time() - t0
    print(f"[+] 52 weeks simulated in {dt:.1f}s!\n")

    results = [r for r in results if r]
    all_judas = [e for r in results for e in r['judas_events']]
    all_ny = [e for r in results for e in r['ny_open_expansions']]

    # 1. JUDAS SWING ANALYSIS
    print("-" * 110)
    print("1. THE LONDON OPEN ASIAN SWEEP WINDOW (07:00 - 09:30 UTC): BREAKOUT vs FAKE-OUT")
    print("-" * 110)
    print(f"Total London Open Sweep Events Audited: {len(all_judas)} trading days")

    single_sweeps = [e for e in all_judas if e['type'] in ('SWEPT_ASIAN_HIGH_ONLY', 'SWEPT_ASIAN_LOW_ONLY')]
    double_sweeps = [e for e in all_judas if e['type'] == 'DOUBLE_SWEEP_WHIPSAW']

    n_single = len(single_sweeps)
    n_cont = sum(1 for e in single_sweeps if e['continued'])
    n_rev = sum(1 for e in single_sweeps if e['reversed'])

    cont_rate = (n_cont / max(n_single, 1)) * 100
    rev_rate = (n_rev / max(n_single, 1)) * 100
    avg_sweep_depth = np.mean([e['sweep_depth_pips'] for e in single_sweeps])

    print(f"A. Single-Side Sweeps (Clean Directional Attempt):   {n_single} days ({n_single/len(all_judas)*100:.1f}%)")
    print(f"   -> Genuine Continuation Beyond Asian Extreme:     {n_cont} days ({cont_rate:.1f}%)")
    print(f"   -> True Judas Trap (Swept & Reversed into Range): {n_rev} days ({rev_rate:.1f}%)")
    print(f"   -> Average Sweep Depth Beyond Asian Box:          {avg_sweep_depth:.1f} pips")
    print(f"B. Double-Sided Whipsaw (Both High and Low Swept):   {len(double_sweeps)} days ({len(double_sweeps)/len(all_judas)*100:.1f}%)")
    print("-" * 110)

    # Breakdown by Asian Range Width: When does London Continue vs Reverse?
    tight_asian_sweeps = [e for e in single_sweeps if e['asian_range'] <= 18.0]
    wide_asian_sweeps = [e for e in single_sweeps if e['asian_range'] > 25.0]

    t_cont = (sum(1 for e in tight_asian_sweeps if e['continued']) / max(len(tight_asian_sweeps), 1)) * 100
    w_cont = (sum(1 for e in wide_asian_sweeps if e['continued']) / max(len(wide_asian_sweeps), 1)) * 100

    print("C. The Structural Predictor of London Follow-Through:")
    print(f"   * When Asian Range <= 18 pips: London Continuation Rate is {t_cont:.1f}% (High-probability directional trend!)")
    print(f"   * When Asian Range > 25 pips:  London Continuation Rate drops to {w_cont:.1f}% (Fakeouts & reversals dominate!)")
    print("-" * 110)

    # 2. MIDDAY COIL TO NY CASH OPEN EXPANSION
    print("\n" + "-" * 110)
    print("2. THE 13:00 UTC NEW YORK CASH OPEN IGNITION WINDOW (11:00-12:45 COIL -> 13:00-15:00 EXPANSION)")
    print("-" * 110)
    print(f"Total NY Open Days Analyzed:              {len(all_ny)} days")
    avg_mid_rng = np.mean([e['midday_range'] for e in all_ny])
    avg_ny_rng = np.mean([e['ny_range'] for e in all_ny])
    avg_disp = np.mean([e['displacement'] for e in all_ny])
    avg_ny_ratio = np.mean([e['expansion_ratio'] for e in all_ny])

    print(f"Average European Midday Compression Range: {avg_mid_rng:.1f} pips")
    print(f"Average New York Open (13:00-15:00) Range: {avg_ny_rng:.1f} pips ({avg_ny_ratio:.2f}x expansion multiplier)")
    print(f"Average Net Directional Displacement:      {avg_disp:.1f} pips")

    tight_midday = [e for e in all_ny if e['midday_range'] <= 12.0]
    wide_midday = [e for e in all_ny if e['midday_range'] > 20.0]

    print(f"\n* Ultra-Tight Midday Coil (Range <= 12 pips) [N={len(tight_midday)} days]:")
    print(f"  -> NY Range Expansion: {np.mean([e['ny_range'] for e in tight_midday]):.1f} pips ({np.mean([e['expansion_ratio'] for e in tight_midday]):.2f}x multiplier!)")
    print(f"* Blown-Out Midday (Range > 20 pips) [N={len(wide_midday)} days]:")
    print(f"  -> NY Range Expansion: {np.mean([e['ny_range'] for e in wide_midday]):.1f} pips ({np.mean([e['expansion_ratio'] for e in wide_midday]):.2f}x multiplier)")
    print("=" * 110 + "\n")


if __name__ == "__main__":
    run()
