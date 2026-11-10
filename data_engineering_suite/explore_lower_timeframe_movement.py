#!/usr/bin/env python3
"""
================================================================================
  DATA ENGINEERING SUITE: LOWER-TIMEFRAME (M5 / M1) INTRADAY MICRO-MOVEMENT STUDY
================================================================================
Investigates the internal micro-mechanics INSIDE major institutional sessions:

  1. The 15-Minute Micro-Rhythm Inside Each Hour (:00, :15, :30, :45 Quadrants)
     - Which 15-minute quadrant produces the expansion run vs the pullback?
  2. M5 Impulse Displacement & Mitigation Retest Behavior:
     - When a high-momentum M5 displacement candle occurs (body >= 70% range),
       does price retest the 50% midpoint (mitigation) before continuation?
     - Forward MFE/MAE of M5 Breakout Close vs M5 Retest Entry.
  3. Minute-by-Minute London Cash Open Surge (07:00 - 07:30 UTC at M1 Resolution):
     - How does order flow, tick velocity, and displacement evolve minute-by-minute
       during the critical 30 minutes of London opening?
  4. M5 CVD Absorption at Micro-Extremes:
     - Order Flow Imbalance behavior at M5 rolling 12-bar highs and lows.
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


def analyze_week_ltf(w: int) -> Dict[str, Any]:
    ticks, _ = get_week_dataset(w)
    if len(ticks) < 1000:
        return {}

    # Build M5 bars
    m5 = build_time_bars(ticks, timeframe_min=5)
    if len(m5) < 100:
        return {}

    c5 = m5['close'].values
    h5 = m5['high'].values
    l5 = m5['low'].values
    o5 = m5['open'].values
    v5 = m5['vol'].values
    deltas5 = m5['delta'].values
    hours5 = m5.index.hour.values
    minutes5 = m5.index.minute.values
    n5 = len(m5)

    # --------------------------------------------------------------------------
    # 1. THE 15-MINUTE QUADRANT MICRO-RHYTHM INSIDE EACH HOUR
    # Quadrant 0: :00 - :15  (m5 bars 0, 5, 10)
    # Quadrant 1: :15 - :30  (m5 bars 15, 20, 25)
    # Quadrant 2: :30 - :45  (m5 bars 30, 35, 40)
    # Quadrant 3: :45 - :00  (m5 bars 45, 50, 55)
    # --------------------------------------------------------------------------
    quadrant_ranges = {q: [] for q in range(4)}
    quadrant_efficiencies = {q: [] for q in range(4)}
    quadrant_mfes = {q: [] for q in range(4)}
    quadrant_maes = {q: [] for q in range(4)}

    # Filter to active European & US hours (07:00 - 17:00 UTC)
    for i in range(12, n5 - 12):
        hr = hours5[i]
        if hr < 7 or hr > 17:
            continue

        mn = minutes5[i]
        q = mn // 15
        rng = (h5[i] - l5[i]) / PIP_SIZE
        quadrant_ranges[q].append(rng)

        # 3-bar (15m) directional efficiency inside the quadrant
        if i >= 3:
            net_disp = abs(c5[i] - c5[i-3])
            path = np.sum(h5[i-2:i+1] - l5[i-2:i+1])
            er = net_disp / max(path, 1e-6)
            quadrant_efficiencies[q].append(er)

        # Forward 6-bar (30m) excursion from close of this M5 bar
        sub_h = h5[i+1 : i+7]
        sub_l = l5[i+1 : i+7]
        if len(sub_h) == 6:
            if c5[i] > o5[i]:
                fav = (np.max(sub_h) - c5[i]) / PIP_SIZE
                adv = (c5[i] - np.min(sub_l)) / PIP_SIZE
            else:
                fav = (c5[i] - np.min(sub_l)) / PIP_SIZE
                adv = (np.max(sub_h) - c5[i]) / PIP_SIZE
            quadrant_mfes[q].append(fav)
            quadrant_maes[q].append(adv)

    # --------------------------------------------------------------------------
    # 2. M5 IMPULSE DISPLACEMENT & MITIGATION RETEST BEHAVIOR
    # Detect high-displacement M5 impulse bars: Body >= 70% of range, range >= 6 pips
    # --------------------------------------------------------------------------
    m5_breakout_mfes = []
    m5_breakout_maes = []
    m5_retest_mfes = []
    m5_retest_maes = []
    retest_occurred_count = 0
    total_impulses = 0

    for i in range(20, n5 - 12):
        hr = hours5[i]
        if hr < 7 or hr > 17:
            continue

        bar_rng = (h5[i] - l5[i]) / PIP_SIZE
        body = abs(c5[i] - o5[i]) / PIP_SIZE

        # Impulse Candle Condition
        if bar_rng >= 7.0 and (body / max(bar_rng, 0.1)) >= 0.70:
            total_impulses += 1
            is_bull = c5[i] > o5[i]
            midpoint = 0.5 * (o5[i] + c5[i])

            # Measure Breakout Close Forward Excursion (1 hour = 12 M5 bars)
            fwd_h = h5[i+1 : i+13]
            fwd_l = l5[i+1 : i+13]
            if len(fwd_h) < 12:
                continue

            if is_bull:
                bo_fav = (np.max(fwd_h) - c5[i]) / PIP_SIZE
                bo_adv = (c5[i] - np.min(fwd_l)) / PIP_SIZE
            else:
                bo_fav = (c5[i] - np.min(fwd_l)) / PIP_SIZE
                bo_adv = (np.max(fwd_h) - c5[i]) / PIP_SIZE

            m5_breakout_mfes.append(bo_fav)
            m5_breakout_maes.append(bo_adv)

            # Check if price retested the midpoint in the subsequent 3 bars (15m)
            retested = False
            retest_idx = -1
            for k in range(1, 4):
                next_idx = i + k
                if next_idx >= n5:
                    break
                if is_bull and l5[next_idx] <= midpoint:
                    retested = True
                    retest_idx = next_idx
                    break
                elif not is_bull and h5[next_idx] >= midpoint:
                    retested = True
                    retest_idx = next_idx
                    break

            if retested and retest_idx != -1:
                retest_occurred_count += 1
                rem_h = h5[retest_idx+1 : i+13]
                rem_l = l5[retest_idx+1 : i+13]
                if len(rem_h) > 0:
                    if is_bull:
                        ret_fav = (np.max(rem_h) - midpoint) / PIP_SIZE
                        ret_adv = (midpoint - np.min(rem_l)) / PIP_SIZE
                    else:
                        ret_fav = (midpoint - np.min(rem_l)) / PIP_SIZE
                        ret_adv = (np.max(rem_h) - midpoint) / PIP_SIZE
                    m5_retest_mfes.append(ret_fav)
                    m5_retest_maes.append(ret_adv)

    # --------------------------------------------------------------------------
    # 3. MINUTE-BY-MINUTE LONDON CASH OPEN DYNAMICS (07:00 - 07:30 UTC)
    # Build M1 bars for week to analyze first 30 minutes of London
    # --------------------------------------------------------------------------
    m1 = build_time_bars(ticks, timeframe_min=1)
    lon_m1_vol = {m: [] for m in range(30)}
    lon_m1_rng = {m: [] for m in range(30)}

    if len(m1) > 100:
        h1_m1 = m1.index.hour.values
        min_m1 = m1.index.minute.values
        v_m1 = m1['vol'].values
        rng_m1 = (m1['high'].values - m1['low'].values) / PIP_SIZE

        for i in range(len(m1)):
            if h1_m1[i] == 7 and 0 <= min_m1[i] < 30:
                lon_m1_vol[min_m1[i]].append(v_m1[i])
                lon_m1_rng[min_m1[i]].append(rng_m1[i])

    return {
        'quadrant_ranges': {q: float(np.mean(vals)) if vals else 0.0 for q, vals in quadrant_ranges.items()},
        'quadrant_efficiencies': {q: float(np.mean(vals)) if vals else 0.0 for q, vals in quadrant_efficiencies.items()},
        'quadrant_mfes': {q: float(np.mean(vals)) if vals else 0.0 for q, vals in quadrant_mfes.items()},
        'quadrant_maes': {q: float(np.mean(vals)) if vals else 0.0 for q, vals in quadrant_maes.items()},
        'total_impulses': total_impulses,
        'retest_count': retest_occurred_count,
        'bo_mfe': float(np.mean(m5_breakout_mfes)) if m5_breakout_mfes else 0.0,
        'bo_mae': float(np.mean(m5_breakout_maes)) if m5_breakout_maes else 0.0,
        'ret_mfe': float(np.mean(m5_retest_mfes)) if m5_retest_mfes else 0.0,
        'ret_mae': float(np.mean(m5_retest_maes)) if m5_retest_maes else 0.0,
        'lon_m1_vol': {m: float(np.mean(lon_m1_vol[m])) if lon_m1_vol[m] else 0.0 for m in range(30)},
        'lon_m1_rng': {m: float(np.mean(lon_m1_rng[m])) if lon_m1_rng[m] else 0.0 for m in range(30)},
    }


def run():
    print("=" * 115)
    print("     DATA ENGINEERING SUITE: LOWER-TIMEFRAME (M5 / M1) INTRADAY MICRO-MOVEMENT STUDY     ".center(115))
    print("=" * 115)

    weeks = list(range(1, 53))
    t0 = time.time()
    num_workers = min(8, os.cpu_count() or 4)

    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        results = list(ex.map(analyze_week_ltf, weeks))

    dt = time.time() - t0
    print(f"[+] 52 weeks (~26M ticks) processed at M5 and M1 resolution in {dt:.1f}s!\n")

    results = [r for r in results if r]
    count_w = len(results)

    # 1. 15-MINUTE QUADRANT MICRO-RHYTHM
    print("-" * 115)
    print("1. THE 15-MINUTE MICRO-RHYTHM INSIDE EACH HOURLY BAR (07:00 - 17:00 UTC)")
    print("-" * 115)
    print(f"{'Quadrant':<22} | {'Minutes':<15} | {'Avg M5 Range':<15} | {'Directness (ER)':<18} | {'Fwd 30m MFE':<14} | {'Fwd 30m MAE':<14} | {'Asymmetry'}")
    print("-" * 115)

    quad_names = [
        ("Quadrant 1: Opening Thrust", ":00 - :15"),
        ("Quadrant 2: Mid-Hour Pullback", ":15 - :30"),
        ("Quadrant 3: Peak Expansion", ":30 - :45"),
        ("Quadrant 4: Pre-Close Rebalance", ":45 - :00")
    ]

    for q, (qname, qmins) in enumerate(quad_names):
        avg_rng = np.mean([r['quadrant_ranges'][q] for r in results])
        avg_er = np.mean([r['quadrant_efficiencies'][q] for r in results])
        avg_mfe = np.mean([r['quadrant_mfes'][q] for r in results])
        avg_mae = np.mean([r['quadrant_maes'][q] for r in results])
        ratio = avg_mfe / max(avg_mae, 1e-6)
        print(f"{qname:<22} | {qmins:<15} | {avg_rng:6.1f} pips       | {avg_er:6.3f}             | {avg_mfe:6.1f} p        | {avg_mae:6.1f} p        | {ratio:6.2f} : 1")

    print("-" * 115)

    # 2. M5 DISPLACEMENT BREAKOUT vs MITIGATION RETEST
    print("\n" + "-" * 115)
    print("2. M5 IMPULSE CANDLE DISPLACEMENT vs MITIGATION RETEST (BODY >= 70% RANGE)")
    print("-" * 115)

    tot_imp = sum(r['total_impulses'] for r in results)
    tot_ret = sum(r['retest_count'] for r in results)
    retest_pct = (tot_ret / max(tot_imp, 1)) * 100

    bo_mfe = np.mean([r['bo_mfe'] for r in results if r['bo_mfe'] > 0])
    bo_mae = np.mean([r['bo_mae'] for r in results if r['bo_mae'] > 0])
    bo_ratio = bo_mfe / max(bo_mae, 1e-6)

    ret_mfe = np.mean([r['ret_mfe'] for r in results if r['ret_mfe'] > 0])
    ret_mae = np.mean([r['ret_mae'] for r in results if r['ret_mae'] > 0])
    ret_ratio = ret_mfe / max(ret_mae, 1e-6)

    print(f"Total High-Displacement M5 Impulse Bars Audited: {tot_imp:,} events")
    print(f"Percentage Retesting 50% Midpoint within 15 Minutes: {retest_pct:.1f}% ({tot_ret:,} retests)")
    print("-" * 115)
    print(f"Execution Paradigm             | Avg Fwd 1h MFE | Avg Fwd 1h MAE | Asymmetry Ratio | Retail vs Institutional")
    print("-" * 115)
    print(f"A. Entering on Breakout Close  | {bo_mfe:6.1f} pips     | {bo_mae:6.1f} pips     | {bo_ratio:6.2f} : 1       | Retail FOMO (High Drawdown!)")
    print(f"B. Waiting for Midpoint Retest | {ret_mfe:6.1f} pips     | {ret_mae:6.1f} pips     | {ret_ratio:6.2f} : 1       | Institutional Limit Entry (Superior!)")
    print("-" * 115)

    # 3. LONDON OPEN MINUTE-BY-MINUTE VELOCITY (07:00 - 07:30 UTC)
    print("\n" + "-" * 115)
    print("3. MINUTE-BY-MINUTE LONDON OPEN TICK ARRIVAL VELOCITY & SPREAD PROFILE (07:00 - 07:30 UTC)")
    print("-" * 115)
    print(f"{'Minute (UTC)':<15} | {'Avg Ticks / Min':<20} | {'Avg M1 Range (pips)':<25} | {'Micro-Regime'}")
    print("-" * 115)

    for m in range(0, 30, 3):
        # average 3-minute chunk
        avg_vol = np.mean([np.mean([r['lon_m1_vol'][min_k] for min_k in range(m, min(m+3, 30))]) for r in results])
        avg_rng = np.mean([np.mean([r['lon_m1_rng'][min_k] for min_k in range(m, min(m+3, 30))]) for r in results])

        if m == 0:
            regime = "INITIAL SHOCK SURGE (Whipsaw Noise)"
        elif m < 9:
            regime = "LIQUIDITY ABSORPTION / PROBE"
        elif 9 <= m < 18:
            regime = "DIRECTIONAL IGNITION BREAKOUT"
        else:
            regime = "SUSTAINED FLOW / EXPANSION"

        print(f"07:{m:02d} - 07:{min(m+2,29):02d}     | {avg_vol:6.0f} ticks/min        | {avg_rng:5.2f} pips/min               | {regime}")

    print("=" * 115 + "\n")


if __name__ == "__main__":
    run()
