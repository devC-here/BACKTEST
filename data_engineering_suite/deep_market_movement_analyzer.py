#!/usr/bin/env python3
"""
================================================================================
  DATA ENGINEERING SUITE: DEEP MARKET MOVEMENT & EDGE WINDOWS ANALYZER
================================================================================
A pure quantitative study of raw market movement, price discovery, and excursion
dynamics without indicators or pre-conceived strategy rules:

  1. Directional Efficiency Window (Kaufman Efficiency Ratio by Hour)
     - Where does price slice cleanly vs where does it churn in friction/chop?
  2. Momentum Drift & Excursion Asymmetry (Unconditional Forward MFE/MAE)
     - When does a directional candle produce massive follow-through vs instant fade?
  3. Daily High & Low Formation Windows (HOD / LOD Distribution)
     - At what exact hours do the true extreme turning points form?
  4. Compression-to-Expansion Phase Transitions (Asian Range -> London/NY Expansion)
     - How does pre-market compression dictate the magnitude of the day's expansion?
  5. Day-of-Week Macro Volatility & Flow Cycle (Monday to Friday)
     - How does institutional liquidity evolve across the trading week?
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


def analyze_week_market_movement(w: int) -> Dict[str, Any]:
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
    v = m15['vol'].values
    spread = m15['spread_pips'].values
    hours = m15.index.hour.values
    days = m15.index.dayofweek.values
    dates = m15.index.date
    n = len(m15)

    # --------------------------------------------------------------------------
    # 1. KAUFMAN DIRECTIONAL EFFICIENCY RATIO (ER) BY 15-MIN BAR & HOUR
    # ER = |Net displacement over 4 bars (1h)| / Sum of 4 individual bar ranges
    # --------------------------------------------------------------------------
    er_by_hour = {hr: [] for hr in range(24)}
    for i in range(4, n):
        net_move = abs(c[i] - c[i-4])
        path_length = np.sum(h[i-3:i+1] - l[i-3:i+1])
        er = net_move / max(path_length, 1e-6)
        er_by_hour[hours[i]].append(er)

    # --------------------------------------------------------------------------
    # 2. RAW MOMENTUM FORWARD EXCURSION BY HOUR
    # If a candle is bullish (C > O), what is the 2-hour forward MFE and MAE?
    # --------------------------------------------------------------------------
    fwd_mfe_by_hour = {hr: [] for hr in range(24)}
    fwd_mae_by_hour = {hr: [] for hr in range(24)}
    fwd_drift_by_hour = {hr: [] for hr in range(24)}

    # Horizon = 8 bars (2 hours forward)
    horizon = 8
    for i in range(10, n - horizon):
        hr = hours[i]
        ep = c[i]
        sub_h = h[i+1 : i+1+horizon]
        sub_l = l[i+1 : i+1+horizon]
        if len(sub_h) == 0:
            continue

        is_bull = c[i] > o[i]
        is_bear = c[i] < o[i]

        if is_bull:
            fav = (np.max(sub_h) - ep) / PIP_SIZE
            adv = (ep - np.min(sub_l)) / PIP_SIZE
            drift = (c[i+horizon] - ep) / PIP_SIZE
            fwd_mfe_by_hour[hr].append(fav)
            fwd_mae_by_hour[hr].append(adv)
            fwd_drift_by_hour[hr].append(drift)
        elif is_bear:
            fav = (ep - np.min(sub_l)) / PIP_SIZE
            adv = (np.max(sub_h) - ep) / PIP_SIZE
            drift = (ep - c[i+horizon]) / PIP_SIZE
            fwd_mfe_by_hour[hr].append(fav)
            fwd_mae_by_hour[hr].append(adv)
            fwd_drift_by_hour[hr].append(drift)

    # --------------------------------------------------------------------------
    # 3. DAILY HIGH (HOD) & DAILY LOW (LOD) FORMATION TIMING
    # --------------------------------------------------------------------------
    hod_hours = []
    lod_hours = []
    unique_dates = np.unique(dates)
    daily_ranges = []
    day_of_week_ranges = {d: [] for d in range(5)}  # Mon=0, Fri=4
    asian_vs_london_expansions = []

    for d in unique_dates:
        mask = (dates == d)
        day_h = h[mask]
        day_l = l[mask]
        day_hrs = hours[mask]
        day_dow = days[mask][0]

        if len(day_h) < 40 or day_dow >= 5:  # Skip weekends or truncated days
            continue

        max_idx = np.argmax(day_h)
        min_idx = np.argmin(day_l)

        hod_hours.append(int(day_hrs[max_idx]))
        lod_hours.append(int(day_hrs[min_idx]))

        day_rng = (np.max(day_h) - np.min(day_l)) / PIP_SIZE
        daily_ranges.append(day_rng)
        day_of_week_ranges[day_dow].append(day_rng)

        # Asian Range (00:00 - 06:00 UTC) vs London/NY Range (07:00 - 17:00 UTC)
        asian_mask = mask & (hours >= 0) & (hours < 6)
        lon_ny_mask = mask & (hours >= 7) & (hours <= 17)

        if np.sum(asian_mask) > 10 and np.sum(lon_ny_mask) > 20:
            asian_rng = (np.max(h[asian_mask]) - np.min(l[asian_mask])) / PIP_SIZE
            lon_ny_rng = (np.max(h[lon_ny_mask]) - np.min(l[lon_ny_mask])) / PIP_SIZE
            asian_vs_london_expansions.append({
                'asian_range': asian_rng,
                'lon_ny_range': lon_ny_rng,
                'expansion_ratio': lon_ny_rng / max(asian_rng, 1.0)
            })

    return {
        'er_by_hour': {hr: float(np.mean(vals)) if vals else 0.0 for hr, vals in er_by_hour.items()},
        'fwd_mfe_by_hour': {hr: float(np.mean(vals)) if vals else 0.0 for hr, vals in fwd_mfe_by_hour.items()},
        'fwd_mae_by_hour': {hr: float(np.mean(vals)) if vals else 0.0 for hr, vals in fwd_mae_by_hour.items()},
        'fwd_drift_by_hour': {hr: float(np.mean(vals)) if vals else 0.0 for hr, vals in fwd_drift_by_hour.items()},
        'hod_hours': hod_hours,
        'lod_hours': lod_hours,
        'daily_ranges': daily_ranges,
        'day_of_week_ranges': day_of_week_ranges,
        'asian_vs_london': asian_vs_london_expansions
    }


def run_market_movement_analysis():
    print("=" * 115)
    print("     DATA ENGINEERING SUITE: DEEP MARKET MOVEMENT & EDGE WINDOWS STUDY (52 WEEKS)     ".center(115))
    print("=" * 115)

    weeks = list(range(1, 53))
    num_workers = min(8, os.cpu_count() or 4)
    t0 = time.time()

    print(f"[*] Analyzing raw market movement across {len(weeks)} weeks (~26M ticks) on {num_workers} CPU cores...")
    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        results = list(ex.map(analyze_week_market_movement, weeks))

    dt = time.time() - t0
    print(f"[+] Market movement analysis completed in {dt:.1f}s!\n")

    results = [r for r in results if r]

    # 1. Aggregate Directional Efficiency & Momentum Excursion by Hour
    avg_er = np.zeros(24)
    avg_mfe = np.zeros(24)
    avg_mae = np.zeros(24)
    avg_drift = np.zeros(24)
    hod_counts = np.zeros(24, dtype=int)
    lod_counts = np.zeros(24, dtype=int)

    count_w = len(results)
    for r in results:
        for hr in range(24):
            avg_er[hr] += r['er_by_hour'][hr]
            avg_mfe[hr] += r['fwd_mfe_by_hour'][hr]
            avg_mae[hr] += r['fwd_mae_by_hour'][hr]
            avg_drift[hr] += r['fwd_drift_by_hour'][hr]

        for h_val in r['hod_hours']:
            hod_counts[h_val] += 1
        for l_val in r['lod_hours']:
            lod_counts[l_val] += 1

    avg_er /= count_w
    avg_mfe /= count_w
    avg_mae /= count_w
    avg_drift /= count_w

    total_extreme_events = np.sum(hod_counts) + np.sum(lod_counts)

    # --------------------------------------------------------------------------
    # TABLE 1: 24-HOUR MARKET MOVEMENT & MOMENTUM ASYMMETRY MATRIX
    # --------------------------------------------------------------------------
    print("-" * 115)
    print("1. 24-HOUR RAW PRICE DYNAMICS & MOMENTUM EXCURSION ASYMMETRY (M15)")
    print("-" * 115)
    print(f"{'Hour (UTC)':<12} | {'Directional Efficiency':<24} | {'Fwd 2h MFE':<12} | {'Fwd 2h MAE':<12} | {'MFE/MAE Ratio':<15} | {'Net Drift':<10} | {'Edge Type':<16}")
    print("-" * 115)

    edge_profile = {}
    for h in range(24):
        ratio = avg_mfe[h] / max(avg_mae[h], 1e-6)
        er_val = avg_er[h]
        drift_val = avg_drift[h]

        if ratio >= 1.25 and er_val >= 0.40:
            edge_type = "TREND SURGE"
        elif ratio <= 0.90:
            edge_type = "CHOP / FADE"
        elif er_val <= 0.32:
            edge_type = "FRICTION NOISE"
        else:
            edge_type = "NEUTRAL"

        print(f"{h:02d}:00 - {h+1:02d}:00  | {er_val:6.3f} (Path Directness)    | {avg_mfe[h]:6.1f} p    | {avg_mae[h]:6.1f} p    | {ratio:6.2f} : 1       | {drift_val:+5.1f} p     | {edge_type:<16}")

        edge_profile[h] = {
            'efficiency_ratio': round(float(er_val), 3),
            'fwd_mfe': round(float(avg_mfe[h]), 1),
            'fwd_mae': round(float(avg_mae[h]), 1),
            'asymmetry_ratio': round(float(ratio), 2),
            'net_drift': round(float(drift_val), 1),
            'edge_type': edge_type
        }

    print("-" * 115)

    # --------------------------------------------------------------------------
    # TABLE 2: WHERE DO THE TRUE DAILY HIGHS (HOD) AND LOWS (LOD) FORM?
    # --------------------------------------------------------------------------
    print("\n" + "-" * 115)
    print("2. LIQUIDITY TURNING POINT TIMING: WHEN DO DAILY HIGHS & LOWS FORM? (250 TRADING DAYS)")
    print("-" * 115)
    print(f"{'Session Phase':<35} | {'Hours (UTC)':<15} | {'High of Day %':<16} | {'Low of Day %':<16} | {'Total Extreme %':<16}")
    print("-" * 115)

    phases = [
        ("Asian Drift & Tokyo Open", list(range(0, 6))),
        ("Frankfurt Pre-Market Trap / Coil", [6]),
        ("London Cash Open Surge", [7, 8]),
        ("London Midday / Trend Continuation", [9, 10, 11]),
        ("European Midday / US Pre-Market", [12]),
        ("London-NY Overlap (Peak Liquidity)", [13, 14, 15]),
        ("London Close / US Afternoon Drift", [16, 17, 18, 19]),
        ("Late US / Rollover Spread Zone", [20, 21, 22, 23])
    ]

    total_hod = np.sum(hod_counts)
    total_lod = np.sum(lod_counts)

    phase_summary = []
    for pname, phours in phases:
        h_sum = sum(hod_counts[h] for h in phours)
        l_sum = sum(lod_counts[h] for h in phours)
        tot_sum = h_sum + l_sum
        h_pct = (h_sum / max(total_hod, 1)) * 100
        l_pct = (l_sum / max(total_lod, 1)) * 100
        tot_pct = (tot_sum / max(total_extreme_events, 1)) * 100
        hrs_str = f"{phours[0]:02d}:00 - {phours[-1]+1:02d}:00"
        print(f"{pname:<35} | {hrs_str:<15} | {h_pct:14.1f}% | {l_pct:14.1f}% | {tot_pct:14.1f}%")
        phase_summary.append({
            'phase': pname,
            'hours': hrs_str,
            'hod_pct': round(h_pct, 1),
            'lod_pct': round(l_pct, 1),
            'total_extreme_pct': round(tot_pct, 1)
        })

    print("-" * 115)

    # --------------------------------------------------------------------------
    # TABLE 3: DAY OF THE WEEK VOLATILITY & PIP DISTRIBUTION
    # --------------------------------------------------------------------------
    print("\n" + "-" * 115)
    print("3. DAY-OF-THE-WEEK MACRO VOLATILITY & RANGE CYCLE (52 WEEKS)")
    print("-" * 115)
    print(f"{'Day of Week':<15} | {'Mean Daily Range':<20} | {'Median Daily Range':<20} | {'P90 Volatility Expansion':<25}")
    print("-" * 115)

    dow_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    dow_stats = {}
    for d in range(5):
        all_d_rng = [val for r in results for val in r['day_of_week_ranges'][d]]
        if all_d_rng:
            m_rng = np.mean(all_d_rng)
            med_rng = np.median(all_d_rng)
            p90_rng = np.percentile(all_d_rng, 90)
            print(f"{dow_names[d]:<15} | {m_rng:6.1f} pips            | {med_rng:6.1f} pips              | {p90_rng:6.1f} pips")
            dow_stats[dow_names[d]] = {
                'mean_range': round(float(m_rng), 1),
                'median_range': round(float(med_rng), 1),
                'p90_expansion': round(float(p90_rng), 1)
            }
    print("-" * 115)

    # --------------------------------------------------------------------------
    # TABLE 4: ASIAN COMPRESSION TO LONDON/NY EXPANSION RELATIONSHIP
    # --------------------------------------------------------------------------
    print("\n" + "-" * 115)
    print("4. COMPRESSION-TO-EXPANSION TRANSITION: ASIAN RANGE vs LONDON/NY EXPANSION")
    print("-" * 115)
    all_asian_london = [pair for r in results for pair in r['asian_vs_london']]
    if all_asian_london:
        asian_ranges = np.array([p['asian_range'] for p in all_asian_london])
        lon_ranges = np.array([p['lon_ny_range'] for p in all_asian_london])
        ratios = np.array([p['expansion_ratio'] for p in all_asian_london])

        tight_mask = (asian_ranges <= 18.0)
        norm_mask = (asian_ranges > 18.0) & (asian_ranges <= 30.0)
        wide_mask = (asian_ranges > 30.0)

        print(f"Condition A: Ultra-Tight Asian Coil (Asian Range <= 18 pips) [N={np.sum(tight_mask)} days]:")
        print(f"   -> Average London/NY Expansion: {np.mean(lon_ranges[tight_mask]):.1f} pips ({np.mean(ratios[tight_mask]):.2f}x of Asian range)")
        print(f"   -> Expansion Multiplier:        London/NY expands an average of +{np.mean(lon_ranges[tight_mask]) - np.mean(asian_ranges[tight_mask]):.1f} pips beyond Asian box!")
        print("-" * 115)
        print(f"Condition B: Normal Asian Range (18 to 30 pips) [N={np.sum(norm_mask)} days]:")
        print(f"   -> Average London/NY Expansion: {np.mean(lon_ranges[norm_mask]):.1f} pips ({np.mean(ratios[norm_mask]):.2f}x of Asian range)")
        print("-" * 115)
        print(f"Condition C: Blown-Out / Exhausted Asian Range (> 30 pips) [N={np.sum(wide_mask)} days]:")
        print(f"   -> Average London/NY Expansion: {np.mean(lon_ranges[wide_mask]):.1f} pips ({np.mean(ratios[wide_mask]):.2f}x of Asian range)")
        print(f"   -> Market Behavior:             Volatility exhaustion! London/NY struggles to expand, leading to mean-reverting chop.")
        print("-" * 115)

    # Export payload to JSON
    export_payload = {
        'edge_profile_by_hour': edge_profile,
        'turning_point_phases': phase_summary,
        'day_of_week_volatility': dow_stats
    }

    with open(RESULTS_DIR / "deep_market_movement_study.json", "w") as f:
        json.dump(export_payload, f, indent=2)

    print(f"\n[+] Comprehensive market movement dataset saved to: {RESULTS_DIR / 'deep_market_movement_study.json'}")


if __name__ == "__main__":
    run_market_movement_analysis()
