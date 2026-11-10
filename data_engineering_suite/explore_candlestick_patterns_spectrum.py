#!/usr/bin/env python3
"""
================================================================================
  DATA ENGINEERING SUITE: CANDLESTICK PATTERN STUDY ACROSS TIMEFRAMES (M5 - H4)
================================================================================
Investigates the statistical validity of classical & institutional candlestick
patterns across all 6 timeframes:
  - M5  (5 min)
  - M15 (15 min)
  - M30 (30 min)
  - H1  (60 min)
  - H2  (120 min)
  - H4  (240 min)

Patterns Tested:
  1. Pin Bar / Hammer & Shooting Star (Rejection wicks >= 60% of candle range)
  2. Bullish & Bearish Engulfing Bars (Outside reversal bars)
  3. Inside Bar Mother-Bar Breakout (Volatility compression expansion)
  4. Morning Star & Evening Star (3-candle institutional reversal sequences)
  5. Three Consecutive Momentum Bars (Three Soldiers / Three Crows)

Key Questions Answered:
  - Do candlestick patterns work standalone (naive retail)?
  - How does win rate and asymmetry scale as timeframe rises from M5 to H4?
  - What is the predictive power when conditioned on Macro Trend (50-EMA)?
================================================================================
"""

from __future__ import annotations
import sys
import os
import time
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple

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

TIMEFRAMES = {
    'H4': 240,
    'H2': 120,
    'H1': 60,
    'M30': 30,
    'M15': 15,
    'M5': 5
}


def load_all_timeframe_bars(num_weeks: int = 52) -> Dict[str, pd.DataFrame]:
    print(f"[*] Aggregating timeframe spectrum across {num_weeks} weeks...")
    t0 = time.time()
    weekly_bars: Dict[str, List[pd.DataFrame]] = {tf: [] for tf in TIMEFRAMES}

    for w in range(1, num_weeks + 1):
        try:
            ticks, _ = get_week_dataset(w)
            if len(ticks) < 1000:
                continue
            for tf, mins in TIMEFRAMES.items():
                b = build_time_bars(ticks, timeframe_min=mins)
                weekly_bars[tf].append(b)
        except Exception as e:
            continue

    dfs = {}
    for tf, blist in weekly_bars.items():
        dfs[tf] = pd.concat(blist).sort_index()

    dt = time.time() - t0
    print(f"[+] Loaded {num_weeks} weeks in {dt:.1f}s across H4, H2, H1, M30, M15, M5.\n")
    return dfs


def audit_patterns_for_timeframe(df: pd.DataFrame, tf_name: str) -> Dict[str, Any]:
    c = df['close'].values
    h = df['high'].values
    l = df['low'].values
    o = df['open'].values
    n = len(df)

    # 50-EMA for macro trend
    ema50 = pd.Series(c).ewm(span=50, adjust=False).mean().values
    macro = np.where(c > ema50, 1, -1)

    rng = (h - l) / PIP_SIZE
    body = np.abs(c - o) / PIP_SIZE
    upper_wick = (h - np.maximum(o, c)) / PIP_SIZE
    lower_wick = (np.minimum(o, c) - l) / PIP_SIZE

    # Determine forward horizon based on timeframe
    # M5: 12 bars (1 hr), M15: 12 bars (3 hrs), M30: 12 bars (6 hrs), H1: 12 bars (12 hrs), H2: 12 bars (24 hrs), H4: 12 bars (48 hrs)
    fwd_len = 12

    pattern_stats = {
        'Pin_Bar': {'total': 0, 'wins_naive': 0, 'wins_trend': 0, 'total_trend': 0, 'mfes': [], 'maes': []},
        'Engulfing': {'total': 0, 'wins_naive': 0, 'wins_trend': 0, 'total_trend': 0, 'mfes': [], 'maes': []},
        'Inside_Bar': {'total': 0, 'wins_naive': 0, 'wins_trend': 0, 'total_trend': 0, 'mfes': [], 'maes': []},
        'Morning_Evening_Star': {'total': 0, 'wins_naive': 0, 'wins_trend': 0, 'total_trend': 0, 'mfes': [], 'maes': []},
        'Three_Momentum_Bars': {'total': 0, 'wins_naive': 0, 'wins_trend': 0, 'total_trend': 0, 'mfes': [], 'maes': []},
    }

    # Minimum pattern range to filter sub-pip noise
    min_bar_rng = 3.0 if tf_name in ('M5', 'M15') else (5.0 if tf_name == 'M30' else 8.0)

    for i in range(15, n - fwd_len):
        if rng[i] < min_bar_rng:
            continue

        m_dir = macro[i]

        # ----------------------------------------------------------------------
        # 1. PIN BAR / HAMMER & SHOOTING STAR
        # ----------------------------------------------------------------------
        is_bull_pin = (lower_wick[i] >= 0.58 * rng[i]) and (body[i] <= 0.32 * rng[i]) and (c[i] > o[i])
        is_bear_pin = (upper_wick[i] >= 0.58 * rng[i]) and (body[i] <= 0.32 * rng[i]) and (c[i] < o[i])

        if is_bull_pin or is_bear_pin:
            dir_sign = 1 if is_bull_pin else -1
            entry = c[i]
            sl_dist = (c[i] - l[i]) / PIP_SIZE if dir_sign == 1 else (h[i] - c[i]) / PIP_SIZE
            sl_dist = max(sl_dist, 3.0)
            tp_dist = 1.50 * sl_dist

            sub_h = h[i+1 : i+1+fwd_len]
            sub_l = l[i+1 : i+1+fwd_len]

            mfe = (np.max(sub_h) - entry) / PIP_SIZE if dir_sign == 1 else (entry - np.min(sub_l)) / PIP_SIZE
            mae = (entry - np.min(sub_l)) / PIP_SIZE if dir_sign == 1 else (np.max(sub_h) - entry) / PIP_SIZE

            # Check 1.5R outcome
            win = False
            loss = False
            for step in range(i + 1, i + 1 + fwd_len):
                if dir_sign == 1:
                    if (entry - l[step]) / PIP_SIZE >= sl_dist:
                        loss = True
                        break
                    if (h[step] - entry) / PIP_SIZE >= tp_dist:
                        win = True
                        break
                else:
                    if (h[step] - entry) / PIP_SIZE >= sl_dist:
                        loss = True
                        break
                    if (entry - l[step]) / PIP_SIZE >= tp_dist:
                        win = True
                        break

            pattern_stats['Pin_Bar']['total'] += 1
            pattern_stats['Pin_Bar']['mfes'].append(mfe)
            pattern_stats['Pin_Bar']['maes'].append(mae)
            if win and not loss:
                pattern_stats['Pin_Bar']['wins_naive'] += 1

            # Conditioned on Trend:
            if (dir_sign == 1 and m_dir > 0) or (dir_sign == -1 and m_dir < 0):
                pattern_stats['Pin_Bar']['total_trend'] += 1
                if win and not loss:
                    pattern_stats['Pin_Bar']['wins_trend'] += 1

        # ----------------------------------------------------------------------
        # 2. ENGULFING BAR (OUTSIDE REVERSAL)
        # ----------------------------------------------------------------------
        is_bull_engulf = (c[i-1] < o[i-1]) and (c[i] > o[i]) and (o[i] <= c[i-1]) and (c[i] >= o[i-1]) and (body[i] >= 1.15 * body[i-1])
        is_bear_engulf = (c[i-1] > o[i-1]) and (c[i] < o[i]) and (o[i] >= c[i-1]) and (c[i] <= o[i-1]) and (body[i] >= 1.15 * body[i-1])

        if is_bull_engulf or is_bear_engulf:
            dir_sign = 1 if is_bull_engulf else -1
            entry = c[i]
            sl_dist = max((c[i] - l[i]) / PIP_SIZE if dir_sign == 1 else (h[i] - c[i]) / PIP_SIZE, 3.0)
            tp_dist = 1.50 * sl_dist

            sub_h = h[i+1 : i+1+fwd_len]
            sub_l = l[i+1 : i+1+fwd_len]
            mfe = (np.max(sub_h) - entry) / PIP_SIZE if dir_sign == 1 else (entry - np.min(sub_l)) / PIP_SIZE
            mae = (entry - np.min(sub_l)) / PIP_SIZE if dir_sign == 1 else (np.max(sub_h) - entry) / PIP_SIZE

            win = False
            loss = False
            for step in range(i + 1, i + 1 + fwd_len):
                if dir_sign == 1:
                    if (entry - l[step]) / PIP_SIZE >= sl_dist:
                        loss = True
                        break
                    if (h[step] - entry) / PIP_SIZE >= tp_dist:
                        win = True
                        break
                else:
                    if (h[step] - entry) / PIP_SIZE >= sl_dist:
                        loss = True
                        break
                    if (entry - l[step]) / PIP_SIZE >= tp_dist:
                        win = True
                        break

            pattern_stats['Engulfing']['total'] += 1
            pattern_stats['Engulfing']['mfes'].append(mfe)
            pattern_stats['Engulfing']['maes'].append(mae)
            if win and not loss:
                pattern_stats['Engulfing']['wins_naive'] += 1

            if (dir_sign == 1 and m_dir > 0) or (dir_sign == -1 and m_dir < 0):
                pattern_stats['Engulfing']['total_trend'] += 1
                if win and not loss:
                    pattern_stats['Engulfing']['wins_trend'] += 1

        # ----------------------------------------------------------------------
        # 3. INSIDE BAR BREAKOUT
        # ----------------------------------------------------------------------
        is_inside = (h[i] < h[i-1]) and (l[i] > l[i-1])
        if is_inside and (i + 1 + fwd_len < n):
            # Check bar i+1 for breakout of mother bar (i-1)
            dir_sign = 0
            if c[i+1] > h[i-1]:
                dir_sign = 1
            elif c[i+1] < l[i-1]:
                dir_sign = -1

            if dir_sign != 0:
                entry = c[i+1]
                sl_dist = max((c[i+1] - l[i]) / PIP_SIZE if dir_sign == 1 else (h[i] - c[i+1]) / PIP_SIZE, 3.0)
                tp_dist = 1.50 * sl_dist

                sub_h = h[i+2 : i+2+fwd_len]
                sub_l = l[i+2 : i+2+fwd_len]
                mfe = (np.max(sub_h) - entry) / PIP_SIZE if dir_sign == 1 else (entry - np.min(sub_l)) / PIP_SIZE
                mae = (entry - np.min(sub_l)) / PIP_SIZE if dir_sign == 1 else (np.max(sub_h) - entry) / PIP_SIZE

                win = False
                loss = False
                for step in range(i + 2, min(i + 2 + fwd_len, n)):
                    if dir_sign == 1:
                        if (entry - l[step]) / PIP_SIZE >= sl_dist:
                            loss = True
                            break
                        if (h[step] - entry) / PIP_SIZE >= tp_dist:
                            win = True
                            break
                    else:
                        if (h[step] - entry) / PIP_SIZE >= sl_dist:
                            loss = True
                            break
                        if (entry - l[step]) / PIP_SIZE >= tp_dist:
                            win = True
                            break

                pattern_stats['Inside_Bar']['total'] += 1
                pattern_stats['Inside_Bar']['mfes'].append(mfe)
                pattern_stats['Inside_Bar']['maes'].append(mae)
                if win and not loss:
                    pattern_stats['Inside_Bar']['wins_naive'] += 1

                if (dir_sign == 1 and m_dir > 0) or (dir_sign == -1 and m_dir < 0):
                    pattern_stats['Inside_Bar']['total_trend'] += 1
                    if win and not loss:
                        pattern_stats['Inside_Bar']['wins_trend'] += 1

        # ----------------------------------------------------------------------
        # 4. MORNING STAR & EVENING STAR
        # ----------------------------------------------------------------------
        is_morning_star = (c[i-2] < o[i-2]) and (body[i-2] >= 0.50 * rng[i-2]) and \
                           (body[i-1] <= 0.35 * rng[i-1]) and \
                           (c[i] > o[i]) and (c[i] >= 0.5 * (o[i-2] + c[i-2]))

        is_evening_star = (c[i-2] > o[i-2]) and (body[i-2] >= 0.50 * rng[i-2]) and \
                           (body[i-1] <= 0.35 * rng[i-1]) and \
                           (c[i] < o[i]) and (c[i] <= 0.5 * (o[i-2] + c[i-2]))

        if is_morning_star or is_evening_star:
            dir_sign = 1 if is_morning_star else -1
            entry = c[i]
            sl_dist = max((c[i] - min(l[i-2], l[i-1], l[i])) / PIP_SIZE if dir_sign == 1 else (max(h[i-2], h[i-1], h[i]) - c[i]) / PIP_SIZE, 3.0)
            tp_dist = 1.50 * sl_dist

            sub_h = h[i+1 : i+1+fwd_len]
            sub_l = l[i+1 : i+1+fwd_len]
            mfe = (np.max(sub_h) - entry) / PIP_SIZE if dir_sign == 1 else (entry - np.min(sub_l)) / PIP_SIZE
            mae = (entry - np.min(sub_l)) / PIP_SIZE if dir_sign == 1 else (np.max(sub_h) - entry) / PIP_SIZE

            win = False
            loss = False
            for step in range(i + 1, i + 1 + fwd_len):
                if dir_sign == 1:
                    if (entry - l[step]) / PIP_SIZE >= sl_dist:
                        loss = True
                        break
                    if (h[step] - entry) / PIP_SIZE >= tp_dist:
                        win = True
                        break
                else:
                    if (h[step] - entry) / PIP_SIZE >= sl_dist:
                        loss = True
                        break
                    if (entry - l[step]) / PIP_SIZE >= tp_dist:
                        win = True
                        break

            pattern_stats['Morning_Evening_Star']['total'] += 1
            pattern_stats['Morning_Evening_Star']['mfes'].append(mfe)
            pattern_stats['Morning_Evening_Star']['maes'].append(mae)
            if win and not loss:
                pattern_stats['Morning_Evening_Star']['wins_naive'] += 1

            if (dir_sign == 1 and m_dir > 0) or (dir_sign == -1 and m_dir < 0):
                pattern_stats['Morning_Evening_Star']['total_trend'] += 1
                if win and not loss:
                    pattern_stats['Morning_Evening_Star']['wins_trend'] += 1

        # ----------------------------------------------------------------------
        # 5. THREE CONSECUTIVE MOMENTUM BARS (THREE SOLDIERS / CROWS)
        # ----------------------------------------------------------------------
        is_three_soldiers = (c[i-2] > o[i-2]) and (c[i-1] > o[i-1]) and (c[i] > o[i]) and (c[i] > c[i-1] > c[i-2]) and (h[i] > h[i-1] > h[i-2])
        is_three_crows = (c[i-2] < o[i-2]) and (c[i-1] < o[i-1]) and (c[i] < o[i]) and (c[i] < c[i-1] < c[i-2]) and (l[i] < l[i-1] < l[i-2])

        if is_three_soldiers or is_three_crows:
            dir_sign = 1 if is_three_soldiers else -1
            entry = c[i]
            sl_dist = max((c[i] - l[i-2]) / PIP_SIZE if dir_sign == 1 else (h[i-2] - c[i]) / PIP_SIZE, 3.0)
            tp_dist = 1.50 * sl_dist

            sub_h = h[i+1 : i+1+fwd_len]
            sub_l = l[i+1 : i+1+fwd_len]
            mfe = (np.max(sub_h) - entry) / PIP_SIZE if dir_sign == 1 else (entry - np.min(sub_l)) / PIP_SIZE
            mae = (entry - np.min(sub_l)) / PIP_SIZE if dir_sign == 1 else (np.max(sub_h) - entry) / PIP_SIZE

            win = False
            loss = False
            for step in range(i + 1, i + 1 + fwd_len):
                if dir_sign == 1:
                    if (entry - l[step]) / PIP_SIZE >= sl_dist:
                        loss = True
                        break
                    if (h[step] - entry) / PIP_SIZE >= tp_dist:
                        win = True
                        break
                else:
                    if (h[step] - entry) / PIP_SIZE >= sl_dist:
                        loss = True
                        break
                    if (entry - l[step]) / PIP_SIZE >= tp_dist:
                        win = True
                        break

            pattern_stats['Three_Momentum_Bars']['total'] += 1
            pattern_stats['Three_Momentum_Bars']['mfes'].append(mfe)
            pattern_stats['Three_Momentum_Bars']['maes'].append(mae)
            if win and not loss:
                pattern_stats['Three_Momentum_Bars']['wins_naive'] += 1

            if (dir_sign == 1 and m_dir > 0) or (dir_sign == -1 and m_dir < 0):
                pattern_stats['Three_Momentum_Bars']['total_trend'] += 1
                if win and not loss:
                    pattern_stats['Three_Momentum_Bars']['wins_trend'] += 1

    # Format summaries
    tf_summary = {}
    for pat, s in pattern_stats.items():
        tot = s['total']
        wr_naive = (s['wins_naive'] / max(tot, 1)) * 100.0
        tot_tr = s['total_trend']
        wr_trend = (s['wins_trend'] / max(tot_tr, 1)) * 100.0
        ev_trend = (wr_trend / 100.0 * 1.5) - ((100.0 - wr_trend) / 100.0 * 1.0)
        asym = float(np.mean(s['mfes']) / max(np.mean(s['maes']), 0.1)) if s['mfes'] else 0.0

        tf_summary[pat] = {
            'total_occurrences': tot,
            'win_rate_naive': wr_naive,
            'total_trend_aligned': tot_tr,
            'win_rate_trend': wr_trend,
            'expectancy_trend': ev_trend,
            'asymmetry': asym,
            'avg_mfe': float(np.mean(s['mfes'])) if s['mfes'] else 0.0,
            'avg_mae': float(np.mean(s['maes'])) if s['maes'] else 0.0
        }

    return tf_summary


def run():
    print("=" * 115)
    print("  DATA ENGINEERING SUITE: CANDLESTICK PATTERN STUDY ACROSS ALL TIMEFRAMES (M5 - H4)  ".center(115))
    print("=" * 115)

    dfs = load_all_timeframe_bars(num_weeks=52)

    all_results = {}
    tfs = ['H4', 'H2', 'H1', 'M30', 'M15', 'M5']
    for tf in tfs:
        all_results[tf] = audit_patterns_for_timeframe(dfs[tf], tf)

    # Print Comparative Tables for Each Pattern Across Timeframes
    patterns = [
        ("PIN BAR / HAMMER & SHOOTING STAR", "Pin_Bar"),
        ("ENGULFING BARS (OUTSIDE REVERSAL)", "Engulfing"),
        ("INSIDE BAR BREAKOUT", "Inside_Bar"),
        ("MORNING & EVENING STARS (3-BAR REVERSAL)", "Morning_Evening_Star"),
        ("THREE CONSECUTIVE MOMENTUM BARS", "Three_Momentum_Bars")
    ]

    for p_title, p_key in patterns:
        print("\n" + "-" * 115)
        print(f"PATTERN: {p_title} ACROSS TIMEFRAMES (1.50R TARGET)")
        print("-" * 115)
        print(f"{'Timeframe':<10} | {'Total Setups':<14} | {'Naive WR (%)':<14} | {'Trend WR (%)':<14} | {'Trend Expectancy':<18} | {'Asymmetry (MFE/MAE)'}")
        print("-" * 115)
        for tf in tfs:
            data = all_results[tf][p_key]
            print(f"{tf:<10} | {data['total_occurrences']:>10}     | {data['win_rate_naive']:>10.1f}%   | {data['win_rate_trend']:>10.1f}%   | +{data['expectancy_trend']:>6.2f}R           | {data['asymmetry']:>5.2f} : 1 ({data['avg_mfe']:.1f}p / {data['avg_mae']:.1f}p)")
        print("-" * 115)

    # Save to JSON
    out_file = RESULTS_DIR / "candlestick_patterns_spectrum_study.json"
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[+] Saved detailed candlestick pattern spectrum study to {out_file}")


if __name__ == '__main__':
    run()
