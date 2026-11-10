#!/usr/bin/env python3
"""
================================================================================
  DATA ENGINEERING SUITE: M5 LIQUIDITY SWEEP & FAIR VALUE GAP (FVG) STUDY
================================================================================
Investigates two institutional lower-timeframe (M5) micro-structures:

  1. M5 Fair Value Gap (FVG) Imbalance Mitigation:
     - Bullish FVG: Bar 1 High < Bar 3 Low (unfilled liquidity gap between 1 and 3).
       Setup: Macro trend is bullish, Bar 4-8 pulls back into the gap to mitigate,
       and prints a bullish rejection candle.
     - Bearish FVG: Bar 1 Low > Bar 3 High.
  
  2. M5 Asian Extreme Sweep & Market Structure Shift (MSS):
     - M5 price probes beyond Asian High/Low during 07:15 - 09:00 UTC London Open.
     - Within 3 bars, prints a sharp displacement candle closing back inside the range,
       breaking the most recent opposing M5 swing point (Market Structure Shift).
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


def evaluate_week_fvg_and_sweeps(w: int) -> List[Dict[str, Any]]:
    ticks, _ = get_week_dataset(w)
    if len(ticks) < 1000:
        return []

    m5 = build_time_bars(ticks, timeframe_min=5)
    if len(m5) < 100:
        return []

    c = m5['close'].values
    h = m5['high'].values
    l = m5['low'].values
    o = m5['open'].values
    spread = m5['spread_pips'].values
    hours = m5.index.hour.values
    minutes = m5.index.minute.values
    dates = m5.index.date
    days = m5.index.dayofweek.values
    n = len(m5)

    # Macro Trend (H1 50-EMA)
    ts = pd.to_datetime(ticks['timestamp'], unit='ms')
    mid = 0.5 * (ticks['bid'] + ticks['ask'])
    base = pd.DataFrame({'close': mid}, index=ts)
    h1 = base.resample('1h').last().dropna()
    h1['ema50'] = h1['close'].ewm(span=50, adjust=False).mean()
    h1['macro'] = np.where(h1['close'] > h1['ema50'], 1, -1)
    macro = h1['macro'].reindex(m5.index, method='ffill').fillna(0).values

    events = []

    # --- 1. M5 FAIR VALUE GAP (FVG) MITIGATION IN TREND ---
    # Active operational hours (07:30 - 16:30 UTC)
    for i in range(10, n - 24):
        hr = hours[i]
        if not ((7 <= hr <= 11) or (13 <= hr <= 16)):
            continue

        if spread[i] > 0.60:
            continue

        m_dir = macro[i]

        # Check for 3-bar Bullish FVG created at [i-2, i-1, i]:
        # Bar i-2 High < Bar i Low with gap >= 1.5 pips
        gap_up = (l[i] - h[i-2]) / PIP_SIZE
        if gap_up >= 1.5 and m_dir > 0 and c[i-1] > o[i-1]:
            fvg_top = l[i]
            fvg_bot = h[i-2]
            # Track forward 8 bars (40 mins) to see if price retests gap and bounces
            for k in range(1, 9):
                next_k = i + k
                if next_k >= n:
                    break
                # Retest touch
                if l[next_k] <= fvg_top and c[next_k] >= fvg_bot and c[next_k] > o[next_k]:
                    events.append({
                        'setup': 'M5_FVG_IMBALANCE_MITIGATION',
                        'idx': next_k,
                        'direction': 1,
                        'entry_price': c[next_k],
                        'risk_pips': max(min((c[next_k] - fvg_bot) / PIP_SIZE + 1.2, 10.0), 5.0)
                    })
                    break

        # Check for 3-bar Bearish FVG:
        # Bar i-2 Low > Bar i High with gap >= 1.5 pips
        gap_dn = (l[i-2] - h[i]) / PIP_SIZE
        if gap_dn >= 1.5 and m_dir < 0 and c[i-1] < o[i-1]:
            fvg_top = l[i-2]
            fvg_bot = h[i]
            for k in range(1, 9):
                next_k = i + k
                if next_k >= n:
                    break
                if h[next_k] >= fvg_bot and c[next_k] <= fvg_top and c[next_k] < o[next_k]:
                    events.append({
                        'setup': 'M5_FVG_IMBALANCE_MITIGATION',
                        'idx': next_k,
                        'direction': -1,
                        'entry_price': c[next_k],
                        'risk_pips': max(min((fvg_top - c[next_k]) / PIP_SIZE + 1.2, 10.0), 5.0)
                    })
                    break

    # --- 2. M5 ASIAN SWEEP & MARKET STRUCTURE SHIFT (MSS) AT LONDON OPEN ---
    unique_dates = np.unique(dates)
    for d_val in unique_dates:
        mask = (dates == d_val)
        day_indices = np.where(mask)[0]
        if len(day_indices) < 50:
            continue

        asian_indices = [i for i in day_indices if 0 <= hours[i] < 7]
        if len(asian_indices) < 30:
            continue

        asian_high = np.max(h[asian_indices])
        asian_low = np.min(l[asian_indices])
        asian_rng = (asian_high - asian_low) / PIP_SIZE

        # Monitor 07:15 to 09:15 UTC for sweep & shift
        lon_indices = [i for i in day_indices if (hours[i] == 7 and minutes[i] >= 15) or (hours[i] == 8) or (hours[i] == 9 and minutes[i] <= 15)]

        swept_h = False
        swept_l = False
        sweep_h_bar = -1
        sweep_l_bar = -1

        for i in lon_indices:
            if h[i] > asian_high and not swept_h:
                swept_h = True
                sweep_h_bar = i
            if l[i] < asian_low and not swept_l:
                swept_l = True
                sweep_l_bar = i

        # Look for Bearish Market Structure Shift after sweeping Asian High
        if swept_h and sweep_h_bar != -1:
            # Look at subsequent 4 bars for displacement closing below recent M5 swing low
            for k in range(1, 6):
                chk_idx = sweep_h_bar + k
                if chk_idx >= n or hours[chk_idx] > 9:
                    break
                recent_low = np.min(l[max(0, chk_idx-3):chk_idx])
                if c[chk_idx] < recent_low and c[chk_idx] < asian_high and (h[chk_idx] - l[chk_idx]) >= 4.0 * PIP_SIZE:
                    events.append({
                        'setup': 'M5_ASIAN_SWEEP_MSS_REVERSAL',
                        'idx': chk_idx,
                        'direction': -1,
                        'entry_price': c[chk_idx],
                        'risk_pips': max(min((np.max(h[sweep_h_bar:chk_idx+1]) - c[chk_idx]) / PIP_SIZE + 1.2, 12.0), 5.5)
                    })
                    break

        # Look for Bullish Market Structure Shift after sweeping Asian Low
        if swept_l and sweep_l_bar != -1:
            for k in range(1, 6):
                chk_idx = sweep_l_bar + k
                if chk_idx >= n or hours[chk_idx] > 9:
                    break
                recent_high = np.max(h[max(0, chk_idx-3):chk_idx])
                if c[chk_idx] > recent_high and c[chk_idx] > asian_low and (h[chk_idx] - l[chk_idx]) >= 4.0 * PIP_SIZE:
                    events.append({
                        'setup': 'M5_ASIAN_SWEEP_MSS_REVERSAL',
                        'idx': chk_idx,
                        'direction': 1,
                        'entry_price': c[chk_idx],
                        'risk_pips': max(min((c[chk_idx] - np.min(l[sweep_l_bar:chk_idx+1])) / PIP_SIZE + 1.2, 12.0), 5.5)
                    })
                    break

    # Calculate 1-hour Forward Excursions (12 M5 bars)
    for e in events:
        idx = e['idx']
        d = e['direction']
        ep = e['entry_price']
        risk = e['risk_pips']
        end_idx = min(idx + 1 + 12, n)

        sub_h = h[idx + 1 : end_idx]
        sub_l = l[idx + 1 : end_idx]
        if len(sub_h) == 0:
            e['mfe'] = 0.0
            e['mae'] = 0.0
            e['hit_target'] = 0
            continue

        target_pips = risk * 1.70
        hit_tp = 0

        if d == 1:
            fav = (np.max(sub_h) - ep) / PIP_SIZE
            adv = (ep - np.min(sub_l)) / PIP_SIZE
            e['mfe'] = max(0.0, fav)
            e['mae'] = max(0.0, adv)

            for k in range(len(sub_h)):
                p_up = (sub_h[k] - ep) / PIP_SIZE
                p_dn = (ep - sub_l[k]) / PIP_SIZE
                if p_up >= target_pips and p_dn < risk:
                    hit_tp = 1
                    break
                elif p_dn >= risk:
                    break
            e['hit_target'] = hit_tp

        else:
            fav = (ep - np.min(sub_l)) / PIP_SIZE
            adv = (np.max(sub_h) - ep) / PIP_SIZE
            e['mfe'] = max(0.0, fav)
            e['mae'] = max(0.0, adv)

            for k in range(len(sub_h)):
                p_dn = (ep - sub_l[k]) / PIP_SIZE
                p_up = (sub_h[k] - ep) / PIP_SIZE
                if p_dn >= target_pips and p_up < risk:
                    hit_tp = 1
                    break
                elif p_up >= risk:
                    break
            e['hit_target'] = hit_tp

    return events


def run():
    print("=" * 115)
    print("     DATA ENGINEERING SUITE: M5 LIQUIDITY SWEEP & FVG MITIGATION BENCHMARK (52 WEEKS)     ".center(115))
    print("=" * 115)

    weeks = list(range(1, 53))
    t0 = time.time()
    num_workers = min(8, os.cpu_count() or 4)

    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        results = list(ex.map(evaluate_week_fvg_and_sweeps, weeks))

    dt = time.time() - t0
    all_events = [e for r in results for e in r]
    print(f"[+] 52 weeks (~70,500 M5 bars) audited in {dt:.1f}s!\n")

    if not all_events:
        print("[!] No events found.")
        return

    df = pd.DataFrame(all_events)

    print("-" * 115)
    print(f"{'M5 Microstructure Setup':<42} | {'Signals':<8} | {'Avg MFE':<9} | {'Avg MAE':<9} | {'Edge Ratio':<12} | {'1.7R Win Rate':<12}")
    print("-" * 115)

    for setup_key in sorted(df['setup'].unique()):
        sub_df = df[df['setup'] == setup_key]
        n_ev = len(sub_df)
        if n_ev > 0:
            avg_mfe = sub_df['mfe'].mean()
            avg_mae = sub_df['mae'].mean()
            ratio = avg_mfe / max(avg_mae, 1e-6)
            wr = (sub_df['hit_target'].sum() / n_ev) * 100
            print(f"{setup_key:<42} | {n_ev:<8} | {avg_mfe:6.1f} p  | {avg_mae:6.1f} p  | {ratio:6.2f} : 1   | {wr:6.1f}%")

    print("-" * 115)


if __name__ == "__main__":
    run()
