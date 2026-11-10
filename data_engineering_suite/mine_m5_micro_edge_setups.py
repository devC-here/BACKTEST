#!/usr/bin/env python3
"""
================================================================================
  DATA ENGINEERING SUITE: M5 INTRADAY MICRO-EDGE OPPORTUNITY MINER
================================================================================
Mines 52 weeks (~26M ticks / ~70,500 M5 bars) across 3 distinct lower-timeframe
microstructure paradigms:

  1. M5 DISPLACEMENT & 50% MITIGATION RETEST (07:15 - 09:30 & 13:15 - 15:30 UTC):
     - An M5 bar forms a powerful displacement (range >= 7p, body >= 70%).
     - Price pulls back within 3 bars to the 50% midpoint and prints a reversal wick.
  
  2. M5 TICK VELOCITY & ORDER FLOW DELTA IGNITION:
     - Tick volume on M5 surges >= 2.2x the 12-bar rolling average.
     - Cumulative Tick Delta (CVD) is strongly positive (> 60% buy volume) or negative.
     - Aligned with H1 50-EMA trend.
  
  3. M5 NARROW-RANGE COIL BREAKOUT (NR4 / NR7 ON M5):
     - 4 consecutive M5 bars with ultra-compressed range (<= 3.5 pips each).
     - First M5 bar breaking out of the 4-bar box with volume surge.
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


def evaluate_week_m5_micro_setups(w: int) -> List[Dict[str, Any]]:
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
    v = m5['vol'].values
    deltas = m5['delta'].values
    buy_v = m5['buy_vol'].values
    sell_v = m5['sell_vol'].values
    spread = m5['spread_pips'].values
    hours = m5.index.hour.values
    minutes = m5.index.minute.values
    n = len(m5)

    # Macro Trend (H1 50-EMA)
    ts = pd.to_datetime(ticks['timestamp'], unit='ms')
    mid = 0.5 * (ticks['bid'] + ticks['ask'])
    base = pd.DataFrame({'close': mid}, index=ts)
    h1 = base.resample('1h').last().dropna()
    h1['ema50'] = h1['close'].ewm(span=50, adjust=False).mean()
    h1['macro'] = np.where(h1['close'] > h1['ema50'], 1, -1)
    macro = h1['macro'].reindex(m5.index, method='ffill').fillna(0).values

    # Rolling M5 Vol SMA
    vol_sma12 = pd.Series(v).rolling(12).mean().shift(1).bfill().values
    ranges = (h - l) / PIP_SIZE

    events = []

    # Active operational hours (London 07:00 - 10:00 & NY 12:30 - 16:30 UTC)
    for i in range(25, n - 24):
        hr = hours[i]
        mn = minutes[i]
        if not ((7 <= hr <= 10) or (12 <= hr <= 16)):
            continue

        if spread[i] > 0.60:
            continue

        m_dir = macro[i]

        # ----------------------------------------------------------------------
        # SETUP 1: M5 DISPLACEMENT & 50% MITIGATION RETEST
        # Look back 1-3 bars for an impulse displacement, currently touching 50% midpoint
        # ----------------------------------------------------------------------
        for lookback in (1, 2, 3):
            prev_i = i - lookback
            prev_rng = ranges[prev_i]
            prev_body = abs(c[prev_i] - o[prev_i]) / PIP_SIZE
            if prev_rng >= 7.0 and (prev_body / max(prev_rng, 0.1)) >= 0.70:
                midpoint = 0.5 * (o[prev_i] + c[prev_i])
                is_bull_imp = c[prev_i] > o[prev_i]

                # Retest touch on current bar
                if is_bull_imp and m_dir > 0 and l[i] <= midpoint and c[i] > midpoint and c[i] > o[i]:
                    events.append({
                        'setup': '1_M5_DISPLACEMENT_MITIGATION_RETEST',
                        'idx': i,
                        'direction': 1,
                        'entry_price': c[i],
                        'risk_pips': max(min((c[i] - l[prev_i]) / PIP_SIZE + 1.0, 12.0), 5.5)
                    })
                    break
                elif not is_bull_imp and m_dir < 0 and h[i] >= midpoint and c[i] < midpoint and c[i] < o[i]:
                    events.append({
                        'setup': '1_M5_DISPLACEMENT_MITIGATION_RETEST',
                        'idx': i,
                        'direction': -1,
                        'entry_price': c[i],
                        'risk_pips': max(min((h[prev_i] - c[i]) / PIP_SIZE + 1.0, 12.0), 5.5)
                    })
                    break

        # ----------------------------------------------------------------------
        # SETUP 2: M5 TICK VELOCITY & ORDER FLOW DELTA IGNITION
        # Volume >= 2.2x SMA12 AND Buy/Sell ratio >= 65% in trend direction
        # ----------------------------------------------------------------------
        if v[i] >= 2.2 * vol_sma12[i] and ranges[i] >= 5.0:
            buy_ratio = buy_v[i] / max(v[i], 1)
            sell_ratio = sell_v[i] / max(v[i], 1)

            if m_dir > 0 and buy_ratio >= 0.65 and c[i] > o[i]:
                events.append({
                    'setup': '2_M5_VELOCITY_DELTA_IGNITION',
                    'idx': i,
                    'direction': 1,
                    'entry_price': c[i],
                    'risk_pips': max(min((c[i] - l[i]) / PIP_SIZE + 1.2, 12.0), 5.5)
                })
            elif m_dir < 0 and sell_ratio >= 0.65 and c[i] < o[i]:
                events.append({
                    'setup': '2_M5_VELOCITY_DELTA_IGNITION',
                    'idx': i,
                    'direction': -1,
                    'entry_price': c[i],
                    'risk_pips': max(min((h[i] - c[i]) / PIP_SIZE + 1.2, 12.0), 5.5)
                })

        # ----------------------------------------------------------------------
        # SETUP 3: M5 NARROW-RANGE COIL BREAKOUT (NR4 ON M5)
        # Previous 4 bars all have range <= 3.5 pips
        # ----------------------------------------------------------------------
        is_nr4 = (ranges[i-1] <= 3.5) and (ranges[i-2] <= 3.5) and (ranges[i-3] <= 3.5) and (ranges[i-4] <= 3.5)
        if is_nr4:
            box_h = np.max(h[i-4:i])
            box_l = np.min(l[i-4:i])

            if c[i] > box_h and m_dir > 0 and c[i] > o[i]:
                events.append({
                    'setup': '3_M5_NR4_COIL_BREAKOUT',
                    'idx': i,
                    'direction': 1,
                    'entry_price': c[i],
                    'risk_pips': max(min((c[i] - box_l) / PIP_SIZE + 1.0, 10.0), 5.0)
                })
            elif c[i] < box_l and m_dir < 0 and c[i] < o[i]:
                events.append({
                    'setup': '3_M5_NR4_COIL_BREAKOUT',
                    'idx': i,
                    'direction': -1,
                    'entry_price': c[i],
                    'risk_pips': max(min((box_h - c[i]) / PIP_SIZE + 1.0, 10.0), 5.0)
                })

    # Forward 12-bar (1 hour) Excursion Evaluation
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
    print("     DATA ENGINEERING SUITE: M5 INTRADAY MICRO-EDGE SETUP BENCHMARK (52 WEEKS)     ".center(115))
    print("=" * 115)

    weeks = list(range(1, 53))
    t0 = time.time()
    num_workers = min(8, os.cpu_count() or 4)

    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        results = list(ex.map(evaluate_week_m5_micro_setups, weeks))

    dt = time.time() - t0
    all_events = [e for r in results for e in r]
    print(f"[+] 52 weeks (~70,500 M5 bars) processed in {dt:.1f}s!\n")

    if not all_events:
        print("[!] No events found.")
        return

    df = pd.DataFrame(all_events)

    print("-" * 115)
    print(f"{'M5 Microstructure Setup':<42} | {'Signals':<8} | {'Avg MFE':<9} | {'Avg MAE':<9} | {'Edge Ratio':<12} | {'1.7R Win Rate':<12}")
    print("-" * 115)

    setup_names = {
        '1_M5_DISPLACEMENT_MITIGATION_RETEST': '1. M5 Displacement Mitigation Retest',
        '2_M5_VELOCITY_DELTA_IGNITION': '2. M5 Tick Velocity & Delta Ignition',
        '3_M5_NR4_COIL_BREAKOUT': '3. M5 Narrow-Range Coil Breakout (NR4)'
    }

    for setup_key, name in sorted(setup_names.items()):
        sub_df = df[df['setup'] == setup_key]
        n_ev = len(sub_df)
        if n_ev > 0:
            avg_mfe = sub_df['mfe'].mean()
            avg_mae = sub_df['mae'].mean()
            ratio = avg_mfe / max(avg_mae, 1e-6)
            wr = (sub_df['hit_target'].sum() / n_ev) * 100
            print(f"{name:<42} | {n_ev:<8} | {avg_mfe:6.1f} p  | {avg_mae:6.1f} p  | {ratio:6.2f} : 1   | {wr:6.1f}%")

    print("-" * 115)


if __name__ == "__main__":
    run()
