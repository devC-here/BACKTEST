#!/usr/bin/env python3
"""
================================================================================
  DATA ENGINEERING SUITE: ALGORITHM & INDICATOR MATCHING FOR EDGE WINDOWS
================================================================================
Systematically tests targeted algorithmic mechanisms against the 4 empirically
proven structural Edge Windows discovered from pure price movement:

  EDGE WINDOW 1: Asian Ultra-Tight Coil (<= 18p) -> London Breakout Expansion (07:30-08:45 UTC)
    - Algo A: ATR Velocity Surge + H1 50-EMA Trend Continuation
  
  EDGE WINDOW 2: The Frankfurt False Extreme Trap (06:00-07:00 UTC)
    - Algo B: London Liquidity Sweep Rejection Wick Fade (07:00-08:30 UTC)
  
  EDGE WINDOW 3: Thursday Peak Volatility Expansion (97.8p avg range)
    - Algo C: Thursday Post-Macro Donchian Ignition (13:00-15:00 UTC)
  
  EDGE WINDOW 4: European Midday Ultra-Coil (<= 12p) -> NY Cash Open Thrust (13:00 UTC)
    - Algo D: Midday VWAP / Keltner Deviation Band Ignition
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
import ta

sys.dont_write_bytecode = True

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tick_engine.core.dataloader import get_week_dataset
from data_engineering_suite.bars import build_time_bars

PIP_SIZE = 0.0001
RESULTS_DIR = ROOT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_week_edge_algos(w: int) -> List[Dict[str, Any]]:
    ticks, _ = get_week_dataset(w)
    if len(ticks) < 1000:
        return []

    m15 = build_time_bars(ticks, timeframe_min=15)
    if len(m15) < 60:
        return []

    c = m15['close'].values
    h = m15['high'].values
    l = m15['low'].values
    o = m15['open'].values
    v = m15['vol'].values
    spread = m15['spread_pips'].values
    hours = m15.index.hour.values
    minutes = m15.index.minute.values
    days = m15.index.dayofweek.values
    dates = m15.index.date
    n = len(m15)

    # Macro Trend (H1 50-EMA)
    ts = pd.to_datetime(ticks['timestamp'], unit='ms')
    mid = 0.5 * (ticks['bid'] + ticks['ask'])
    base = pd.DataFrame({'close': mid}, index=ts)
    h1 = base.resample('1h').last().dropna()
    h1['ema50'] = h1['close'].ewm(span=50, adjust=False).mean()
    h1['macro'] = np.where(h1['close'] > h1['ema50'], 1, -1)
    macro = h1['macro'].reindex(m15.index, method='ffill').fillna(0).values

    # Indicators
    # ATR (14)
    tr = np.maximum(h - l, np.maximum(abs(h - pd.Series(c).shift(1).bfill().values), abs(l - pd.Series(c).shift(1).bfill().values)))
    atr14 = pd.Series(tr).rolling(14).mean().bfill().values

    # ADX (14)
    adx_ind = ta.trend.ADXIndicator(high=m15['high'], low=m15['low'], close=m15['close'], window=14)
    adx = adx_ind.adx().values

    # Donchian 20
    donchian_h20 = pd.Series(h).rolling(20).max().shift(1).bfill().values
    donchian_l20 = pd.Series(l).rolling(20).min().shift(1).bfill().values

    events = []

    unique_dates = np.unique(dates)
    for d_val in unique_dates:
        mask = (dates == d_val)
        day_indices = np.where(mask)[0]
        if len(day_indices) < 40:
            continue

        dow = days[day_indices[0]]
        if dow >= 5:
            continue

        # ----------------------------------------------------------------------
        # ALGO A: ASIAN TIGHT COIL (<= 18p) -> LONDON BREAKOUT EXPANSION (07:30 - 09:00 UTC)
        # ----------------------------------------------------------------------
        asian_idx = [i for i in day_indices if 0 <= hours[i] < 7]
        if len(asian_idx) >= 15:
            asian_h = np.max(h[asian_idx])
            asian_l = np.min(l[asian_idx])
            asian_rng = (asian_h - asian_l) / PIP_SIZE

            if asian_rng <= 18.0:
                # Monitor 07:30 to 09:00 UTC
                lon_exp_idx = [i for i in day_indices if (hours[i] == 7 and minutes[i] >= 30) or (hours[i] == 8)]
                triggered_a = False
                for i in lon_exp_idx:
                    if triggered_a or spread[i] > 0.60:
                        continue
                    m_dir = macro[i]
                    bar_rng = (h[i] - l[i]) / PIP_SIZE

                    # Bullish Expansion: close > asian_h, aligned with macro, bar range > 1.2 * ATR
                    if c[i] > asian_h and m_dir > 0 and c[i] > o[i] and bar_rng >= 1.2 * (atr14[i] / PIP_SIZE):
                        events.append({
                            'algo': 'ALGO_A_ASIAN_COIL_LONDON_EXPANSION',
                            'idx': i,
                            'direction': 1,
                            'entry_price': c[i],
                            'risk_pips': max(min((c[i] - asian_l) / PIP_SIZE * 0.5, 14.0), 6.5)
                        })
                        triggered_a = True
                    elif c[i] < asian_l and m_dir < 0 and c[i] < o[i] and bar_rng >= 1.2 * (atr14[i] / PIP_SIZE):
                        events.append({
                            'algo': 'ALGO_A_ASIAN_COIL_LONDON_EXPANSION',
                            'idx': i,
                            'direction': -1,
                            'entry_price': c[i],
                            'risk_pips': max(min((asian_h - c[i]) / PIP_SIZE * 0.5, 14.0), 6.5)
                        })
                        triggered_a = True

        # ----------------------------------------------------------------------
        # ALGO B: FRANKFURT PRE-MARKET TRAP SWEEP & REVERSAL (07:00 - 08:30 UTC)
        # ----------------------------------------------------------------------
        ff_idx = [i for i in day_indices if hours[i] == 6]
        if len(ff_idx) == 4:
            ff_h = np.max(h[ff_idx])
            ff_l = np.min(l[ff_idx])
            ff_rng = (ff_h - ff_l) / PIP_SIZE

            lon_sweep_idx = [i for i in day_indices if hours[i] == 7 or (hours[i] == 8 and minutes[i] <= 30)]
            triggered_b = False
            for i in lon_sweep_idx:
                if triggered_b or spread[i] > 0.60:
                    continue
                bar_rng = h[i] - l[i]
                if bar_rng < 5.0 * PIP_SIZE:
                    continue

                # Bearish Reversal: Price swept above Frankfurt High by 2-15 pips, but rejected with upper wick >= 35% and closed < ff_h
                sweep_up = (h[i] - ff_h) / PIP_SIZE
                upper_wick = h[i] - max(o[i], c[i])
                if 2.0 <= sweep_up <= 15.0 and upper_wick >= 0.35 * bar_rng and c[i] < ff_h and c[i] < o[i]:
                    events.append({
                        'algo': 'ALGO_B_FRANKFURT_TRAP_SWEEP_REVERSAL',
                        'idx': i,
                        'direction': -1,
                        'entry_price': c[i],
                        'risk_pips': max(min((h[i] - c[i]) / PIP_SIZE + 1.5, 14.0), 6.0)
                    })
                    triggered_b = True

                # Bullish Reversal: Price swept below Frankfurt Low by 2-15 pips, rejected with lower wick >= 35% and closed > ff_l
                sweep_dn = (ff_l - l[i]) / PIP_SIZE
                lower_wick = min(o[i], c[i]) - l[i]
                if 2.0 <= sweep_dn <= 15.0 and lower_wick >= 0.35 * bar_rng and c[i] > ff_l and c[i] > o[i]:
                    events.append({
                        'algo': 'ALGO_B_FRANKFURT_TRAP_SWEEP_REVERSAL',
                        'idx': i,
                        'direction': 1,
                        'entry_price': c[i],
                        'risk_pips': max(min((c[i] - l[i]) / PIP_SIZE + 1.5, 14.0), 6.0)
                    })
                    triggered_b = True

        # ----------------------------------------------------------------------
        # ALGO C: THURSDAY PEAK VOLATILITY MACRO BREAKOUT (13:00 - 15:30 UTC)
        # ----------------------------------------------------------------------
        if dow == 3:  # Thursday only
            thu_ny_idx = [i for i in day_indices if 13 <= hours[i] <= 15]
            triggered_c = False
            for i in thu_ny_idx:
                if triggered_c or spread[i] > 0.60:
                    continue
                m_dir = macro[i]

                # Donchian 20 breakout + ADX >= 22
                if adx[i] >= 22.0:
                    if c[i] > donchian_h20[i] and m_dir > 0 and c[i] > o[i]:
                        sl_ref = np.min(l[max(0, i-3):i+1])
                        events.append({
                            'algo': 'ALGO_C_THURSDAY_MACRO_IGNITION',
                            'idx': i,
                            'direction': 1,
                            'entry_price': c[i],
                            'risk_pips': max(min((c[i] - sl_ref) / PIP_SIZE, 15.0), 7.0)
                        })
                        triggered_c = True
                    elif c[i] < donchian_l20[i] and m_dir < 0 and c[i] < o[i]:
                        sl_ref = np.max(h[max(0, i-3):i+1])
                        events.append({
                            'algo': 'ALGO_C_THURSDAY_MACRO_IGNITION',
                            'idx': i,
                            'direction': -1,
                            'entry_price': c[i],
                            'risk_pips': max(min((sl_ref - c[i]) / PIP_SIZE, 15.0), 7.0)
                        })
                        triggered_c = True

        # ----------------------------------------------------------------------
        # ALGO D: MIDDAY ULTRA-COIL (<= 12p) -> 13:00 UTC EXPANSION THRUST
        # ----------------------------------------------------------------------
        midday_idx = [i for i in day_indices if 11 <= hours[i] < 13]
        if len(midday_idx) >= 6:
            mid_h = np.max(h[midday_idx])
            mid_l = np.min(l[midday_idx])
            mid_rng = (mid_h - mid_l) / PIP_SIZE

            if mid_rng <= 12.0:
                ny_open_idx = [i for i in day_indices if 13 <= hours[i] <= 14]
                triggered_d = False
                for i in ny_open_idx:
                    if triggered_d or spread[i] > 0.60:
                        continue
                    m_dir = macro[i]

                    if c[i] > mid_h and m_dir > 0 and c[i] > o[i]:
                        events.append({
                            'algo': 'ALGO_D_MIDDAY_ULTRACOIL_THRUST',
                            'idx': i,
                            'direction': 1,
                            'entry_price': c[i],
                            'risk_pips': max(min((c[i] - mid_l) / PIP_SIZE, 14.0), 6.0)
                        })
                        triggered_d = True
                    elif c[i] < mid_l and m_dir < 0 and c[i] < o[i]:
                        events.append({
                            'algo': 'ALGO_D_MIDDAY_ULTRACOIL_THRUST',
                            'idx': i,
                            'direction': -1,
                            'entry_price': c[i],
                            'risk_pips': max(min((mid_h - c[i]) / PIP_SIZE, 14.0), 6.0)
                        })
                        triggered_d = True

    # Compute 4-hour Forward Excursions
    for e in events:
        idx = e['idx']
        d = e['direction']
        ep = e['entry_price']
        risk = e['risk_pips']
        end_idx = min(idx + 1 + 16, n)

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


def run_algo_benchmark():
    print("=" * 115)
    print("     DATA ENGINEERING SUITE: MATCHING ALGORITHMIC MECHANISMS TO STRUCTURAL EDGE WINDOWS     ".center(115))
    print("=" * 115)

    weeks = list(range(1, 53))
    num_workers = min(8, os.cpu_count() or 4)
    t0 = time.time()

    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        results = list(ex.map(evaluate_week_edge_algos, weeks))

    dt = time.time() - t0
    all_events = [e for r in results for e in r]
    print(f"[+] Evaluated across 52 weeks (~26M ticks) in {dt:.1f}s!\n")

    if not all_events:
        print("[!] No events found.")
        return

    df = pd.DataFrame(all_events)

    print("-" * 115)
    print(f"{'Targeted Algorithmic Mechanism':<42} | {'Signals':<8} | {'Avg MFE':<9} | {'Avg MAE':<9} | {'Edge Ratio':<12} | {'1.7R Win Rate':<12}")
    print("-" * 115)

    algo_names = {
        'ALGO_A_ASIAN_COIL_LONDON_EXPANSION': 'Algo A: Asian Coil -> London Expansion',
        'ALGO_B_FRANKFURT_TRAP_SWEEP_REVERSAL': 'Algo B: Frankfurt Trap Sweep Reversal Fade',
        'ALGO_C_THURSDAY_MACRO_IGNITION': 'Algo C: Thursday Macro Trend Ignition',
        'ALGO_D_MIDDAY_ULTRACOIL_THRUST': 'Algo D: Midday Ultra-Coil NY Thrust'
    }

    for algo_key, name in sorted(algo_names.items()):
        sub_df = df[df['algo'] == algo_key]
        n_ev = len(sub_df)
        if n_ev > 0:
            avg_mfe = sub_df['mfe'].mean()
            avg_mae = sub_df['mae'].mean()
            ratio = avg_mfe / max(avg_mae, 1e-6)
            wr = (sub_df['hit_target'].sum() / n_ev) * 100
            print(f"{name:<42} | {n_ev:<8} | {avg_mfe:6.1f} p  | {avg_mae:6.1f} p  | {ratio:6.2f} : 1   | {wr:6.1f}%")
        else:
            print(f"{name:<42} | 0        |    -    |    -    |    -         |    -")

    print("-" * 115)


if __name__ == "__main__":
    run_algo_benchmark()
