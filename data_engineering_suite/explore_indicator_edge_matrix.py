#!/usr/bin/env python3
"""
================================================================================
  DATA ENGINEERING SUITE: QUANTITATIVE INDICATOR EDGE MATRIX SCANNER
================================================================================
Systematically mines the EUR/USD tick dataset (~26M ticks / 52 weeks) to identify
and measure the exact mathematical edge across 3 core indicator paradigms:

  ZONE 1: Volatility Compression -> Session Expansion (Bollinger Bandwidth <= 15th pct)
  ZONE 2: Trend Momentum Pullbacks (H1 50-EMA + M15 20-EMA + ADX >= 22)
  ZONE 3: Session Extreme Reversal / Fades (RSI Exhaustion at Session High/Low)

For every setup, we compute:
  - Forward Maximum Favorable Excursion (MFE in pips)
  - Forward Maximum Adverse Excursion (MAE in pips)
  - Edge Asymmetry Ratio (MFE / MAE)
  - Probability of reaching +10 pips before -10 pips (Win Probability)
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
import numba

sys.dont_write_bytecode = True

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tick_engine.core.dataloader import get_week_dataset
from data_engineering_suite.bars import build_time_bars

PIP_SIZE = 0.0001
RESULTS_DIR = ROOT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------------------
# FORWARD EXCURSION CALCULATOR (MFE / MAE)
# ------------------------------------------------------------------------------
def compute_forward_excursions(
    entry_prices: np.ndarray,
    directions: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    horizon_bars: int = 16  # 16 bars * 15m = 4 hours forward
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes forward MFE (Max Favorable Excursion) and MAE (Max Adverse Excursion) in pips
    over the subsequent `horizon_bars`.
    """
    n = len(entry_prices)
    mfe = np.zeros(n, dtype=np.float64)
    mae = np.zeros(n, dtype=np.float64)
    hit_10p_first = np.zeros(n, dtype=np.int8)  # 1 if +10p before -10p, 0 otherwise

    for i in range(n):
        if directions[i] == 0:
            continue
        ep = entry_prices[i]
        d = directions[i]
        end_idx = min(i + 1 + horizon_bars, n)

        sub_h = highs[i+1 : end_idx]
        sub_l = lows[i+1 : end_idx]
        if len(sub_h) == 0:
            continue

        if d == 1:  # LONG
            fav = (np.max(sub_h) - ep) / PIP_SIZE
            adv = (ep - np.min(sub_l)) / PIP_SIZE
            mfe[i] = max(0.0, fav)
            mae[i] = max(0.0, adv)

            # Check +10p before -10p sequence
            hit_tp = False
            for j in range(len(sub_h)):
                p_up = (sub_h[j] - ep) / PIP_SIZE
                p_dn = (ep - sub_l[j]) / PIP_SIZE
                if p_up >= 10.0 and p_dn < 10.0:
                    hit_tp = True
                    break
                elif p_dn >= 10.0:
                    break
            if hit_tp:
                hit_10p_first[i] = 1

        elif d == -1:  # SHORT
            fav = (ep - np.min(sub_l)) / PIP_SIZE
            adv = (np.max(sub_h) - ep) / PIP_SIZE
            mfe[i] = max(0.0, fav)
            mae[i] = max(0.0, adv)

            # Check +10p before -10p sequence
            hit_tp = False
            for j in range(len(sub_h)):
                p_dn = (ep - sub_l[j]) / PIP_SIZE
                p_up = (sub_h[j] - ep) / PIP_SIZE
                if p_dn >= 10.0 and p_up < 10.0:
                    hit_tp = True
                    break
                elif p_up >= 10.0:
                    break
            if hit_tp:
                hit_10p_first[i] = 1

    return mfe, mae, hit_10p_first


# ------------------------------------------------------------------------------
# WEEKLY SCAN WORKER
# ------------------------------------------------------------------------------
def scan_week_indicators(w: int):
    ticks, _ = get_week_dataset(w)
    if len(ticks) < 1000:
        return []

    m15 = build_time_bars(ticks, timeframe_min=15)
    if len(m15) < 50:
        return []

    c = m15['close']
    h = m15['high']
    l = m15['low']
    o = m15['open']

    # 1. Bollinger Bands (20, 2.0)
    sma20 = c.rolling(20).mean().bfill()
    std20 = c.rolling(20).std().bfill()
    bb_up = sma20 + (2.0 * std20)
    bb_low = sma20 - (2.0 * std20)
    bb_width = (bb_up - bb_low) / PIP_SIZE
    bb_w_min80 = bb_width.rolling(80).min().bfill()
    is_bb_compressed = bb_width <= 1.35 * bb_w_min80

    # 2. ADX (14) & ATR (14)
    tr = np.maximum(h - l, np.maximum(abs(h - c.shift(1)), abs(l - c.shift(1))))
    atr = tr.rolling(14).mean().bfill()
    up_m = h.diff()
    dn_m = -l.diff()
    p_dm = np.where((up_m > dn_m) & (up_m > 0), up_m, 0.0)
    m_dm = np.where((dn_m > up_m) & (dn_m > 0), dn_m, 0.0)
    smooth_tr = tr.ewm(alpha=1.0/14, adjust=False).mean()
    p_di = (pd.Series(p_dm, index=m15.index).ewm(alpha=1.0/14, adjust=False).mean() / np.maximum(smooth_tr, 1e-9)) * 100
    m_di = (pd.Series(m_dm, index=m15.index).ewm(alpha=1.0/14, adjust=False).mean() / np.maximum(smooth_tr, 1e-9)) * 100
    dx = (abs(p_di - m_di) / np.maximum(p_di + m_di, 1e-9)) * 100
    adx = dx.ewm(alpha=1.0/14, adjust=False).mean().values

    # 3. Moving Averages & Trend
    ema20 = c.ewm(span=20, adjust=False).mean().values
    ts = pd.to_datetime(ticks['timestamp'], unit='ms')
    mid = 0.5 * (ticks['bid'] + ticks['ask'])
    base_df = pd.DataFrame({'close': mid}, index=ts)
    h1 = base_df.resample('1h').last().dropna()
    h1['ema50'] = h1['close'].ewm(span=50, adjust=False).mean()
    h1['macro'] = np.where(h1['close'] > h1['ema50'], 1, -1)
    m15['macro'] = h1['macro'].reindex(m15.index, method='ffill').fillna(0)

    # 4. RSI (14)
    delta = c.diff()
    gain = delta.clip(lower=0.0).ewm(alpha=1.0/14, adjust=False).mean()
    loss = (-delta.clip(upper=0.0)).ewm(alpha=1.0/14, adjust=False).mean()
    rs = gain / np.maximum(loss, 1e-9)
    rsi = (100.0 - (100.0 / (1.0 + rs))).values

    hours = m15.index.hour.values
    highs = h.values
    lows = l.values
    closes = c.values
    opens = o.values
    macros = m15['macro'].values
    compressed = is_bb_compressed.values

    # 20-bar Rolling Extremes
    roll_h20 = pd.Series(highs).rolling(20).max().shift(1).bfill().values
    roll_l20 = pd.Series(lows).rolling(20).min().shift(1).bfill().values

    events = []

    for i in range(30, len(m15) - 20):
        hr = hours[i]

        # -------------------------------------------------------------
        # SETUP ZONE 1: Volatility Compression Breakout (Bollinger Squeeze)
        # -------------------------------------------------------------
        if compressed[i-1]:
            # Bullish expansion
            if closes[i] > bb_up.iloc[i-1] and closes[i] > opens[i] and macros[i] > 0:
                events.append({
                    'zone': 'ZONE_1_VOLATILITY_SQUEEZE',
                    'hour': hr,
                    'direction': 1,
                    'idx': i,
                    'entry_price': closes[i]
                })
            # Bearish expansion
            elif closes[i] < bb_low.iloc[i-1] and closes[i] < opens[i] and macros[i] < 0:
                events.append({
                    'zone': 'ZONE_1_VOLATILITY_SQUEEZE',
                    'hour': hr,
                    'direction': -1,
                    'idx': i,
                    'entry_price': closes[i]
                })

        # -------------------------------------------------------------
        # SETUP ZONE 2: Trend Momentum Pullback to 20-EMA with ADX >= 22
        # -------------------------------------------------------------
        if adx[i] >= 22.0:
            # Bullish pullback touch
            if macros[i] > 0 and lows[i] <= ema20[i] and closes[i] > ema20[i] and closes[i] > opens[i]:
                events.append({
                    'zone': 'ZONE_2_TREND_PULLBACK',
                    'hour': hr,
                    'direction': 1,
                    'idx': i,
                    'entry_price': closes[i]
                })
            # Bearish pullback touch
            elif macros[i] < 0 and highs[i] >= ema20[i] and closes[i] < ema20[i] and closes[i] < opens[i]:
                events.append({
                    'zone': 'ZONE_2_TREND_PULLBACK',
                    'hour': hr,
                    'direction': -1,
                    'idx': i,
                    'entry_price': closes[i]
                })

        # -------------------------------------------------------------
        # SETUP ZONE 3: Session Extreme Reversal / Fade (RSI Exhaustion)
        # -------------------------------------------------------------
        # Probing above 20-bar high with RSI >= 65
        if highs[i] >= roll_h20[i] and closes[i] < roll_h20[i] and rsi[i] >= 65.0 and closes[i] < opens[i]:
            events.append({
                'zone': 'ZONE_3_SESSION_EXTREME_FADE',
                'hour': hr,
                'direction': -1,
                'idx': i,
                'entry_price': closes[i]
            })
        # Probing below 20-bar low with RSI <= 35
        elif lows[i] <= roll_l20[i] and closes[i] > roll_l20[i] and rsi[i] <= 35.0 and closes[i] > opens[i]:
            events.append({
                'zone': 'ZONE_3_SESSION_EXTREME_FADE',
                'hour': hr,
                'direction': 1,
                'idx': i,
                'entry_price': closes[i]
            })

    if not events:
        return []

    # Compute MFE and MAE for all detected events in this week
    event_indices = [e['idx'] for e in events]
    event_dirs = np.array([e['direction'] for e in events])
    event_prices = np.array([e['entry_price'] for e in events])

    # Calculate excursions over next 16 bars (4 hours)
    for k, e in enumerate(events):
        idx = e['idx']
        d = e['direction']
        ep = e['entry_price']
        end_idx = min(idx + 1 + 16, len(m15))

        sub_h = highs[idx + 1 : end_idx]
        sub_l = lows[idx + 1 : end_idx]
        if len(sub_h) == 0:
            e['mfe'] = 0.0
            e['mae'] = 0.0
            e['win_10p'] = 0
            continue

        if d == 1:
            fav = (np.max(sub_h) - ep) / PIP_SIZE
            adv = (ep - np.min(sub_l)) / PIP_SIZE
            e['mfe'] = max(0.0, fav)
            e['mae'] = max(0.0, adv)

            # Check +10p before -10p
            hit_tp = 0
            for bar_k in range(len(sub_h)):
                p_up = (sub_h[bar_k] - ep) / PIP_SIZE
                p_dn = (ep - sub_l[bar_k]) / PIP_SIZE
                if p_up >= 10.0 and p_dn < 10.0:
                    hit_tp = 1
                    break
                elif p_dn >= 10.0:
                    break
            e['win_10p'] = hit_tp

        else:
            fav = (ep - np.min(sub_l)) / PIP_SIZE
            adv = (np.max(sub_h) - ep) / PIP_SIZE
            e['mfe'] = max(0.0, fav)
            e['mae'] = max(0.0, adv)

            hit_tp = 0
            for bar_k in range(len(sub_h)):
                p_dn = (ep - sub_l[bar_k]) / PIP_SIZE
                p_up = (sub_h[bar_k] - ep) / PIP_SIZE
                if p_dn >= 10.0 and p_up < 10.0:
                    hit_tp = 1
                    break
                elif p_up >= 10.0:
                    break
            e['win_10p'] = hit_tp

    return events


def run_indicator_edge_matrix():
    print("=" * 110)
    print("     DATA ENGINEERING SUITE: 52-WEEK QUANTITATIVE INDICATOR EDGE MATRIX SCAN     ".center(110))
    print("=" * 110)

    t0 = time.time()
    weeks = list(range(1, 53))
    num_workers = min(8, os.cpu_count() or 4)

    print(f"[*] Scanning 52 weeks (~26M ticks) across 3 Core Indicator Edge Paradigms...")
    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        results = list(ex.map(scan_week_indicators, weeks))

    all_events = [e for r in results for e in r]
    dt = time.time() - t0
    print(f"[+] Processed {len(all_events):,} signal events in {dt:.1f}s!\n")

    df = pd.DataFrame(all_events)
    if df.empty:
        print("[!] No events found.")
        return

    # Aggregate by Zone
    print("-" * 110)
    print(f"{'Indicator Zone':<35} | {'Events':<8} | {'Avg MFE (p)':<12} | {'Avg MAE (p)':<12} | {'MFE/MAE Ratio':<15} | {'10p Win Rate':<12}")
    print("-" * 110)

    zone_names = {
        'ZONE_1_VOLATILITY_SQUEEZE': 'Zone 1: Volatility Squeeze Breakout',
        'ZONE_2_TREND_PULLBACK': 'Zone 2: 20-EMA Pullback (ADX >= 22)',
        'ZONE_3_SESSION_EXTREME_FADE': 'Zone 3: Extreme Fade (RSI Exhaustion)'
    }

    zone_summaries = {}
    for z, z_name in zone_names.items():
        sub = df[df['zone'] == z]
        cnt = len(sub)
        if cnt == 0:
            continue
        avg_mfe = sub['mfe'].mean()
        avg_mae = sub['mae'].mean()
        ratio = avg_mfe / max(avg_mae, 1e-6)
        win_rate = (sub['win_10p'].sum() / cnt) * 100.0

        zone_summaries[z] = {
            'name': z_name,
            'events': cnt,
            'avg_mfe': round(avg_mfe, 2),
            'avg_mae': round(avg_mae, 2),
            'ratio': round(ratio, 2),
            'win_rate': round(win_rate, 1)
        }

        print(f"{z_name:<35} | {cnt:<8,d} | {avg_mfe:8.1f} p   | {avg_mae:8.1f} p   | {ratio:6.2f} : 1       | {win_rate:5.1f}%")

    print("-" * 110)

    # Hourly Edge Breakdown for Zone 1 and Zone 3
    print("\n" + "=" * 110)
    print("HOURLY BREAKDOWN: WHERE DOES EACH ZONE HAVE PEAK ASYMMETRIC EDGE?")
    print("=" * 110)
    print(f"{'Hour (UTC)':<12} | {'Zone 1 Ratio (Squeeze)':<25} | {'Zone 2 Ratio (Pullback)':<25} | {'Zone 3 Ratio (Fade)':<25}")
    print("-" * 110)

    hourly_breakdown = {}
    for h in range(24):
        h_df = df[df['hour'] == h]
        
        def get_stat(z):
            s = h_df[h_df['zone'] == z]
            if len(s) < 5:
                return 0.0, 0
            m_ratio = s['mfe'].mean() / max(s['mae'].mean(), 1e-6)
            return m_ratio, len(s)

        r1, n1 = get_stat('ZONE_1_VOLATILITY_SQUEEZE')
        r2, n2 = get_stat('ZONE_2_TREND_PULLBACK')
        r3, n3 = get_stat('ZONE_3_SESSION_EXTREME_FADE')

        s1_str = f"{r1:4.2f} : 1 ({n1:2d} ev)" if n1 > 0 else "   -   "
        s2_str = f"{r2:4.2f} : 1 ({n2:2d} ev)" if n2 > 0 else "   -   "
        s3_str = f"{r3:4.2f} : 1 ({n3:2d} ev)" if n3 > 0 else "   -   "

        tag1 = " *" if r1 >= 1.60 else ""
        tag3 = " #" if r3 >= 1.50 else ""

        hourly_breakdown[h] = {
            'squeeze_ratio': round(r1, 2),
            'pullback_ratio': round(r2, 2),
            'fade_ratio': round(r3, 2),
            'n1': n1, 'n2': n2, 'n3': n3
        }

        print(f"{h:02d}:00 - {h+1:02d}:00  | {s1_str:<23}{tag1:<2} | {s2_str:<23}   | {s3_str:<23}{tag3:<2}")

    print("-" * 110)
    print("Legend: (*) = High Asymmetry Breakout Edge (Ratio >= 1.6:1)")
    print("        (#) = High Asymmetry Mean-Reversion Edge (Ratio >= 1.5:1)")
    print("=" * 110 + "\n")

    # Export to JSON
    with open(RESULTS_DIR / "indicator_edge_matrix_scan.json", "w") as f:
        json.dump({
            'zone_summaries': zone_summaries,
            'hourly_breakdown': hourly_breakdown
        }, f, indent=2)

    print(f"[+] Full Edge Matrix results saved to: {RESULTS_DIR / 'indicator_edge_matrix_scan.json'}")


if __name__ == "__main__":
    run_indicator_edge_matrix()
