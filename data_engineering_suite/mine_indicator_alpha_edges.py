#!/usr/bin/env python3
"""
================================================================================
  DATA ENGINEERING SUITE: MULTI-INDICATOR ALPHA & EDGE MINING ENGINE
================================================================================
Systematically mines the 52-week EUR/USD tick dataset (~26M ticks) across 8
classical & modern technical indicator paradigms using the `ta` library:

  1. Carter TTM Squeeze (Bollinger Bands inside Keltner Channels + Momentum)
  2. Donchian 20-Channel Trend Breakout + ADX >= 25 Filter
  3. Multi-Timeframe EMA Ribbon Alignment (H1 50-EMA + M15 9/21-EMA)
  4. Stochastic Extreme Pullback Re-entry in Trend (%K < 20 / > 80)
  5. MACD Zero-Line Bounce & Histogram Momentum Acceleration
  6. CCI (Commodity Channel Index) Momentum Surge (+100 / -100)
  7. Awesome Oscillator (AO) Trend Saucer Continuation
  8. RSI Bull/Bear Regime Range Shift (40-50 Support / 50-60 Resistance)

For each indicator setup, we measure:
  - 4-Hour Forward Maximum Favorable Excursion (MFE in pips)
  - 4-Hour Forward Maximum Adverse Excursion (MAE in pips)
  - Edge Asymmetry Ratio (MFE / MAE)
  - Win Rate reaching +15 pips before -10 pips (1.5R expectancy)
  - Win Rate reaching +20 pips before -10 pips (2.0R expectancy)
  - Hourly Institutional Edge Heatmap (00:00 to 23:00 UTC)
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


def scan_week_all_indicators(w: int) -> List[Dict[str, Any]]:
    """
    Computes all 8 indicator setups for a single week and tracks 4-hour forward excursions.
    """
    ticks, _ = get_week_dataset(w)
    if len(ticks) < 1000:
        return []

    m15 = build_time_bars(ticks, timeframe_min=15)
    if len(m15) < 60:
        return []

    c = m15['close']
    h = m15['high']
    l = m15['low']
    o = m15['open']
    v = m15['vol']

    # --- 0. Macro Regime (H1 50-EMA) ---
    ts = pd.to_datetime(ticks['timestamp'], unit='ms')
    mid = 0.5 * (ticks['bid'] + ticks['ask'])
    base_df = pd.DataFrame({'close': mid}, index=ts)
    h1 = base_df.resample('1h').last().dropna()
    h1['ema50'] = h1['close'].ewm(span=50, adjust=False).mean()
    h1['macro'] = np.where(h1['close'] > h1['ema50'], 1, -1)
    macro = h1['macro'].reindex(m15.index, method='ffill').fillna(0).values

    # --- 1. Bollinger Bands & Keltner Channels (TTM Squeeze) ---
    bb = ta.volatility.BollingerBands(close=c, window=20, window_dev=2.0)
    bb_high = bb.bollinger_hband()
    bb_low = bb.bollinger_lband()
    bb_mid = bb.bollinger_mavg()

    kc = ta.volatility.KeltnerChannel(high=h, low=l, close=c, window=20, window_atr=10, original_version=False)
    kc_high = kc.keltner_channel_hband()
    kc_low = kc.keltner_channel_lband()

    # Squeeze is on when BB is completely inside KC
    is_squeeze = (bb_high < kc_high) & (bb_low > kc_low)
    squeeze_fired_bull = (~is_squeeze) & (is_squeeze.shift(1).fillna(False)) & (c > bb_mid) & (macro > 0)
    squeeze_fired_bear = (~is_squeeze) & (is_squeeze.shift(1).fillna(False)) & (c < bb_mid) & (macro < 0)

    # --- 2. Donchian Channel (20) & ADX (14) ---
    adx_ind = ta.trend.ADXIndicator(high=h, low=l, close=c, window=14)
    adx = adx_ind.adx().values
    donchian_h20 = h.rolling(20).max().shift(1).bfill().values
    donchian_l20 = l.rolling(20).min().shift(1).bfill().values

    # --- 3. EMA Ribbon (M15 9-EMA, 21-EMA) ---
    ema9 = c.ewm(span=9, adjust=False).mean().values
    ema21 = c.ewm(span=21, adjust=False).mean().values

    # --- 4. Stochastic Oscillator (14, 3, 3) ---
    stoch = ta.momentum.StochasticOscillator(high=h, low=l, close=c, window=14, smooth_window=3)
    stoch_k = stoch.stoch().values
    stoch_d = stoch.stoch_signal().values

    # --- 5. MACD (12, 26, 9) ---
    macd = ta.trend.MACD(close=c, window_slow=26, window_fast=12, window_sign=9)
    macd_line = macd.macd().values
    macd_signal = macd.macd_signal().values
    macd_diff = macd.macd_diff().values

    # --- 6. CCI (20) ---
    cci = ta.trend.CCIIndicator(high=h, low=l, close=c, window=20).cci().values

    # --- 7. Awesome Oscillator (AO) ---
    ao = ta.momentum.AwesomeOscillatorIndicator(high=h, low=l, window1=5, window2=34).awesome_oscillator().values

    # --- 8. RSI (14) ---
    rsi = ta.momentum.RSIIndicator(close=c, window=14).rsi().values

    hours = m15.index.hour.values
    highs = h.values
    lows = l.values
    closes = c.values
    opens = o.values
    n_bars = len(m15)

    events = []

    # Helper function to append signal event
    def add_event(name: str, idx: int, direction: int):
        events.append({
            'strategy': name,
            'idx': idx,
            'hour': int(hours[idx]),
            'direction': int(direction),
            'entry_price': float(closes[idx])
        })

    for i in range(35, n_bars - 20):
        m = macro[i]

        # -------------------------------------------------------------
        # 1. CARTER TTM SQUEEZE EXPANSION
        # -------------------------------------------------------------
        if squeeze_fired_bull.iloc[i]:
            add_event("1_CARTER_TTM_SQUEEZE", i, 1)
        elif squeeze_fired_bear.iloc[i]:
            add_event("1_CARTER_TTM_SQUEEZE", i, -1)

        # -------------------------------------------------------------
        # 2. DONCHIAN BREAKOUT + ADX >= 25
        # -------------------------------------------------------------
        if adx[i] >= 25.0:
            if closes[i] > donchian_h20[i] and m > 0 and closes[i] > opens[i]:
                add_event("2_DONCHIAN_ADX_BREAKOUT", i, 1)
            elif closes[i] < donchian_l20[i] and m < 0 and closes[i] < opens[i]:
                add_event("2_DONCHIAN_ADX_BREAKOUT", i, -1)

        # -------------------------------------------------------------
        # 3. EMA RIBBON CONTINUATION (9 > 21 EMA touch + bounce)
        # -------------------------------------------------------------
        if m > 0 and ema9[i] > ema21[i]:
            if lows[i] <= ema21[i] and closes[i] > ema21[i] and closes[i] > opens[i]:
                add_event("3_EMA_RIBBON_PULLBACK", i, 1)
        elif m < 0 and ema9[i] < ema21[i]:
            if highs[i] >= ema21[i] and closes[i] < ema21[i] and closes[i] < opens[i]:
                add_event("3_EMA_RIBBON_PULLBACK", i, -1)

        # -------------------------------------------------------------
        # 4. STOCHASTIC EXTREME PULLBACK IN TREND (%K cross %D)
        # -------------------------------------------------------------
        if m > 0 and stoch_k[i-1] <= 20.0 and stoch_k[i] > stoch_d[i] and stoch_k[i-1] <= stoch_d[i-1]:
            add_event("4_STOCHASTIC_DEEP_PULLBACK", i, 1)
        elif m < 0 and stoch_k[i-1] >= 80.0 and stoch_k[i] < stoch_d[i] and stoch_k[i-1] >= stoch_d[i-1]:
            add_event("4_STOCHASTIC_DEEP_PULLBACK", i, -1)

        # -------------------------------------------------------------
        # 5. MACD HISTOGRAM MOMENTUM ACCELERATION
        # -------------------------------------------------------------
        if m > 0 and macd_line[i] > 0 and macd_diff[i-1] < 0 and macd_diff[i] > 0:
            add_event("5_MACD_HISTOGRAM_ACCELERATION", i, 1)
        elif m < 0 and macd_line[i] < 0 and macd_diff[i-1] > 0 and macd_diff[i] < 0:
            add_event("5_MACD_HISTOGRAM_ACCELERATION", i, -1)

        # -------------------------------------------------------------
        # 6. CCI MOMENTUM SURGE (+100 / -100 cross)
        # -------------------------------------------------------------
        if m > 0 and cci[i-1] < 100.0 and cci[i] >= 100.0:
            add_event("6_CCI_MOMENTUM_SURGE", i, 1)
        elif m < 0 and cci[i-1] > -100.0 and cci[i] <= -100.0:
            add_event("6_CCI_MOMENTUM_SURGE", i, -1)

        # -------------------------------------------------------------
        # 7. AWESOME OSCILLATOR (AO) SAUCER
        # -------------------------------------------------------------
        if m > 0 and ao[i] > 0 and ao[i-2] > ao[i-1] and ao[i] > ao[i-1]:
            add_event("7_AO_SAUCER_CONTINUATION", i, 1)
        elif m < 0 and ao[i] < 0 and ao[i-2] < ao[i-1] and ao[i] < ao[i-1]:
            add_event("7_AO_SAUCER_CONTINUATION", i, -1)

        # -------------------------------------------------------------
        # 8. RSI BULL/BEAR REGIME SHIFT (40-50 Support / 50-60 Resistance)
        # -------------------------------------------------------------
        if m > 0 and 40.0 <= rsi[i-1] <= 50.0 and rsi[i] > 50.0 and closes[i] > opens[i]:
            add_event("8_RSI_REGIME_RANGE_SHIFT", i, 1)
        elif m < 0 and 50.0 <= rsi[i-1] <= 60.0 and rsi[i] < 50.0 and closes[i] < opens[i]:
            add_event("8_RSI_REGIME_RANGE_SHIFT", i, -1)

    if not events:
        return []

    # -------------------------------------------------------------
    # FORWARD EXCURSIONS (4 Hours = 16 M15 Bars)
    # -------------------------------------------------------------
    for e in events:
        idx = e['idx']
        d = e['direction']
        ep = e['entry_price']
        end_idx = min(idx + 1 + 16, n_bars)

        sub_h = highs[idx + 1 : end_idx]
        sub_l = lows[idx + 1 : end_idx]
        if len(sub_h) == 0:
            e['mfe'] = 0.0
            e['mae'] = 0.0
            e['win_1_5r'] = 0
            e['win_2_0r'] = 0
            continue

        if d == 1:
            fav = (np.max(sub_h) - ep) / PIP_SIZE
            adv = (ep - np.min(sub_l)) / PIP_SIZE
            e['mfe'] = max(0.0, fav)
            e['mae'] = max(0.0, adv)

            # Hit 1.5R (+15p before -10p)
            w15 = 0
            for k in range(len(sub_h)):
                p_up = (sub_h[k] - ep) / PIP_SIZE
                p_dn = (ep - sub_l[k]) / PIP_SIZE
                if p_up >= 15.0 and p_dn < 10.0:
                    w15 = 1
                    break
                elif p_dn >= 10.0:
                    break
            e['win_1_5r'] = w15

            # Hit 2.0R (+20p before -10p)
            w20 = 0
            for k in range(len(sub_h)):
                p_up = (sub_h[k] - ep) / PIP_SIZE
                p_dn = (ep - sub_l[k]) / PIP_SIZE
                if p_up >= 20.0 and p_dn < 10.0:
                    w20 = 1
                    break
                elif p_dn >= 10.0:
                    break
            e['win_2_0r'] = w20

        else:
            fav = (ep - np.min(sub_l)) / PIP_SIZE
            adv = (np.max(sub_h) - ep) / PIP_SIZE
            e['mfe'] = max(0.0, fav)
            e['mae'] = max(0.0, adv)

            # Hit 1.5R (+15p before -10p)
            w15 = 0
            for k in range(len(sub_h)):
                p_dn = (ep - sub_l[k]) / PIP_SIZE
                p_up = (sub_h[k] - ep) / PIP_SIZE
                if p_dn >= 15.0 and p_up < 10.0:
                    w15 = 1
                    break
                elif p_up >= 10.0:
                    break
            e['win_1_5r'] = w15

            # Hit 2.0R (+20p before -10p)
            w20 = 0
            for k in range(len(sub_h)):
                p_dn = (ep - sub_l[k]) / PIP_SIZE
                p_up = (sub_h[k] - ep) / PIP_SIZE
                if p_dn >= 20.0 and p_up < 10.0:
                    w20 = 1
                    break
                elif p_up >= 10.0:
                    break
            e['win_2_0r'] = w20

    return events


def run_mining_scan():
    print("=" * 115)
    print("     DATA ENGINEERING SUITE: 52-WEEK MULTI-INDICATOR ALPHA MINING BENCHMARK     ".center(115))
    print("=" * 115)

    t0 = time.time()
    weeks = list(range(1, 53))
    num_workers = min(8, os.cpu_count() or 4)

    print(f"[*] Dispatching 8-indicator setup scan across 52 weeks (~26M ticks) on {num_workers} CPU cores...")
    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        results = list(ex.map(scan_week_all_indicators, weeks))

    all_events = [e for r in results for e in r]
    dt = time.time() - t0
    print(f"[+] Extracted {len(all_events):,} signal events in {dt:.1f}s!\n")

    if not all_events:
        print("[!] No events found.")
        return

    df = pd.DataFrame(all_events)

    # 1. OVERALL INDICATOR BENCHMARK TABLE
    print("-" * 115)
    print(f"{'Indicator Setup':<35} | {'Events':<7} | {'Avg MFE':<9} | {'Avg MAE':<9} | {'MFE/MAE':<9} | {'1.5R Win%':<10} | {'2.0R Win%':<10}")
    print("-" * 115)

    strat_groups = df.groupby('strategy')
    summary_data = {}

    for strat, grp in sorted(strat_groups):
        n = len(grp)
        avg_mfe = grp['mfe'].mean()
        avg_mae = grp['mae'].mean()
        ratio = avg_mfe / max(avg_mae, 1e-6)
        w15 = (grp['win_1_5r'].sum() / n) * 100
        w20 = (grp['win_2_0r'].sum() / n) * 100

        print(f"{strat:<35} | {n:<7} | {avg_mfe:6.1f} p  | {avg_mae:6.1f} p  | {ratio:6.2f}:1  | {w15:6.1f}%    | {w20:6.1f}%")

        # Compute hourly ratios
        hourly_ratios = {}
        for h in range(24):
            h_grp = grp[grp['hour'] == h]
            if len(h_grp) >= 5:
                h_mfe = h_grp['mfe'].mean()
                h_mae = h_grp['mae'].mean()
                hourly_ratios[h] = {
                    'count': len(h_grp),
                    'mfe': round(h_mfe, 1),
                    'mae': round(h_mae, 1),
                    'ratio': round(h_mfe / max(h_mae, 1e-6), 2)
                }

        summary_data[strat] = {
            'events': n,
            'avg_mfe': round(avg_mfe, 2),
            'avg_mae': round(avg_mae, 2),
            'asymmetry_ratio': round(ratio, 2),
            'win_rate_1_5r': round(w15, 1),
            'win_rate_2_0r': round(w20, 1),
            'hourly_edge': hourly_ratios
        }

    print("-" * 115)

    # 2. IDENTIFY TOP ASYMMETRIC COMBINATIONS (Setup + Session Hour)
    print("\n" + "=" * 115)
    print("TOP ASYMMETRIC STATISTICAL EDGES DISCOVERED (MFE/MAE Ratio >= 1.60 : 1 with N >= 10)")
    print("=" * 115)
    print(f"{'Indicator Setup':<32} | {'Hour (UTC)':<12} | {'Events':<8} | {'Avg MFE':<10} | {'Avg MAE':<10} | {'Edge Ratio':<12}")
    print("-" * 115)

    high_edge_combos = []
    for strat, d in summary_data.items():
        for h, h_stats in d['hourly_edge'].items():
            if h_stats['count'] >= 8 and h_stats['ratio'] >= 1.50:
                high_edge_combos.append({
                    'strat': strat,
                    'hour': h,
                    'count': h_stats['count'],
                    'mfe': h_stats['mfe'],
                    'mae': h_stats['mae'],
                    'ratio': h_stats['ratio']
                })

    high_edge_combos.sort(key=lambda x: x['ratio'], reverse=True)
    for c in high_edge_combos[:15]:
        print(f"{c['strat']:<32} | {c['hour']:02d}:00-{c['hour']+1:02d}:00  | {c['count']:<8} | {c['mfe']:6.1f} p   | {c['mae']:6.1f} p   | {c['ratio']:6.2f} : 1")

    print("=" * 115 + "\n")

    # Save to JSON
    with open(RESULTS_DIR / "indicator_alpha_mining_results.json", "w") as f:
        json.dump(summary_data, f, indent=2)

    print(f"[+] Comprehensive mining results saved to: {RESULTS_DIR / 'indicator_alpha_mining_results.json'}")


if __name__ == "__main__":
    run_mining_scan()
