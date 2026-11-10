#!/usr/bin/env python3
"""
================================================================================
  DATA ENGINEERING SUITE: DEEP DATA ANALYTICS & STATISTICAL PROFILER
================================================================================
Performs institutional statistical profiling across 5 years of tick data:
  1. 24-Hour Intraday Diurnal Curves (Spread, Velocity, Volatility, Hurst)
  2. Order Flow Imbalance (CVD) Absorption Dynamics
  3. Tick Velocity Anomaly / Ignition Bursts
  4. Time Bars vs Range/Tick Bars Information Density Comparison
================================================================================
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Any, List

from data_engineering_suite.bars import build_time_bars, build_range_bars, build_tick_bars
from data_engineering_suite.features import (
    compute_spread_metrics,
    compute_parkinson_volatility,
    compute_garman_klass_volatility,
    compute_hurst_exponent,
    compute_ou_halflife,
    PIP_SIZE
)


def analyze_week_microstructure(ticks: np.ndarray, week_num: int = 1) -> Dict[str, Any]:
    """
    Analyzes a single week of high-frequency tick data across multiple dimensions.
    """
    n_ticks = len(ticks)
    if n_ticks < 1000:
        return {}

    ts = pd.to_datetime(ticks['timestamp'], unit='ms')
    mid = 0.5 * (ticks['bid'] + ticks['ask'])
    spread_pips = (ticks['ask'] - ticks['bid']) / PIP_SIZE

    # Build M15 bars
    m15 = build_time_bars(ticks, timeframe_min=15)
    if len(m15) < 30:
        return {}

    m15['hour'] = m15.index.hour
    m15['weekday'] = m15.index.weekday

    # 1. 24-Hour Hourly Profiles
    hourly_stats = {}
    for h in range(24):
        h_bars = m15[m15['hour'] == h]
        if len(h_bars) > 0:
            h_spread = float(h_bars['spread_pips'].mean())
            h_vol = float(h_bars['vol'].mean())
            h_range = float(h_bars['range_pips'].mean())
            h_p_vol = compute_parkinson_volatility(h_bars['high'].values, h_bars['low'].values)
            h_hurst = compute_hurst_exponent(h_bars['close'].values, max_lag=30) if len(h_bars) >= 20 else 0.5
        else:
            h_spread, h_vol, h_range, h_p_vol, h_hurst = 0.0, 0.0, 0.0, 0.0, 0.5

        hourly_stats[h] = {
            'spread_pips': h_spread,
            'tick_count': h_vol,
            'range_pips': h_range,
            'parkinson_vol': h_p_vol,
            'hurst': h_hurst
        }

    # 2. Information Compression: Range Bars vs Time Bars
    range_bars_10p = build_range_bars(ticks, range_pips=10.0)
    tick_bars_1000t = build_tick_bars(ticks, ticks_per_bar=1000)

    # 3. Overall Week Summary
    spread_m = compute_spread_metrics(ticks)
    overall_hurst = compute_hurst_exponent(m15['close'].values)
    overall_halflife = compute_ou_halflife(m15['close'].values)
    gk_vol = compute_garman_klass_volatility(m15['open'].values, m15['high'].values, m15['low'].values, m15['close'].values)

    return {
        'week': week_num,
        'ticks': n_ticks,
        'm15_bars_count': len(m15),
        'range_bars_10p_count': len(range_bars_10p),
        'tick_bars_1000t_count': len(tick_bars_1000t),
        'spread_metrics': spread_m,
        'hourly_profiles': hourly_stats,
        'overall_hurst': overall_hurst,
        'overall_halflife': overall_halflife,
        'garman_klass_vol': gk_vol,
        'total_range_pips': float((m15['high'].max() - m15['low'].min()) / PIP_SIZE),
        'net_direction_pips': float((m15['close'].iloc[-1] - m15['open'].iloc[0]) / PIP_SIZE),
    }
