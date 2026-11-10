#!/usr/bin/env python3
"""
================================================================================
  DATA ENGINEERING SUITE: ALTERNATIVE INFORMATION-DRIVEN BAR GENERATORS
================================================================================
Numba-accelerated bar generators implementing multiple sampling paradigms:
  1. Time Bars (1m, 5m, 15m, 1h)
  2. Tick Bars (Constant information arrival)
  3. Volume / Activity Bars (Constant activity)
  4. Range Bars (Constant price excursion, H - L = constant pips)
  5. Renko Bricks (Momentum bricks with 2-brick reversals)
  6. Cumulative Tick Delta (CTD) Imbalance Bars
================================================================================
"""

from __future__ import annotations
from typing import Optional
import numpy as np
import pandas as pd
import numba

PIP_SIZE = 0.0001

BAR_DTYPE = np.dtype([
    ('start_time', np.int64),
    ('end_time', np.int64),
    ('open', np.float64),
    ('high', np.float64),
    ('low', np.float64),
    ('close', np.float64),
    ('volume', np.int64),
    ('tick_count', np.int64),
    ('buy_ticks', np.int64),
    ('sell_ticks', np.int64),
    ('delta', np.int64)
])


# ------------------------------------------------------------------------------
# 1. TIME BARS (M15, M5, H1, etc.)
# ------------------------------------------------------------------------------
def build_time_bars(ticks: np.ndarray, timeframe_min: int = 15, pip_size: Optional[float] = None) -> pd.DataFrame:
    """
    Standard high-speed time bar aggregation with tick-rule aggressor classification.
    """
    n = len(ticks)
    if n == 0:
        return pd.DataFrame()

    eff_pip_size = pip_size if pip_size is not None else (0.01 if ticks['bid'][0] > 20.0 else PIP_SIZE)

    ts = pd.to_datetime(ticks['timestamp'], unit='ms')
    mid = 0.5 * (ticks['bid'] + ticks['ask'])
    spread_pips = (ticks['ask'] - ticks['bid']) / eff_pip_size

    # Tick rule aggressor: +1 if mid > prev mid, -1 if mid < prev mid
    diff = np.diff(mid, prepend=mid[0])
    # zero-diff maintains previous direction
    signs = np.sign(diff)
    for i in range(1, len(signs)):
        if signs[i] == 0:
            signs[i] = signs[i-1]
    signs[0] = 1 if signs[0] == 0 else signs[0]

    buy_ticks = np.where(signs > 0, 1, 0)
    sell_ticks = np.where(signs < 0, 1, 0)
    delta = signs

    df = pd.DataFrame({
        'open': mid,
        'high': mid,
        'low': mid,
        'close': mid,
        'spread_pips': spread_pips,
        'vol': 1,
        'buy_vol': buy_ticks,
        'sell_vol': sell_ticks,
        'delta': delta
    }, index=ts)

    rule = f"{timeframe_min}min" if timeframe_min < 60 else f"{timeframe_min//60}h"
    bars = df.resample(rule).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'spread_pips': 'mean',
        'vol': 'count',
        'buy_vol': 'sum',
        'sell_vol': 'sum',
        'delta': 'sum'
    }).dropna()

    bars['volume'] = bars['vol']
    bars['cvd'] = bars['delta'].cumsum()
    bars['range_pips'] = (bars['high'] - bars['low']) / eff_pip_size
    bars['body_pips'] = abs(bars['close'] - bars['open']) / eff_pip_size
    return bars


# ------------------------------------------------------------------------------
# 2. NUMBA ACCELERATED TICK BARS
# ------------------------------------------------------------------------------
@numba.njit(fastmath=True)
def _numba_build_tick_bars(
    timestamps: np.ndarray,
    bids: np.ndarray,
    asks: np.ndarray,
    ticks_per_bar: int
):
    n = len(timestamps)
    n_bars = (n + ticks_per_bar - 1) // ticks_per_bar

    start_times = np.empty(n_bars, dtype=np.int64)
    end_times = np.empty(n_bars, dtype=np.int64)
    opens = np.empty(n_bars, dtype=np.float64)
    highs = np.empty(n_bars, dtype=np.float64)
    lows = np.empty(n_bars, dtype=np.float64)
    closes = np.empty(n_bars, dtype=np.float64)
    tick_counts = np.empty(n_bars, dtype=np.int64)
    buy_ticks = np.empty(n_bars, dtype=np.int64)
    sell_ticks = np.empty(n_bars, dtype=np.int64)
    deltas = np.empty(n_bars, dtype=np.int64)

    bar_idx = 0
    i = 0
    prev_mid = 0.5 * (bids[0] + asks[0])
    cur_aggressor = 1

    while i < n:
        chunk_end = min(i + ticks_per_bar, n)
        cur_mid = 0.5 * (bids[i] + asks[i])

        b_start_time = timestamps[i]
        b_end_time = timestamps[chunk_end - 1]
        b_open = cur_mid
        b_high = cur_mid
        b_low = cur_mid
        b_close = cur_mid
        b_buy = 0
        b_sell = 0

        for j in range(i, chunk_end):
            m = 0.5 * (bids[j] + asks[j])
            if m > b_high:
                b_high = m
            if m < b_low:
                b_low = m
            b_close = m

            # Tick rule
            if m > prev_mid:
                cur_aggressor = 1
            elif m < prev_mid:
                cur_aggressor = -1
            prev_mid = m

            if cur_aggressor == 1:
                b_buy += 1
            else:
                b_sell += 1

        start_times[bar_idx] = b_start_time
        end_times[bar_idx] = b_end_time
        opens[bar_idx] = b_open
        highs[bar_idx] = b_high
        lows[bar_idx] = b_low
        closes[bar_idx] = b_close
        tick_counts[bar_idx] = chunk_end - i
        buy_ticks[bar_idx] = b_buy
        sell_ticks[bar_idx] = b_sell
        deltas[bar_idx] = b_buy - b_sell

        bar_idx += 1
        i = chunk_end

    return (
        start_times[:bar_idx],
        end_times[:bar_idx],
        opens[:bar_idx],
        highs[:bar_idx],
        lows[:bar_idx],
        closes[:bar_idx],
        tick_counts[:bar_idx],
        buy_ticks[:bar_idx],
        sell_ticks[:bar_idx],
        deltas[:bar_idx]
    )


def build_tick_bars(ticks: np.ndarray, ticks_per_bar: int = 1000) -> pd.DataFrame:
    """
    Generates tick bars sampling at every N ticks (e.g. 1000 ticks/bar).
    """
    res = _numba_build_tick_bars(ticks['timestamp'], ticks['bid'], ticks['ask'], ticks_per_bar)
    df = pd.DataFrame({
        'start_time': pd.to_datetime(res[0], unit='ms'),
        'end_time': pd.to_datetime(res[1], unit='ms'),
        'open': res[2],
        'high': res[3],
        'low': res[4],
        'close': res[5],
        'tick_count': res[6],
        'buy_ticks': res[7],
        'sell_ticks': res[8],
        'delta': res[9]
    })
    df['cvd'] = df['delta'].cumsum()
    df['range_pips'] = (df['high'] - df['low']) / PIP_SIZE
    df['duration_sec'] = (res[1] - res[0]) / 1000.0
    return df


# ------------------------------------------------------------------------------
# 3. NUMBA ACCELERATED RANGE BARS (TIME-INVARIANT)
# ------------------------------------------------------------------------------
@numba.njit(fastmath=True)
def _numba_build_range_bars(
    timestamps: np.ndarray,
    bids: np.ndarray,
    asks: np.ndarray,
    range_pips: float
):
    n = len(timestamps)
    r_val = range_pips * PIP_SIZE
    max_bars = n

    start_times = np.empty(max_bars, dtype=np.int64)
    end_times = np.empty(max_bars, dtype=np.int64)
    opens = np.empty(max_bars, dtype=np.float64)
    highs = np.empty(max_bars, dtype=np.float64)
    lows = np.empty(max_bars, dtype=np.float64)
    closes = np.empty(max_bars, dtype=np.float64)
    tick_counts = np.empty(max_bars, dtype=np.int64)
    buy_ticks = np.empty(max_bars, dtype=np.int64)
    sell_ticks = np.empty(max_bars, dtype=np.int64)
    deltas = np.empty(max_bars, dtype=np.int64)

    bar_cnt = 0
    cur_mid = 0.5 * (bids[0] + asks[0])
    cur_open = cur_mid
    cur_high = cur_mid
    cur_low = cur_mid
    bar_start_ts = timestamps[0]
    t_cnt = 0
    b_buy = 0
    b_sell = 0
    prev_mid = cur_mid
    aggressor = 1

    for i in range(n):
        p = 0.5 * (bids[i] + asks[i])
        t = timestamps[i]

        if p > prev_mid:
            aggressor = 1
        elif p < prev_mid:
            aggressor = -1
        prev_mid = p

        if aggressor == 1:
            b_buy += 1
        else:
            b_sell += 1

        if p > cur_high:
            cur_high = p
        if p < cur_low:
            cur_low = p
        t_cnt += 1

        span = cur_high - cur_low
        if span >= r_val:
            start_times[bar_cnt] = bar_start_ts
            end_times[bar_cnt] = t
            opens[bar_cnt] = cur_open
            highs[bar_cnt] = cur_high
            lows[bar_cnt] = cur_low
            closes[bar_cnt] = p
            tick_counts[bar_cnt] = t_cnt
            buy_ticks[bar_cnt] = b_buy
            sell_ticks[bar_cnt] = b_sell
            deltas[bar_cnt] = b_buy - b_sell

            bar_cnt += 1
            cur_open = p
            cur_high = p
            cur_low = p
            bar_start_ts = t
            t_cnt = 0
            b_buy = 0
            b_sell = 0

    return (
        start_times[:bar_cnt],
        end_times[:bar_cnt],
        opens[:bar_cnt],
        highs[:bar_cnt],
        lows[:bar_cnt],
        closes[:bar_cnt],
        tick_counts[:bar_cnt],
        buy_ticks[:bar_cnt],
        sell_ticks[:bar_cnt],
        deltas[:bar_cnt]
    )


def build_range_bars(ticks: np.ndarray, range_pips: float = 10.0) -> pd.DataFrame:
    """
    Generates constant price excursion Range Bars (e.g. exactly 10 pips high-low).
    Completely eliminates time dilation to reveal pure market geometry.
    """
    res = _numba_build_range_bars(ticks['timestamp'], ticks['bid'], ticks['ask'], range_pips)
    df = pd.DataFrame({
        'start_time': pd.to_datetime(res[0], unit='ms'),
        'end_time': pd.to_datetime(res[1], unit='ms'),
        'open': res[2],
        'high': res[3],
        'low': res[4],
        'close': res[5],
        'tick_count': res[6],
        'buy_ticks': res[7],
        'sell_ticks': res[8],
        'delta': res[9]
    })
    df['cvd'] = df['delta'].cumsum()
    df['duration_sec'] = (res[1] - res[0]) / 1000.0
    return df
