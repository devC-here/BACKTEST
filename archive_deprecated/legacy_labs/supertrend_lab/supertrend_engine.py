"""
Supertrend Engine: High-Speed Candle Resampler and Authentic Supertrend Calculator.
Implements the exact recursive, non-repainting Supertrend formula:
- ATR calculation (Wilder's or EMA)
- Upper/Lower basic & final bands
- Trend state tracking (+1 Bullish, -1 Bearish)
- Strong candle pattern recognition
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any

def resample_ticks_to_candles(ticks: np.ndarray, timeframe: str = "5min") -> pd.DataFrame:
    """
    Resamples raw Dukascopy ticks into OHLCV candles based on mid price.
    Tracks tick volume, open time, close time, and the last tick index for each candle.
    """
    n = len(ticks)
    if n == 0:
        return pd.DataFrame()

    ts = pd.to_datetime(ticks['timestamp'], unit='ms')
    bids = ticks['bid']
    asks = ticks['ask']
    mid = 0.5 * (bids + asks)

    df_ticks = pd.DataFrame({
        'mid': mid,
        'bid': bids,
        'ask': asks,
        'tick_idx': np.arange(n, dtype=np.int64)
    }, index=ts)

    # Resample to OHLC
    res = df_ticks.resample(timeframe).agg({
        'mid': ['first', 'max', 'min', 'last', 'count'],
        'tick_idx': 'last'
    }).dropna()

    if len(res) == 0:
        return pd.DataFrame()

    candles = pd.DataFrame({
        'open': res[('mid', 'first')].to_numpy(dtype=np.float64),
        'high': res[('mid', 'max')].to_numpy(dtype=np.float64),
        'low': res[('mid', 'min')].to_numpy(dtype=np.float64),
        'close': res[('mid', 'last')].to_numpy(dtype=np.float64),
        'volume': res[('mid', 'count')].to_numpy(dtype=np.float64),
        'tick_idx': res[('tick_idx', 'last')].to_numpy(dtype=np.int64)
    }, index=res.index)

    return candles

def calculate_supertrend(
    candles: pd.DataFrame,
    atr_period: int = 10,
    multiplier: float = 3.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculates the exact, non-repainting Supertrend indicator.
    Returns:
    - supertrend_line: np.ndarray of float64
    - trend_direction: np.ndarray of int8 (+1 for Bullish, -1 for Bearish)
    - atr_values: np.ndarray of float64
    """
    n = len(candles)
    if n == 0:
        return np.array([]), np.array([]), np.array([])

    highs = candles['high'].to_numpy(dtype=np.float64)
    lows = candles['low'].to_numpy(dtype=np.float64)
    closes = candles['close'].to_numpy(dtype=np.float64)

    # 1. Calculate True Range (TR)
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = highs[0] - lows[0]
    for i in range(1, n):
        hl = highs[i] - lows[i]
        hc = abs(highs[i] - closes[i - 1])
        lc = abs(lows[i] - closes[i - 1])
        tr[i] = max(hl, max(hc, lc))

    # 2. Wilder's ATR or EMA ATR
    atr = pd.Series(tr).ewm(span=atr_period, adjust=False).mean().to_numpy()

    # 3. Supertrend Bands
    hl2 = (highs + lows) / 2.0
    basic_ub = hl2 + (multiplier * atr)
    basic_lb = hl2 - (multiplier * atr)

    final_ub = np.zeros(n, dtype=np.float64)
    final_lb = np.zeros(n, dtype=np.float64)
    trend = np.zeros(n, dtype=np.int8)
    st_line = np.zeros(n, dtype=np.float64)

    final_ub[0] = basic_ub[0]
    final_lb[0] = basic_lb[0]
    trend[0] = 1 if closes[0] >= basic_lb[0] else -1
    st_line[0] = final_lb[0] if trend[0] == 1 else final_ub[0]

    for i in range(1, n):
        # Final Upper Band
        if basic_ub[i] < final_ub[i - 1] or closes[i - 1] > final_ub[i - 1]:
            final_ub[i] = basic_ub[i]
        else:
            final_ub[i] = final_ub[i - 1]

        # Final Lower Band
        if basic_lb[i] > final_lb[i - 1] or closes[i - 1] < final_lb[i - 1]:
            final_lb[i] = basic_lb[i]
        else:
            final_lb[i] = final_lb[i - 1]

        # Trend Decision
        prev_trend = trend[i - 1]
        if prev_trend == 1:
            if closes[i] < final_lb[i]:
                cur_trend = -1
            else:
                cur_trend = 1
        else:
            if closes[i] > final_ub[i]:
                cur_trend = 1
            else:
                cur_trend = -1

        trend[i] = cur_trend
        st_line[i] = final_lb[i] if cur_trend == 1 else final_ub[i]

    return st_line, trend, atr

def identify_strong_candles(
    candles: pd.DataFrame,
    atr_values: np.ndarray,
    min_body_ratio: float = 0.55,
    min_atr_factor: float = 0.50
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Identifies strong bullish and bearish candles based on:
    1. Body Ratio: (Abs(Close - Open) / (High - Low)) >= min_body_ratio
    2. Body Size relative to ATR: Abs(Close - Open) >= min_atr_factor * ATR
    3. Proper Direction: Bullish (Close > Open), Bearish (Close < Open)
    """
    n = len(candles)
    if n == 0:
        return np.array([], dtype=bool), np.array([], dtype=bool)

    opens = candles['open'].to_numpy(dtype=np.float64)
    highs = candles['high'].to_numpy(dtype=np.float64)
    lows = candles['low'].to_numpy(dtype=np.float64)
    closes = candles['close'].to_numpy(dtype=np.float64)

    total_range = np.maximum(highs - lows, 1e-7)
    body = closes - opens
    abs_body = np.abs(body)
    body_ratio = abs_body / total_range

    is_bullish = closes > opens
    is_bearish = closes < opens

    has_body_strength = body_ratio >= min_body_ratio
    has_size_strength = abs_body >= (min_atr_factor * atr_values)

    strong_bull = is_bullish & has_body_strength & has_size_strength
    strong_bear = is_bearish & has_body_strength & has_size_strength

    return strong_bull, strong_bear
