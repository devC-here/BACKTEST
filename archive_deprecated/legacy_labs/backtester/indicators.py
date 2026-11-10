"""
Vectorized Technical Indicators optimized for Pandas and NumPy.
Fast, reliable, zero third-party C-bindings required.
"""

import numpy as np
import pandas as pd


def sma(series: pd.Series, period: int) -> pd.Series:
    """Simple Moving Average."""
    return series.rolling(window=period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential Moving Average."""
    return series.ewm(span=period, adjust=False).mean()


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """True Range calculation."""
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average True Range (Wilder's exponential smoothing)."""
    tr = true_range(high, low, close)
    return tr.ewm(alpha=1.0 / period, adjust=False).mean()


def atr_pips(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14, pip_size: float = 0.0001) -> pd.Series:
    """ATR denominated directly in pips."""
    return atr(high, low, close, period=period) / pip_size


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index (RSI)."""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0.0)).ewm(alpha=1.0 / period, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0.0)).ewm(alpha=1.0 / period, adjust=False).mean()
    rs = gain / (loss.replace(0, np.nan))
    rsi_val = 100.0 - (100.0 / (1.0 + rs))
    return rsi_val.fillna(50.0)


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """Moving Average Convergence Divergence (MACD)."""
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def bollinger_bands(series: pd.Series, period: int = 20, std_dev: float = 2.0):
    """Bollinger Bands (Middle, Upper, Lower)."""
    middle = sma(series, period)
    std = series.rolling(window=period).std()
    upper = middle + (std * std_dev)
    lower = middle - (std * std_dev)
    return upper, middle, lower


def highest(series: pd.Series, period: int) -> pd.Series:
    """Rolling highest value over period."""
    return series.rolling(window=period).max()


def lowest(series: pd.Series, period: int) -> pd.Series:
    """Rolling lowest value over period."""
    return series.rolling(window=period).min()


def stochastic(high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 14, d_period: int = 3):
    """Stochastic Oscillator %K and %D."""
    lowest_low = lowest(low, k_period)
    highest_high = highest(high, k_period)
    denom = (highest_high - lowest_low).replace(0, np.nan)
    k = 100.0 * ((close - lowest_low) / denom)
    k = k.fillna(50.0)
    d = sma(k, d_period)
    return k, d


def supertrend(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 10, multiplier: float = 3.0):
    """Supertrend indicator (trend direction 1=BUY, -1=SELL and line level)."""
    atr_val = atr(high, low, close, period)
    hl2 = (high + low) / 2.0

    basic_upper = hl2 + (multiplier * atr_val)
    basic_lower = hl2 - (multiplier * atr_val)

    n = len(close)
    final_upper = np.zeros(n)
    final_lower = np.zeros(n)
    st = np.zeros(n)
    trend = np.zeros(n, dtype=int)

    c = close.values
    bu = basic_upper.values
    bl = basic_lower.values

    for i in range(1, n):
        if bu[i] < final_upper[i - 1] or c[i - 1] > final_upper[i - 1]:
            final_upper[i] = bu[i]
        else:
            final_upper[i] = final_upper[i - 1]

        if bl[i] > final_lower[i - 1] or c[i - 1] < final_lower[i - 1]:
            final_lower[i] = bl[i]
        else:
            final_lower[i] = final_lower[i - 1]

        prev_trend = trend[i - 1] if i > 1 else 1
        if prev_trend == 1:
            if c[i] < final_lower[i]:
                trend[i] = -1
                st[i] = final_upper[i]
            else:
                trend[i] = 1
                st[i] = final_lower[i]
        else:
            if c[i] > final_upper[i]:
                trend[i] = 1
                st[i] = final_lower[i]
            else:
                trend[i] = -1
                st[i] = final_upper[i]

    return pd.Series(trend, index=close.index), pd.Series(st, index=close.index)
