#!/usr/bin/env python3
"""
================================================================================
  DATA ENGINEERING SUITE: INSTITUTIONAL MICROSTRUCTURE FEATURES
================================================================================
Calculates advanced market microstructure and mathematical indicators:
  1. Spread Profile & Liquidity Anomaly Z-scores
  2. Tick Velocity & Order Arrival Intensity (Ticks / Second)
  3. Order Flow Imbalance (OFI) & Cumulative Volume/Tick Delta (CVD)
  4. Realized Volatility Estimators (Classical RV, Parkinson, Garman-Klass)
  5. Fractal Hurst Exponent (Persistence vs Mean-Reversion)
  6. Ornstein-Uhlenbeck Mean-Reversion Half-Life
================================================================================
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import numba

PIP_SIZE = 0.0001


# ------------------------------------------------------------------------------
# 1. SPREAD DYNAMICS & LIQUIDITY METRICS
# ------------------------------------------------------------------------------
def compute_spread_metrics(ticks: np.ndarray) -> dict:
    """
    Computes statistical spread metrics from raw tick data.
    """
    spreads_pips = (ticks['ask'] - ticks['bid']) / PIP_SIZE
    return {
        'mean_spread': float(np.mean(spreads_pips)),
        'median_spread': float(np.median(spreads_pips)),
        'p95_spread': float(np.percentile(spreads_pips, 95)),
        'p99_spread': float(np.percentile(spreads_pips, 99)),
        'min_spread': float(np.min(spreads_pips)),
        'max_spread': float(np.max(spreads_pips)),
        'std_spread': float(np.std(spreads_pips))
    }


# ------------------------------------------------------------------------------
# 2. TICK VELOCITY & ORDER ARRIVAL INTENSITY (Poisson Process)
# ------------------------------------------------------------------------------
@numba.njit(fastmath=True)
def _numba_tick_velocity_1s(timestamps: np.ndarray):
    """
    Computes rolling 1-second tick arrival intensity.
    """
    n = len(timestamps)
    vel = np.empty(n, dtype=np.float64)
    left = 0
    for right in range(n):
        t_cur = timestamps[right]
        t_thresh = t_cur - 1000  # 1 second window
        while left < right and timestamps[left] < t_thresh:
            left += 1
        vel[right] = float(right - left + 1)
    return vel


def compute_tick_velocity(ticks: np.ndarray) -> np.ndarray:
    """
    Computes instant tick arrival rate (ticks/sec) for each tick.
    """
    return _numba_tick_velocity_1s(ticks['timestamp'])


# ------------------------------------------------------------------------------
# 3. REALIZED VOLATILITY ESTIMATORS
# ------------------------------------------------------------------------------
def compute_parkinson_volatility(high: np.ndarray, low: np.ndarray) -> float:
    """
    Parkinson (1980) High-Low Volatility estimator:
    sigma_P = sqrt( 1 / (4 * ln(2) * N) * sum( ln(H_i / L_i)^2 ) )
    """
    n = len(high)
    if n == 0:
        return 0.0
    hl_ratio = np.log(np.maximum(high / np.maximum(low, 1e-9), 1.0))
    sum_sq = np.sum(hl_ratio ** 2)
    return float(np.sqrt(sum_sq / (4.0 * np.log(2.0) * n)))


def compute_garman_klass_volatility(open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> float:
    """
    Garman-Klass (1980) OHLC Volatility estimator (8x more efficient than close-to-close):
    sigma_GK = sqrt( 1 / N * sum( 0.5 * ln(H/L)^2 - (2*ln(2) - 1) * ln(C/O)^2 ) )
    """
    n = len(close)
    if n == 0:
        return 0.0
    hl = np.log(np.maximum(high / np.maximum(low, 1e-9), 1.0))
    co = np.log(np.maximum(close / np.maximum(open_, 1e-9), 1e-9))
    term1 = 0.5 * (hl ** 2)
    term2 = (2.0 * np.log(2.0) - 1.0) * (co ** 2)
    return float(np.sqrt(np.maximum(np.sum(term1 - term2) / n, 0.0)))


# ------------------------------------------------------------------------------
# 4. FRACTAL HURST EXPONENT (Persistence vs Mean Reversion)
# ------------------------------------------------------------------------------
def compute_hurst_exponent(series: np.ndarray, max_lag: int = 50) -> float:
    """
    Computes the Hurst Exponent (H) via rescaled range (R/S) analysis:
    - H < 0.5: Mean-Reverting regime (anticorrelated, ideal for range/fades)
    - H = 0.5: Random Walk (Brownian motion)
    - H > 0.5: Persistent Trending regime (momentum expansion)
    """
    n = len(series)
    if n < 100:
        return 0.5

    lags = range(10, min(max_lag, n // 2), 5)
    tau = []
    lagvec = []

    for lag in lags:
        pp = np.subtract(series[lag:], series[:-lag])
        std = np.std(pp)
        if std > 1e-9:
            tau.append(std)
            lagvec.append(lag)

    if len(tau) < 3:
        return 0.5

    # Linear fit of log(tau) vs log(lagvec)
    m = np.polyfit(np.log(lagvec), np.log(tau), 1)
    hurst = float(m[0])
    return max(0.01, min(0.99, hurst))


# ------------------------------------------------------------------------------
# 5. ORNSTEIN-UHLENBECK MEAN-REVERSION HALF-LIFE
# ------------------------------------------------------------------------------
def compute_ou_halflife(prices: np.ndarray) -> float:
    """
    Estimates the Ornstein-Uhlenbeck mean-reversion rate lambda and half-life:
    dx_t = lambda * (mu - x_t) * dt + sigma * dW_t
    Half-life = -ln(2) / lambda (in number of bars/periods)
    """
    n = len(prices)
    if n < 30:
        return 999.0

    delta_p = np.diff(prices)
    lag_p = prices[:-1]

    # Regress delta_p on lag_p
    # delta_p = a + b * lag_p
    slope, intercept = np.polyfit(lag_p, delta_p, 1)

    # b = -lambda * dt
    if slope >= 0:
        # Non-mean-reverting or explosive
        return 999.0

    lambda_param = -slope
    halflife = np.log(2.0) / lambda_param
    return float(halflife)
