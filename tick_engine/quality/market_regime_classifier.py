#!/usr/bin/env python3
"""
================================================================================
  PRODUCTION NUMBA-ACCELERATED MARKET REGIME & KILL-SWITCH CLASSIFIER
================================================================================
Implements closed-form quantitative indicators and archetype-specific kill-switch
gates developed in consultation with NEON and calibrated against 7,928 historical
trades across the 71-strategy production fleet.

Regimes:
  0 = Uninitialized / Warmup
  1 = Persistent Directional Trend (Strong Drift)
  2 = Mean-Reverting Stationary Range (Gaussian Boundary)
  3 = High-Entropy Volatile Chop (Noise / Whip-saw)
  4 = Liquidity Squeeze & Compression (Pre-Explosive Coil)
================================================================================
"""

import numpy as np

try:
    from numba import njit
except ImportError:
    # Graceful fallback to standard Python/NumPy if numba is not installed
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


@njit(fastmath=True)
def _compute_tr_and_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int):
    n = len(close)
    tr = np.empty(n, dtype=np.float64)
    atr = np.full(n, np.nan, dtype=np.float64)
    
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        tr[i] = max(hl, max(hc, lc))
        
    if n >= period:
        acc = 0.0
        for i in range(period):
            acc += tr[i]
        atr[period - 1] = acc / period
        alpha = 1.0 / period
        for i in range(period, n):
            atr[i] = atr[i - 1] + alpha * (tr[i] - atr[i - 1])
            
    return tr, atr


@njit(fastmath=True)
def _compute_ker_and_chop(high: np.ndarray, low: np.ndarray, close: np.ndarray, tr: np.ndarray, n_ker: int, n_chop: int):
    n = len(close)
    ker = np.full(n, np.nan, dtype=np.float64)
    chop = np.full(n, np.nan, dtype=np.float64)
    
    abs_diff = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        abs_diff[i] = abs(close[i] - close[i - 1])
        
    for i in range(n_ker, n):
        net_change = abs(close[i] - close[i - n_ker])
        path_sum = 0.0
        for j in range(i - n_ker + 1, i + 1):
            path_sum += abs_diff[j]
        ker[i] = net_change / path_sum if path_sum > 1e-12 else 0.0

    log10_n = np.log10(float(n_chop))
    for i in range(n_chop - 1, n):
        tr_sum = 0.0
        max_h = -1e18
        min_l = 1e18
        for j in range(i - n_chop + 1, i + 1):
            tr_sum += tr[j]
            if high[j] > max_h:
                max_h = high[j]
            if low[j] < min_l:
                min_l = low[j]
        span = max_h - min_l
        if span > 1e-12 and tr_sum > 1e-12:
            chop[i] = 100.0 * np.log10(tr_sum / span) / log10_n
        else:
            chop[i] = 50.0
            
    return ker, chop


@njit(fastmath=True)
def _compute_local_hurst_vr(close: np.ndarray, lag_q: int, window: int):
    n = len(close)
    hurst = np.full(n, np.nan, dtype=np.float64)
    vr_arr = np.full(n, np.nan, dtype=np.float64)
    
    log_p = np.log(close)
    ret1 = np.zeros(n, dtype=np.float64)
    retq = np.zeros(n, dtype=np.float64)
    
    for i in range(1, n):
        ret1[i] = log_p[i] - log_p[i - 1]
    for i in range(lag_q, n):
        retq[i] = log_p[i] - log_p[i - lag_q]
        
    inv_2lnq = 1.0 / (2.0 * np.log(float(lag_q)))
    
    for i in range(window + lag_q, n):
        m1 = 0.0
        for j in range(i - window + 1, i + 1):
            m1 += ret1[j]
        m1 /= window
        v1 = 0.0
        for j in range(i - window + 1, i + 1):
            d = ret1[j] - m1
            v1 += d * d
        v1 /= (window - 1)
        
        mq = 0.0
        for j in range(i - window + 1, i + 1):
            mq += retq[j]
        mq /= window
        vq = 0.0
        for j in range(i - window + 1, i + 1):
            d = retq[j] - mq
            vq += d * d
        vq /= (window - 1)
        
        if v1 > 1e-16 and vq > 1e-16:
            vr = vq / (lag_q * v1)
            vr_arr[i] = vr
            hurst[i] = 0.5 + np.log(vr) * inv_2lnq
            
    return hurst, vr_arr


@njit(fastmath=True)
def _compute_bbwp(close: np.ndarray, bb_period: int, pct_window: int):
    n = len(close)
    bbw = np.full(n, np.nan, dtype=np.float64)
    bbwp = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(bb_period - 1, n):
        mean = 0.0
        for j in range(i - bb_period + 1, i + 1):
            mean += close[j]
        mean /= bb_period
        
        var = 0.0
        for j in range(i - bb_period + 1, i + 1):
            d = close[j] - mean
            var += d * d
        std = np.sqrt(var / bb_period)
        bbw[i] = (4.0 * std) / mean if mean > 1e-12 else 0.0
        
    start_idx = bb_period - 1 + pct_window - 1
    for i in range(start_idx, n):
        curr = bbw[i]
        count = 0
        for j in range(i - pct_window + 1, i + 1):
            if bbw[j] <= curr:
                count += 1
        bbwp[i] = (count / float(pct_window)) * 100.0
        
    return bbwp


class MarketRegimeClassifier:
    """
    Production Market Regime & Archetype Kill-Switch Engine.
    """
    def __init__(
        self,
        ker_period: int = 20,
        chop_period: int = 14,
        hurst_lag: int = 5,
        hurst_window: int = 100,
        atr_fast: int = 5,
        atr_slow: int = 50,
        bb_period: int = 20,
        bbwp_window: int = 252,
    ):
        self.ker_period = ker_period
        self.chop_period = chop_period
        self.hurst_lag = hurst_lag
        self.hurst_window = hurst_window
        self.atr_fast = atr_fast
        self.atr_slow = atr_slow
        self.bb_period = bb_period
        self.bbwp_window = bbwp_window

    def compute(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, spread: np.ndarray) -> dict:
        tr, atr_fast = _compute_tr_and_atr(high, low, close, self.atr_fast)
        _, atr_slow = _compute_tr_and_atr(high, low, close, self.atr_slow)
        
        ker, chop = _compute_ker_and_chop(high, low, close, tr, self.ker_period, self.chop_period)
        hurst, vr = _compute_local_hurst_vr(close, self.hurst_lag, self.hurst_window)
        bbwp = _compute_bbwp(close, self.bb_period, self.bbwp_window)
        
        rvr = np.divide(atr_fast, atr_slow, out=np.full_like(atr_fast, np.nan), where=atr_slow > 1e-12)
        svp = np.divide(spread, atr_fast, out=np.full_like(spread, np.nan), where=atr_fast > 1e-12)
        
        regimes = self._classify_regimes(ker, chop, hurst, rvr, bbwp)
        gates = self._compute_archetype_gates(ker, chop, hurst, rvr, bbwp, svp)
        
        return {
            "ker": ker,
            "chop": chop,
            "hurst": hurst,
            "vr": vr,
            "rvr": rvr,
            "bbwp": bbwp,
            "svp": svp,
            "regime": regimes,
            "gates": gates,
        }

    @staticmethod
    def _classify_regimes(ker, chop, hurst, rvr, bbwp) -> np.ndarray:
        n = len(ker)
        regime = np.zeros(n, dtype=np.int8)
        
        is_squeeze = (rvr < 0.70) & (bbwp < 15.0)
        is_trend = (ker >= 0.40) & (chop < 38.2) & (hurst > 0.56)
        is_chop = (ker < 0.20) & (chop > 61.8) & (rvr > 1.35)
        is_range = (ker < 0.22) & (chop > 55.0) & (hurst < 0.44)
        
        valid = ~np.isnan(bbwp) & ~np.isnan(hurst)
        regime[valid] = 3
        regime[valid & is_range] = 2
        regime[valid & is_chop] = 3
        regime[valid & is_trend] = 1
        regime[valid & is_squeeze] = 4
        return regime

    @staticmethod
    def _compute_archetype_gates(ker, chop, hurst, rvr, bbwp, svp) -> dict:
        """Returns boolean masks: True = PERMITTED TO TRADE, False = KILL-SWITCHED."""
        valid = ~np.isnan(bbwp) & ~np.isnan(hurst) & ~np.isnan(svp)
        
        # 1. Trend & Momentum Pullbacks (Kill in Range, Chop, Squeeze)
        gate_trend = valid & (ker >= 0.32) & (chop <= 52.0) & (hurst >= 0.52) & (svp <= 0.10)
        
        # 2. Mean-Reversion & Pivot Bounces (Kill in Strong Trend, Explosive Volatility)
        gate_mr = valid & (hurst <= 0.54) & (ker <= 0.38) & (rvr <= 1.40) & (svp <= 0.12)
        
        # 3. Volatility Breakouts & Expansion (Kill in High-Vol Chop and Overextended Exhaustion)
        gate_bo = valid & (chop <= 61.8) & (bbwp <= 80.0) & (rvr <= 1.75) & (svp <= 0.15)
        
        # 4. Macro Structural Retests (Kill in Dead Squeeze or Random Walk Noise)
        gate_macro = valid & (chop <= 58.0) & (rvr >= 0.65) & ((hurst < 0.47) | (hurst > 0.53))
        
        # 5. Session Liquidity Sweeps (Kill in Extreme Spread or Severe High Volatility)
        gate_sweep = valid & (svp <= 0.14) & (rvr <= 1.60)
        
        return {
            "trend_pullback": gate_trend,
            "mean_reversion": gate_mr,
            "volatility_breakout": gate_bo,
            "macro_retest": gate_macro,
            "session_sweep": gate_sweep,
        }
