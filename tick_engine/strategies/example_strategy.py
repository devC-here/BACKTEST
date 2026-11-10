"""
Example Tick-Level Trading Strategy: Fast/Slow EMA Crossover.
Demonstrates stateful signal generation optimized with Numba JIT.
"""

from __future__ import annotations
import numpy as np
try:
    import numba
    _njit = numba.njit(fastmath=True, nogil=True)
except Exception:
    def _njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]):
            return args[0]
        def dec(fn):
            return fn
        return dec
from .base import BaseStrategy, SIGNAL_HOLD, SIGNAL_LONG, SIGNAL_SHORT, SIGNAL_FORCE_EXIT


@_njit
def compute_ema_crossover_signals(
    bids: np.ndarray,
    asks: np.ndarray,
    fast_period: int,
    slow_period: int
) -> np.ndarray:
    """
    Fast JIT computation of tick-by-tick EMA crossover signals.
    """
    n = len(bids)
    signals = np.zeros(n, dtype=np.int8)

    if n == 0:
        return signals

    alpha_fast = 2.0 / (fast_period + 1.0)
    alpha_slow = 2.0 / (slow_period + 1.0)

    # Initialize EMA with first mid-price
    mid_0 = 0.5 * (bids[0] + asks[0])
    ema_fast = mid_0
    ema_slow = mid_0
    prev_fast = mid_0
    prev_slow = mid_0

    for i in range(1, n):
        mid = 0.5 * (bids[i] + asks[i])
        ema_fast = alpha_fast * mid + (1.0 - alpha_fast) * ema_fast
        ema_slow = alpha_slow * mid + (1.0 - alpha_slow) * ema_slow

        # Check for crossover
        if prev_fast <= prev_slow and ema_fast > ema_slow:
            signals[i] = SIGNAL_LONG
        elif prev_fast >= prev_slow and ema_fast < ema_slow:
            signals[i] = SIGNAL_SHORT
        else:
            signals[i] = SIGNAL_HOLD

        prev_fast = ema_fast
        prev_slow = ema_slow

    return signals


class EMACrossoverStrategy(BaseStrategy):
    """
    Fast Tick EMA Crossover Strategy.
    """
    name = "EMACrossover"

    def __init__(
        self,
        fast_period: int = 150,
        slow_period: int = 600,
        sl_pips: float = 15.0,
        tp_pips: float = 30.0,
        **kwargs
    ):
        super().__init__(sl_pips=sl_pips, tp_pips=tp_pips, **kwargs)
        self.fast_period = int(fast_period)
        self.slow_period = int(slow_period)

    def generate_signals(self, ticks: np.ndarray) -> np.ndarray:
        bids = np.ascontiguousarray(ticks['bid'], dtype=np.float64)
        asks = np.ascontiguousarray(ticks['ask'], dtype=np.float64)

        return compute_ema_crossover_signals(
            bids,
            asks,
            self.fast_period,
            self.slow_period
        )
