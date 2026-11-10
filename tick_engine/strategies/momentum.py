"""
Tick Momentum Strategy.
Enters in the direction of strong tick price momentum over a short tick lookback window.
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
def compute_tick_momentum_signals(
    bids: np.ndarray,
    asks: np.ndarray,
    lookback: int,
    threshold_pips: float,
    pip_size: float
) -> np.ndarray:
    """
    Computes momentum breakout signals based on price velocity over `lookback` ticks.
    """
    n = len(bids)
    signals = np.zeros(n, dtype=np.int8)

    if n <= lookback:
        return signals

    threshold_price = threshold_pips * pip_size

    for i in range(lookback, n):
        # Current mid vs lookback mid
        curr_mid = 0.5 * (bids[i] + asks[i])
        prev_mid = 0.5 * (bids[i - lookback] + asks[i - lookback])
        delta = curr_mid - prev_mid

        if delta >= threshold_price:
            signals[i] = SIGNAL_LONG
        elif delta <= -threshold_price:
            signals[i] = SIGNAL_SHORT
        else:
            signals[i] = SIGNAL_HOLD

    return signals


class TickMomentumStrategy(BaseStrategy):
    """
    Tick-by-tick Momentum Strategy.
    """
    name = "TickMomentum"

    def __init__(
        self,
        lookback: int = 50,
        threshold_pips: float = 3.0,
        sl_pips: float = 12.0,
        tp_pips: float = 24.0,
        pip_size: float = 0.0001,
        **kwargs
    ):
        super().__init__(sl_pips=sl_pips, tp_pips=tp_pips, **kwargs)
        self.lookback = int(lookback)
        self.threshold_pips = float(threshold_pips)
        self.pip_size = float(pip_size)

    def generate_signals(self, ticks: np.ndarray) -> np.ndarray:
        bids = np.ascontiguousarray(ticks['bid'], dtype=np.float64)
        asks = np.ascontiguousarray(ticks['ask'], dtype=np.float64)

        return compute_tick_momentum_signals(
            bids,
            asks,
            self.lookback,
            self.threshold_pips,
            self.pip_size
        )
