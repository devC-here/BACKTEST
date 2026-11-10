"""
Strategy Base Interface and Signal Constants.
Defines standard strategy signatures for tick-level evaluation.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np

# Standard Strategy Signal Actions
SIGNAL_HOLD = 0          # 0: Do nothing / Hold current state
SIGNAL_LONG = 1          # 1: Enter Long (fill at current Ask)
SIGNAL_SHORT = -1        # -1: Enter Short (fill at current Bid)
SIGNAL_FORCE_EXIT = 2    # 2: Force Exit Active Position (Long at Bid, Short at Ask)


class BaseStrategy(ABC):
    """
    Abstract base class for tick-level trading strategies.
    Ensures clean separation between strategy signal generation and execution engine.
    """
    name: str = "BaseStrategy"
    sl_pips: float = 20.0
    tp_pips: float = 40.0

    def __init__(self, sl_pips: float = 20.0, tp_pips: float = 40.0, **kwargs):
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        for key, val in kwargs.items():
            setattr(self, key, val)

    @abstractmethod
    def generate_signals(self, ticks: np.ndarray) -> np.ndarray:
        """
        Processes structured tick array and returns contiguous int8 array of signals:
        - 0: Hold
        - 1: Enter Long
        - -1: Enter Short
        - 2: Force Exit

        Parameters:
            ticks: np.ndarray structured array with TICK_DTYPE ('timestamp', 'bid', 'ask')

        Returns:
            np.ndarray: 1D array of dtype np.int8 with length equal to len(ticks)
        """
        raise NotImplementedError("Strategies must implement generate_signals(ticks)")
