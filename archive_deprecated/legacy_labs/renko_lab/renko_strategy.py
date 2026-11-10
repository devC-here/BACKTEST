"""
Renko SMA-10 + OBV Trend Strategy.
User Specification:
- Buy (Long) : Green Renko brick forms above 10 SMA, with both 10 SMA and OBV rising.
- Sell (Short): Red Renko brick forms below 10 SMA, with both 10 SMA and OBV falling.
"""

from __future__ import annotations
import sys
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tick_engine.strategies.base import (
    BaseStrategy,
    SIGNAL_HOLD,
    SIGNAL_LONG,
    SIGNAL_SHORT,
    SIGNAL_FORCE_EXIT
)
from renko_lab.renko_builder import build_renko_bricks, compute_renko_indicators


class RenkoSMAOBVStrategy(BaseStrategy):
    """
    Renko Trend Strategy with 10 SMA and On-Balance Volume (OBV).
    """
    def __init__(
        self,
        brick_size_pips: float = 5.0,
        sma_period: int = 10,
        sl_bricks: float = 2.0,       # Stop loss in multiples of brick_size
        tp_bricks: float = 3.0,       # Take profit in multiples of brick_size
        pip_size: float = 0.0001,
        exit_on_opposite_brick: bool = False,
        **kwargs
    ):
        self.brick_size_pips = float(brick_size_pips)
        self.sma_period = int(sma_period)
        self.sl_bricks = float(sl_bricks)
        self.tp_bricks = float(tp_bricks)
        self.pip_size = float(pip_size)
        self.exit_on_opposite_brick = exit_on_opposite_brick

        super().__init__(**kwargs)
        self.name = f"Renko_SMA{self.sma_period}_OBV_{int(self.brick_size_pips)}p"

    def generate_signals(self, ticks: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n_ticks = len(ticks)
        signals = np.zeros(n_ticks, dtype=np.int8)
        sl_pips_arr = np.zeros(n_ticks, dtype=np.float64)
        tp_pips_arr = np.zeros(n_ticks, dtype=np.float64)

        if n_ticks < 100:
            return signals, sl_pips_arr, tp_pips_arr

        # 1. Build Renko Bricks from ticks
        bricks = build_renko_bricks(ticks, brick_size_pips=self.brick_size_pips, pip_size=self.pip_size)
        n_bricks = len(bricks)

        if n_bricks < self.sma_period + 2:
            return signals, sl_pips_arr, tp_pips_arr

        # 2. Compute 10 SMA and OBV
        closes, sma, obv = compute_renko_indicators(bricks, sma_period=self.sma_period)

        sl_pips_val = self.sl_bricks * self.brick_size_pips
        tp_pips_val = self.tp_bricks * self.brick_size_pips

        # 3. Evaluate Trading Rules on Brick Completion
        for i in range(self.sma_period, n_bricks):
            b = bricks[i]
            tick_idx = b.tick_idx

            if tick_idx >= n_ticks:
                continue

            # Indicators state
            cur_close = closes[i]
            cur_sma = sma[i]
            prev_sma = sma[i - 1]
            cur_obv = obv[i]
            prev_obv = obv[i - 1]

            if np.isnan(cur_sma) or np.isnan(prev_sma):
                continue

            sma_rising = cur_sma > prev_sma
            sma_falling = cur_sma < prev_sma
            obv_rising = cur_obv > prev_obv
            obv_falling = cur_obv < prev_obv

            # BUY CONDITION:
            # - Green brick forms (direction == 1)
            # - Brick close is above 10 SMA
            # - 10 SMA is going up
            # - OBV is going up
            if b.direction == 1 and cur_close > cur_sma and sma_rising and obv_rising:
                signals[tick_idx] = SIGNAL_LONG
                sl_pips_arr[tick_idx] = sl_pips_val
                tp_pips_arr[tick_idx] = tp_pips_val

            # SELL CONDITION:
            # - Red brick forms (direction == -1)
            # - Brick close is below 10 SMA
            # - 10 SMA is going down
            # - OBV is going down
            elif b.direction == -1 and cur_close < cur_sma and sma_falling and obv_falling:
                signals[tick_idx] = SIGNAL_SHORT
                sl_pips_arr[tick_idx] = sl_pips_val
                tp_pips_arr[tick_idx] = tp_pips_val

            elif self.exit_on_opposite_brick:
                # Optional exit on brick direction reversal
                if b.direction != bricks[i - 1].direction:
                    signals[tick_idx] = SIGNAL_FORCE_EXIT

        return signals, sl_pips_arr, tp_pips_arr
