"""
Supertrend Directional Strategy with Strong Candle Confirmation.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any

from tick_engine.strategies.base import (
    BaseStrategy,
    SIGNAL_HOLD,
    SIGNAL_LONG,
    SIGNAL_SHORT,
    SIGNAL_FORCE_EXIT
)
from supertrend_lab.supertrend_engine import (
    resample_ticks_to_candles,
    calculate_supertrend,
    identify_strong_candles
)


class SupertrendStrongCandleStrategy(BaseStrategy):
    """
    Supertrend Directional Strategy.
    Buys when Supertrend is Bullish (+1) AND a strong bullish candle forms.
    Sells when Supertrend is Bearish (-1) AND a strong bearish candle forms.
    Holds until opposing condition (or exit mode criteria).
    """
    def __init__(
        self,
        timeframe: str = "5min",
        atr_period: int = 10,
        multiplier: float = 3.0,
        min_body_ratio: float = 0.55,
        min_atr_factor: float = 0.50,
        exit_mode: str = "pure_reversal",  # "pure_reversal", "st_flip", "st_trailing_sl", "target_rr"
        target_rr: float = 2.0,
        sl_buffer_pips: float = 2.0,
        session_filter: bool = True,
        session_start_hour: int = 7,
        session_end_hour: int = 18,
        pip_size: float = 0.0001,
        **kwargs
    ):
        self.timeframe = str(timeframe)
        self.atr_period = int(atr_period)
        self.multiplier = float(multiplier)
        self.min_body_ratio = float(min_body_ratio)
        self.min_atr_factor = float(min_atr_factor)
        self.exit_mode = str(exit_mode)
        self.target_rr = float(target_rr)
        self.sl_buffer_pips = float(sl_buffer_pips)
        self.session_filter = bool(session_filter)
        self.session_start_hour = int(session_start_hour)
        self.session_end_hour = int(session_end_hour)
        self.pip_size = float(pip_size)

        super().__init__(**kwargs)
        self.name = f"Supertrend_{timeframe}_ATR{atr_period}_M{multiplier}_{exit_mode}"

    def generate_signals(
        self,
        ticks: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n_ticks = len(ticks)
        signals = np.zeros(n_ticks, dtype=np.int8)
        sl_pips_arr = np.zeros(n_ticks, dtype=np.float64)
        tp_pips_arr = np.zeros(n_ticks, dtype=np.float64)

        if n_ticks < 100:
            return signals, sl_pips_arr, tp_pips_arr

        bids = ticks['bid']
        asks = ticks['ask']

        # 1. Resample ticks to candles
        candles = resample_ticks_to_candles(ticks, timeframe=self.timeframe)
        n_c = len(candles)
        if n_c < self.atr_period + 5:
            return signals, sl_pips_arr, tp_pips_arr

        # 2. Compute Supertrend and Strong Candle flags
        st_line, trend, atr = calculate_supertrend(
            candles,
            atr_period=self.atr_period,
            multiplier=self.multiplier
        )
        strong_bull, strong_bear = identify_strong_candles(
            candles,
            atr_values=atr,
            min_body_ratio=self.min_body_ratio,
            min_atr_factor=self.min_atr_factor
        )

        candle_tick_indices = candles['tick_idx'].to_numpy(dtype=np.int64)
        candle_hours = candles.index.hour.to_numpy()

        pos_state = 0  # 0 = Flat, 1 = Long, -1 = Short

        for i in range(self.atr_period + 2, n_c):
            t_idx = candle_tick_indices[i]
            if t_idx + 1 >= n_ticks:
                break
            exec_idx = t_idx + 1

            h = candle_hours[i]
            in_session = (self.session_start_hour <= h < self.session_end_hour) if self.session_filter else True

            cur_trend = trend[i]
            prev_trend = trend[i - 1]
            st_val = st_line[i]

            # Entry condition: Supertrend matches trend AND strong confirmation candle
            buy_cond = (cur_trend == 1) and strong_bull[i] and in_session
            sell_cond = (cur_trend == -1) and strong_bear[i] and in_session

            # Handling exits based on exit_mode
            if pos_state == 1:
                if self.exit_mode == "pure_reversal":
                    if sell_cond:
                        # Reversal: close Long and open Short
                        signals[exec_idx] = SIGNAL_SHORT
                        sl_pips_arr[exec_idx] = (abs(bids[exec_idx] - st_val) / self.pip_size) + self.sl_buffer_pips
                        tp_pips_arr[exec_idx] = sl_pips_arr[exec_idx] * self.target_rr if self.target_rr > 0 else 0.0
                        pos_state = -1
                        continue

                elif self.exit_mode == "st_flip":
                    if cur_trend == -1:
                        # Supertrend flipped bearish: exit long
                        signals[exec_idx] = SIGNAL_FORCE_EXIT
                        pos_state = 0
                        continue

                elif self.exit_mode == "st_trailing_sl":
                    # If price drops below Supertrend lower band, exit
                    if bids[exec_idx] < st_val:
                        signals[exec_idx] = SIGNAL_FORCE_EXIT
                        pos_state = 0
                        continue

            elif pos_state == -1:
                if self.exit_mode == "pure_reversal":
                    if buy_cond:
                        # Reversal: close Short and open Long
                        signals[exec_idx] = SIGNAL_LONG
                        sl_pips_arr[exec_idx] = (abs(st_val - asks[exec_idx]) / self.pip_size) + self.sl_buffer_pips
                        tp_pips_arr[exec_idx] = sl_pips_arr[exec_idx] * self.target_rr if self.target_rr > 0 else 0.0
                        pos_state = 1
                        continue

                elif self.exit_mode == "st_flip":
                    if cur_trend == 1:
                        signals[exec_idx] = SIGNAL_FORCE_EXIT
                        pos_state = 0
                        continue

                elif self.exit_mode == "st_trailing_sl":
                    if asks[exec_idx] > st_val:
                        signals[exec_idx] = SIGNAL_FORCE_EXIT
                        pos_state = 0
                        continue

            # If Flat, check entries
            if pos_state == 0:
                if buy_cond:
                    signals[exec_idx] = SIGNAL_LONG
                    sl_dist = (abs(asks[exec_idx] - st_val) / self.pip_size) + self.sl_buffer_pips
                    sl_pips_arr[exec_idx] = max(sl_dist, 5.0)
                    tp_pips_arr[exec_idx] = sl_pips_arr[exec_idx] * self.target_rr if self.exit_mode == "target_rr" else 0.0
                    pos_state = 1

                elif sell_cond:
                    signals[exec_idx] = SIGNAL_SHORT
                    sl_dist = (abs(st_val - bids[exec_idx]) / self.pip_size) + self.sl_buffer_pips
                    sl_pips_arr[exec_idx] = max(sl_dist, 5.0)
                    tp_pips_arr[exec_idx] = sl_pips_arr[exec_idx] * self.target_rr if self.exit_mode == "target_rr" else 0.0
                    pos_state = -1

        return signals, sl_pips_arr, tp_pips_arr
