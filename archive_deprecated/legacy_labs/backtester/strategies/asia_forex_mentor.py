"""
Strategy 34: Asia Forex Mentor (Ezekiel Chew) Smoothed Heikin Ashi.
Asymmetric Positive Risk-to-Reward Model (Target R:R = 1:2.0+).

Rules:
1. Double Smoothed Heikin Ashi calculation.
2. Trend Direction:
   - Bullish: Smoothed HA Close > Smoothed HA Open (Green)
   - Bearish: Smoothed HA Close < Smoothed HA Open (Red)
3. Pullback & Rejection:
   - Price tests the Smoothed HA baseline and closes with strong confirmation.
4. Risk Management:
   - Stop Loss: Below/above recent swing low/high.
   - Take Profit: 1:2.0 Risk-to-Reward Ratio (Reward > Risk).
"""

from typing import Optional
import pandas as pd
import numpy as np

from backtester.strategy import Strategy
from backtester.indicators import ema, highest, lowest, atr_pips


class AsiaForexMentorStrategy(Strategy):
    name = "Asia Forex Mentor Smoothed Heikin Ashi (High R:R)"
    timeframe = "M15"

    def __init__(
        self,
        smooth_period: int = 10,
        risk_reward: float = 2.0,
        swing_lookback: int = 15,
        sl_buffer_pips: float = 2.0,
        min_sl_pips: float = 8.0,
        max_sl_pips: float = 25.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.smooth_period = smooth_period
        self.risk_reward = risk_reward
        self.swing_lookback = swing_lookback
        self.sl_buffer_pips = sl_buffer_pips
        self.min_sl_pips = min_sl_pips
        self.max_sl_pips = max_sl_pips

    def init(self, data: pd.DataFrame) -> pd.DataFrame:
        # Pre-smooth prices
        o = ema(data["open"], self.smooth_period)
        c = ema(data["close"], self.smooth_period)
        h = ema(data["high"], self.smooth_period)
        l = ema(data["low"], self.smooth_period)

        ha_close = (o + h + l + c) / 4.0
        ha_open = pd.Series(index=data.index, dtype=float)
        ha_open.iloc[0] = (data["open"].iloc[0] + data["close"].iloc[0]) / 2.0

        for i in range(1, len(data)):
            ha_open.iloc[i] = (ha_open.iloc[i - 1] + ha_close.iloc[i - 1]) / 2.0

        # Post-smooth Heikin Ashi
        data["ha_open_smooth"] = ema(ha_open, self.smooth_period)
        data["ha_close_smooth"] = ema(ha_close, self.smooth_period)

        data["swing_high"] = highest(data["high"].shift(1), self.swing_lookback)
        data["swing_low"] = lowest(data["low"].shift(1), self.swing_lookback)
        data["atr_p"] = atr_pips(data["high"], data["low"], data["close"], 14, self.pip_size)
        return data

    def on_bar(self, index: int, row: pd.Series, prev_row: Optional[pd.Series], engine):
        if prev_row is None or index < self.smooth_period * 2 + 5:
            return

        if engine.has_position():
            return

        c = float(row["close"])
        o = float(row["open"])
        curr_ha_c = float(row["ha_close_smooth"])
        curr_ha_o = float(row["ha_open_smooth"])
        prev_ha_c = float(prev_row["ha_close_smooth"])
        prev_ha_o = float(prev_row["ha_open_smooth"])

        # Bullish: HA turns green or continues green, price bounces above baseline
        if curr_ha_c > curr_ha_o:
            if prev_ha_c <= prev_ha_o or (c > o and float(row["low"]) <= curr_ha_o and c > curr_ha_c):
                swing_l = float(prev_row["swing_low"])
                sl_dist = round(((c - swing_l) / self.pip_size) + self.sl_buffer_pips, 1)

                if self.min_sl_pips <= sl_dist <= self.max_sl_pips:
                    tp_pips = round(sl_dist * self.risk_reward, 1)
                    engine.buy(
                        sl_pips=sl_dist,
                        tp_pips=tp_pips,
                        comment="AFM_HA_BULL_BUY"
                    )

        # Bearish: HA turns red or continues red, price rejects below baseline
        elif curr_ha_c < curr_ha_o:
            if prev_ha_c >= prev_ha_o or (c < o and float(row["high"]) >= curr_ha_o and c < curr_ha_c):
                swing_h = float(prev_row["swing_high"])
                sl_dist = round(((swing_h - c) / self.pip_size) + self.sl_buffer_pips, 1)

                if self.min_sl_pips <= sl_dist <= self.max_sl_pips:
                    tp_pips = round(sl_dist * self.risk_reward, 1)
                    engine.sell(
                        sl_pips=sl_dist,
                        tp_pips=tp_pips,
                        comment="AFM_HA_BEAR_SELL"
                    )
