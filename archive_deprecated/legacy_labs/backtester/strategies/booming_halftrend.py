"""
Strategy 51: Booming Bulls HalfTrend + DMI Intraday Breakout Strategy.
Asymmetric Positive Risk-to-Reward Model (Target R:R = 1:2.0+).

Rules:
1. HalfTrend (Amplitude=2, Channel Deviation=2.0):
   - Bullish Trend: HalfTrend line is green / below price.
   - Bearish Trend: HalfTrend line is red / above price.
2. DMI Momentum Filter:
   - Bullish: +DI > -DI
   - Bearish: -DI > +DI
3. Risk Management:
   - Stop Loss: Placed at HalfTrend line / swing extreme (min 8p, max 25p).
   - Take Profit: Strictly 1:2.0 Risk-to-Reward Ratio (Reward > Risk).
"""

from typing import Optional
import pandas as pd
import numpy as np

from backtester.strategy import Strategy
from backtester.indicators import atr, highest, lowest, atr_pips


class BoomingHalfTrendStrategy(Strategy):
    name = "Booming Bulls HalfTrend + DMI (High R:R)"
    timeframe = "M15"

    def __init__(
        self,
        amplitude: int = 2,
        channel_deviation: float = 2.0,
        dmi_period: int = 14,
        risk_reward: float = 2.0,
        swing_lookback: int = 15,
        sl_buffer_pips: float = 2.0,
        min_sl_pips: float = 8.0,
        max_sl_pips: float = 25.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.amplitude = amplitude
        self.channel_deviation = channel_deviation
        self.dmi_period = dmi_period
        self.risk_reward = risk_reward
        self.swing_lookback = swing_lookback
        self.sl_buffer_pips = sl_buffer_pips
        self.min_sl_pips = min_sl_pips
        self.max_sl_pips = max_sl_pips

    def init(self, data: pd.DataFrame) -> pd.DataFrame:
        high = data["high"]
        low = data["low"]
        close = data["close"]

        # DMI calculation
        up_move = high.diff()
        down_move = -low.diff()
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

        tr = atr(high, low, close, self.dmi_period)
        plus_di = 100.0 * pd.Series(plus_dm, index=data.index).ewm(alpha=1.0 / self.dmi_period, adjust=False).mean() / tr.replace(0, np.nan)
        minus_di = 100.0 * pd.Series(minus_dm, index=data.index).ewm(alpha=1.0 / self.dmi_period, adjust=False).mean() / tr.replace(0, np.nan)

        data["plus_di"] = plus_di.fillna(0.0)
        data["minus_di"] = minus_di.fillna(0.0)

        # HalfTrend simplified vectorized tracking
        atr_val = atr(high, low, close, 100)
        dev = self.channel_deviation * atr_val
        high_low_half = (highest(high, self.amplitude) + lowest(low, self.amplitude)) / 2.0
        data["ht_upper"] = high_low_half + dev
        data["ht_lower"] = high_low_half - dev
        data["ht_mid"] = high_low_half

        data["swing_high"] = highest(high.shift(1), self.swing_lookback)
        data["swing_low"] = lowest(low.shift(1), self.swing_lookback)
        data["atr_p"] = atr_pips(high, low, close, 14, self.pip_size)
        return data

    def on_bar(self, index: int, row: pd.Series, prev_row: Optional[pd.Series], engine):
        if prev_row is None or index < self.dmi_period * 2 + 5:
            return

        if engine.has_position():
            return

        c = float(row["close"])
        o = float(row["open"])
        plus_di = float(row["plus_di"])
        minus_di = float(row["minus_di"])
        prev_plus = float(prev_row["plus_di"])
        prev_minus = float(prev_row["minus_di"])
        ht_mid = float(row["ht_mid"])

        # Bullish Trigger: +DI crosses above -DI while price is above HalfTrend midpoint
        if plus_di > minus_di and prev_plus <= prev_minus and c > ht_mid and c > o:
            swing_l = float(prev_row["swing_low"])
            sl_dist = round(((c - min(swing_l, ht_mid)) / self.pip_size) + self.sl_buffer_pips, 1)

            if self.min_sl_pips <= sl_dist <= self.max_sl_pips:
                tp_pips = round(sl_dist * self.risk_reward, 1)
                engine.buy(
                    sl_pips=sl_dist,
                    tp_pips=tp_pips,
                    comment="BOOMING_HALFTREND_BUY"
                )

        # Bearish Trigger: -DI crosses above +DI while price is below HalfTrend midpoint
        elif minus_di > plus_di and prev_minus <= prev_plus and c < ht_mid and c < o:
            swing_h = float(prev_row["swing_high"])
            sl_dist = round(((max(swing_h, ht_mid) - c) / self.pip_size) + self.sl_buffer_pips, 1)

            if self.min_sl_pips <= sl_dist <= self.max_sl_pips:
                tp_pips = round(sl_dist * self.risk_reward, 1)
                engine.sell(
                    sl_pips=sl_dist,
                    tp_pips=tp_pips,
                    comment="BOOMING_HALFTREND_SELL"
                )
