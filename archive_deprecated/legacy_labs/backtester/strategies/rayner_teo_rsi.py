"""
Strategy 17: Rayner Teo 200 SMA + RSI(10) Swing Pullback Strategy.
Asymmetric Positive Risk-to-Reward Model (Target R:R = 1:2.0+).

Rules:
1. Macro Trend:
   - Bullish: Close > 200 SMA
   - Bearish: Close < 200 SMA
2. Pullback Extreme:
   - Bullish: RSI(10) dipped below 30 (oversold in uptrend)
   - Bearish: RSI(10) climbed above 70 (overbought in downtrend)
3. Reclaim Trigger:
   - Bullish: RSI crosses back above 30 with a bullish candle (close > open)
   - Bearish: RSI crosses back below 70 with a bearish candle (close < open)
4. Risk Management:
   - Stop Loss: Below/above recent swing extreme (buffered with ATR, min 8p, max 25p)
   - Take Profit: Strictly 1:2.0 Risk-to-Reward multiple (Reward > Risk)
"""

from typing import Optional
import pandas as pd
import numpy as np

from backtester.strategy import Strategy
from backtester.indicators import sma, rsi, highest, lowest, atr_pips


class RaynerTeoRSIStrategy(Strategy):
    name = "Rayner Teo 200 SMA + RSI Pullback (High R:R)"
    timeframe = "M15"

    def __init__(
        self,
        sma_period: int = 200,
        rsi_period: int = 10,
        rsi_oversold: float = 30.0,
        rsi_overbought: float = 70.0,
        risk_reward: float = 2.0,
        swing_lookback: int = 15,
        sl_buffer_pips: float = 2.0,
        min_sl_pips: float = 8.0,
        max_sl_pips: float = 25.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.sma_period = sma_period
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.risk_reward = risk_reward
        self.swing_lookback = swing_lookback
        self.sl_buffer_pips = sl_buffer_pips
        self.min_sl_pips = min_sl_pips
        self.max_sl_pips = max_sl_pips

    def init(self, data: pd.DataFrame) -> pd.DataFrame:
        data["sma200"] = sma(data["close"], self.sma_period)
        data["rsi10"] = rsi(data["close"], self.rsi_period)
        data["swing_high"] = highest(data["high"].shift(1), self.swing_lookback)
        data["swing_low"] = lowest(data["low"].shift(1), self.swing_lookback)
        data["atr_p"] = atr_pips(data["high"], data["low"], data["close"], 14, self.pip_size)
        return data

    def on_bar(self, index: int, row: pd.Series, prev_row: Optional[pd.Series], engine):
        if prev_row is None or index < self.sma_period + 5:
            return

        if engine.has_position():
            return

        c = float(row["close"])
        o = float(row["open"])
        sma_val = float(row["sma200"])
        curr_rsi = float(row["rsi10"])
        prev_rsi = float(prev_row["rsi10"])

        # 1. Bullish Swing Pullback Setup (Close > 200 SMA)
        if c > sma_val:
            # RSI was below oversold threshold and now hooks back up on a bullish candle
            if prev_rsi < self.rsi_oversold and curr_rsi >= self.rsi_oversold and c > o:
                swing_l = float(prev_row["swing_low"])
                sl_dist = round(((c - swing_l) / self.pip_size) + self.sl_buffer_pips, 1)

                if self.min_sl_pips <= sl_dist <= self.max_sl_pips:
                    tp_pips = round(sl_dist * self.risk_reward, 1)
                    engine.buy(
                        sl_pips=sl_dist,
                        tp_pips=tp_pips,
                        comment="RAYNER_SMA_RSI_BUY"
                    )

        # 2. Bearish Swing Pullback Setup (Close < 200 SMA)
        elif c < sma_val:
            # RSI was above overbought threshold and now hooks back down on a bearish candle
            if prev_rsi > self.rsi_overbought and curr_rsi <= self.rsi_overbought and c < o:
                swing_h = float(prev_row["swing_high"])
                sl_dist = round(((swing_h - c) / self.pip_size) + self.sl_buffer_pips, 1)

                if self.min_sl_pips <= sl_dist <= self.max_sl_pips:
                    tp_pips = round(sl_dist * self.risk_reward, 1)
                    engine.sell(
                        sl_pips=sl_dist,
                        tp_pips=tp_pips,
                        comment="RAYNER_SMA_RSI_SELL"
                    )
