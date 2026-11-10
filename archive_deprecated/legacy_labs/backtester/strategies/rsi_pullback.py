"""
50 EMA Trend + RSI Pullback Strategy.
Macro Trend: 50 EMA
Momentum Trigger: RSI(14) oversold (<35) / overbought (>65) recovery
Risk Management: Standard 15.0 pip SL, 30.0 pip TP (1:2.0 RR).
"""

from typing import Optional
import pandas as pd
from backtester.strategy import Strategy
from backtester.indicators import ema, rsi


class RSIPullbackStrategy(Strategy):
    name = "50 EMA + RSI Pullback"
    timeframe = "M5"

    def __init__(
        self,
        ema_period: int = 50,
        rsi_period: int = 14,
        sl_pips: float = 15.0,
        tp_pips: float = 30.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.ema_period = ema_period
        self.rsi_period = rsi_period
        self.sl_pips = sl_pips
        self.tp_pips = tp_pips

    def init(self, data: pd.DataFrame) -> pd.DataFrame:
        data["ema50"] = ema(data["close"], self.ema_period)
        data["rsi"] = rsi(data["close"], self.rsi_period)
        return data

    def on_bar(self, index: int, row: pd.Series, prev_row: Optional[pd.Series], engine):
        if prev_row is None or pd.isna(row["ema50"]) or pd.isna(row["rsi"]):
            return

        if engine.has_position():
            return

        c = float(row["close"])
        ema_val = float(row["ema50"])
        curr_rsi = float(row["rsi"])
        prev_rsi = float(prev_row["rsi"])

        # Long: Above 50 EMA, RSI crosses up through 35 from oversold
        if c > ema_val and prev_rsi < 35.0 and curr_rsi >= 35.0:
            engine.buy(sl_pips=self.sl_pips, tp_pips=self.tp_pips, comment="RSI_OVERSOLD_RECLAIM_BUY")

        # Short: Below 50 EMA, RSI crosses down through 65 from overbought
        elif c < ema_val and prev_rsi > 65.0 and curr_rsi <= 65.0:
            engine.sell(sl_pips=self.sl_pips, tp_pips=self.tp_pips, comment="RSI_OVERBOUGHT_REJECT_SELL")
