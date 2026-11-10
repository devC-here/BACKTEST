"""
20/200 Dual EMA + 2x ATR Dynamic Stop Loss Strategy.
Macro Trend: 200 EMA
Micro Pullback Trigger: 20 EMA Reclaim
Risk Management: 2x ATR Volatility Envelope SL, 1:2.0 Risk-to-Reward TP.
"""

from typing import Optional
import pandas as pd
from backtester.strategy import Strategy
from backtester.indicators import ema, atr_pips


class DualEMAStrategy(Strategy):
    name = "20/200 Dual EMA + ATR Envelope"
    timeframe = "M5"

    def __init__(
        self,
        ema_fast: int = 20,
        ema_slow: int = 200,
        atr_period: int = 14,
        atr_multiplier: float = 2.0,
        risk_reward: float = 2.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.ema_fast_len = ema_fast
        self.ema_slow_len = ema_slow
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
        self.risk_reward = risk_reward

    def init(self, data: pd.DataFrame) -> pd.DataFrame:
        data["ema_fast"] = ema(data["close"], self.ema_fast_len)
        data["ema_slow"] = ema(data["close"], self.ema_slow_len)
        data["atr_p"] = atr_pips(data["high"], data["low"], data["close"], self.atr_period, self.pip_size)
        return data

    def on_bar(self, index: int, row: pd.Series, prev_row: Optional[pd.Series], engine):
        if prev_row is None or pd.isna(row["ema_slow"]) or pd.isna(row["atr_p"]):
            return

        if engine.has_position():
            return

        c = float(row["close"])
        ef = float(row["ema_fast"])
        es = float(row["ema_slow"])
        prev_c = float(prev_row["close"])
        prev_ef = float(prev_row["ema_fast"])

        atr_val = max(float(row["atr_p"]), 5.0)  # Minimum 5 pips stop
        sl_pips = round(atr_val * self.atr_multiplier, 1)
        tp_pips = round(sl_pips * self.risk_reward, 1)

        # Bullish: Macro uptrend (c > es), previous candle tested below/at fast EMA, current closes above
        if c > es and ef > es:
            if prev_c <= prev_ef and c > ef:
                engine.buy(sl_pips=sl_pips, tp_pips=tp_pips, comment="20_EMA_RECLAIM_BUY")

        # Bearish: Macro downtrend (c < es), previous candle tested above/at fast EMA, current closes below
        elif c < es and ef < es:
            if prev_c >= prev_ef and c < ef:
                engine.sell(sl_pips=sl_pips, tp_pips=tp_pips, comment="20_EMA_REJECT_SELL")
