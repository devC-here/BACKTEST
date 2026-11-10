"""
Supertrend Trend-Flip Reversal Strategy.
Indicators: Supertrend (10, 3.0) + 200 EMA Baseline Filter
"""

from typing import Optional
import pandas as pd
from backtester.strategy import Strategy
from backtester.indicators import supertrend, ema, atr_pips


class SupertrendStrategy(Strategy):
    name = "Supertrend Trend Flip + 200 EMA"
    timeframe = "M15"

    def __init__(
        self,
        st_period: int = 10,
        st_multiplier: float = 3.0,
        ema_baseline: int = 200,
        sl_atr_mult: float = 2.0,
        tp_risk_reward: float = 2.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.st_period = st_period
        self.st_multiplier = st_multiplier
        self.ema_baseline = ema_baseline
        self.sl_atr_mult = sl_atr_mult
        self.tp_risk_reward = tp_risk_reward

    def init(self, data: pd.DataFrame) -> pd.DataFrame:
        trend, st_line = supertrend(data["high"], data["low"], data["close"], self.st_period, self.st_multiplier)
        data["st_trend"] = trend
        data["st_line"] = st_line
        data["ema200"] = ema(data["close"], self.ema_baseline)
        data["atr_p"] = atr_pips(data["high"], data["low"], data["close"], 14, self.pip_size)
        return data

    def on_bar(self, index: int, row: pd.Series, prev_row: Optional[pd.Series], engine):
        if prev_row is None or pd.isna(row["st_trend"]) or pd.isna(row["ema200"]):
            return

        curr_trend = int(row["st_trend"])
        prev_trend = int(prev_row["st_trend"])
        c = float(row["close"])
        ema_val = float(row["ema200"])
        atr_val = max(float(row["atr_p"]), 5.0)

        sl_pips = round(atr_val * self.sl_atr_mult, 1)
        tp_pips = round(sl_pips * self.tp_risk_reward, 1)

        # Bullish flip (from -1 to 1) and above 200 EMA baseline
        if prev_trend == -1 and curr_trend == 1 and c > ema_val:
            if engine.has_position():
                engine.close_position(comment="OPPOSITE_FLIP")
            engine.buy(sl_pips=sl_pips, tp_pips=tp_pips, comment="SUPERTREND_BULL_FLIP")

        # Bearish flip (from 1 to -1) and below 200 EMA baseline
        elif prev_trend == 1 and curr_trend == -1 and c < ema_val:
            if engine.has_position():
                engine.close_position(comment="OPPOSITE_FLIP")
            engine.sell(sl_pips=sl_pips, tp_pips=tp_pips, comment="SUPERTREND_BEAR_FLIP")
