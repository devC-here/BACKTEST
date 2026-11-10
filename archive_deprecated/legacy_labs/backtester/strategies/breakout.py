"""
Donchian Channel Breakout Strategy with Dynamic ATR Trailing Stop.
Breakout: 20-period highest high / lowest low breakout
Exit: 1.5x ATR dynamic trailing stop
"""

from typing import Optional
import pandas as pd
from backtester.strategy import Strategy
from backtester.indicators import highest, lowest, atr_pips


class DonchianBreakoutStrategy(Strategy):
    name = "Donchian 20-Period Breakout"
    timeframe = "M15"

    def __init__(
        self,
        lookback: int = 20,
        trailing_atr_mult: float = 2.0,
        initial_sl_pips: float = 20.0,
        tp_pips: float = 40.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.lookback = lookback
        self.trailing_atr_mult = trailing_atr_mult
        self.initial_sl_pips = initial_sl_pips
        self.tp_pips = tp_pips

    def init(self, data: pd.DataFrame) -> pd.DataFrame:
        data["hh"] = highest(data["high"].shift(1), self.lookback)
        data["ll"] = lowest(data["low"].shift(1), self.lookback)
        data["atr_p"] = atr_pips(data["high"], data["low"], data["close"], 14, self.pip_size)
        return data

    def on_bar(self, index: int, row: pd.Series, prev_row: Optional[pd.Series], engine):
        if prev_row is None or pd.isna(row["hh"]) or pd.isna(row["ll"]):
            return

        if engine.has_position():
            return

        c = float(row["close"])
        hh = float(row["hh"])
        ll = float(row["ll"])
        atr_val = max(float(row["atr_p"]), 5.0)
        trail_pips = round(atr_val * self.trailing_atr_mult, 1)

        # Bullish Breakout
        if c > hh:
            engine.buy(
                sl_pips=self.initial_sl_pips,
                tp_pips=self.tp_pips,
                trailing_pips=trail_pips,
                comment="DONCHIAN_BREAKOUT_BUY"
            )

        # Bearish Breakout
        elif c < ll:
            engine.sell(
                sl_pips=self.initial_sl_pips,
                tp_pips=self.tp_pips,
                trailing_pips=trail_pips,
                comment="DONCHIAN_BREAKOUT_SELL"
            )
