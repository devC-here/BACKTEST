"""
London Session Range Breakout Strategy (Asian Killzone Expansion).
Institutional Model:
1. Asian Consolidation Range: Tracks High and Low formed between 00:00 and 06:45 UTC.
2. London Killzone Trigger (07:00 - 10:30 UTC):
   - Bullish Breakout: Candle closes above Asian High -> BUY.
   - Bearish Breakout: Candle closes below Asian Low -> SELL.
3. Max 1 breakout trade per session.
4. Risk Management: 1.5x ATR dynamic Stop Loss; 1:2.0 Risk-to-Reward Take Profit.
"""

from typing import Optional
import pandas as pd
from backtester.strategy import Strategy
from backtester.indicators import atr_pips, ema


class LondonBreakoutStrategy(Strategy):
    name = "London Session Range Breakout"
    timeframe = "M15"

    def __init__(
        self,
        risk_reward: float = 2.0,
        sl_atr_mult: float = 1.5,
        min_sl_pips: float = 8.0,
        max_sl_pips: float = 25.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.risk_reward = risk_reward
        self.sl_atr_mult = sl_atr_mult
        self.min_sl_pips = min_sl_pips
        self.max_sl_pips = max_sl_pips

        # Session tracking state
        self.current_day = None
        self.asian_high = None
        self.asian_low = None
        self.traded_today = False

    def init(self, data: pd.DataFrame) -> pd.DataFrame:
        data["atr_p"] = atr_pips(data["high"], data["low"], data["close"], 14, self.pip_size)
        data["ema50"] = ema(data["close"], 50)
        return data

    def on_bar(self, index: int, row: pd.Series, prev_row: Optional[pd.Series], engine):
        if prev_row is None:
            return

        bar_time = pd.to_datetime(row["time"])
        bar_day = bar_time.date()
        hour = bar_time.hour
        minute = bar_time.minute

        # Reset session range on new calendar day
        if bar_day != self.current_day:
            self.current_day = bar_day
            self.asian_high = None
            self.asian_low = None
            self.traded_today = False

        c = float(row["close"])
        h = float(row["high"])
        l = float(row["low"])
        atr_val = max(float(row["atr_p"]), 6.0)

        # 1. Asian Session (00:00 to 06:45 UTC): Accumulate High / Low
        if hour < 7:
            if self.asian_high is None or h > self.asian_high:
                self.asian_high = h
            if self.asian_low is None or l < self.asian_low:
                self.asian_low = l
            return

        # 2. London Killzone Window (07:00 to 11:00 UTC): Trade Breakouts
        if 7 <= hour <= 11 and not self.traded_today and not engine.has_position():
            if self.asian_high is not None and self.asian_low is not None:
                range_pips = (self.asian_high - self.asian_low) / self.pip_size

                # Avoid trading if Asian range was abnormally massive (>60 pips)
                if 10.0 <= range_pips <= 60.0:
                    sl_pips = max(round(atr_val * self.sl_atr_mult, 1), self.min_sl_pips)
                    sl_pips = min(sl_pips, self.max_sl_pips)
                    tp_pips = round(sl_pips * self.risk_reward, 1)

                    # Bullish Breakout above Asian High
                    if c > self.asian_high and float(prev_row["close"]) <= self.asian_high:
                        self.traded_today = True
                        engine.buy(sl_pips=sl_pips, tp_pips=tp_pips, comment="LONDON_ASIAN_HIGH_BREAK")

                    # Bearish Breakout below Asian Low
                    elif c < self.asian_low and float(prev_row["close"]) >= self.asian_low:
                        self.traded_today = True
                        engine.sell(sl_pips=sl_pips, tp_pips=tp_pips, comment="LONDON_ASIAN_LOW_BREAK")
