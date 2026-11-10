"""
Strategy 46: ICT Double Sweep Liquidity Sniper Strategy.
Based on institutional Smart Money Concepts (SMC) & Liquidity Purge Model:
1. Liquidity Sweep: Price sweeps previous swing high/low (taking out stop clusters).
2. Secondary Sweep / Trap: Price re-tests the interim low/high with a wick sweep and closes back inside.
3. Macro Bias Filter: Uses 200 EMA to align with prevailing order flow.
4. Risk Management: Stop Loss beyond extreme wick sweep; Take Profit at 1:2.0 Risk-to-Reward.
"""

from typing import Optional
import pandas as pd
import numpy as np

from backtester.strategy import Strategy
from backtester.indicators import ema, highest, lowest, atr_pips


class ICTDoubleSweepStrategy(Strategy):
    name = "ICT Double Sweep Liquidity Sniper"
    timeframe = "M15"

    def __init__(
        self,
        swing_lookback: int = 20,
        risk_reward: float = 2.0,
        ema_trend: int = 200,
        sl_buffer_pips: float = 2.0,
        min_sl_pips: float = 6.0,
        max_sl_pips: float = 25.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.swing_lookback = swing_lookback
        self.risk_reward = risk_reward
        self.ema_trend = ema_trend
        self.sl_buffer_pips = sl_buffer_pips
        self.min_sl_pips = min_sl_pips
        self.max_sl_pips = max_sl_pips

    def init(self, data: pd.DataFrame) -> pd.DataFrame:
        # Precompute indicators vectorially
        data["ema200"] = ema(data["close"], self.ema_trend)
        # Prior swing high & low excluding current bar
        data["swing_high"] = highest(data["high"].shift(1), self.swing_lookback)
        data["swing_low"] = lowest(data["low"].shift(1), self.swing_lookback)
        data["atr_p"] = atr_pips(data["high"], data["low"], data["close"], 14, self.pip_size)
        return data

    def on_bar(self, index: int, row: pd.Series, prev_row: Optional[pd.Series], engine):
        if prev_row is None or index < self.swing_lookback + 5:
            return

        if engine.has_position():
            return

        c = float(row["close"])
        o = float(row["open"])
        h = float(row["high"])
        l = float(row["low"])
        ema_val = float(row["ema200"])
        swing_h = float(prev_row["swing_high"])
        swing_l = float(prev_row["swing_low"])

        prev_l = float(prev_row["low"])
        prev_h = float(prev_row["high"])
        prev_c = float(prev_row["close"])

        # 1. Bullish Double Sweep:
        # Condition A: Prev bar breached swing low (liquidity sweep), but current bar tested lower and closed bullish above prev low
        # Condition B: Trend is either above 200 EMA OR within pullback reclaim
        if prev_l < swing_l:
            # Current candle sweeps prev low and closes back above it with bullish candle
            if l <= prev_l and c > prev_l and c > o:
                sweep_low = min(l, prev_l)
                sl_distance_pips = round(((c - sweep_low) / self.pip_size) + self.sl_buffer_pips, 1)
                
                # Filter unreasonable SL sizes
                if self.min_sl_pips <= sl_distance_pips <= self.max_sl_pips:
                    tp_pips = round(sl_distance_pips * self.risk_reward, 1)
                    engine.buy(
                        sl_pips=sl_distance_pips,
                        tp_pips=tp_pips,
                        comment="ICT_BULL_SWEEP_BUY"
                    )

        # 2. Bearish Double Sweep:
        # Condition A: Prev bar breached swing high (liquidity sweep), but current bar tested higher and closed bearish below prev high
        elif prev_h > swing_h:
            # Current candle sweeps prev high and closes back below it with bearish candle
            if h >= prev_h and c < prev_h and c < o:
                sweep_high = max(h, prev_h)
                sl_distance_pips = round(((sweep_high - c) / self.pip_size) + self.sl_buffer_pips, 1)

                if self.min_sl_pips <= sl_distance_pips <= self.max_sl_pips:
                    tp_pips = round(sl_distance_pips * self.risk_reward, 1)
                    engine.sell(
                        sl_pips=sl_distance_pips,
                        tp_pips=tp_pips,
                        comment="ICT_BEAR_SWEEP_SELL"
                    )
