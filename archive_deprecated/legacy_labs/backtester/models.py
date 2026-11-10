"""
Trade and Position Data Models for Forex Backtesting.
All pip calculations strictly follow EURUSD standard: 1 pip = 0.0001 (10 points).
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import datetime
import pandas as pd


@dataclass
class Trade:
    trade_id: str
    week: int
    strategy_name: str
    symbol: str
    direction: str  # 'BUY' or 'SELL'
    entry_time: pd.Timestamp
    entry_price: float
    sl_price: Optional[float] = None
    tp_price: Optional[float] = None
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    exit_reason: str = "OPEN"  # 'TP', 'SL', 'TRAILING_STOP', 'SIGNAL_REVERSAL', 'WEEK_END'
    pnl_pips: float = 0.0
    pnl_usd: float = 0.0
    mfe_pips: float = 0.0  # Maximum Favorable Excursion (peak profit)
    mae_pips: float = 0.0  # Maximum Adverse Excursion (peak drawdown)
    bars_held: int = 0
    lot_size: float = 0.1
    commission_pips: float = 0.0
    comment: str = ""

    def to_dict(self) -> Dict[str, Any]:
        duration_mins = 0.0
        if self.exit_time and self.entry_time:
            duration_mins = round((self.exit_time - self.entry_time).total_seconds() / 60.0, 1)

        return {
            "trade_id": self.trade_id,
            "week": self.week,
            "strategy": self.strategy_name,
            "symbol": self.symbol,
            "direction": self.direction,
            "entry_time": str(self.entry_time),
            "entry_price": round(self.entry_price, 5),
            "sl_price": round(self.sl_price, 5) if self.sl_price is not None else None,
            "tp_price": round(self.tp_price, 5) if self.tp_price is not None else None,
            "exit_time": str(self.exit_time) if self.exit_time else "",
            "exit_price": round(self.exit_price, 5) if self.exit_price is not None else None,
            "exit_reason": self.exit_reason,
            "pnl_pips": round(self.pnl_pips, 1),
            "pnl_usd": round(self.pnl_usd, 2),
            "mfe_pips": round(self.mfe_pips, 1),
            "mae_pips": round(self.mae_pips, 1),
            "bars_held": self.bars_held,
            "duration_minutes": duration_mins,
            "lot_size": self.lot_size,
            "comment": self.comment
        }


@dataclass
class Position:
    trade_id: str
    week: int
    strategy_name: str
    symbol: str
    direction: str  # 'BUY' or 'SELL'
    entry_time: pd.Timestamp
    entry_price: float
    sl_price: Optional[float] = None
    tp_price: Optional[float] = None
    trailing_sl_pips: Optional[float] = None
    lot_size: float = 0.1
    comment: str = ""

    # Live tracking
    current_price: float = 0.0
    mfe_pips: float = 0.0
    mae_pips: float = 0.0
    bars_held: int = 0
    highest_price: float = field(default_factory=lambda: 0.0)
    lowest_price: float = field(default_factory=lambda: 999999.0)

    def __post_init__(self):
        self.current_price = self.entry_price
        self.highest_price = self.entry_price
        self.lowest_price = self.entry_price

    def update_bar(self, bar_high: float, bar_low: float, bar_close: float, pip_size: float = 0.0001):
        """Updates MFE/MAE excursions and trailing stop based on latest bar price movement."""
        self.bars_held += 1
        self.current_price = bar_close

        if self.direction == "BUY":
            fav = (bar_high - self.entry_price) / pip_size
            adv = (self.entry_price - bar_low) / pip_size
            if fav > self.mfe_pips:
                self.mfe_pips = fav
            if adv > self.mae_pips:
                self.mae_pips = adv

            if bar_high > self.highest_price:
                self.highest_price = bar_high
                if self.trailing_sl_pips is not None:
                    trail_level = self.highest_price - (self.trailing_sl_pips * pip_size)
                    if self.sl_price is None or trail_level > self.sl_price:
                        self.sl_price = trail_level

        else:  # SELL
            fav = (self.entry_price - bar_low) / pip_size
            adv = (bar_high - self.entry_price) / pip_size
            if fav > self.mfe_pips:
                self.mfe_pips = fav
            if adv > self.mae_pips:
                self.mae_pips = adv

            if bar_low < self.lowest_price:
                self.lowest_price = bar_low
                if self.trailing_sl_pips is not None:
                    trail_level = self.lowest_price + (self.trailing_sl_pips * pip_size)
                    if self.sl_price is None or trail_level < self.sl_price:
                        self.sl_price = trail_level
