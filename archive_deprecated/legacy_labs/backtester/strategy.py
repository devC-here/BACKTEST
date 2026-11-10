"""
Base Strategy Interface for the Forex Backtester.
Strategies can define custom indicators in `init()`, and trading rules in `on_bar()`.
"""

from typing import Dict, Any, Optional
import pandas as pd


class Strategy:
    name: str = "BaseStrategy"
    timeframe: str = "M5"
    pip_size: float = 0.0001
    default_lot: float = 0.1

    def __init__(self, **params):
        self.params = params
        self.engine = None

    def bind_engine(self, engine: Any):
        """Binds the strategy to a running backtest engine."""
        self.engine = engine

    def init(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Precompute indicators on the candle DataFrame.
        Return the enriched DataFrame with added indicator columns.
        """
        return data

    def on_bar(self, index: int, row: pd.Series, prev_row: Optional[pd.Series], engine: Any):
        """
        Called sequentially on every bar.
        `row` contains current bar attributes and precomputed indicators.
        `prev_row` contains previous bar (or None on the very first bar).
        """
        pass

    # Helper methods that delegate to the bound engine
    def buy(
        self,
        sl_pips: Optional[float] = None,
        tp_pips: Optional[float] = None,
        trailing_pips: Optional[float] = None,
        lot_size: Optional[float] = None,
        comment: str = ""
    ):
        """Open a LONG position."""
        if self.engine is not None:
            return self.engine.buy(
                sl_pips=sl_pips,
                tp_pips=tp_pips,
                trailing_pips=trailing_pips,
                lot_size=lot_size or self.default_lot,
                comment=comment
            )

    def sell(
        self,
        sl_pips: Optional[float] = None,
        tp_pips: Optional[float] = None,
        trailing_pips: Optional[float] = None,
        lot_size: Optional[float] = None,
        comment: str = ""
    ):
        """Open a SHORT position."""
        if self.engine is not None:
            return self.engine.sell(
                sl_pips=sl_pips,
                tp_pips=tp_pips,
                trailing_pips=trailing_pips,
                lot_size=lot_size or self.default_lot,
                comment=comment
            )

    def close_position(self, comment: str = "SIGNAL_CLOSE"):
        """Close any currently open position."""
        if self.engine is not None:
            return self.engine.close_position(comment=comment)

    def has_position(self) -> bool:
        """Check if an open position exists."""
        if self.engine is not None:
            return self.engine.has_position()
        return False

    def get_position(self):
        """Return the current open position object or None."""
        if self.engine is not None:
            return self.engine.current_position
        return None
