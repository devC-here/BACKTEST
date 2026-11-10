"""
High-Speed Simulation and Execution Engine for Forex Backtesting.
Handles order lifecycle, intra-bar SL/TP execution, trailing stops,
spread modeling, and pip PnL accounting.
"""

from typing import List, Optional, Dict, Any
import datetime
import pandas as pd
import numpy as np

from backtester.models import Trade, Position
from backtester.strategy import Strategy


class BacktestEngine:
    def __init__(
        self,
        strategy: Strategy,
        spread_pips: Optional[float] = 0.1,
        slippage_pips: float = 0.0,
        lot_size: float = 0.1,
        pip_size: float = 0.0001,
        close_on_week_end: bool = True,
        excluded_hours: Optional[List[int]] = None
    ):
        self.strategy = strategy
        self.spread_pips = spread_pips  # None = dynamic Dukascopy spread from data
        self.slippage_pips = slippage_pips
        self.lot_size = lot_size
        self.pip_size = pip_size
        self.close_on_week_end = close_on_week_end
        self.excluded_hours = set(excluded_hours) if excluded_hours is not None else None

        # Bind engine to strategy
        self.strategy.bind_engine(self)

        # Simulation state
        self.current_position: Optional[Position] = None
        self.closed_trades: List[Trade] = []
        self.current_bar: Optional[pd.Series] = None
        self.current_bar_index: int = 0
        self.current_week: int = 0
        self.trade_counter: int = 0

    def reset(self):
        """Resets engine state between backtest runs."""
        self.current_position = None
        self.closed_trades = []
        self.current_bar = None
        self.current_bar_index = 0
        self.current_week = 0
        self.trade_counter = 0

    def has_position(self) -> bool:
        """Returns True if a position is currently open."""
        return self.current_position is not None

    def _get_spread_pips(self) -> float:
        """Determines the spread in pips for the current bar."""
        if self.spread_pips is not None:
            return float(self.spread_pips)
        if self.current_bar is not None and "spread_pips" in self.current_bar:
            return float(self.current_bar["spread_pips"])
        return 0.1

    def buy(
        self,
        sl_pips: Optional[float] = None,
        tp_pips: Optional[float] = None,
        trailing_pips: Optional[float] = None,
        lot_size: Optional[float] = None,
        comment: str = ""
    ) -> Optional[Position]:
        """
        Executes a BUY market order on the close of the current bar.
        Entry price incorporates ask price / spread.
        """
        if self.has_position():
            return None  # Single position per strategy by default

        bar = self.current_bar

        # Session / Hourly filter: Reject entries in toxic hours
        if self.excluded_hours is not None and "time" in bar:
            bar_hour = pd.to_datetime(bar["time"]).hour
            if bar_hour in self.excluded_hours:
                return None

        spread_pips = self._get_spread_pips()

        # Use fixed spread if specified by user, else use real dynamic ask_close from data
        if self.spread_pips is not None:
            entry_price = float(bar["close"]) + (spread_pips * self.pip_size) + (self.slippage_pips * self.pip_size)
        elif "ask_close" in bar and pd.notna(bar["ask_close"]):
            entry_price = float(bar["ask_close"]) + (self.slippage_pips * self.pip_size)
        else:
            entry_price = float(bar["close"]) + (spread_pips * self.pip_size) + (self.slippage_pips * self.pip_size)

        sl_price = (entry_price - (sl_pips * self.pip_size)) if sl_pips is not None else None
        tp_price = (entry_price + (tp_pips * self.pip_size)) if tp_pips is not None else None

        self.trade_counter += 1
        trade_id = f"TRD-{self.current_week:02d}-{self.trade_counter:04d}"

        pos = Position(
            trade_id=trade_id,
            week=self.current_week,
            strategy_name=self.strategy.name,
            symbol="EURUSD",
            direction="BUY",
            entry_time=bar["time"],
            entry_price=entry_price,
            sl_price=sl_price,
            tp_price=tp_price,
            trailing_sl_pips=trailing_pips,
            lot_size=lot_size or self.lot_size,
            comment=comment
        )
        self.current_position = pos
        return pos

    def sell(
        self,
        sl_pips: Optional[float] = None,
        tp_pips: Optional[float] = None,
        trailing_pips: Optional[float] = None,
        lot_size: Optional[float] = None,
        comment: str = ""
    ) -> Optional[Position]:
        """
        Executes a SELL market order on the close of the current bar.
        Entry price executes at Bid (bar.close).
        """
        if self.has_position():
            return None

        bar = self.current_bar

        # Session / Hourly filter: Reject entries in toxic hours
        if self.excluded_hours is not None and "time" in bar:
            bar_hour = pd.to_datetime(bar["time"]).hour
            if bar_hour in self.excluded_hours:
                return None

        entry_price = float(bar["close"]) - (self.slippage_pips * self.pip_size)

        sl_price = (entry_price + (sl_pips * self.pip_size)) if sl_pips is not None else None
        tp_price = (entry_price - (tp_pips * self.pip_size)) if tp_pips is not None else None

        self.trade_counter += 1
        trade_id = f"TRD-{self.current_week:02d}-{self.trade_counter:04d}"

        pos = Position(
            trade_id=trade_id,
            week=self.current_week,
            strategy_name=self.strategy.name,
            symbol="EURUSD",
            direction="SELL",
            entry_time=bar["time"],
            entry_price=entry_price,
            sl_price=sl_price,
            tp_price=tp_price,
            trailing_sl_pips=trailing_pips,
            lot_size=lot_size or self.lot_size,
            comment=comment
        )
        self.current_position = pos
        return pos

    def close_position(self, exit_price: Optional[float] = None, comment: str = "SIGNAL_CLOSE") -> Optional[Trade]:
        """Closes the current position and logs the completed Trade."""
        pos = self.current_position
        if pos is None:
            return None

        bar = self.current_bar
        spread_pips = self._get_spread_pips()

        if exit_price is None:
            if pos.direction == "BUY":
                # BUY closes at Bid
                exit_price = float(bar["close"])
            else:
                # SELL closes at Ask (Bid + Spread)
                if self.spread_pips is not None:
                    exit_price = float(bar["close"]) + (spread_pips * self.pip_size)
                elif "ask_close" in bar and pd.notna(bar["ask_close"]):
                    exit_price = float(bar["ask_close"])
                else:
                    exit_price = float(bar["close"]) + (spread_pips * self.pip_size)

        # Pip Calculation
        if pos.direction == "BUY":
            raw_pips = (exit_price - pos.entry_price) / self.pip_size
        else:
            raw_pips = (pos.entry_price - exit_price) / self.pip_size

        net_pips = round(raw_pips, 1)
        # USD value: for EURUSD, 1 lot = $10/pip, 0.1 lot = $1/pip
        pnl_usd = round(net_pips * (pos.lot_size * 10.0), 2)

        trade = Trade(
            trade_id=pos.trade_id,
            week=pos.week,
            strategy_name=pos.strategy_name,
            symbol=pos.symbol,
            direction=pos.direction,
            entry_time=pos.entry_time,
            entry_price=pos.entry_price,
            sl_price=pos.sl_price,
            tp_price=pos.tp_price,
            exit_time=bar["time"],
            exit_price=exit_price,
            exit_reason=comment,
            pnl_pips=net_pips,
            pnl_usd=pnl_usd,
            mfe_pips=round(pos.mfe_pips, 1),
            mae_pips=round(pos.mae_pips, 1),
            bars_held=pos.bars_held,
            lot_size=pos.lot_size,
            comment=pos.comment
        )

        self.closed_trades.append(trade)
        self.current_position = None
        return trade

    def _check_intra_bar_exits(self, bar: pd.Series) -> bool:
        """
        Evaluates High and Low of the incoming bar against active SL and TP levels.
        Returns True if the position was closed during this bar.
        """
        pos = self.current_position
        if pos is None:
            return False

        b_high = float(bar["high"])
        b_low = float(bar["low"])

        if pos.direction == "BUY":
            tp_hit = (pos.tp_price is not None) and (b_high >= pos.tp_price)
            sl_hit = (pos.sl_price is not None) and (b_low <= pos.sl_price)

            if tp_hit and sl_hit:
                # Conservative rule: if both reached within same bar, assume Stop Loss hit first
                reason = "TRAILING_STOP" if pos.trailing_sl_pips is not None else "SL"
                self.close_position(exit_price=pos.sl_price, comment=reason)
                return True
            elif sl_hit:
                reason = "TRAILING_STOP" if pos.trailing_sl_pips is not None else "SL"
                self.close_position(exit_price=pos.sl_price, comment=reason)
                return True
            elif tp_hit:
                self.close_position(exit_price=pos.tp_price, comment="TP")
                return True

        else:  # SELL
            # For a SELL, exit is at Ask price (Bid + Spread)
            spread_pips = self._get_spread_pips()
            ask_high = float(bar["ask_high"]) if "ask_high" in bar else (b_high + spread_pips * self.pip_size)
            ask_low = float(bar["ask_low"]) if "ask_low" in bar else (b_low + spread_pips * self.pip_size)

            tp_hit = (pos.tp_price is not None) and (ask_low <= pos.tp_price)
            sl_hit = (pos.sl_price is not None) and (ask_high >= pos.sl_price)

            if tp_hit and sl_hit:
                reason = "TRAILING_STOP" if pos.trailing_sl_pips is not None else "SL"
                self.close_position(exit_price=pos.sl_price, comment=reason)
                return True
            elif sl_hit:
                reason = "TRAILING_STOP" if pos.trailing_sl_pips is not None else "SL"
                self.close_position(exit_price=pos.sl_price, comment=reason)
                return True
            elif tp_hit:
                self.close_position(exit_price=pos.tp_price, comment="TP")
                return True

        return False

    def run_week(self, week_data: pd.DataFrame, week_num: int) -> List[Trade]:
        """
        Executes backtest for a single weekly candle dataset.
        Returns the list of trades closed during this week.
        """
        self.current_week = week_num

        # Precompute indicators via strategy
        df = self.strategy.init(week_data.copy())

        n_bars = len(df)
        trades_start_count = len(self.closed_trades)

        for i in range(n_bars):
            row = df.iloc[i]
            prev_row = df.iloc[i - 1] if i > 0 else None
            self.current_bar = row
            self.current_bar_index = i

            # 1. Intra-bar SL/TP check on existing open position
            closed_in_bar = self._check_intra_bar_exits(row)

            # 2. If position is still open, update its excursions and trailing stop
            if not closed_in_bar and self.current_position is not None:
                self.current_position.update_bar(
                    bar_high=float(row["high"]),
                    bar_low=float(row["low"]),
                    bar_close=float(row["close"]),
                    pip_size=self.pip_size
                )

            # 3. Strategy decision on closed bar
            self.strategy.on_bar(i, row, prev_row, self)

        # End of week: close any open position if configured
        if self.close_on_week_end and self.has_position():
            self.close_position(comment="WEEK_END")

        week_closed_trades = self.closed_trades[trades_start_count:]
        return week_closed_trades
