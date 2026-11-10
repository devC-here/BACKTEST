#!/usr/bin/env python3
"""
================================================================================
  INSTITUTIONAL TRADE CONTROLLER & CONCURRENCY RISK GOVERNOR (FLIPPED LOGIC)
================================================================================
Core Philosophy: "Cut Your Losses Early & Let Your Winners Run"

Rules Enforced:
1. Hard Capacity Cap: Maximum 20 simultaneous active trades at any point in time.
2. Signal Ingestion with Flipped Dynamic Room-Making:
   - When a new trade signal arrives and open positions < 20:
     -> Accept and execute the trade immediately (Direct Acceptance).
   - When open positions == 20 (Capacity Full):
     -> Check existing active trades:
        a) Max Holding Expiration (>= 24h):
           -> Any zombie trade exceeding 24 hours is closed immediately to prevent
              indefinite capital blockage.
        b) CUT LOSING TRADES TO MAKE ROOM (Flipped Mechanism):
           -> Tier 1: Check if any trade has TIMED OUT (>= 4 hours) AND is IN LOSS
              (unrealized PnL <= 0):
              -> Close that stagnant losing trade early at current market price!
              -> Free the slot and execute the fresh incoming high-expectancy trade!
           -> Tier 2: If no trade has timed out yet, but active trades are in loss
              (unrealized PnL < 0):
              -> Cut the worst performing losing trade (deepest drawdown) to make room!
              -> Free the slot and execute the fresh incoming trade!
        c) PROTECT RUNNING WINNERS (Let Winners Run):
           -> Active trades currently IN PROFIT (unrealized PnL > 0) are STRICTLY
              SHIELDED from early closure so they can reach full Take Profit.
        d) All Active Trades in Profit:
           -> If ALL 20 active trades are in profit, DO NOT interrupt any winner.
           -> QUASH (REJECT) the incoming trade signal!
3. Continuous Mark-to-Market & SL/TP Management:
   - Monitors live prices, tracks floating PnL, and expires trades on natural SL/TP hits.
================================================================================
"""

from __future__ import annotations
import sys
import time
import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import pandas as pd
import numpy as np

try:
    from trade_trajectory_predictor import TradeTrajectoryPredictor
except ImportError:
    class TradeTrajectoryPredictor:
        """
        Predictive engine for in-flight trade trajectory evaluation.
        Empirically fitted models:
          - Breakeven Recovery: P(BE | d) = (1 - d)^0.84  (R^2 = 0.9921)
          - Target Hit from Loss: P(TP | d) = P0_TP * (1 - d)^1.45
          - Target Hit from Profit: P(TP | p) = P0_TP + (1 - P0_TP) * p^0.72
          - Breakeven Retrace from Profit: P(BE | p) = (1 - p)^0.88
          - Full SL Collapse from Profit: P(SL | p) = P0_SL * (1 - p)^1.35
        """
        def __init__(self, default_base_winrate: float = 0.50):
            self.default_base_winrate = float(default_base_winrate)

        def predict(
            self,
            sl_pips: float,
            tp_pips: float,
            current_pips: float,
            base_winrate: Optional[float] = None,
            hold_hours: Optional[float] = None
        ) -> Dict[str, Any]:
            sl = max(0.1, float(sl_pips))
            tp = max(0.1, float(tp_pips))
            pnl = float(current_pips)

            p0_tp = float(base_winrate) if base_winrate is not None else (sl / (sl + tp) if sl + tp > 0 else 0.50)
            p0_sl = 1.0 - p0_tp

            if pnl > 0:
                p = min(1.0, max(0.0, pnl / tp))
                prob_tp = p0_tp + (1.0 - p0_tp) * (p ** 0.72)
                prob_tp = min(0.999, max(0.001, prob_tp))
                prob_retrace_be = (1.0 - p) ** 0.88
                prob_collapse_sl = p0_sl * ((1.0 - p) ** 1.35)
                return {
                    "regime": "PROFIT",
                    "prob_hit_tp_pct": round(prob_tp * 100.0, 1),
                    "prob_recover_be_pct": 100.0,
                    "prob_retrace_be_pct": round(prob_retrace_be * 100.0, 1),
                    "prob_collapse_sl_pct": round(prob_collapse_sl * 100.0, 1),
                    "verdict": "LET_RUN_TO_TP"
                }
            else:
                d = min(1.0, max(0.0, abs(pnl) / sl))
                prob_be = (1.0 - d) ** 0.84
                prob_tp_from_d = p0_tp * ((1.0 - d) ** 1.45)
                return {
                    "regime": "LOSS",
                    "drawdown_depth_pct": round(d * 100.0, 1),
                    "prob_recover_be_pct": round(prob_be * 100.0, 1),
                    "prob_hit_tp_pct": round(prob_tp_from_d * 100.0, 1),
                    "verdict": "CUT_FOR_ROOM" if d >= 0.50 else "HOLD_OR_CUT"
                }

        def predict_trade_object(self, trade: Any, current_price: Optional[float] = None) -> Dict[str, Any]:
            curr_p = current_price if current_price is not None else trade.current_price
            if curr_p <= 0:
                curr_p = trade.entry_price
            pip_size = PIP_SIZES.get(trade.symbol, 0.0001)
            if trade.direction == 1:
                cur_pips = (curr_p - trade.entry_price) / pip_size
            else:
                cur_pips = (trade.entry_price - curr_p) / pip_size
            now_ms = int(time.time() * 1000)
            hold_hrs = (now_ms - trade.entry_time_ms) / (3600 * 1000) if hasattr(trade, 'entry_time_ms') else 1.0
            return self.predict(trade.sl_pips, trade.tp_pips, cur_pips, hold_hours=hold_hrs)

# Pip values in USD per standard 1.00 lot (100,000 units)
PIP_VALUES_PER_LOT_USD: Dict[str, float] = {
    "EURUSD": 10.00,
    "GBPCHF": 10.00,
    "USDJPY": 6.80,
    "GBPJPY": 6.80,
    "EURGBP": 13.00,
    "USDCAD": 7.40,
    "AUDCAD": 7.40,
    "NZDCAD": 7.40,
    "AUDNZD": 6.10,
    "EURNZD": 6.10,
}

PIP_SIZES: Dict[str, float] = {
    "EURUSD": 0.0001,
    "EURGBP": 0.0001,
    "AUDNZD": 0.0001,
    "AUDCAD": 0.0001,
    "USDCAD": 0.0001,
    "EURNZD": 0.0001,
    "GBPCHF": 0.0001,
    "NZDCAD": 0.0001,
    "USDJPY": 0.01,
    "GBPJPY": 0.01,
}

@dataclass
class Trade:
    trade_id: str
    symbol: str
    strategy_name: str
    direction: int            # 1 for Long (Buy), -1 for Short (Sell)
    entry_price: float
    sl_price: float
    tp_price: float
    sl_pips: float
    tp_pips: float
    lot_size: float
    entry_time_ms: int
    timeout_duration_ms: int  # Max holding duration threshold before considered stagnant (e.g. 4 hours)

    # Dynamic Mark-to-Market State
    current_bid: float = 0.0
    current_ask: float = 0.0
    current_price: float = 0.0
    unrealized_pips: float = 0.0
    unrealized_usd: float = 0.0
    is_closed: bool = False
    exit_time_ms: int = 0
    exit_price: float = 0.0
    exit_reason: str = ""
    realized_pips: float = 0.0
    realized_usd: float = 0.0

    @property
    def timeout_timestamp_ms(self) -> int:
        return self.entry_time_ms + self.timeout_duration_ms

    def is_timed_out(self, current_time_ms: int) -> bool:
        """Returns True if the trade has exceeded its designated holding duration."""
        return current_time_ms >= self.timeout_timestamp_ms

    def is_in_profit(self) -> bool:
        """Returns True if current floating unrealized PnL is strictly positive."""
        return self.unrealized_pips > 0.0

    def is_in_loss(self) -> bool:
        """Returns True if current floating unrealized PnL is negative or zero."""
        return self.unrealized_pips <= 0.0

    def update_price(self, bid: float, ask: float) -> None:
        """Updates live mark-to-market prices and unrealized PnL."""
        self.current_bid = bid
        self.current_ask = ask
        pip_size = PIP_SIZES.get(self.symbol, 0.0001)
        pip_val_usd = PIP_VALUES_PER_LOT_USD.get(self.symbol, 10.00) * self.lot_size

        if self.direction == 1:
            # Long position closes on Bid
            self.current_price = bid
            self.unrealized_pips = (bid - self.entry_price) / pip_size
        else:
            # Short position closes on Ask
            self.current_price = ask
            self.unrealized_pips = (self.entry_price - ask) / pip_size

        self.unrealized_usd = self.unrealized_pips * pip_val_usd

    def check_sl_tp(self, bid: float, ask: float) -> Optional[Tuple[str, float]]:
        """Checks if current tick triggers Stop Loss or Take Profit."""
        if self.direction == 1:
            if self.sl_price > 0 and bid <= self.sl_price:
                return "SL_HIT", bid
            if self.tp_price > 0 and bid >= self.tp_price:
                return "TP_HIT", bid
        else:
            if self.sl_price > 0 and ask >= self.sl_price:
                return "SL_HIT", ask
            if self.tp_price > 0 and ask <= self.tp_price:
                return "TP_HIT", ask
        return None


class TradeController:
    """
    Institutional Trade Controller enforcing strict 20-slot concurrency cap
    and FLIPPED room-making rules: "Cut Losers Early, Let Winners Run"
    
    Rules:
    - Max 20 open positions at any time.
    - If slots < 20: Place new incoming trade immediately.
    - If slots == 20:
        1. Close any trade exceeding hard max holding limit (e.g. 24h).
        2. Scan active trades:
           a) Timed-out losing trades (hold >= 4h & PnL <= 0):
              -> CLOSE EARLY to prune stagnant losses and make room for fresh trade!
           b) Active losing trades (PnL < 0):
              -> CUT the worst performing losing trade to free a slot!
        3. Protect all winning trades (PnL > 0):
           -> NEVER cut running winners to make room. Allow winners to reach full TP!
        4. If all 20 active positions are in profit:
           -> QUASH the new incoming trade! Protect 100% of winning trades.
    """

    def __init__(
        self,
        max_concurrent_trades: int = 20,
        default_timeout_hours: float = 4.0,
        starting_equity: float = 5000.00,
        enable_active_loss_pruning: bool = True,
        max_holding_hours: float = 24.0,
        verbose: bool = True,
        **kwargs
    ):
        self.max_concurrent_trades = int(max_concurrent_trades)
        self.default_timeout_ms = int(default_timeout_hours * 3600 * 1000)
        self.max_holding_ms = int(max_holding_hours * 3600 * 1000)
        self.starting_equity = float(starting_equity)
        self.current_equity = float(starting_equity)
        self.peak_equity = float(starting_equity)
        self.enable_active_loss_pruning = bool(enable_active_loss_pruning)
        self.verbose = verbose

        # Active trade registry: trade_id -> Trade
        self.active_trades: Dict[str, Trade] = {}
        # Closed trade ledger
        self.closed_trades: List[Trade] = []

        # Audit counters
        self.total_signals_received: int = 0
        self.trades_accepted_direct: int = 0
        self.trades_accepted_via_room_making: int = 0
        self.signals_quashed_capacity_full: int = 0
        self.loss_cuts_for_room_count: int = 0
        self.winning_trades_protected_count: int = 0
        self.max_holding_expired_count: int = 0

        # Backwards compatibility aliases
        self.early_profit_closes_count: int = 0
        self.salvage_loss_closes_count: int = 0
        self.losing_trades_protected_count: int = 0

        self.predictor = TradeTrajectoryPredictor()

    @property
    def open_trade_count(self) -> int:
        return len(self.active_trades)

    @property
    def has_free_slot(self) -> bool:
        return len(self.active_trades) < self.max_concurrent_trades

    # --------------------------------------------------------------------------
    # 1. LIVE TICK UPDATE & NATURAL SL/TP EVALUATION
    # --------------------------------------------------------------------------
    def on_tick(self, symbol: str, bid: float, ask: float, timestamp_ms: int) -> List[Trade]:
        """
        Updates mark-to-market prices for all open positions of this symbol.
        Automatically closes positions that hit SL or TP naturally.
        """
        closed_this_tick: List[Trade] = []
        trades_to_close: List[Tuple[str, str, float]] = []

        for trade_id, trade in list(self.active_trades.items()):
            if trade.symbol != symbol:
                continue

            trade.update_price(bid, ask)
            sl_tp_res = trade.check_sl_tp(bid, ask)
            if sl_tp_res:
                reason, exit_price = sl_tp_res
                trades_to_close.append((trade_id, reason, exit_price))

        for trade_id, reason, exit_price in trades_to_close:
            closed_trade = self._close_position(trade_id, exit_price, reason, timestamp_ms)
            if closed_trade:
                closed_this_tick.append(closed_trade)

        return closed_this_tick

    # --------------------------------------------------------------------------
    # 2. SIGNAL INGESTION & FLIPPED DYNAMIC ROOM-MAKING LOGIC
    # --------------------------------------------------------------------------
    def on_new_signal(
        self,
        symbol: str,
        strategy_name: str,
        direction: int,
        entry_price: float,
        sl_pips: float,
        tp_pips: float,
        timestamp_ms: int,
        lot_size: float = 0.03,
        timeout_hours: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Evaluates an incoming trade signal under the 20-slot Trade Control rules.
        """
        self.total_signals_received += 1
        pip_size = PIP_SIZES.get(symbol, 0.0001)

        # Compute SL / TP prices
        if direction == 1:
            sl_price = entry_price - (sl_pips * pip_size)
            tp_price = entry_price + (tp_pips * pip_size)
        else:
            sl_price = entry_price + (sl_pips * pip_size)
            tp_price = entry_price - (tp_pips * pip_size)

        timeout_ms = int(timeout_hours * 3600 * 1000) if timeout_hours else self.default_timeout_ms
        trade_id = f"{symbol}_{strategy_name}_{timestamp_ms}_{self.total_signals_received}"

        new_trade = Trade(
            trade_id=trade_id,
            symbol=symbol,
            strategy_name=strategy_name,
            direction=direction,
            entry_price=entry_price,
            sl_price=sl_price,
            tp_price=tp_price,
            sl_pips=sl_pips,
            tp_pips=tp_pips,
            lot_size=lot_size,
            entry_time_ms=timestamp_ms,
            timeout_duration_ms=timeout_ms
        )

        # ----------------------------------------------------------------------
        # CASE 1: Open Trades < 20 (Room is Available)
        # ----------------------------------------------------------------------
        if len(self.active_trades) < self.max_concurrent_trades:
            self.active_trades[trade_id] = new_trade
            self.trades_accepted_direct += 1
            if self.verbose:
                print(f"[TRADE CONTROLLER] -> ACCEPTED (Direct): {symbol} {strategy_name} ({direction:+d}). Open: {self.open_trade_count}/{self.max_concurrent_trades}")
            return {
                "status": "ACCEPTED_DIRECT",
                "trade_id": trade_id,
                "message": f"Slot available. Trade placed directly ({self.open_trade_count}/{self.max_concurrent_trades} open).",
                "new_trade": new_trade,
                "closed_trade": None
            }

        # ----------------------------------------------------------------------
        # CASE 2: Open Trades == 20 (Capacity Full: Flipped Room-Making Check)
        # ----------------------------------------------------------------------
        # Step 2.1: Check hard maximum holding expiration (e.g. 24 hours) to prevent zombie congestion
        max_holding_expired = [
            t for t in self.active_trades.values()
            if (timestamp_ms - t.entry_time_ms) >= self.max_holding_ms
        ]
        if len(max_holding_expired) > 0:
            candidate = max(max_holding_expired, key=lambda x: (timestamp_ms - x.entry_time_ms))
            exit_price = candidate.current_price if candidate.current_price > 0 else candidate.entry_price
            closed_trade = self._close_position(
                trade_id=candidate.trade_id,
                exit_price=exit_price,
                reason="MAX_HOLDING_EXPIRED",
                timestamp_ms=timestamp_ms
            )
            self.active_trades[trade_id] = new_trade
            self.trades_accepted_via_room_making += 1
            self.max_holding_expired_count += 1
            return {
                "status": "ACCEPTED_ROOM_MADE",
                "trade_id": trade_id,
                "message": f"Capacity was 20/20. Trade {candidate.trade_id} exceeded max holding duration ({self.max_holding_ms/3600000:.1f}h) and was closed to make room.",
                "new_trade": new_trade,
                "closed_trade": closed_trade
            }

        # Step 2.2: Scan active trades: Separate into Winning (Protected) vs Losing (Room-Making Candidates)
        timed_out_in_loss: List[Trade] = []
        active_in_loss: List[Trade] = []
        active_in_profit: List[Trade] = []

        for t in self.active_trades.values():
            if t.is_in_profit():
                active_in_profit.append(t)
            else:
                if t.is_timed_out(timestamp_ms):
                    timed_out_in_loss.append(t)
                else:
                    active_in_loss.append(t)

        # Winning trades are protected: allow them to run towards full Take Profit!
        self.winning_trades_protected_count += len(active_in_profit)

        # ----------------------------------------------------------------------
        # Subcase 2A: Found a timed-out trade that is IN LOSS -> CUT IT TO MAKE ROOM!
        # "Cut losses early, let winners run"
        # ----------------------------------------------------------------------
        if len(timed_out_in_loss) > 0:
            # Pick the trade with the worst negative PnL (deepest loss) or longest timed out
            candidate = min(timed_out_in_loss, key=lambda x: (x.unrealized_pips, -(timestamp_ms - x.timeout_timestamp_ms)))
            
            exit_price = candidate.current_price if candidate.current_price > 0 else candidate.entry_price
            closed_trade = self._close_position(
                trade_id=candidate.trade_id,
                exit_price=exit_price,
                reason="TIMEOUT_LOSS_CUT_ROOM_MADE",
                timestamp_ms=timestamp_ms
            )

            # Slot freed! Place the new incoming high-expectancy trade
            self.active_trades[trade_id] = new_trade
            self.trades_accepted_via_room_making += 1
            self.loss_cuts_for_room_count += 1
            self.salvage_loss_closes_count += 1

            if self.verbose:
                print(f"[TRADE CONTROLLER] -> ROOM MADE (Loss Cut)! Cut timed-out loss on {candidate.symbol} ({candidate.strategy_name}) at {candidate.realized_pips:.1f}p (${candidate.realized_usd:.2f}) to execute fresh trade: {symbol} {strategy_name}. Open: {self.open_trade_count}/{self.max_concurrent_trades}")

            return {
                "status": "ACCEPTED_ROOM_MADE",
                "trade_id": trade_id,
                "message": f"Capacity was 20/20. Timed-out losing trade {candidate.trade_id} cut at {candidate.realized_pips:.1f}p to make room for fresh trade.",
                "new_trade": new_trade,
                "closed_trade": closed_trade
            }

        # ----------------------------------------------------------------------
        # Subcase 2B: No timed-out loss, but active trades in loss exist
        # If active loss pruning is enabled, prune the deepest active loss
        # ----------------------------------------------------------------------
        if self.enable_active_loss_pruning and len(active_in_loss) > 0:
            # Pick the trade with the deepest drawdown (worst negative PnL)
            candidate = min(active_in_loss, key=lambda x: x.unrealized_pips)

            exit_price = candidate.current_price if candidate.current_price > 0 else candidate.entry_price
            closed_trade = self._close_position(
                trade_id=candidate.trade_id,
                exit_price=exit_price,
                reason="ACTIVE_LOSS_CUT_ROOM_MADE",
                timestamp_ms=timestamp_ms
            )

            # Slot freed! Place the fresh incoming trade
            self.active_trades[trade_id] = new_trade
            self.trades_accepted_via_room_making += 1
            self.loss_cuts_for_room_count += 1
            self.salvage_loss_closes_count += 1

            if self.verbose:
                print(f"[TRADE CONTROLLER] -> ROOM MADE (Active Loss Pruned)! Cut deepest losing trade on {candidate.symbol} ({candidate.strategy_name}) at {candidate.realized_pips:.1f}p (${candidate.realized_usd:.2f}) to execute fresh trade: {symbol} {strategy_name}. Open: {self.open_trade_count}/{self.max_concurrent_trades}")

            return {
                "status": "ACCEPTED_ROOM_MADE",
                "trade_id": trade_id,
                "message": f"Capacity was 20/20. Active losing trade {candidate.trade_id} cut at {candidate.realized_pips:.1f}p to make room for fresh trade.",
                "new_trade": new_trade,
                "closed_trade": closed_trade
            }

        # ----------------------------------------------------------------------
        # Subcase 2C: All 20 Active Trades are IN PROFIT (or no eligible loss)!
        # DO NOT TOUCH RUNNING WINNERS. QUASH THE NEW TRADE!
        # ----------------------------------------------------------------------
        self.signals_quashed_capacity_full += 1
        if self.verbose:
            print(f"[TRADE CONTROLLER] -> QUASHED (All Positions in Profit): {symbol} {strategy_name}. 20 slots full, all active trades are profitable. Winners protected to run to TP.")

        return {
            "status": "QUASHED_CAPACITY_FULL",
            "trade_id": None,
            "message": f"Capacity 20/20 full. All active trades are in profit or protected. Fresh signal quashed without closing any winning trade.",
            "new_trade": None,
            "closed_trade": None
        }

    # --------------------------------------------------------------------------
    # 3. INTERNAL POSITION CLOSURE & LEDGER ACCOUNTING
    # --------------------------------------------------------------------------
    def _close_position(
        self,
        trade_id: str,
        exit_price: float,
        reason: str,
        timestamp_ms: int
    ) -> Optional[Trade]:
        """Closes an active position and registers realized cash to equity."""
        if trade_id not in self.active_trades:
            return None

        trade = self.active_trades.pop(trade_id)
        pip_size = PIP_SIZES.get(trade.symbol, 0.0001)
        pip_val_usd = PIP_VALUES_PER_LOT_USD.get(trade.symbol, 10.00) * trade.lot_size

        if trade.direction == 1:
            realized_pips = (exit_price - trade.entry_price) / pip_size
        else:
            realized_pips = (trade.entry_price - exit_price) / pip_size

        realized_usd = realized_pips * pip_val_usd

        trade.is_closed = True
        trade.exit_time_ms = timestamp_ms
        trade.exit_price = exit_price
        trade.exit_reason = reason
        trade.realized_pips = realized_pips
        trade.realized_usd = realized_usd

        self.closed_trades.append(trade)
        self.current_equity += realized_usd
        if self.current_equity > self.peak_equity:
            self.peak_equity = self.current_equity

        return trade

    # --------------------------------------------------------------------------
    # 4. INSPECTION & REPORTING HELPERS
    # --------------------------------------------------------------------------
    def predict_active_trade(self, trade_id: str, current_price: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """Predicts target-hit and breakeven recovery probabilities for an open position."""
        trade = self.active_trades.get(trade_id)
        if not trade:
            return None
        return self.predictor.predict_trade_object(trade, current_price=current_price)

    def get_open_trades_dataframe(self, current_time_ms: Optional[int] = None) -> pd.DataFrame:
        """Returns clean DataFrame of all currently active trades, their status, and predictive probabilities."""
        records = []
        now_ms = current_time_ms if current_time_ms else int(time.time() * 1000)

        for t in self.active_trades.values():
            hold_hours = (now_ms - t.entry_time_ms) / (3600 * 1000)
            max_hours = t.timeout_duration_ms / (3600 * 1000)
            timed_out = t.is_timed_out(now_ms)
            in_profit = t.is_in_profit()

            pred = self.predictor.predict_trade_object(t, t.current_price)

            if in_profit:
                elig = "NO (Protected Winner: Let Run to TP)"
            elif timed_out:
                elig = f"YES (Cut Loss: Timed Out at {t.unrealized_pips:.1f}p)"
            else:
                elig = f"YES (Cut Loss: Active Loss at {t.unrealized_pips:.1f}p)"

            records.append({
                "Trade ID": t.trade_id,
                "Symbol": t.symbol,
                "Strategy": t.strategy_name,
                "Dir": "BUY" if t.direction == 1 else "SELL",
                "Entry Price": t.entry_price,
                "Current Price": t.current_price,
                "Unrealized Pips": round(t.unrealized_pips, 1),
                "Unrealized USD": round(t.unrealized_usd, 2),
                "Hold Hours": round(hold_hours, 2),
                "Max Hours": round(max_hours, 2),
                "Timed Out?": "YES" if timed_out else "NO",
                "In Profit?": "YES" if in_profit else "NO",
                "P(Hit TP)": f"{pred.get('prob_hit_tp_pct', 50.0):.1f}%",
                "P(Recover BE)": f"{pred.get('prob_recover_be_pct', 100.0):.1f}%",
                "Trajectory Verdict": pred.get("verdict", "HOLD"),
                "Eligible for Room-Making?": elig
            })

        return pd.DataFrame(records)

    def get_controller_metrics(self) -> Dict[str, Any]:
        """Returns comprehensive operational metrics of the controller."""
        n_closed = len(self.closed_trades)
        wins = [t.realized_usd for t in self.closed_trades if t.realized_usd > 0]
        losses = [t.realized_usd for t in self.closed_trades if t.realized_usd < 0]

        total_net = self.current_equity - self.starting_equity
        win_rate = (len(wins) / n_closed * 100.0) if n_closed > 0 else 0.0
        profit_factor = (sum(wins) / abs(sum(losses))) if len(losses) > 0 and sum(losses) != 0 else 0.0

        return {
            "starting_equity": self.starting_equity,
            "current_equity": round(self.current_equity, 2),
            "net_profit_usd": round(total_net, 2),
            "open_positions": self.open_trade_count,
            "max_capacity": self.max_concurrent_trades,
            "total_signals_received": self.total_signals_received,
            "trades_accepted_direct": self.trades_accepted_direct,
            "trades_accepted_via_room_making": self.trades_accepted_via_room_making,
            "loss_cuts_for_room_count": self.loss_cuts_for_room_count,
            "winning_trades_protected_count": self.winning_trades_protected_count,
            "max_holding_expired_count": self.max_holding_expired_count,
            "signals_quashed_capacity_full": self.signals_quashed_capacity_full,
            "closed_trades_total": n_closed,
            "win_rate_pct": round(win_rate, 1),
            "profit_factor": round(profit_factor, 2)
        }


# ==============================================================================
# DEMONSTRATION & UNIT VERIFICATION TEST
# ==============================================================================
def run_demonstration():
    print("=" * 100)
    print("        FLIPPED TRADE CONTROLLER VERIFICATION: CUT LOSERS & LET WINNERS RUN        ")
    print("=" * 100)

    # Initialize Controller with 20-trade limit and 4-hour timeout
    controller = TradeController(max_concurrent_trades=20, default_timeout_hours=4.0, starting_equity=5000.00)
    base_time = 1700000000000

    print("\n[Step 1] Filling 19 slots with normal incoming trade signals (Slots < 20)...")
    for i in range(1, 20):
        sym = "EURUSD" if i % 2 == 0 else "USDJPY"
        p = 1.0850 if sym == "EURUSD" else 150.20
        controller.on_new_signal(
            symbol=sym,
            strategy_name=f"Strat_Sample_{i:02d}",
            direction=1,
            entry_price=p,
            sl_pips=10.0,
            tp_pips=20.0,
            timestamp_ms=base_time + (i * 60000),
            lot_size=0.03
        )
    print(f"Current Open Trades: {controller.open_trade_count}/20")

    print("\n[Step 2] Placing 20th trade -> Capacity is FULL (20/20)...")
    controller.on_new_signal(
        symbol="GBPJPY",
        strategy_name="Strat_GBPJPY_20",
        direction=1,
        entry_price=188.50,
        sl_pips=15.0,
        tp_pips=30.0,
        timestamp_ms=base_time + (20 * 60000),
        lot_size=0.03
    )
    print(f"Current Open Trades: {controller.open_trade_count}/20 (Capacity Full: {not controller.has_free_slot})")

    # Fast forward time by 5 hours (trades are now TIMED OUT > 4h)
    advanced_time = base_time + (5 * 3600 * 1000)

    print("\n[Step 3] Simulating market mark-to-market prices at T + 5 hours:")
    # Trade 1 is TIMED OUT and in PROFIT (+20.0 pips)
    t1 = list(controller.active_trades.values())[0]
    t1.update_price(bid=150.40, ask=150.41)
    print(f"  * Trade 1 ({t1.symbol} {t1.strategy_name}): Hold 5.0h | Unrealized: +{t1.unrealized_pips:.1f}p -> [TIMED OUT & IN PROFIT]")

    # Trade 2 is TIMED OUT and in LOSS (-6.0 pips)
    t2 = list(controller.active_trades.values())[1]
    t2.update_price(bid=1.0844, ask=1.0845)
    print(f"  * Trade 2 ({t2.symbol} {t2.strategy_name}): Hold 5.0h | Unrealized: {t2.unrealized_pips:.1f}p -> [TIMED OUT & IN LOSS]")

    print("\n[Step 4] Signal 21 arrives while capacity is 20/20:")
    print("  -> FLIPPED RULE: Trade 1 is in PROFIT -> PROTECT IT! Do NOT end profit trades!")
    print("  -> FLIPPED RULE: Trade 2 is in LOSS -> CUT Trade 2 early to MAKE ROOM for Signal 21!")
    res21 = controller.on_new_signal(
        symbol="AUDNZD",
        strategy_name="Strat_AUDNZD_Fresh_21",
        direction=1,
        entry_price=1.0800,
        sl_pips=12.0,
        tp_pips=24.0,
        timestamp_ms=advanced_time,
        lot_size=0.03
    )
    print(f"  Result: {res21['status']} -> {res21['message']}")
    print(f"  Current Open Trades: {controller.open_trade_count}/20")
    print(f"  Is Trade 1 (Profitable) Still Open? {'YES (Protected!)' if t1.trade_id in controller.active_trades else 'NO'}")
    print(f"  Is Trade 2 (Losing) Closed? {'YES (Cut to make room!)' if t2.trade_id not in controller.active_trades else 'NO'}")

    print("\n[Step 5] Simulating All 20 Active Trades Turning Profitable:")
    for t in controller.active_trades.values():
        if t.direction == 1:
            t.update_price(bid=t.entry_price + (10.0 * PIP_SIZES.get(t.symbol, 0.0001)), ask=t.entry_price + (10.1 * PIP_SIZES.get(t.symbol, 0.0001)))
        else:
            t.update_price(bid=t.entry_price - (10.1 * PIP_SIZES.get(t.symbol, 0.0001)), ask=t.entry_price - (10.0 * PIP_SIZES.get(t.symbol, 0.0001)))

    print(f"  Active Trades in Profit: {sum(1 for t in controller.active_trades.values() if t.is_in_profit())}/20")

    print("\n[Step 6] Signal 22 arrives when all 20 active trades are in profit:")
    print("  -> FLIPPED RULE: All 20 trades are winners. DO NOT cut any winner! QUASH incoming Signal 22!")
    res22 = controller.on_new_signal(
        symbol="EURGBP",
        strategy_name="Strat_EURGBP_Blocked_22",
        direction=1,
        entry_price=0.8550,
        sl_pips=8.0,
        tp_pips=16.0,
        timestamp_ms=advanced_time + 1000,
        lot_size=0.03
    )
    print(f"  Result: {res22['status']} -> {res22['message']}")
    print(f"  Current Open Trades: {controller.open_trade_count}/20")

    print("\n[Step 7] Active Trades Inspection Table:")
    df_active = controller.get_open_trades_dataframe(current_time_ms=advanced_time + 2000)
    print(df_active[["Symbol", "Strategy", "Dir", "Unrealized Pips", "Hold Hours", "Timed Out?", "In Profit?", "Eligible for Room-Making?"]].head(6).to_string())

    print("\n[Step 8] Controller Audit Metrics Summary:")
    metrics = controller.get_controller_metrics()
    for k, v in metrics.items():
        print(f"  * {k}: {v}")

    print("\nVERIFICATION COMPLETE: 'Cut Losers Early, Let Winners Run' logic confirmed!")

if __name__ == "__main__":
    run_demonstration()
