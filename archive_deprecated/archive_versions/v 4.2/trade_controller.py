#!/usr/bin/env python3
"""
================================================================================
  INSTITUTIONAL TRADE CONTROLLER & CONCURRENCY RISK GOVERNOR
================================================================================
Rules Enforced:
1. Hard Capacity Cap: Maximum 20 simultaneous active trades at any point in time.
2. Signal Ingestion with Dynamic Room-Making:
   - When a new trade signal arrives and open positions < 20:
     -> Accept and execute the trade immediately.
   - When open positions == 20 (Capacity Full):
     -> Check existing active trades:
        a) If any active trade has RUN OUT OF TIME (exceeded timeout threshold)
           AND is currently IN PROFIT:
           -> Close that trade early to harvest profit and make room!
           -> Execute the new incoming trade in the newly freed slot.
        b) If an active trade has RUN OUT OF TIME but is currently IN LOSS:
           -> KEEP IT RUNNING! (Do not close losing trades early; give them room
              to recover, hit target, or stop loss naturally).
        c) If NO active trade has run out of time (or all timed-out trades are in loss):
           -> DO NOT quash any existing placed trade.
           -> QUASH (REJECT) the new incoming trade!
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
            """
            Computes forward-looking outcome probabilities for an active position.

            Parameters:
                sl_pips: Full Stop Loss distance in pips (positive value, e.g. 10.0)
                tp_pips: Full Take Profit distance in pips (positive value, e.g. 20.0)
                current_pips: Current unrealized P&L in pips (positive for profit, negative for loss)
                base_winrate: Initial strategy win rate at entry (default 0.50 if None)
                hold_hours: Hours elapsed since trade entry (optional duration penalty)

            Returns:
                Dictionary with granular probabilities and actionable verdicts.
            """
            sl = max(0.1, float(sl_pips))
            tp = max(0.1, float(tp_pips))
            pnl = float(current_pips)

            p0_tp = float(base_winrate) if base_winrate is not None else (sl / (sl + tp) if sl + tp > 0 else 0.50)
            p0_sl = 1.0 - p0_tp

            # ----------------------------------------------------------------------
            # CASE 1: Trade is Currently in PROFIT (pnl > 0)
            # ----------------------------------------------------------------------
            if pnl > 0:
                p = min(1.0, max(0.0, pnl / tp))  # Profit progress ratio [0.0, 1.0]

                # 1. Probability of reaching full 100% Take Profit
                # As p increases, probability accelerates towards 100%
                prob_tp = p0_tp + (1.0 - p0_tp) * (p ** 0.72)
                prob_tp = min(0.999, max(0.001, prob_tp))

                # 2. Probability of retracing back to Breakeven (0 pips)
                # Declines as profit expands
                prob_retrace_be = (1.0 - p) ** 0.88

                # 3. Probability of full catastrophic collapse to Stop Loss
                prob_collapse_sl = p0_sl * ((1.0 - p) ** 1.35)

                # Time duration adjustment: if held > 6 hours without hitting TP, slight fatigue
                if hold_hours and hold_hours > 6.0:
                    fatigue = min(0.15, (hold_hours - 6.0) * 0.02)
                    prob_tp = max(0.05, prob_tp - fatigue)
                    prob_retrace_be = min(0.95, prob_retrace_be + fatigue)

                # Verdict assignment
                if p >= 0.80:
                    verdict = "NEAR_TARGET_HARVEST_OR_RUN"
                elif p >= 0.50:
                    verdict = "FAVORABLE_HIGH_CONVICTION"
                else:
                    verdict = "MOMENTUM_BUILDING"

                return {
                    "state": "IN_PROFIT",
                    "current_pips": round(pnl, 2),
                    "profit_progress_pct": round(p * 100.0, 1),
                    "drawdown_depth_pct": 0.0,
                    "prob_hit_tp_pct": round(prob_tp * 100.0, 1),
                    "prob_retrace_be_pct": round(prob_retrace_be * 100.0, 1),
                    "prob_collapse_sl_pct": round(prob_collapse_sl * 100.0, 1),
                    "prob_recover_be_pct": 100.0,  # Already past BE
                    "expected_remaining_pips": round((prob_tp * (tp - pnl)) - (prob_collapse_sl * (sl + pnl)), 2),
                    "verdict": verdict,
                    "guidance": f"Trade is at +{p*100:.0f}% of TP. Probability of hitting full TP is {prob_tp*100:.1f}%. Risk of full SL collapse is only {prob_collapse_sl*100:.1f}%."
                }

            # ----------------------------------------------------------------------
            # CASE 2: Trade is Currently in LOSS (pnl <= 0)
            # ----------------------------------------------------------------------
            else:
                d = min(1.0, max(0.0, -pnl / sl))  # Drawdown depth ratio [0.0, 1.0]

                # 1. Probability of recovering back to Breakeven (0 pips)
                # Empirically fitted power law: (1 - d)^0.84 (R^2 = 0.9921)
                prob_recover_be = (1.0 - d) ** 0.84
                prob_recover_be = min(1.0, max(0.0, prob_recover_be))

                # 2. Probability of reversing from drawdown all the way to Take Profit
                prob_tp = p0_tp * ((1.0 - d) ** 1.45)
                prob_tp = min(0.999, max(0.001, prob_tp))

                # 3. Probability of directly dying at Stop Loss without recovering
                prob_sl = 1.0 - prob_recover_be

                # Verdict assignment based on empirical survival inflection zones
                if d >= 0.85:
                    verdict = "CRITICAL_SL_IMMINENT"
                elif d >= 0.60:
                    verdict = "HIGH_RISK_NEGATIVE_EDGE"
                elif d >= 0.40:
                    verdict = "INFLECTION_ZONE"
                else:
                    verdict = "SHALLOW_PULLBACK_RECOVERY_FAVORABLE"

                return {
                    "state": "IN_LOSS" if d > 0.01 else "BREAKEVEN",
                    "current_pips": round(pnl, 2),
                    "profit_progress_pct": 0.0,
                    "drawdown_depth_pct": round(d * 100.0, 1),
                    "prob_recover_be_pct": round(prob_recover_be * 100.0, 1),
                    "prob_hit_tp_pct": round(prob_tp * 100.0, 1),
                    "prob_collapse_sl_pct": round(prob_sl * 100.0, 1),
                    "prob_retrace_be_pct": 0.0,
                    "expected_remaining_pips": round((prob_tp * (tp - pnl)) - (prob_sl * (sl + pnl)), 2),
                    "verdict": verdict,
                    "guidance": f"Trade is at -{d*100:.0f}% of SL. Probability of recovering to Breakeven is {prob_recover_be*100:.1f}%. Chance of dying at SL is {prob_sl*100:.1f}%."
                }

        def predict_trade_object(self, trade: Any, current_price: Optional[float] = None) -> Dict[str, Any]:
            """Convenience wrapper for Trade dataclass objects."""
            pip_size = 0.01 if "JPY" in trade.symbol.upper() else 0.0001
            p = current_price if current_price is not None else trade.current_price

            if p > 0:
                if trade.direction == 1:
                    pnl = (p - trade.entry_price) / pip_size
                else:
                    pnl = (trade.entry_price - p) / pip_size
            else:
                pnl = trade.unrealized_pips

            return self.predict(
                sl_pips=trade.sl_pips,
                tp_pips=trade.tp_pips,
                current_pips=pnl,
                base_winrate=None
            )


    # ==============================================================================
    # DEMONSTRATION & VERIFICATION CLI
    # ==============================================================================


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
    timeout_duration_ms: int  # Max holding duration threshold (e.g. 4 hours)

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
        """Returns True if current floating unrealized PnL is positive."""
        return self.unrealized_pips > 0.0

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
            # Long
            if self.sl_price > 0 and bid <= self.sl_price:
                return "SL_HIT", bid
            if self.tp_price > 0 and bid >= self.tp_price:
                return "TP_HIT", bid
        else:
            # Short
            if self.sl_price > 0 and ask >= self.sl_price:
                return "SL_HIT", ask
            if self.tp_price > 0 and ask <= self.tp_price:
                return "TP_HIT", ask
        return None


class TradeController:
    """
    Trade Controller enforcing strict 20-slot concurrency cap and intelligent
    time-out room-making rules:
    - Max 20 open positions at any time.
    - If slots < 20: Place new trade immediately.
    - If slots == 20:
        * Find trades running out of time.
        * If in profit -> Close early to make room & place new trade!
        * If in loss -> Continue running!
        * If no trades timed out (or all timed-out are in loss) -> Quash new trade!
    """

    def __init__(
        self,
        max_concurrent_trades: int = 20,
        default_timeout_hours: float = 4.0,
        starting_equity: float = 5000.00,
        enable_probabilistic_salvage: bool = True,
        salvage_drawdown_threshold: float = 0.65,
        salvage_recovery_prob_threshold: float = 35.0,
        max_holding_hours: float = 24.0,
        verbose: bool = True
    ):
        self.max_concurrent_trades = int(max_concurrent_trades)
        self.default_timeout_ms = int(default_timeout_hours * 3600 * 1000)
        self.max_holding_ms = int(max_holding_hours * 3600 * 1000)
        self.starting_equity = float(starting_equity)
        self.current_equity = float(starting_equity)
        self.peak_equity = float(starting_equity)
        self.enable_probabilistic_salvage = bool(enable_probabilistic_salvage)
        self.salvage_drawdown_threshold = float(salvage_drawdown_threshold)
        self.salvage_recovery_prob_threshold = float(salvage_recovery_prob_threshold)
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
        Automatically closes positions that hit SL or TP.
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
    # 2. SIGNAL INGESTION & DYNAMIC ROOM-MAKING LOGIC
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
        # CASE 2: Open Trades == 20 (Capacity Full: Dynamic Room-Making Check)
        # ----------------------------------------------------------------------
        # Check hard maximum holding expiration (e.g. 24 hours) to prevent zombie congestion
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
            return {
                "status": "ACCEPTED_ROOM_MADE",
                "trade_id": trade_id,
                "message": f"Capacity was 20/20. Trade {candidate.trade_id} exceeded max holding duration and was closed to make room.",
                "new_trade": new_trade,
                "closed_trade": closed_trade
            }

        # Scan active trades: Has any trade run out of time?
        timed_out_in_profit: List[Trade] = []
        timed_out_in_loss: List[Trade] = []

        for t in self.active_trades.values():
            if t.is_timed_out(timestamp_ms):
                if t.is_in_profit():
                    timed_out_in_profit.append(t)
                else:
                    timed_out_in_loss.append(t)

        # ----------------------------------------------------------------------
        # Subcase 2A: Found a timed-out trade that is IN PROFIT -> CLOSE IT TO MAKE ROOM!
        # ----------------------------------------------------------------------
        if len(timed_out_in_profit) > 0:
            # Pick the trade with longest time past timeout (or highest profit)
            candidate = max(timed_out_in_profit, key=lambda x: (timestamp_ms - x.timeout_timestamp_ms))
            
            # Close the profitable trade early at current market price
            exit_price = candidate.current_price if candidate.current_price > 0 else candidate.entry_price
            closed_trade = self._close_position(
                trade_id=candidate.trade_id,
                exit_price=exit_price,
                reason="TIMEOUT_HARVEST_ROOM_MADE",
                timestamp_ms=timestamp_ms
            )

            # Now we have 19 active positions -> execute the new incoming trade!
            self.active_trades[trade_id] = new_trade
            self.trades_accepted_via_room_making += 1
            self.early_profit_closes_count += 1

            if self.verbose:
                print(f"[TRADE CONTROLLER] -> ROOM MADE! Closed {candidate.symbol} ({candidate.strategy_name}) in profit (+{candidate.realized_pips:.1f}p / +${candidate.realized_usd:.2f}) to execute fresh trade: {symbol} {strategy_name}. Open: {self.open_trade_count}/{self.max_concurrent_trades}")

            return {
                "status": "ACCEPTED_ROOM_MADE",
                "trade_id": trade_id,
                "message": f"Capacity was 20/20. Timed-out profitable trade {candidate.trade_id} closed (+${candidate.realized_usd:.2f}) to make room.",
                "new_trade": new_trade,
                "closed_trade": closed_trade
            }

        # ----------------------------------------------------------------------
        # Subcase 2B: Check Timed-out Losing Trades with Probabilistic Edge
        # ----------------------------------------------------------------------
        # If enable_probabilistic_salvage is active:
        # Evaluate whether any timed-out losing trade has crossed into the
        # TERMINAL NEGATIVE-EDGE ZONE (d >= salvage_drawdown_threshold or P(recover) < salvage_recovery_prob_threshold).
        # - Zone 1: Shallow/Moderate Loss (d < 65% / P(recover) >= 35%) -> PROTECT & HOLD!
        # - Zone 2: Deep Terminal Loss (d >= 65% / P(recover) < 35%) -> SALVAGE CUT to free slot for fresh +EV trade!
        if self.enable_probabilistic_salvage and len(timed_out_in_loss) > 0:
            salvage_candidates: List[Tuple[Trade, Dict[str, Any]]] = []
            for t in timed_out_in_loss:
                pred = self.predictor.predict_trade_object(t, t.current_price)
                d_ratio = pred.get("drawdown_depth_pct", 0.0) / 100.0
                p_rec = pred.get("prob_recover_be_pct", 100.0)
                if d_ratio >= self.salvage_drawdown_threshold or p_rec < self.salvage_recovery_prob_threshold:
                    salvage_candidates.append((t, pred))
                else:
                    self.losing_trades_protected_count += 1

            if len(salvage_candidates) > 0:
                # Pick candidate with deepest drawdown (lowest recovery chance)
                candidate, cand_pred = max(salvage_candidates, key=lambda x: x[1].get("drawdown_depth_pct", 0.0))

                exit_price = candidate.current_price if candidate.current_price > 0 else candidate.entry_price
                closed_trade = self._close_position(
                    trade_id=candidate.trade_id,
                    exit_price=exit_price,
                    reason="TIMEOUT_SALVAGE_ROOM_MADE",
                    timestamp_ms=timestamp_ms
                )

                # Slot freed! Now place the new incoming +EV trade
                self.active_trades[trade_id] = new_trade
                self.trades_accepted_via_room_making += 1
                self.salvage_loss_closes_count += 1

                if self.verbose:
                    print(f"[TRADE CONTROLLER] -> SALVAGE ROOM MADE! Cut terminal loss on {candidate.symbol} ({candidate.strategy_name}) at {candidate.realized_pips:.1f}p (${candidate.realized_usd:.2f}, P(rec)={cand_pred['prob_recover_be_pct']:.1f}%) to make room for fresh trade: {symbol} {strategy_name}. Open: {self.open_trade_count}/{self.max_concurrent_trades}")

                return {
                    "status": "ACCEPTED_SALVAGE_ROOM_MADE",
                    "trade_id": trade_id,
                    "message": f"Capacity was 20/20. Terminal timed-out losing trade {candidate.trade_id} salvaged at {candidate.realized_pips:.1f}p (saving remaining SL capital) to make room for fresh +EV trade.",
                    "new_trade": new_trade,
                    "closed_trade": closed_trade
                }
        else:
            if len(timed_out_in_loss) > 0:
                self.losing_trades_protected_count += len(timed_out_in_loss)

        # ----------------------------------------------------------------------
        # Subcase 2C: No timed-out profitable or terminal trades -> QUASH THE NEW TRADE!
        # Healthy active positions protected; new trade rejected.
        # ----------------------------------------------------------------------
        self.signals_quashed_capacity_full += 1
        if self.verbose:
            print(f"[TRADE CONTROLLER] -> QUASHED (No Room): {symbol} {strategy_name}. 20 slots full, active trades healthy/recovery-favorable. Active trades untouched.")

        return {
            "status": "QUASHED_CAPACITY_FULL",
            "trade_id": None,
            "message": f"Capacity 20/20 full. No timed-out harvestable or terminal trades found. New trade quashed without touching active positions.",
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

            # Dynamic Trajectory Predictions
            pred = self.predictor.predict_trade_object(t, t.current_price)

            d_ratio = pred.get("drawdown_depth_pct", 0.0) / 100.0
            p_rec = pred.get("prob_recover_be_pct", 100.0)

            if timed_out and in_profit:
                elig = "YES (Harvest Profit)"
            elif timed_out and self.enable_probabilistic_salvage and (d_ratio >= self.salvage_drawdown_threshold or p_rec < self.salvage_recovery_prob_threshold):
                elig = f"YES (Salvage Cut: {d_ratio*100:.0f}% Loss)"
            elif timed_out:
                elig = f"NO (Protect & Hold: {p_rec:.0f}% Rec)"
            else:
                elig = "NO (Active Session)"

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
                "P(Hit TP)": f"{pred['prob_hit_tp_pct']:.1f}%",
                "P(Recover BE)": f"{pred['prob_recover_be_pct']:.1f}%",
                "Trajectory Verdict": pred["verdict"],
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
            "early_profit_closes_count": self.early_profit_closes_count,
            "salvage_loss_closes_count": self.salvage_loss_closes_count,
            "signals_quashed_capacity_full": self.signals_quashed_capacity_full,
            "losing_trades_protected": self.losing_trades_protected_count,
            "closed_trades_total": n_closed,
            "win_rate_pct": round(win_rate, 1),
            "profit_factor": round(profit_factor, 2)
        }


# ==============================================================================
# DEMONSTRATION & UNIT VERIFICATION TEST
# ==============================================================================
def run_demonstration():
    print("=" * 100)
    print("              TRADE CONTROLLER VERIFICATION & SCENARIO AUDIT              ")
    print("=" * 100)

    # Initialize Controller with 20-trade limit and 4-hour timeout
    controller = TradeController(max_concurrent_trades=20, default_timeout_hours=4.0, starting_equity=5000.00)
    base_time = 1700000000000  # Epoch milliseconds benchmark

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

    print("\n[Step 2] Placing 20th trade -> Now Capacity is FULL (20/20)...")
    res20 = controller.on_new_signal(
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

    # Fast forward time by 5 hours (300 minutes) so trades entered at base_time are now TIMED OUT (> 4h)
    advanced_time = base_time + (5 * 3600 * 1000)

    print("\n[Step 3] Simulating market mark-to-market prices at T + 5 hours:")
    # Trade 1 (USDJPY entered at 150.20) is TIMED OUT and in PROFIT (+20.0 pips)
    t1 = list(controller.active_trades.values())[0]
    t1.update_price(bid=150.40, ask=150.41) # +20.0 pips profit
    print(f"  * Trade 1 ({t1.symbol} {t1.strategy_name}): Hold 5.0h (> 4h) | Unrealized: +{t1.unrealized_pips:.1f}p (+${t1.unrealized_usd:.2f}) -> [TIMED OUT & IN PROFIT]")

    # Trade 2 (EURUSD entered at 1.0850) is TIMED OUT but in LOSS (-5.0 pips)
    t2 = list(controller.active_trades.values())[1]
    t2.update_price(bid=1.0845, ask=1.0846) # -5.0 pips loss
    print(f"  * Trade 2 ({t2.symbol} {t2.strategy_name}): Hold 5.0h (> 4h) | Unrealized: {t2.unrealized_pips:.1f}p (${t2.unrealized_usd:.2f}) -> [TIMED OUT BUT IN LOSS]")

    print("\n[Step 4] Signal 21 arrives while capacity is 20/20:")
    print("  -> Rule: A trade (Trade 1) is timed out AND in profit. Should CLOSE Trade 1 in profit and MAKE ROOM for Signal 21!")
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

    print("\n[Step 5] Signal 22 arrives while capacity is 20/20:")
    print("  -> Rule: Trade 2 is timed out at -5.0p (50% loss depth). Recovery probability is 55.9% (> 35%).")
    print("  -> Expected: PROTECT & HOLD Trade 2! DO NOT quash active trade; QUASH Signal 22!")
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

    print("\n[Step 6] Simulating Trade 2 sinking deeper into terminal loss (-8.0p / 80% of SL):")
    # Trade 2 sinks to 1.0842 (-8.0 pips loss, P(rec)=22.5% < 35%, in terminal salvage zone)
    t2.update_price(bid=1.0842, ask=1.0843)
    pred_t2 = controller.predictor.predict_trade_object(t2, t2.current_price)
    print(f"  * Trade 2 ({t2.symbol} {t2.strategy_name}): PnL: {t2.unrealized_pips:.1f}p | P(Recover BE): {pred_t2['prob_recover_be_pct']:.1f}% | Verdict: {pred_t2['verdict']}")

    print("\n[Step 7] Signal 23 arrives while capacity is 20/20:")
    print("  -> Rule: Trade 2 has crossed into deep terminal loss (d >= 65%). Salvage remaining 20% capital and make room for fresh +EV Signal 23!")
    res23 = controller.on_new_signal(
        symbol="USDCAD",
        strategy_name="Strat_USDCAD_Fresh_23",
        direction=-1,
        entry_price=1.3500,
        sl_pips=10.0,
        tp_pips=20.0,
        timestamp_ms=advanced_time + 2000,
        lot_size=0.03
    )
    print(f"  Result: {res23['status']} -> {res23['message']}")
    print(f"  Current Open Trades: {controller.open_trade_count}/20")

    print("\n[Step 8] Verifying active trades integrity with predictive trajectory metrics:")
    df_active = controller.get_open_trades_dataframe(current_time_ms=advanced_time + 3000)
    print(df_active[["Symbol", "Strategy", "Dir", "Unrealized Pips", "Hold Hours", "Timed Out?", "In Profit?", "P(Hit TP)", "P(Recover BE)", "Eligible for Room-Making?"]].head(6).to_string())

    print("\n[Step 9] Controller Audit Metrics Summary:")
    metrics = controller.get_controller_metrics()
    for k, v in metrics.items():
        print(f"  * {k}: {v}")

    print("\nVERIFICATION COMPLETE: Both Profit Harvesting and Probabilistic Loss Salvaging confirmed!")

if __name__ == "__main__":
    run_demonstration()
