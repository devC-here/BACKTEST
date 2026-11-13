#!/usr/bin/env python3
"""
================================================================================
  PRODUCTION PORTFOLIO SIMULATION: 100-STRATEGY CENTURION FLEET (v4.5)
================================================================================
Simulates the live $5,000 account trading the 100 production strategies over
the 90 Quarantined Holdout Weeks (~42M ticks) under the 20-slot Concurrency
Governor with the Flipped Room-Making Mechanism ("Cut Losses Early & Let Winners Run").

Pricing Friction Included:
- Live MT5 Raw ECN Spreads (Floating, tick-by-tick)
- $4.00/Lot Round-Turn Flat Commission ($0.40/pip per standard lot)
- 0.20 Pips Average Slippage Deduction
================================================================================
"""

import sys
import os
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np
import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent
RESULTS_DIR = ROOT_DIR / "results"
TRADES_FILE = RESULTS_DIR / "unseen_holdout_trades_v45_broker_spread_4usd.json"

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

def simulate_portfolio(
    trades: List[Dict[str, Any]],
    lot_size: float = 0.04,
    initial_equity: float = 5000.0,
    max_slots: int = 20,
    timeout_hours: float = 4.0,
    max_holding_hours: float = 24.0,
    enable_room_making: bool = True
) -> Dict[str, Any]:
    sorted_trades = sorted(trades, key=lambda x: x["entry_time"])
    timeout_ms = int(timeout_hours * 3600 * 1000)
    max_holding_ms = int(max_holding_hours * 3600 * 1000)

    active_slots: List[Dict[str, Any]] = []
    accepted_trades: List[Dict[str, Any]] = []
    blocked_trades = 0
    cut_trades = 0
    zombie_closes = 0

    equity = initial_equity
    peak_equity = initial_equity
    max_dd_usd = 0.0
    max_dd_pct = 0.0

    daily_pnl: Dict[str, float] = {}
    monthly_pnl: Dict[str, float] = {}

    for t in sorted_trades:
        e_time = t["entry_time"]
        sym = t.get("symbol", "EURUSD")
        dollar_per_pip = (PIP_VALUES_PER_LOT_USD.get(sym, 10.00)) * lot_size

        # 1. Natural exit expiry & Zombie checks
        surviving = []
        for s in active_slots:
            s_sym = s.get("symbol", "EURUSD")
            s_dp_pip = (PIP_VALUES_PER_LOT_USD.get(s_sym, 10.00)) * lot_size

            if s["exit_time"] <= e_time:
                # Trade finished naturally
                pnl_usd = s["pips"] * s_dp_pip
                equity += pnl_usd
                if equity > peak_equity: peak_equity = equity
                dd_usd = peak_equity - equity
                dd_pct = (dd_usd / peak_equity) * 100.0 if peak_equity > 0 else 0.0
                if dd_usd > max_dd_usd: max_dd_usd = dd_usd
                if dd_pct > max_dd_pct: max_dd_pct = dd_pct

                dt = pd.to_datetime(s["exit_time"], unit="ms")
                d_str = dt.strftime("%Y-%m-%d")
                m_str = dt.strftime("%Y-%m")
                daily_pnl[d_str] = daily_pnl.get(d_str, 0.0) + pnl_usd
                monthly_pnl[m_str] = monthly_pnl.get(m_str, 0.0) + pnl_usd
            else:
                # Check Max Holding (24h)
                if (e_time - s["entry_time"]) >= max_holding_ms:
                    # Zombie trade forced close
                    pnl_usd = s["pips"] * s_dp_pip
                    equity += pnl_usd
                    if equity > peak_equity: peak_equity = equity
                    dd_usd = peak_equity - equity
                    dd_pct = (dd_usd / peak_equity) * 100.0 if peak_equity > 0 else 0.0
                    if dd_usd > max_dd_usd: max_dd_usd = dd_usd
                    if dd_pct > max_dd_pct: max_dd_pct = dd_pct
                    zombie_closes += 1
                else:
                    surviving.append(s)
        active_slots = surviving

        # 2. Capacity Check & Room-Making
        if len(active_slots) < max_slots:
            active_slots.append(t)
            accepted_trades.append(t)
        else:
            if not enable_room_making:
                blocked_trades += 1
                continue

            # Flipped dynamic room-making: Cut losing trades, shield winners
            candidates_for_cut = []
            for idx, s in enumerate(active_slots):
                is_stagnant_loss = (e_time - s["entry_time"] >= timeout_ms) and (s["pips"] <= 0)
                is_active_loss = (s["pips"] < 0)
                if is_stagnant_loss:
                    candidates_for_cut.append((0, s["pips"], idx)) # Priority 0: Stagnant loss
                elif is_active_loss:
                    candidates_for_cut.append((1, s["pips"], idx)) # Priority 1: Deepest loss

            if candidates_for_cut:
                candidates_for_cut.sort(key=lambda x: (x[0], x[1]))
                best_cut_idx = candidates_for_cut[0][2]
                cut_trade = active_slots.pop(best_cut_idx)

                # Realize loss
                cut_sym = cut_trade.get("symbol", "EURUSD")
                cut_dp = (PIP_VALUES_PER_LOT_USD.get(cut_sym, 10.00)) * lot_size
                pnl_usd = cut_trade["pips"] * cut_dp
                equity += pnl_usd
                if equity > peak_equity: peak_equity = equity
                dd_usd = peak_equity - equity
                dd_pct = (dd_usd / peak_equity) * 100.0 if peak_equity > 0 else 0.0
                if dd_usd > max_dd_usd: max_dd_usd = dd_usd
                if dd_pct > max_dd_pct: max_dd_pct = dd_pct
                cut_trades += 1

                # Admit new trade
                active_slots.append(t)
                accepted_trades.append(t)
            else:
                # All 20 active trades in profit: shield them all!
                blocked_trades += 1

    # Close any lingering positions at end of dataset
    for s in active_slots:
        s_sym = s.get("symbol", "EURUSD")
        s_dp = (PIP_VALUES_PER_LOT_USD.get(s_sym, 10.00)) * lot_size
        pnl_usd = s["pips"] * s_dp
        equity += pnl_usd
        if equity > peak_equity: peak_equity = equity
        dd_usd = peak_equity - equity
        dd_pct = (dd_usd / peak_equity) * 100.0 if peak_equity > 0 else 0.0
        if dd_usd > max_dd_usd: max_dd_usd = dd_usd
        if dd_pct > max_dd_pct: max_dd_pct = dd_pct

    # Portfolio metrics
    net_pips = sum(t["pips"] for t in accepted_trades)
    net_profit_usd = equity - initial_equity
    roi_pct = (net_profit_usd / initial_equity) * 100.0

    # Compute daily Sharpe
    daily_returns = list(daily_pnl.values())
    if len(daily_returns) > 1:
        mean_ret = np.mean(daily_returns)
        std_ret = np.std(daily_returns)
        sharpe = (mean_ret / std_ret) * np.sqrt(252) if std_ret > 0 else 0.0
    else:
        sharpe = 0.0

    return {
        "lot_size": lot_size,
        "initial_equity": initial_equity,
        "final_equity": round(equity, 2),
        "net_profit_usd": round(net_profit_usd, 2),
        "roi_pct": round(roi_pct, 2),
        "total_incoming_trades": len(sorted_trades),
        "accepted_trades": len(accepted_trades),
        "acceptance_rate_pct": round(len(accepted_trades) / len(sorted_trades) * 100.0, 1),
        "blocked_trades": blocked_trades,
        "cut_trades": cut_trades,
        "zombie_closes": zombie_closes,
        "net_pips": round(net_pips, 1),
        "max_drawdown_usd": round(max_dd_usd, 2),
        "max_drawdown_pct": round(max_dd_pct, 2),
        "annualized_sharpe": round(sharpe, 2)
    }

def main():
    if not TRADES_FILE.exists():
        print(f"Error: Holdout trades file {TRADES_FILE} not found!")
        return 1

    with open(TRADES_FILE, "r", encoding="utf-8") as f:
        trades = json.load(f)

    print(f"Loaded {len(trades)} quarantined holdout trades.")
    print("=" * 72)
    print("  SIMULATION RESULTS: 100-STRATEGY FLEET (20-SLOT GOVERNOR)")
    print("=" * 72)

    for lot in [0.03, 0.04]:
        res = simulate_portfolio(trades, lot_size=lot, initial_equity=5000.0, max_slots=20)
        print(f"\n[Lot Size: {lot:.2f} Lots | Capital: ${res['initial_equity']:,.2f}]")
        print(f"  Accepted Trades:       {res['accepted_trades']:,} / {res['total_incoming_trades']:,} ({res['acceptance_rate_pct']}%)")
        print(f"  Room-Making Cuts:      {res['cut_trades']:,} trades (Cut early to make room)")
        print(f"  Blocked Trades:        {res['blocked_trades']:,} (Quashed when all 20 in profit)")
        print(f"  Banked Net Pips:       +{res['net_pips']:,.1f} pips")
        print(f"  Net Cash Profit:       +${res['net_profit_usd']:,.2f} ({res['roi_pct']:+.1f}% ROI)")
        print(f"  Final Account Equity:  ${res['final_equity']:,.2f}")
        print(f"  Peak Max Drawdown:     {res['max_drawdown_pct']:.2f}% (-${res['max_drawdown_usd']:,.2f})")
        print(f"  Annualized Sharpe:     {res['annualized_sharpe']:.2f}")

    print("\n" + "=" * 72)
    print("Simulation completed successfully!")
    return 0

if __name__ == "__main__":
    sys.exit(main())

