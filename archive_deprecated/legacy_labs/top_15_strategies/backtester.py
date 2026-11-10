"""
========================================================================================
TOP 15 HIGH-CONVICTION STRATEGY FLEET BACKTESTER (EURUSD)
========================================================================================
Executes fast, institutional bar-by-bar backtesting across the Top 15 verified models:
  S08: Smart Money H1 Demand/Supply Zone + 8 EMA Cross (M15)
  S26: Mamba FX Forex H4 S/R + 8/50 SMA Scalper (M15)
  S28: Chandelier Exit (22, 3.0) + 32 ZLSMA Trend Rider (H1)
  S54: Renko Brick Trend + 10 SMA + OBV Alignment (M15)
  S34: Asia Forex Mentor Smoothed Heikin Ashi (H1)
  S01: ICT Market Structure Shift & Breaker Block Mitigation (M15)
  S06: The Trading School 0.618 - 0.786 Golden Phantom Zone (M15)
  S23: Gomathi Shankar Central Pivot Range + R1/PDH Break (M5)
  S43: HalfTrend (2, 2.0) ATR Dynamic Channel Breakout (M15)
  S18: Multi-Timeframe MACD 62% Win Rate Momentum (M15)
  S36: Trade Like Berlin 'Final Boss' 7 EMA System (M15)
  S10: 20/200 Dual EMA Macro Trend + 20 EMA Pullback (M5)
  S03: TG Capital London Killzone EMA Stack + FVG 50% Eq (M30)
  S14: Rayner Teo 10 EMA + Inside Bar Momentum Breakout (H1)
  S51: Booming Bulls HalfTrend + DMI Intraday (M15)

Features:
- Bar-by-bar simulation with realistic spread deduction (default 0.8 pips).
- Authentic Stop-Loss (SL), Take-Profit (TP1/TP2), Breakeven Trailing, and ST Exits.
- Toggleable Empirical Hours-Filtering (V3.8 mode vs V3.7 24/7 continuous mode).
- Institutional performance scorecards denominated strictly in PIPS and R-MULTIPLES.
========================================================================================
"""

import os
import sys
import json
import argparse
import datetime
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np

# Ensure current directory is on path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

try:
    from evaluators import evaluate_strategy_detail
except ImportError:
    from strategy_evaluators import evaluate_strategy_detail

PIP_SIZE = 0.0001

# The Top 15 High-Conviction Fleet Registry
TOP_15_STRATEGIES = [
    {"id": 8,  "code": "S08", "name": "Smart Money H1 Demand/Supply + 8 EMA", "tf": "M15", "allowed_hours": [0, 2, 3, 5, 6, 7, 8, 9, 12, 14, 15, 17]},
    {"id": 26, "code": "S26", "name": "Mamba FX Forex H4 S/R + 8/50 SMA",    "tf": "M15", "allowed_hours": [1, 2, 3, 7, 8, 9, 12, 13, 14, 15]},
    {"id": 28, "code": "S28", "name": "Chandelier Exit (22, 3.0) + ZLSMA",    "tf": "H1",  "allowed_hours": [2, 3, 7, 8, 9, 13, 14, 15]},
    {"id": 54, "code": "S54", "name": "Renko 10 SMA + OBV Volume Trend",      "tf": "M15", "allowed_hours": [2, 3, 7, 8, 9, 13, 14, 15]},
    {"id": 34, "code": "S34", "name": "Asia Forex Mentor Smoothed Heikin Ashi","tf": "H1", "allowed_hours": [2, 7, 8, 9, 13, 14, 15]},
    {"id": 1,  "code": "S01", "name": "ICT Market Structure Shift & Breaker",  "tf": "M15", "allowed_hours": [1, 2, 3, 7, 8, 9, 13, 14, 15]},
    {"id": 6,  "code": "S06", "name": "The Trading School Fib Golden Zone",     "tf": "M15", "allowed_hours": [1, 2, 7, 8, 9, 13, 14, 15]},
    {"id": 23, "code": "S23", "name": "Gomathi Shankar CPR + R1/PDH Break",    "tf": "M5",  "allowed_hours": [2, 3, 7, 8, 9, 13, 14, 15]},
    {"id": 43, "code": "S43", "name": "HalfTrend (2, 2.0) ATR Channel Break", "tf": "M15", "allowed_hours": [2, 3, 7, 8, 9, 13, 14, 15]},
    {"id": 18, "code": "S18", "name": "Multi-TF MACD Momentum",                 "tf": "M15", "allowed_hours": [2, 3, 7, 8, 9, 13, 14, 15]},
    {"id": 36, "code": "S36", "name": "Trade Like Berlin 'Final Boss' 7 EMA",   "tf": "M15", "allowed_hours": [2, 3, 7, 8, 9, 13, 14, 15]},
    {"id": 10, "code": "S10", "name": "20/200 Dual EMA Macro Trend Pullback",   "tf": "M5",  "allowed_hours": [2, 3, 7, 8, 9, 13, 14, 15]},
    {"id": 3,  "code": "S03", "name": "TG London Killzone EMA Stack + FVG",    "tf": "M30", "allowed_hours": [2, 3, 7, 8, 9, 13, 14, 15]},
    {"id": 14, "code": "S14", "name": "Rayner Teo 10 EMA + Inside Bar",        "tf": "H1",  "allowed_hours": [2, 7, 8, 9, 13, 14, 15]},
    {"id": 51, "code": "S51", "name": "Booming Bulls HalfTrend + DMI",          "tf": "M15", "allowed_hours": [2, 3, 7, 8, 9, 13, 14, 15]},
]


class Top15BacktestEngine:
    def __init__(
        self,
        df: pd.DataFrame,
        spread_pips: float = 0.8,
        enforce_hours: bool = True,
        strategies: Optional[List[int]] = None
    ):
        self.df = df.copy()
        self.spread_pips = spread_pips
        self.spread = spread_pips * PIP_SIZE
        self.enforce_hours = enforce_hours
        
        # Determine strategy subset
        if strategies:
            self.strategies = [s for s in TOP_15_STRATEGIES if s["id"] in strategies]
        else:
            self.strategies = TOP_15_STRATEGIES
            
        # Parse timestamp
        if "time" in self.df.columns:
            self.df["datetime"] = pd.to_datetime(self.df["time"])
            self.df["hour"] = self.df["datetime"].dt.hour
        else:
            self.df["hour"] = 12

    def run(self) -> List[Dict[str, Any]]:
        trades: List[Dict[str, Any]] = []
        open_positions: Dict[int, Dict[str, Any]] = {}
        
        n_bars = len(self.df)
        min_lookback = 30
        
        print(f"[*] Executing Top 15 Backtest on {n_bars} bars...")
        print(f"    - Strategy Count: {len(self.strategies)} models")
        print(f"    - Hours Filter:   {'ENABLED (V3.8 Hours Filter)' if self.enforce_hours else 'DISABLED (V3.7 24/7 Continuous)'}")
        print(f"    - Spread Friction: {self.spread_pips:.1f} pips")
        
        for i in range(min_lookback, n_bars):
            window_df = self.df.iloc[: i + 1]
            curr_bar = self.df.iloc[i]
            prev_bar = self.df.iloc[i - 1]
            
            bar_time = str(curr_bar.get("time", f"Bar_{i}"))
            bar_hour = int(curr_bar.get("hour", 12))
            bar_open = float(curr_bar["open"])
            bar_high = float(curr_bar["high"])
            bar_low = float(curr_bar["low"])
            bar_close = float(curr_bar["close"])
            
            rates_by_tf = {
                "M5": window_df,
                "M15": window_df,
                "M30": window_df,
                "H1": window_df,
                "H2": window_df,
                "H4": window_df,
                "D1": window_df
            }
            
            # 1. Manage Active Positions
            closed_sids = []
            for sid, pos in open_positions.items():
                direction = pos["direction"]
                entry = pos["entry"]
                sl = pos["sl"]
                tp1 = pos["tp1"]
                tp2 = pos["tp2"]
                be_active = pos["be_active"]
                risk = pos["risk"]
                
                # Update Max Favorable / Adverse
                if direction == "BUY":
                    fav = (bar_high - entry) / PIP_SIZE
                    adv = (entry - bar_low) / PIP_SIZE
                else:
                    fav = (entry - bar_low) / PIP_SIZE
                    adv = (bar_high - entry) / PIP_SIZE
                pos["max_fav"] = max(pos["max_fav"], fav)
                pos["max_adv"] = max(pos["max_adv"], adv)
                
                # Check Breakeven Trigger
                if not be_active and risk > 0:
                    curr_profit_r = (fav * PIP_SIZE) / risk
                    if curr_profit_r >= pos["be_trigger_r"]:
                        pos["be_active"] = True
                        pos["sl"] = entry + (0.5 * PIP_SIZE if direction == "BUY" else -0.5 * PIP_SIZE)
                        sl = pos["sl"]
                
                # Check Exits on this bar
                exit_price = None
                exit_reason = None
                
                if direction == "BUY":
                    # Check SL
                    if bar_low <= sl:
                        exit_price = sl
                        exit_reason = "BREAKEVEN_CUT" if be_active else "SL_HIT"
                    # Check TP2
                    elif bar_high >= tp2:
                        exit_price = tp2
                        exit_reason = "TP2_HIT"
                    # Check TP1 partial/scale
                    elif bar_high >= tp1 and not pos["tp1_hit"]:
                        pos["tp1_hit"] = True
                else:
                    # Check SL
                    if bar_high >= sl:
                        exit_price = sl
                        exit_reason = "BREAKEVEN_CUT" if be_active else "SL_HIT"
                    # Check TP2
                    elif bar_low <= tp2:
                        exit_price = tp2
                        exit_reason = "TP2_HIT"
                    # Check TP1 partial
                    elif bar_low <= tp1 and not pos["tp1_hit"]:
                        pos["tp1_hit"] = True
                        
                if exit_price is not None:
                    # Deduct spread friction on exit
                    if direction == "BUY":
                        raw_pips = (exit_price - entry) / PIP_SIZE
                    else:
                        raw_pips = (entry - exit_price) / PIP_SIZE
                    net_pips = raw_pips - self.spread_pips
                    
                    r_mult = net_pips / (pos["risk_pips"] if pos["risk_pips"] > 0 else 8.0)
                    
                    trades.append({
                        "trade_id": f"BT-{pos['strat_code']}-{len(trades)+1:04d}",
                        "strategy_id": sid,
                        "strategy_name": pos["strat_name"],
                        "direction": direction,
                        "entry_price": entry,
                        "exit_price": exit_price,
                        "open_time": pos["open_time"],
                        "close_time": bar_time,
                        "exit_reason": exit_reason,
                        "pnl_pips": round(net_pips, 2),
                        "r_multiple": round(r_mult, 3),
                        "risk_pips": pos["risk_pips"],
                        "max_favorable_pips": round(pos["max_fav"], 1),
                        "max_adverse_pips": round(pos["max_adv"], 1)
                    })
                    closed_sids.append(sid)
                    
            for sid in closed_sids:
                del open_positions[sid]
                
            # 2. Check Signals for Idle Strategies
            for strat in self.strategies:
                sid = strat["id"]
                if sid in open_positions:
                    continue
                    
                # Time of Day Filter
                if self.enforce_hours:
                    if bar_hour not in strat["allowed_hours"]:
                        continue
                        
                # Evaluate Strategy
                try:
                    dir_eval, steps, metrics, sig, h2, h1 = evaluate_strategy_detail(
                        sid, {}, window_df, window_df, window_df, window_df,
                        live_price=bar_close,
                        spread_pips=self.spread_pips,
                        rates_by_tf=rates_by_tf
                    )
                except Exception:
                    continue
                    
                if sig and dir_eval in ["BUY", "SELL"]:
                    entry_p = float(sig["entry"])
                    sl_p = float(sig["sl"])
                    tp1_p = float(sig["tp1"])
                    tp2_p = float(sig["tp2"])
                    risk_val = abs(entry_p - sl_p)
                    risk_pips = float(sig.get("risk_pips", risk_val / PIP_SIZE))
                    
                    open_positions[sid] = {
                        "strat_id": sid,
                        "strat_code": strat["code"],
                        "strat_name": strat["name"],
                        "direction": dir_eval,
                        "entry": entry_p,
                        "sl": sl_p,
                        "tp1": tp1_p,
                        "tp2": tp2_p,
                        "risk": risk_val,
                        "risk_pips": risk_pips,
                        "be_trigger_r": float(sig.get("be_trigger_r", 0.75)),
                        "be_active": False,
                        "tp1_hit": False,
                        "max_fav": 0.0,
                        "max_adv": 0.0,
                        "open_time": bar_time
                    }
                    
        print(f"[+] Simulation Completed! Executed {len(trades)} completed trades across Top 15.")
        return trades


def compute_backtest_scorecard(trades: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not trades:
        return {"n_trades": 0, "net_pips": 0.0, "win_rate": 0.0, "pf": 0.0, "exp": 0.0, "mdd": 0.0, "sharpe": 0.0}
        
    df_t = pd.DataFrame(trades)
    n = len(df_t)
    wins = df_t[df_t["pnl_pips"] > 0]
    losses = df_t[df_t["pnl_pips"] < 0]
    
    n_wins = len(wins)
    n_losses = len(losses)
    win_rate = (n_wins / n) * 100.0 if n > 0 else 0.0
    
    net_pips = float(df_t["pnl_pips"].sum())
    gross_gains = float(wins["pnl_pips"].sum()) if n_wins > 0 else 0.0
    gross_losses = float(abs(losses["pnl_pips"].sum())) if n_losses > 0 else 0.0
    pf = (gross_gains / gross_losses) if gross_losses > 0 else 99.0
    
    exp_pips = net_pips / n if n > 0 else 0.0
    avg_r = float(df_t["r_multiple"].mean()) if n > 0 else 0.0
    
    # Drawdown
    cum_pips = df_t["pnl_pips"].cumsum()
    peaks = np.maximum.accumulate(cum_pips)
    dds = peaks - cum_pips
    mdd = float(dds.max()) if len(dds) > 0 else 0.0
    
    # Sharpe & Sortino
    pnl = df_t["pnl_pips"].values
    mean_pnl = float(np.mean(pnl)) if n > 0 else 0.0
    std_pnl = float(np.std(pnl, ddof=1)) if n > 1 else 0.0
    sharpe = (mean_pnl / std_pnl * np.sqrt(2500)) if std_pnl > 0 else 0.0
    
    neg_pnl = np.minimum(0.0, pnl)
    down_dev = float(np.sqrt(np.mean(neg_pnl ** 2))) if n > 0 else 0.0
    sortino = (mean_pnl / down_dev * np.sqrt(2500)) if down_dev > 0 else 0.0
    
    return {
        "n_trades": n,
        "n_wins": n_wins,
        "n_losses": n_losses,
        "win_rate": round(win_rate, 1),
        "net_pips": round(net_pips, 1),
        "profit_factor": round(pf, 2),
        "expectancy_pips": round(exp_pips, 2),
        "avg_r_multiple": round(avg_r, 3),
        "max_drawdown_pips": round(mdd, 1),
        "ann_sharpe": round(sharpe, 2),
        "ann_sortino": round(sortino, 2)
    }


def print_scorecard(card: Dict[str, Any], label: str):
    print("\n" + "=" * 80)
    print(f"INSTITUTIONAL PERFORMANCE SCORECARD: {label}")
    print("=" * 80)
    print(f"  Total Trades Executed:   {card['n_trades']} trades ({card['n_wins']} W / {card['n_losses']} L)")
    print(f"  Win Rate (%):            {card['win_rate']:.1f}%")
    print(f"  Net Realized Pips:       {card['net_pips']:>+7.1f} pips")
    print(f"  Expectancy per Bet:      {card['expectancy_pips']:>+6.2f} pips/trade")
    print(f"  Average R-Multiple:      {card['avg_r_multiple']:>+6.3f} R")
    print(f"  Profit Factor (PF):      {card['profit_factor']:.2f}")
    print(f"  Max Drawdown (MDD):      {card['max_drawdown_pips']:.1f} pips")
    print(f"  Annualized Sharpe Ratio: {card['ann_sharpe']:.2f}")
    print(f"  Annualized Sortino:      {card['ann_sortino']:.2f}")
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Top 15 High-Conviction Strategy Fleet Backtester")
    parser.add_argument("--csv", default=os.path.join(CURRENT_DIR, "sample_eurusd_m15.csv"), help="Path to historical bar CSV")
    parser.add_argument("--strategy", default="all", help="Strategy ID to backtest (e.g. S08 or 'all')")
    parser.add_argument("--hours-filter", dest="hours_filter", action="store_true", default=True, help="Enforce V3.8 empirical hours filter")
    parser.add_argument("--no-hours-filter", dest="hours_filter", action="store_false", help="Run 24/7 continuous (V3.7 mode)")
    parser.add_argument("--spread", type=float, default=0.8, help="Spread friction in pips (default: 0.8)")
    parser.add_argument("--out-csv", default=os.path.join(CURRENT_DIR, "top_15_backtest_trades.csv"), help="Output trades CSV path")
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"[!] Historical bar CSV not found at: {args.csv}")
        return

    df = pd.read_csv(args.csv)
    
    # Filter strategies if requested
    strat_ids = None
    if args.strategy.lower() != "all":
        sid = int(args.strategy.upper().replace("S", ""))
        strat_ids = [sid]

    engine = Top15BacktestEngine(
        df=df,
        spread_pips=args.spread,
        enforce_hours=args.hours_filter,
        strategies=strat_ids
    )
    
    trades = engine.run()
    
    if trades:
        df_trades = pd.DataFrame(trades)
        df_trades.to_csv(args.out_csv, index=False)
        print(f"[+] Saved individual trade log to: {args.out_csv}")
        
        # Overall scorecard
        mode_label = "V3.8 Hours Filtered" if args.hours_filter else "V3.7 Continuous 24/7"
        total_card = compute_backtest_scorecard(trades)
        print_scorecard(total_card, f"Top 15 Fleet ({mode_label})")
        
        # Per-Strategy breakdown table
        print("=" * 95)
        print("PER-STRATEGY BREAKDOWN ACROSS TOP 15 FLEET:")
        print("=" * 95)
        fmt = "{:<6} | {:<40} | {:<6} | {:<6} | {:<10} | {:<6} | {:<7} | {:<6}"
        print(fmt.format("ID", "Strategy Name", "Bets", "WR%", "Net Pips", "PF", "Exp(p)", "MDD(p)"))
        print("-" * 95)
        
        for s in TOP_15_STRATEGIES:
            sid = s["id"]
            s_trades = [t for t in trades if t["strategy_id"] == sid]
            if s_trades:
                sc = compute_backtest_scorecard(s_trades)
                print(fmt.format(
                    s["code"], s["name"][:40], sc["n_trades"],
                    f"{sc['win_rate']:.1f}%", f"{sc['net_pips']:>+6.1f} p",
                    f"{sc['profit_factor']:.2f}", f"{sc['expectancy_pips']:>+5.2f} p",
                    f"{sc['max_drawdown_pips']:.1f} p"
                ))
            else:
                print(fmt.format(s["code"], s["name"][:40], 0, "0.0%", "0.0 p", "0.00", "0.00 p", "0.0 p"))
        print("=" * 95 + "\n")
    else:
        print("[!] No trades were generated with the given parameters and data.")


if __name__ == "__main__":
    main()
