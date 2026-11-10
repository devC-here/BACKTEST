#!/usr/bin/env python3
"""
Daily Trade Velocity & Activity Frequency Audit:
Analyzes the exact day-by-day trade distribution across all 260 trading days (52 weeks):
- Percentage of days with 0 trades (idle days)
- Percentage of days with 1 trade
- Percentage of days with 2 trades
- Percentage of days with 3+ trades
- Maximum consecutive idle days (no trades taken)

Evaluates:
1. Squeeze Breakout (40-bar)
2. Squeeze Breakout (20-bar higher frequency)
3. Supertrend VWAP Scalper
4. Asian Range Breakout
5. Market Structure Shift Retest
6. The Unified Multi-Model Daily Active Fleet
"""

from __future__ import annotations
import sys
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Any
import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tick_engine.core.engine import BacktestEngine
from tick_engine.core.dataloader import get_week_dataset
from strategies.comb12_bollinger_squeeze_elite.strategy import BollingerBandwidthExpansionElite
from strategies.comb14_supertrend_vwap_elite.strategy import SupertrendVWAPPullbackElite
from strategies.comb08_asian_breakout_adx_elite.strategy import LondonAsianBreakoutADXElite
from scratch.calibrate_squeeze_high_wr import ParameterizedSqueezeStrategy
from high_winrate_institutional_lab import MarketStructureShiftRetest

PIP_SIZE = 0.0001

MODELS = [
    {"code": "SQZ_40", "name": "Squeeze 40 (52% WR, PF 1.87)", "strat": ParameterizedSqueezeStrategy(lookback=40, squeeze_ratio=1.35, target_rr=1.60, body_ratio=0.80)},
    {"code": "SQZ_20", "name": "Squeeze 20 (42% WR, PF 1.23)", "strat": ParameterizedSqueezeStrategy(lookback=20, squeeze_ratio=1.35, target_rr=1.60, body_ratio=0.70)},
    {"code": "SUP_VWAP", "name": "Supertrend VWAP (+503p)", "strat": SupertrendVWAPPullbackElite()},
    {"code": "ASN_BRK", "name": "Asian Breakout ADX (+235p)", "strat": LondonAsianBreakoutADXElite()},
    {"code": "MSS_RET", "name": "MSS Order Block Retest (+126p)", "strat": MarketStructureShiftRetest()},
]


def run_week(w: int):
    ticks, meta = get_week_dataset(w)
    engine = BacktestEngine(pip_size=PIP_SIZE, use_slippage=True)
    res = {}
    for m in MODELS:
        trades = engine.run(ticks, m["strat"])
        res[m["code"]] = [
            {
                "entry_time": int(t["entry_time"]),
                "exit_time": int(t["exit_time"]),
                "pips": float(t["pips"]),
                "date": pd.to_datetime(t["entry_time"], unit='ms').strftime('%Y-%m-%d')
            }
            for t in trades
        ]
    return res


def main():
    print("Auditing Day-by-Day Trade Activity & Frequency across 52 Weeks...")
    weeks = list(range(1, 53))
    with ProcessPoolExecutor() as executor:
        week_res = list(executor.map(run_week, weeks))

    all_trades: Dict[str, List[Dict[str, Any]]] = {m["code"]: [] for m in MODELS}
    fleet_trades: List[Dict[str, Any]] = []

    for r in week_res:
        for code, t_list in r.items():
            all_trades[code].extend(t_list)
            # Add to combined fleet (SQZ_20 + SUP_VWAP + ASN_BRK + MSS_RET)
            if code in ["SQZ_20", "SUP_VWAP", "ASN_BRK", "MSS_RET"]:
                fleet_trades.extend(t_list)

    # All unique trading dates in 52 weeks
    all_dates = set()
    for t_list in all_trades.values():
        for t in t_list:
            all_dates.add(t["date"])
    
    # Generate full calendar of trading days (52 weeks * 5 days = ~260 days)
    total_trading_days = 260.0

    print("=" * 115)
    print("                     EVERYDAY TRADE FREQUENCY & IDLE DAY AUDIT (52 WEEKS)")
    print("=" * 115)
    print(f"{'Strategy / Portfolio':<35} | {'Total':<6} | {'Trd/Day':<7} | {'0 Trd Days':<11} | {'1 Trd Days':<11} | {'2+ Trd Days':<12} | {'Max Idle Gap':<12}")
    print("-" * 115)

    def analyze_daily(name_str: str, t_list: List[Dict[str, Any]]):
        counts = {}
        for t in t_list:
            d = t["date"]
            counts[d] = counts.get(d, 0) + 1

        n_tot = len(t_list)
        trd_per_day = n_tot / total_trading_days

        days_0 = 260 - len(counts)
        days_1 = sum(1 for c in counts.values() if c == 1)
        days_2plus = sum(1 for c in counts.values() if c >= 2)

        pct_0 = days_0 / total_trading_days * 100.0
        pct_1 = days_1 / total_trading_days * 100.0
        pct_2p = days_2plus / total_trading_days * 100.0

        # Calculate max idle gap (consecutive calendar days without a trade)
        if not t_list:
            return
        sorted_t = sorted(t_list, key=lambda x: x["entry_time"])
        max_gap_days = 0.0
        for idx in range(1, len(sorted_t)):
            gap_d = (sorted_t[idx]["entry_time"] - sorted_t[idx-1]["exit_time"]) / (1000.0 * 86400.0)
            if gap_d > max_gap_days:
                max_gap_days = gap_d

        print(
            f"{name_str:<35} | "
            f"{n_tot:<6} | "
            f"{trd_per_day:<7.2f} | "
            f"{days_0:<4} ({pct_0:<4.1f}%) | "
            f"{days_1:<4} ({pct_1:<4.1f}%) | "
            f"{days_2plus:<4} ({pct_2p:<4.1f}%) | "
            f"{max_gap_days:<5.1f} days"
        )

    for m in MODELS:
        analyze_daily(m["name"], all_trades[m["code"]])

    print("-" * 115)
    analyze_daily("COMBINED DAILY FLEET (4 Models)", fleet_trades)
    print("=" * 115)


if __name__ == "__main__":
    main()
