#!/usr/bin/env python3
"""
================================================================================
  MASTER 80-STRATEGY PORTFOLIO SIMULATION ($5,000 ACCOUNT / 20 CONCURRENT SLOTS)
================================================================================
Parameters:
  - Account Equity: $5,000.00 Starting Balance
  - Total Fleet: 80 Quantitative Strategies (Top 8 Champions across all 10 Symbols)
  - Concurrency Capacity: Maximum 20 simultaneous active strategy instances
  - Position Sizing: Evaluated at 0.03 lots AND 0.04 lots
  - Temporal Horizon: 5.75 Continuous Years (300 Weeks, Jan 2021 – Oct 2026)
  - Data Granularity: Millisecond-accurate event-driven queue from Dukascopy Ticks

Usage:
  .venv\\Scripts\\python.exe data_engineering_suite/simulate_5k_80_strategy_fleet.py
================================================================================
"""

from __future__ import annotations
import sys
import os
import time
import json
import datetime
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Any, Tuple, Optional

import numpy as np
import pandas as pd

# OpenBLAS thread control
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

sys.dont_write_bytecode = True

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

RESULTS_DIR = ROOT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DATA_SPLITS_FILE = ROOT_DIR / "data_splits" / "master_70_30_splits.json"

from tick_engine.core.engine import run_matching_loop
from tick_engine.core.dataloader import get_week_dataset
from data_engineering_suite.bars import build_time_bars
from data_engineering_suite.master_symbol_alpha_discovery_suite import (
    get_symbol_profile,
    StrategySessionVWAPFade,
    StrategyFibGoldenPocket,
    StrategyDailyPivotSniper,
    StrategyMacroDonchianRetest,
    StrategyElderTripleScreen,
    StrategyKeltnerSqueezeExplosion,
    StrategyLondonORB,
    StrategyChaikinVolSurge
)


# ==============================================================================
# 1. CURRENCY CONVERSION & PIP VALUE SPECIFICATIONS
# ==============================================================================

# Value in USD per 1.00 full lot (100,000 units) for 1 standard pip
# Standard pricing conversions based on average multi-year exchange rates:
PIP_VALUES_PER_LOT_USD: Dict[str, float] = {
    "EURUSD": 10.00,  # Quote is USD: 1 pip = $10.00 / lot
    "GBPCHF": 10.00,  # CHF cross approx USD parity
    "USDJPY": 6.80,   # 1,000 JPY / 147 USDJPY avg = $6.80 / lot
    "GBPJPY": 6.80,   # JPY quote = $6.80 / lot
    "EURGBP": 13.00,  # 10 GBP * 1.30 GBPUSD avg = $13.00 / lot
    "USDCAD": 7.40,   # 10 CAD / 1.35 USDCAD avg = $7.40 / lot
    "AUDCAD": 7.40,   # CAD quote = $7.40 / lot
    "NZDCAD": 7.40,   # CAD quote = $7.40 / lot
    "AUDNZD": 6.10,   # 10 NZD / 1.64 NZDUSD avg = $6.10 / lot
    "EURNZD": 6.10,   # NZD quote = $6.10 / lot
}


# ==============================================================================
# 2. DEFINITION OF TOP 8 CHAMPIONS PER SYMBOL
# ==============================================================================

def get_champion_battery_for_symbol(symbol: str) -> List[Any]:
    s = symbol.upper()
    prof = get_symbol_profile(s)
    sl_m = prof["sl_mult"]

    # 1. GBPJPY
    if s == "GBPJPY":
        return [
            StrategyFibGoldenPocket("FibPocket_LB36_SL21.6_TP54.0", sl_pips=12.0*sl_m, tp_pips=30.0*sl_m, lookback_h1=36),
            StrategyFibGoldenPocket("FibPocket_LB36_SL21.6_TP43.2", sl_pips=12.0*sl_m, tp_pips=24.0*sl_m, lookback_h1=36),
            StrategySessionVWAPFade("VWAP_B2.0_SL16.2_TP32.4", sl_pips=9.0*sl_m, tp_pips=18.0*sl_m, band_mult=2.0),
            StrategySessionVWAPFade("VWAP_B2.5_SL16.2_TP32.4", sl_pips=9.0*sl_m, tp_pips=18.0*sl_m, band_mult=2.5),
            StrategyElderTripleScreen("ElderTripleScreen_SL12.6_TP25.2", sl_pips=7.0*sl_m, tp_pips=14.0*sl_m),
            StrategyElderTripleScreen("ElderTripleScreen_SL14.4_TP28.8", sl_pips=8.0*sl_m, tp_pips=16.0*sl_m),
            StrategyDailyPivotSniper("DailyPivot_SL16.2_TP32.4", sl_pips=9.0*sl_m, tp_pips=18.0*sl_m),
            StrategyDailyPivotSniper("DailyPivot_SL12.6_TP25.2", sl_pips=7.0*sl_m, tp_pips=14.0*sl_m),
        ]
    # 2. USDCAD
    elif s == "USDCAD":
        return [
            StrategySessionVWAPFade("VWAP_B2.0_SL9.0_TP18.0", sl_pips=9.0, tp_pips=18.0, band_mult=2.0),
            StrategyFibGoldenPocket("FibPocket_LB36_SL12.0_TP30.0", sl_pips=12.0, tp_pips=30.0, lookback_h1=36),
            StrategySessionVWAPFade("VWAP_B2.0_SL8.0_TP16.0", sl_pips=8.0, tp_pips=16.0, band_mult=2.0),
            StrategyFibGoldenPocket("FibPocket_LB36_SL12.0_TP24.0", sl_pips=12.0, tp_pips=24.0, lookback_h1=36),
            StrategyElderTripleScreen("ElderTripleScreen_SL12.0_TP24.0", sl_pips=12.0, tp_pips=24.0),
            StrategyElderTripleScreen("ElderTripleScreen_SL9.0_TP18.0", sl_pips=9.0, tp_pips=18.0),
            StrategyDailyPivotSniper("DailyPivot_SL9.0_TP18.0", sl_pips=9.0, tp_pips=18.0),
            StrategyDailyPivotSniper("DailyPivot_SL8.0_TP16.0", sl_pips=8.0, tp_pips=16.0),
        ]
    # 3. AUDCAD
    elif s == "AUDCAD":
        return [
            StrategyFibGoldenPocket("FibPocket_LB36_SL14.4_TP36.0", sl_pips=12.0*sl_m, tp_pips=30.0*sl_m, lookback_h1=36),
            StrategyFibGoldenPocket("FibPocket_LB36_SL14.4_TP28.8", sl_pips=12.0*sl_m, tp_pips=24.0*sl_m, lookback_h1=36),
            StrategyElderTripleScreen("ElderTripleScreen_SL14.4_TP28.8", sl_pips=12.0*sl_m, tp_pips=24.0*sl_m),
            StrategyFibGoldenPocket("FibPocket_LB48_SL8.4_TP21.0", sl_pips=7.0*sl_m, tp_pips=17.5*sl_m, lookback_h1=48),
            StrategyFibGoldenPocket("FibPocket_LB48_SL9.6_TP19.2", sl_pips=8.0*sl_m, tp_pips=16.0*sl_m, lookback_h1=48),
            StrategyFibGoldenPocket("FibPocket_LB72_SL8.4_TP21.0", sl_pips=7.0*sl_m, tp_pips=17.5*sl_m, lookback_h1=72),
            StrategyFibGoldenPocket("FibPocket_LB72_SL9.6_TP19.2", sl_pips=8.0*sl_m, tp_pips=16.0*sl_m, lookback_h1=72),
            StrategyFibGoldenPocket("FibPocket_LB48_SL14.4_TP36.0", sl_pips=12.0*sl_m, tp_pips=30.0*sl_m, lookback_h1=48),
        ]
    # 4. GBPCHF
    elif s == "GBPCHF":
        return [
            StrategyFibGoldenPocket("FibPocket_LB48_SL14.4_TP36.0", sl_pips=12.0*sl_m, tp_pips=30.0*sl_m, lookback_h1=48),
            StrategyFibGoldenPocket("FibPocket_LB48_SL14.4_TP28.8", sl_pips=12.0*sl_m, tp_pips=24.0*sl_m, lookback_h1=48),
            StrategySessionVWAPFade("VWAP_B2.5_SL10.8_TP21.6", sl_pips=9.0*sl_m, tp_pips=18.0*sl_m, band_mult=2.5),
            StrategySessionVWAPFade("VWAP_B2.5_SL9.6_TP19.2", sl_pips=8.0*sl_m, tp_pips=16.0*sl_m, band_mult=2.5),
            StrategyFibGoldenPocket("FibPocket_LB36_SL14.4_TP36.0", sl_pips=12.0*sl_m, tp_pips=30.0*sl_m, lookback_h1=36),
            StrategyFibGoldenPocket("FibPocket_LB36_SL14.4_TP28.8", sl_pips=12.0*sl_m, tp_pips=24.0*sl_m, lookback_h1=36),
            StrategyFibGoldenPocket("FibPocket_LB48_SL9.6_TP19.2", sl_pips=8.0*sl_m, tp_pips=16.0*sl_m, lookback_h1=48),
            StrategyFibGoldenPocket("FibPocket_LB48_SL8.4_TP21.0", sl_pips=7.0*sl_m, tp_pips=17.5*sl_m, lookback_h1=48),
        ]
    # 5. EURNZD
    elif s == "EURNZD":
        return [
            StrategyFibGoldenPocket("FibPocket_LB48_SL18.0_TP45.0", sl_pips=12.0*sl_m, tp_pips=30.0*sl_m, lookback_h1=48),
            StrategyFibGoldenPocket("FibPocket_LB36_SL18.0_TP45.0", sl_pips=12.0*sl_m, tp_pips=30.0*sl_m, lookback_h1=36),
            StrategyElderTripleScreen("ElderTripleScreen_SL12.0_TP24.0", sl_pips=8.0*sl_m, tp_pips=16.0*sl_m),
            StrategyElderTripleScreen("ElderTripleScreen_SL10.5_TP21.0", sl_pips=7.0*sl_m, tp_pips=14.0*sl_m),
            StrategyFibGoldenPocket("FibPocket_LB48_SL18.0_TP36.0", sl_pips=12.0*sl_m, tp_pips=24.0*sl_m, lookback_h1=48),
            StrategyFibGoldenPocket("FibPocket_LB36_SL18.0_TP36.0", sl_pips=12.0*sl_m, tp_pips=24.0*sl_m, lookback_h1=36),
            StrategyFibGoldenPocket("FibPocket_LB72_SL12.0_TP24.0", sl_pips=8.0*sl_m, tp_pips=16.0*sl_m, lookback_h1=72),
            StrategyFibGoldenPocket("FibPocket_LB72_SL10.5_TP26.2", sl_pips=7.0*sl_m, tp_pips=17.5*sl_m, lookback_h1=72),
        ]
    # 6. NZDCAD
    elif s == "NZDCAD":
        return [
            StrategyFibGoldenPocket("FibPocket_LB72_SL14.4_TP36.0", sl_pips=12.0*sl_m, tp_pips=30.0*sl_m, lookback_h1=72),
            StrategyFibGoldenPocket("FibPocket_LB48_SL14.4_TP36.0", sl_pips=12.0*sl_m, tp_pips=30.0*sl_m, lookback_h1=48),
            StrategyFibGoldenPocket("FibPocket_LB72_SL14.4_TP28.8", sl_pips=12.0*sl_m, tp_pips=24.0*sl_m, lookback_h1=72),
            StrategyFibGoldenPocket("FibPocket_LB48_SL14.4_TP28.8", sl_pips=12.0*sl_m, tp_pips=24.0*sl_m, lookback_h1=48),
            StrategyFibGoldenPocket("FibPocket_LB36_SL14.4_TP28.8", sl_pips=12.0*sl_m, tp_pips=24.0*sl_m, lookback_h1=36),
            StrategyFibGoldenPocket("FibPocket_LB72_SL9.6_TP19.2", sl_pips=8.0*sl_m, tp_pips=16.0*sl_m, lookback_h1=72),
            StrategyFibGoldenPocket("FibPocket_LB36_SL14.4_TP36.0", sl_pips=12.0*sl_m, tp_pips=30.0*sl_m, lookback_h1=36),
            StrategyFibGoldenPocket("FibPocket_LB72_SL8.4_TP21.0", sl_pips=7.0*sl_m, tp_pips=17.5*sl_m, lookback_h1=72),
        ]
    # 7. AUDNZD
    elif s == "AUDNZD":
        return [
            StrategyFibGoldenPocket("DISC115_FibPocket_LB48", sl_pips=14.0, tp_pips=28.0, lookback_h1=48),
            StrategyFibGoldenPocket("DISC116_FibPocket_Precision", sl_pips=12.0, tp_pips=24.0, lookback_h1=48),
            StrategyFibGoldenPocket("DISC117_FibPocket_HighWR", sl_pips=12.0, tp_pips=20.0, lookback_h1=48),
            StrategyFibGoldenPocket("DISC118_FibPocket_LB36", sl_pips=14.0, tp_pips=28.0, lookback_h1=36),
            StrategySessionVWAPFade("DISC120_SessionVWAP_B2.5", sl_pips=12.0, tp_pips=24.0, band_mult=2.5),
            StrategyMacroDonchianRetest("DISC111_MacroStructure_H4", sl_pips=14.0, tp_pips=28.0, lookback_h4=20),
            StrategyFibGoldenPocket("DISC129_FibPocket_LB72", sl_pips=12.0, tp_pips=30.0, lookback_h1=72),
            StrategyMacroDonchianRetest("DISC112_MacroPrecision_H4", sl_pips=12.0, tp_pips=24.0, lookback_h4=10),
        ]
    # 8. EURGBP
    elif s == "EURGBP":
        return [
            StrategySessionVWAPFade("DISC121_SessionVWAP_Fade", sl_pips=7.0, tp_pips=14.0, band_mult=2.0),
            StrategyDailyPivotSniper("DISC123_DailyPivotSniper", sl_pips=7.5, tp_pips=15.0),
            StrategyFibGoldenPocket("DISC122_FibPocket_LB36", sl_pips=7.0, tp_pips=17.5, lookback_h1=36),
            StrategyElderTripleScreen("DISC56_ElderTripleScreen", sl_pips=7.0, tp_pips=14.0),
            StrategyFibGoldenPocket("DISC125_FibPocket_LB48", sl_pips=8.0, tp_pips=16.0, lookback_h1=48),
            StrategyDailyPivotSniper("DISC48_CamarillaFloor", sl_pips=8.5, tp_pips=14.0),
            StrategyDailyPivotSniper("DISC46_DailyPivotWall", sl_pips=9.0, tp_pips=14.5),
            StrategyKeltnerSqueezeExplosion("DISC58_H1KeltnerSqueeze", sl_pips=8.0, tp_pips=16.0),
        ]
    # 9. EURUSD
    elif s == "EURUSD":
        return [
            StrategyElderTripleScreen("DISC86_EMACrossATRExpansion", sl_pips=10.0, tp_pips=20.0),
            StrategyChaikinVolSurge("DISC105_ChaikinVolSurge", sl_pips=11.0, tp_pips=22.0),
            StrategyDailyPivotSniper("DISC40_RSIRegimeRangeShift", sl_pips=10.0, tp_pips=18.0),
            StrategyFibGoldenPocket("DISC103_DoubleSwingNeckline", sl_pips=9.0, tp_pips=22.5, lookback_h1=36),
            StrategyDailyPivotSniper("DISC45_ContinuousM5Rejection", sl_pips=6.0, tp_pips=8.0),
            StrategyElderTripleScreen("DISC97_CardwellRSI50Momentum", sl_pips=10.0, tp_pips=20.0),
            StrategyLondonORB("DISC85_SessionOpenImpulse", sl_pips=11.0, tp_pips=22.0),
            StrategySessionVWAPFade("DISC93_LondonVWAPPullback", sl_pips=10.0, tp_pips=20.0, band_mult=2.0),
        ]
    # 10. USDJPY
    elif s == "USDJPY":
        return [
            StrategyLondonORB("DISC90_OpeningRangeImpulse", sl_pips=13.0, tp_pips=23.5),
            StrategyKeltnerSqueezeExplosion("DISC102_KeltnerMidlineExpansion", sl_pips=12.0, tp_pips=24.0),
            StrategyChaikinVolSurge("DISC106_ChaikinVolSurge", sl_pips=14.0, tp_pips=28.0),
            StrategyElderTripleScreen("DISC96_CardwellRSI50Momentum", sl_pips=12.0, tp_pips=24.0),
            StrategyFibGoldenPocket("DISC104_ThreeDriveIgnition", sl_pips=14.0, tp_pips=28.0, lookback_h1=48),
            StrategySessionVWAPFade("DISC94_LondonVWAPPullback", sl_pips=12.0, tp_pips=24.0, band_mult=2.0),
            StrategyMacroDonchianRetest("DISC91_SessionDonchianMidline", sl_pips=12.0, tp_pips=24.0, lookback_h4=20),
            StrategyDailyPivotSniper("DISC87_PivotR1S1Breakout", sl_pips=13.0, tp_pips=26.0),
        ]
    else:
        return []


# ==============================================================================
# 3. WORKER: RUN 8 CHAMPIONS ON A SINGLE WEEK
# ==============================================================================

def evaluate_symbol_week_champions_worker(args: Tuple[str, int, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    symbol, week_num, profile = args
    raw_dir = Path(f"data/{symbol}")
    pip_size = profile["pip_size"]
    spread_limit = profile["spread_limit"]

    try:
        ticks, meta = get_week_dataset(week_num, raw_dir=raw_dir, auto_convert=False)
    except Exception:
        return None

    if len(ticks) < 1000:
        return None

    m15 = build_time_bars(ticks, timeframe_min=15, pip_size=pip_size)
    if len(m15) < 30:
        return None
    h1 = m15.resample('1h').agg({'high': 'max', 'low': 'min', 'close': 'last', 'open': 'first'}).dropna()
    h4 = m15.resample('4h').agg({'high': 'max', 'low': 'min', 'close': 'last', 'open': 'first'}).dropna()

    if len(h1) < 15 or len(h4) < 8:
        return None

    champions = get_champion_battery_for_symbol(symbol)
    trades_by_strat: Dict[str, List[Dict[str, Any]]] = {}

    for strat in champions:
        signals = strat.generate_signals(ticks, m15, h1, h4, pip_size, spread_limit)
        if not np.any(signals != 0):
            continue

        raw_trades = run_matching_loop(ticks, signals, pip_size=pip_size, sl_pips=strat.sl_pips, tp_pips=strat.tp_pips)

        tr_list = []
        for t in raw_trades:
            tr_list.append({
                "strategy": strat.name,
                "symbol": symbol,
                "week": week_num,
                "entry_time": int(t["entry_time"]),
                "exit_time": int(t["exit_time"]),
                "direction": int(t["direction"]),
                "entry_price": float(t["entry_price"]),
                "exit_price": float(t["exit_price"]),
                "pips": float(t["pips"]),
                "exit_reason": int(t["exit_reason"])
            })
        trades_by_strat[strat.name] = tr_list

    return {
        "symbol": symbol,
        "week": week_num,
        "trades_by_strat": trades_by_strat
    }


# ==============================================================================
# 4. GENERATE / LOAD ALL 80 CHAMPION TRADES
# ==============================================================================

def load_or_generate_all_80_trades(symbols: List[str], max_workers: int = 12) -> List[Dict[str, Any]]:
    all_trades: List[Dict[str, Any]] = []

    for sym in symbols:
        m_file = RESULTS_DIR / f"master_{sym.lower()}_trades.json"
        c_file = RESULTS_DIR / f"champions_8_{sym.lower()}_trades.json"
        if sym == "EURUSD" and not c_file.exists():
            alt_eur = RESULTS_DIR / "master_eurusd_14_models_trades.json"
            target_file = alt_eur if alt_eur.exists() else None
        else:
            target_file = m_file if m_file.exists() else (c_file if c_file.exists() else None)

        if target_file and target_file.exists():
            print(f"Loading existing trade cache for {sym} from {target_file}...")
            with open(target_file, "r", encoding="utf-8") as f:
                d = json.load(f)
            raw_sym_trades = []
            if isinstance(d, dict):
                for s_name, tr_list in d.items():
                    for tr in tr_list:
                        if "strategy" not in tr:
                            tr["strategy"] = s_name
                        raw_sym_trades.append(tr)
            else:
                raw_sym_trades = d

            sym_trades = []
            for t in raw_sym_trades:
                sym_trades.append({
                    "strategy": t.get("strategy", "Unknown"),
                    "symbol": sym,
                    "week": int(t.get("week", 0)),
                    "entry_time": int(t.get("entry_time", 0)),
                    "exit_time": int(t.get("exit_time", 0)),
                    "pips": float(t.get("pips") if t.get("pips") is not None else t.get("pnl_pips", 0.0)),
                    "direction": int(t.get("direction", 1)),
                    "exit_reason": int(t.get("exit_reason", 0)),
                    "entry_price": float(t.get("entry_price", 0.0)),
                    "exit_price": float(t.get("exit_price", 0.0)),
                    "sl_pips": float(t.get("sl_pips", 0.0)),
                    "tp_pips": float(t.get("tp_pips", 0.0))
                })

            if len(sym_trades) > 0:
                all_trades.extend(sym_trades)
                print(f"  -> {sym}: {len(sym_trades)} trades loaded.")
                continue

        print(f"Simulating 8 Champions for {sym} across 300 weeks ({max_workers} CPU workers)...")
        prof = get_symbol_profile(sym)
        tasks = [(sym, w, prof) for w in range(1, 301)]
        sym_trades: List[Dict[str, Any]] = []
        completed = 0

        t0 = time.time()
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(evaluate_symbol_week_champions_worker, arg): arg for arg in tasks}
            for future in as_completed(futures):
                res = future.result()
                if res and "trades_by_strat" in res:
                    for strat_name, trs in res["trades_by_strat"].items():
                        sym_trades.extend(trs)
                completed += 1
                if completed % 100 == 0 or completed == 300:
                    print(f"  [{sym}] {completed}/300 weeks completed ({time.time() - t0:.1f}s)...")

        # Save symbol trades cache
        with open(c_file, "w", encoding="utf-8") as f:
            json.dump(sym_trades, f)
        print(f"  -> {sym} Complete: {len(sym_trades)} trades generated and saved to {c_file}.\n")
        all_trades.extend(sym_trades)

    return all_trades


# ==============================================================================
# 5. PORTFOLIO ENGINE: $5,000 ACCOUNT / 20 CONCURRENT INSTANCE SIMULATOR
# ==============================================================================

def simulate_portfolio(
    trades: List[Dict[str, Any]],
    starting_equity: float = 5000.0,
    max_concurrent_instances: int = 20,
    lot_size: float = 0.03
) -> Dict[str, Any]:
    """
    Simulates a single unified account running with a hard concurrency cap.
    """
    # Sort all trades strictly chronologically by entry_time
    sorted_trades = sorted(trades, key=lambda x: x["entry_time"])
    n_signals = len(sorted_trades)

    equity = starting_equity
    peak_equity = starting_equity
    max_dd_dollars = 0.0
    max_dd_pct = 0.0

    # Active positions currently open in the market: list of dicts with exit_time and profit
    active_positions: List[Dict[str, Any]] = []

    accepted_trades: List[Dict[str, Any]] = []
    blocked_count = 0

    concurrency_history: List[int] = []
    daily_pnls: Dict[str, float] = {}
    yearly_pnls: Dict[str, float] = {}

    for t in sorted_trades:
        e_time = t["entry_time"]
        sym = t.get("symbol", "EURUSD")
        pip_val = PIP_VALUES_PER_LOT_USD.get(sym, 10.00) * lot_size
        cash_pnl = t["pips"] * pip_val

        # Step 1: Expire all open positions that have closed prior to or at this entry_time
        still_open = []
        for pos in active_positions:
            if pos["exit_time"] <= e_time:
                # Realize PnL
                equity += pos["cash_pnl"]
                if equity > peak_equity:
                    peak_equity = equity
                dd_d = peak_equity - equity
                dd_p = (dd_d / peak_equity * 100) if peak_equity > 0 else 0.0
                if dd_d > max_dd_dollars:
                    max_dd_dollars = dd_d
                if dd_p > max_dd_pct:
                    max_dd_pct = dd_p

                # Daily and yearly tracking
                dt_str = datetime.datetime.fromtimestamp(pos["exit_time"] / 1000.0, tz=datetime.timezone.utc).strftime("%Y-%m-%d")
                yr_str = dt_str[:4]
                daily_pnls[dt_str] = daily_pnls.get(dt_str, 0.0) + pos["cash_pnl"]
                yearly_pnls[yr_str] = yearly_pnls.get(yr_str, 0.0) + pos["cash_pnl"]
            else:
                still_open.append(pos)
        active_positions = still_open

        # Step 2: Check concurrency constraint (Max 20 simultaneous active strategy instances)
        if len(active_positions) >= max_concurrent_instances:
            blocked_count += 1
            continue

        # Step 3: Accept and execute trade
        pos_entry = {
            "strategy": t["strategy"],
            "symbol": sym,
            "entry_time": e_time,
            "exit_time": t["exit_time"],
            "pips": t["pips"],
            "cash_pnl": cash_pnl,
            "lot_size": lot_size
        }
        active_positions.append(pos_entry)
        concurrency_history.append(len(active_positions))
        accepted_trades.append(pos_entry)

    # Step 4: Close remaining active positions at end of backtest
    active_positions.sort(key=lambda x: x["exit_time"])
    for pos in active_positions:
        equity += pos["cash_pnl"]
        if equity > peak_equity:
            peak_equity = equity
        dd_d = peak_equity - equity
        dd_p = (dd_d / peak_equity * 100) if peak_equity > 0 else 0.0
        if dd_d > max_dd_dollars:
            max_dd_dollars = dd_d
        if dd_p > max_dd_pct:
            max_dd_pct = dd_p

        dt_str = datetime.datetime.fromtimestamp(pos["exit_time"] / 1000.0, tz=datetime.timezone.utc).strftime("%Y-%m-%d")
        yr_str = dt_str[:4]
        daily_pnls[dt_str] = daily_pnls.get(dt_str, 0.0) + pos["cash_pnl"]
        yearly_pnls[yr_str] = yearly_pnls.get(yr_str, 0.0) + pos["cash_pnl"]

    # Calculate Summary Metrics
    total_net_profit = equity - starting_equity
    return_pct = (total_net_profit / starting_equity) * 100.0
    n_acc = len(accepted_trades)

    cash_pnls = [x["cash_pnl"] for x in accepted_trades]
    wins = [x for x in cash_pnls if x > 0]
    losses = [-x for x in cash_pnls if x < 0]

    sum_w = sum(wins)
    sum_l = sum(losses)
    pf = round(sum_w / sum_l, 2) if sum_l > 0 else 99.0
    wr = round(len(wins) / n_acc * 100.0, 1) if n_acc > 0 else 0.0
    exp_cash = round(total_net_profit / n_acc, 2) if n_acc > 0 else 0.0

    # Max Daily Drawdown
    worst_day_loss = min(daily_pnls.values()) if daily_pnls else 0.0
    worst_day_loss_pct = (abs(worst_day_loss) / starting_equity * 100) if worst_day_loss < 0 else 0.0

    # Concurrency statistics
    avg_concurrent = float(np.mean(concurrency_history)) if concurrency_history else 0.0
    p95_concurrent = float(np.percentile(concurrency_history, 95)) if concurrency_history else 0.0
    max_concurrent = int(np.max(concurrency_history)) if concurrency_history else 0

    # Margin exposure check
    # 20 concurrent positions * lot_size * 100,000 units = max notional
    max_notional_exposure = max_concurrent_instances * lot_size * 100000.0
    max_leverage_used = max_notional_exposure / starting_equity

    # Ret / DD ratio
    ret_to_dd = round(total_net_profit / max_dd_dollars, 2) if max_dd_dollars > 0 else 99.0

    return {
        "lot_size": lot_size,
        "starting_equity": starting_equity,
        "final_equity": round(equity, 2),
        "net_profit": round(total_net_profit, 2),
        "return_pct": round(return_pct, 1),
        "total_signals_generated": n_signals,
        "trades_executed": n_acc,
        "trades_blocked": blocked_count,
        "acceptance_rate_pct": round(n_acc / n_signals * 100, 1) if n_signals > 0 else 0.0,
        "profit_factor": pf,
        "win_rate_pct": wr,
        "expectancy_dollars": exp_cash,
        "max_drawdown_dollars": round(max_dd_dollars, 2),
        "max_drawdown_pct": round(max_dd_pct, 1),
        "worst_day_loss_dollars": round(worst_day_loss, 2),
        "worst_day_loss_pct": round(worst_day_loss_pct, 2),
        "ret_to_dd_ratio": ret_to_dd,
        "max_concurrent_instances": max_concurrent,
        "avg_concurrent_instances": round(avg_concurrent, 1),
        "p95_concurrent_instances": round(p95_concurrent, 1),
        "max_notional_exposure_usd": round(max_notional_exposure, 2),
        "max_account_leverage": round(max_leverage_used, 1),
        "yearly_breakdown": {k: round(v, 2) for k, v in sorted(yearly_pnls.items())}
    }


# ==============================================================================
# 6. MAIN EXECUTION PIPELINE
# ==============================================================================

def main():
    print("=" * 125)
    print("       MASTER 80-STRATEGY PORTFOLIO SIMULATION: $5,000 ACCOUNT / 20 CONCURRENT INSTANCES       ".center(125))
    print("=" * 125)

    symbols = ["AUDCAD", "USDCAD", "GBPJPY", "EURNZD", "GBPCHF", "NZDCAD", "AUDNZD", "EURGBP", "EURUSD", "USDJPY"]

    # Step 1: Collect / generate all 80 champion trades
    t_start = time.time()
    trades = load_or_generate_all_80_trades(symbols=symbols, max_workers=12)
    print(f"\nTotal Pooled Trade Pool across all 80 Strategies: {len(trades):,} raw signals ({time.time() - t_start:.1f}s).\n")

    # Step 2: Simulate at 0.03 lots
    print("-" * 125)
    print(">>> RUNNING PORTFOLIO SIMULATION: LOT SIZE = 0.03 LOTS ($0.30 / pip) <<<")
    res_003 = simulate_portfolio(trades, starting_equity=5000.0, max_concurrent_instances=20, lot_size=0.03)

    # Step 3: Simulate at 0.04 lots
    print(">>> RUNNING PORTFOLIO SIMULATION: LOT SIZE = 0.04 LOTS ($0.40 / pip) <<<")
    res_004 = simulate_portfolio(trades, starting_equity=5000.0, max_concurrent_instances=20, lot_size=0.04)
    print("-" * 125)

    # Step 4: Display Side-by-Side Comparison
    print("\n" + "=" * 125)
    print("                           $5,000 ACCOUNT PORTFOLIO SCORECARD: 0.03 LOTS vs 0.04 LOTS                         ")
    print("=" * 125)
    print(f"{'Performance Metric':<38} | {'0.03 Lots ($0.30/p)':<35} | {'0.04 Lots ($0.40/p)':<35}")
    print("-" * 125)
    print(f"{'Starting Account Capital':<38} | {'$5,000.00':<35} | {'$5,000.00':<35}")

    fe_03 = f"${res_003['final_equity']:,.2f}"
    fe_04 = f"${res_004['final_equity']:,.2f}"
    print(f"{'Final Account Equity (5.75 Years)':<38} | {fe_03:<35} | {fe_04:<35}")

    np_03 = f"+${res_003['net_profit']:,.2f}"
    np_04 = f"+${res_004['net_profit']:,.2f}"
    print(f"{'Total Net Profit ($)':<38} | {np_03:<35} | {np_04:<35}")

    ret_03 = f"+{res_003['return_pct']:.1f}%"
    ret_04 = f"+{res_004['return_pct']:.1f}%"
    print(f"{'Total Account Return (%)':<38} | {ret_03:<35} | {ret_04:<35}")

    ann_c_03 = f"+${res_003['net_profit']/5.75:,.2f}/yr (+{res_003['return_pct']/5.75:.1f}%/yr)"
    ann_c_04 = f"+${res_004['net_profit']/5.75:,.2f}/yr (+{res_004['return_pct']/5.75:.1f}%/yr)"
    print(f"{'Annualized Cash Return':<38} | {ann_c_03:<35} | {ann_c_04:<35}")

    sig_03 = f"{res_003['total_signals_generated']:,}"
    sig_04 = f"{res_004['total_signals_generated']:,}"
    print(f"{'Total Signals Generated':<38} | {sig_03:<35} | {sig_04:<35}")

    tr_03 = f"{res_003['trades_executed']:,} ({res_003['acceptance_rate_pct']}%)"
    tr_04 = f"{res_004['trades_executed']:,} ({res_004['acceptance_rate_pct']}%)"
    print(f"{'Trades Executed (Passed Instance Gate)':<38} | {tr_03:<35} | {tr_04:<35}")

    blk_03 = f"{res_003['trades_blocked']:,}"
    blk_04 = f"{res_004['trades_blocked']:,}"
    print(f"{'Trades Blocked by 20-Slot Cap':<38} | {blk_03:<35} | {blk_04:<35}")

    pf_03 = f"{res_003['profit_factor']:.2f}"
    pf_04 = f"{res_004['profit_factor']:.2f}"
    print(f"{'Portfolio Profit Factor':<38} | {pf_03:<35} | {pf_04:<35}")

    wr_03 = f"{res_003['win_rate_pct']:.1f}%"
    wr_04 = f"{res_004['win_rate_pct']:.1f}%"
    print(f"{'Portfolio Win Rate (%)':<38} | {wr_03:<35} | {wr_04:<35}")

    exp_03 = f"+${res_003['expectancy_dollars']:.2f}"
    exp_04 = f"+${res_004['expectancy_dollars']:.2f}"
    print(f"{'Expectancy per Trade':<38} | {exp_03:<35} | {exp_04:<35}")

    dd_03 = f"-${res_003['max_drawdown_dollars']:,.2f}"
    dd_04 = f"-${res_004['max_drawdown_dollars']:,.2f}"
    print(f"{'Peak-to-Trough Max Drawdown ($)':<38} | {dd_03:<35} | {dd_04:<35}")

    ddp_03 = f"-{res_003['max_drawdown_pct']:.1f}%"
    ddp_04 = f"-{res_004['max_drawdown_pct']:.1f}%"
    print(f"{'Peak-to-Trough Max Drawdown (%)':<38} | {ddp_03:<35} | {ddp_04:<35}")

    day_03 = f"${res_003['worst_day_loss_dollars']:,.2f} ({res_003['worst_day_loss_pct']:.2f}%)"
    day_04 = f"${res_004['worst_day_loss_dollars']:,.2f} ({res_004['worst_day_loss_pct']:.2f}%)"
    print(f"{'Worst Single Day PnL':<38} | {day_03:<35} | {day_04:<35}")

    mar_03 = f"{res_003['ret_to_dd_ratio']:.2f}x"
    mar_04 = f"{res_004['ret_to_dd_ratio']:.2f}x"
    print(f"{'Return-to-Drawdown Ratio (MAR)':<38} | {mar_03:<35} | {mar_04:<35}")

    inst_03 = f"{res_003['avg_concurrent_instances']:.1f} / {res_003['p95_concurrent_instances']:.1f} of 20 slots"
    inst_04 = f"{res_004['avg_concurrent_instances']:.1f} / {res_004['p95_concurrent_instances']:.1f} of 20 slots"
    print(f"{'Active Instances Utilization (Avg / P95)':<38} | {inst_03:<35} | {inst_04:<35}")

    lev_03 = f"{res_003['max_account_leverage']:.1f}:1 leverage (${res_003['max_notional_exposure_usd']:,.0f})"
    lev_04 = f"{res_004['max_account_leverage']:.1f}:1 leverage (${res_004['max_notional_exposure_usd']:,.0f})"
    print(f"{'Max Leverage Used (20 positions)':<38} | {lev_03:<35} | {lev_04:<35}")
    print("-" * 125)

    # Step 5: Annual Cash Returns Breakdown
    print("\nANNUAL CASH PERFORMANCE BREAKDOWN (USD):")
    years = sorted(list(set(res_003["yearly_breakdown"].keys()) | set(res_004["yearly_breakdown"].keys())))
    print(f"{'Year':<10} | {'0.03 Lots Cash PnL':<22} | {'0.04 Lots Cash PnL':<22}")
    print("-" * 60)
    for y in years:
        p03 = f"+${res_003['yearly_breakdown'].get(y, 0.0):,.2f}"
        p04 = f"+${res_004['yearly_breakdown'].get(y, 0.0):,.2f}"
        print(f"{y:<10} | {p03:<22} | {p04:<22}")
    print("-" * 60)

    # Step 6: Save Results
    out_json = RESULTS_DIR / "portfolio_5k_80_strategies_simulation.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "setup": {
                "starting_equity": 5000.0,
                "total_strategies": 80,
                "symbols": symbols,
                "max_concurrent_instances": 20,
                "tested_lot_sizes": [0.03, 0.04]
            },
            "results_0_03_lots": res_003,
            "results_0_04_lots": res_004
        }, f, indent=2)

    print(f"\nSaved Complete Portfolio Audit to {out_json}\n")


if __name__ == "__main__":
    main()
