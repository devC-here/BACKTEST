#!/usr/bin/env python3
"""
================================================================================
  8-STAR ALL-WEATHER FLEET 5-YEAR AUDIT (262 WEEKS / ~123.22M TICKS)
================================================================================
Audits the consolidated portfolio adding the newly discovered:
  DISC38: London & Midday MACD Acceleration Elite
alongside the 7 established all-weather champions:
  1. DISC03: NY Open Overlap Breakout Elite
  2. INST05: Market Structure Shift Retest Elite
  3. COMB12: Bollinger Squeeze Elite
  4. COMB27: Squeeze Supertrend Hybrid Elite
  5. DISC31: London Lunch Trap Fade Elite
  6. DISC36: Frankfurt Ultra-Coil London ORB Elite
  7. DISC37: European Midday Ultra-Coil NY Cash Open Breakout Elite
  8. DISC38: London & Midday MACD Acceleration Elite
================================================================================
"""

from __future__ import annotations
import sys
import os
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tick_engine.core.engine import BacktestEngine, TRADE_DTYPE
from tick_engine.core.dataloader import get_week_dataset
from tick_engine.core.metrics import compute_metrics

# 7 Champions
from strategies.disc03_ny_open_overlap_elite.strategy import NewYorkOpenOverlapBreakoutElite
from strategies.inst05_market_structure_shift_elite.strategy import MarketStructureShiftRetestElite
from strategies.comb12_bollinger_squeeze_elite.strategy import BollingerBandwidthExpansionElite
from strategies.comb27_squeeze_supertrend_hybrid_elite.strategy import BollingerSupertrendHybridElite
from strategies.disc31_london_lunch_trap_fade_elite.strategy import LondonLunchTrapFadeElite
from strategies.disc36_frankfurt_ultra_coil_orb_elite.strategy import FrankfurtUltraCoilLondonORBElite
from strategies.disc37_european_midday_ultracoil_breakout_elite.strategy import EuropeanMiddayUltraCoilBreakoutElite
from scratch.test_london_macd_acceleration import LondonMACDAccelerationStrategy, PIP_SIZE


def _audit_fleet_week(w: int):
    ticks, _ = get_week_dataset(w)
    engine = BacktestEngine(pip_size=PIP_SIZE, use_slippage=True)

    strats = [
        NewYorkOpenOverlapBreakoutElite(),
        MarketStructureShiftRetestElite(),
        BollingerBandwidthExpansionElite(),
        BollingerSupertrendHybridElite(),
        LondonLunchTrapFadeElite(),
        FrankfurtUltraCoilLondonORBElite(),
        EuropeanMiddayUltraCoilBreakoutElite(),
        LondonMACDAccelerationStrategy(target_rr=1.70, min_risk_pips=7.0, max_risk_pips=14.0, allowed_hours=(8, 11))
    ]

    week_trades = []
    for s in strats:
        tr = engine.run(ticks, s)
        if len(tr) > 0:
            week_trades.append(tr)

    if week_trades:
        return w, np.concatenate(week_trades)
    return w, np.empty(0, dtype=TRADE_DTYPE)


def run_fleet_audit():
    print("=" * 105)
    print("  8-STAR CONSOLIDATED ALL-WEATHER FLEET: 5-YEAR TICK AUDIT (262 WEEKS / ~123.22M TICKS)  ".center(105))
    print("=" * 105)

    weeks = list(range(1, 263))
    t0 = time.time()
    num_workers = min(8, os.cpu_count() or 4)

    print(f"[*] Dispatching backtest across {len(weeks)} weeks on {num_workers} CPU cores...")
    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        results = list(ex.map(_audit_fleet_week, weeks))

    dt = time.time() - t0
    print(f"[+] All 262 weeks simulated across 8 models in {dt:.1f}s!\n")

    results.sort(key=lambda x: x[0])

    train_trades = [r[1] for r in results if r[0] <= 156 and len(r[1]) > 0]
    test_trades = [r[1] for r in results if r[0] > 156 and len(r[1]) > 0]
    all_trades = [r[1] for r in results if len(r[1]) > 0]

    tr_train = np.concatenate(train_trades) if train_trades else np.empty(0, dtype=TRADE_DTYPE)
    tr_test = np.concatenate(test_trades) if test_trades else np.empty(0, dtype=TRADE_DTYPE)
    tr_all = np.concatenate(all_trades) if all_trades else np.empty(0, dtype=TRADE_DTYPE)

    m_train = compute_metrics(tr_train)
    m_test = compute_metrics(tr_test)
    m_all = compute_metrics(tr_all)

    print("-" * 105)
    print(f"{'Metric':<25} | {'3Y Train (W1-156)':<24} | {'2Y Test (W157-262)':<24} | {'5Y Total (W1-262)':<24}")
    print("-" * 105)
    print(f"{'Total Trades':<25} | {m_train['total_trades']:<24} | {m_test['total_trades']:<24} | {m_all['total_trades']:<24}")
    print(f"{'Trades per Week':<25} | {m_train['total_trades']/156:<24.2f} | {m_test['total_trades']/106:<24.2f} | {m_all['total_trades']/262:<24.2f}")
    print(f"{'Win Rate (%)':<25} | {m_train['win_rate_pct']:<23.1f}% | {m_test['win_rate_pct']:<23.1f}% | {m_all['win_rate_pct']:<23.1f}%")
    print(f"{'Profit Factor':<25} | {m_train['profit_factor']:<24.2f} | {m_test['profit_factor']:<24.2f} | {m_all['profit_factor']:<24.2f}")
    print(f"{'Net Cumulative Pips':<25} | {m_train['net_pips']:<+23.1f}p | {m_test['net_pips']:<+23.1f}p | {m_all['net_pips']:<+23.1f}p")
    print(f"{'Max Portfolio DD (p)':<25} | {m_train['max_drawdown_pips']:<23.1f}p | {m_test['max_drawdown_pips']:<23.1f}p | {m_all['max_drawdown_pips']:<23.1f}p")
    print(f"{'Return / DD Ratio':<25} | {m_train['net_pips']/max(m_train['max_drawdown_pips'], 1e-6):<24.2f} | {m_test['net_pips']/max(m_test['max_drawdown_pips'], 1e-6):<24.2f} | {m_all['net_pips']/max(m_all['max_drawdown_pips'], 1e-6):<24.2f}")
    print("-" * 105)


if __name__ == "__main__":
    run_fleet_audit()
