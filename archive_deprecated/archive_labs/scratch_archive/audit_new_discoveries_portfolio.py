#!/usr/bin/env python3
"""
================================================================================
  NEW DISCOVERIES PORTFOLIO 5-YEAR AUDIT (6 MODELS / 262 WEEKS / ~123.22M TICKS)
================================================================================
Evaluates the 6 newly discovered quantitative alpha models in `strategies/new_discoveries/`:
  1. DISC38: London & Midday MACD Acceleration Elite
  2. DISC39: Stochastic Deep Pullback in Trend Elite
  3. DISC40: RSI Bull/Bear Regime Range Shift Elite
  4. DISC41: Thursday Macro Trend Ignition Elite
  5. DISC42: Macro Structure H4 Level + M30 Trigger Elite
  6. DISC43: Intraday Precision H1 Level + M5 Rejection Elite
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

from strategies.new_discoveries import (
    LondonMiddayMACDAccelerationElite,
    StochasticDeepPullbackElite,
    RSIRegimeRangeShiftElite,
    ThursdayMacroTrendIgnitionElite,
    MacroStructureH4M30Elite,
    IntradayPrecisionH1M5Elite
)

PIP_SIZE = 0.0001


def _audit_discoveries_week(w: int):
    ticks, _ = get_week_dataset(w)
    engine = BacktestEngine(pip_size=PIP_SIZE, use_slippage=True)

    strats = [
        LondonMiddayMACDAccelerationElite(),
        StochasticDeepPullbackElite(),
        RSIRegimeRangeShiftElite(),
        ThursdayMacroTrendIgnitionElite(),
        MacroStructureH4M30Elite(),
        IntradayPrecisionH1M5Elite()
    ]

    week_trades = []
    strat_trades = {}
    for s in strats:
        tr = engine.run(ticks, s)
        strat_trades[s.name] = tr
        if len(tr) > 0:
            week_trades.append(tr)

    combined = np.concatenate(week_trades) if week_trades else np.empty(0, dtype=TRADE_DTYPE)
    return w, combined, strat_trades


def run():
    print("=" * 105)
    print("  NEW DISCOVERIES INCUBATOR PORTFOLIO: 5-YEAR AUDIT (6 MODELS / 262 WEEKS / ~123.22M TICKS)  ".center(105))
    print("=" * 105)

    weeks = list(range(1, 263))
    t0 = time.time()
    num_workers = min(8, os.cpu_count() or 4)

    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        results = list(ex.map(_audit_discoveries_week, weeks))

    dt = time.time() - t0
    print(f"[+] All 262 weeks simulated across 6 newly discovered models in {dt:.1f}s!\n")

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
    print(f"{'Combined Discoveries Metric':<30} | {'3Y Train (W1-156)':<22} | {'2Y Test (W157-262)':<22} | {'5Y Total (W1-262)':<22}")
    print("-" * 105)
    print(f"{'Total Trades':<30} | {m_train['total_trades']:<22} | {m_test['total_trades']:<22} | {m_all['total_trades']:<22}")
    print(f"{'Trades per Week':<30} | {m_train['total_trades']/156:<22.2f} | {m_test['total_trades']/106:<22.2f} | {m_all['total_trades']/262:<22.2f}")
    print(f"{'Win Rate (%)':<30} | {m_train['win_rate_pct']:<21.1f}% | {m_test['win_rate_pct']:<21.1f}% | {m_all['win_rate_pct']:<21.1f}%")
    print(f"{'Profit Factor':<30} | {m_train['profit_factor']:<22.2f} | {m_test['profit_factor']:<22.2f} | {m_all['profit_factor']:<22.2f}")
    print(f"{'Net Cumulative Pips':<30} | {m_train['net_pips']:<+21.1f}p | {m_test['net_pips']:<+21.1f}p | {m_all['net_pips']:<+21.1f}p")
    print(f"{'Max Drawdown (p)':<30} | {m_train['max_drawdown_pips']:<21.1f}p | {m_test['max_drawdown_pips']:<21.1f}p | {m_all['max_drawdown_pips']:<21.1f}p")
    print(f"{'Return / DD Ratio':<30} | {m_train['net_pips']/max(m_train['max_drawdown_pips'], 1e-6):<22.2f} | {m_test['net_pips']/max(m_test['max_drawdown_pips'], 1e-6):<22.2f} | {m_all['net_pips']/max(m_all['max_drawdown_pips'], 1e-6):<22.2f}")
    print("-" * 105)

    # Individual breakdown
    print("\n" + "=" * 105)
    print("INDIVIDUAL 5-YEAR BREAKDOWN OF ALL 6 DISCOVERED MODELS (WEEKS 1 - 262)")
    print("=" * 105)
    print(f"{'Model Name':<42} | {'Trades':<8} | {'Win Rate':<10} | {'Profit Factor':<15} | {'Net Pips':<12} | {'Max DD'}")
    print("-" * 105)

    strat_names = [
        "London_Midday_MACD_Acceleration",
        "Stochastic_Deep_Pullback_Elite",
        "RSI_Regime_Range_Shift_Elite",
        "Thursday_Macro_Trend_Ignition_Elite",
        "Macro_Structure_H4_M30_Elite",
        "Intraday_Precision_H1_M5_Elite"
    ]

    for sname in strat_names:
        s_trades = [r[2][sname] for r in results if sname in r[2] and len(r[2][sname]) > 0]
        if s_trades:
            all_s = np.concatenate(s_trades)
            sm = compute_metrics(all_s)
            print(f"{sname:<42} | {sm['total_trades']:<8} | {sm['win_rate_pct']:<9.1f}% | {sm['profit_factor']:<15.2f} | {sm['net_pips']:<+11.1f}p | {sm['max_drawdown_pips']:<8.1f}p")
    print("=" * 105)


if __name__ == '__main__':
    run()
