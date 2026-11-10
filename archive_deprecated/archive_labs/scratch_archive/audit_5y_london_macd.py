#!/usr/bin/env python3
"""
================================================================================
  5-YEAR TICK AUDIT: LONDON & MIDDAY MACD ACCELERATION ELITE (262 WEEKS)
================================================================================
Evaluates the London MACD Acceleration model across the entire 5-year tick dataset:
  - 3-Year In-Sample Train (Weeks 1 - 156, ~73.9M ticks)
  - 2-Year Out-of-Sample Test (Weeks 157 - 262, ~49.3M ticks)
  - Full 5-Year Performance (Weeks 1 - 262, ~123.22M ticks)
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
import ta

sys.dont_write_bytecode = True

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tick_engine.core.engine import BacktestEngine, TRADE_DTYPE
from tick_engine.core.dataloader import get_week_dataset
from tick_engine.core.metrics import compute_metrics
from scratch.test_london_macd_acceleration import LondonMACDAccelerationStrategy, PIP_SIZE


def _audit_week(w: int):
    ticks, _ = get_week_dataset(w)
    strat = LondonMACDAccelerationStrategy(
        target_rr=1.70,
        min_risk_pips=7.0,
        max_risk_pips=14.0,
        allowed_hours=(8, 11)
    )
    engine = BacktestEngine(pip_size=PIP_SIZE, use_slippage=True)
    trades = engine.run(ticks, strat)
    return w, trades


def run_5y_audit():
    print("=" * 95)
    print("  5-YEAR TICK AUDIT: LONDON & MIDDAY MACD ACCELERATION ELITE (262 WEEKS / ~123.22M TICKS)  ".center(95))
    print("=" * 95)

    weeks = list(range(1, 263))
    num_workers = min(8, os.cpu_count() or 4)
    t0 = time.time()

    print(f"[*] Dispatching backtest across {len(weeks)} weeks on {num_workers} CPU cores...")
    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        results = list(ex.map(_audit_week, weeks))

    dt = time.time() - t0
    print(f"[+] All 262 weeks simulated in {dt:.1f}s!\n")

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

    print("-" * 95)
    print(f"{'Metric':<25} | {'3Y Train (W1-156)':<20} | {'2Y Test (W157-262)':<20} | {'5Y Total (W1-262)':<20}")
    print("-" * 95)
    print(f"{'Total Trades':<25} | {m_train['total_trades']:<20} | {m_test['total_trades']:<20} | {m_all['total_trades']:<20}")
    print(f"{'Win Rate (%)':<25} | {m_train['win_rate_pct']:<19.1f}% | {m_test['win_rate_pct']:<19.1f}% | {m_all['win_rate_pct']:<19.1f}%")
    print(f"{'Profit Factor':<25} | {m_train['profit_factor']:<20.2f} | {m_test['profit_factor']:<20.2f} | {m_all['profit_factor']:<20.2f}")
    print(f"{'Net Pips':<25} | {m_train['net_pips']:<+19.1f}p | {m_test['net_pips']:<+19.1f}p | {m_all['net_pips']:<+19.1f}p")
    print(f"{'Max Drawdown (p)':<25} | {m_train['max_drawdown_pips']:<19.1f}p | {m_test['max_drawdown_pips']:<19.1f}p | {m_all['max_drawdown_pips']:<19.1f}p")
    print(f"{'Return / DD Ratio':<25} | {m_train['net_pips']/max(m_train['max_drawdown_pips'], 1e-6):<20.2f} | {m_test['net_pips']/max(m_test['max_drawdown_pips'], 1e-6):<20.2f} | {m_all['net_pips']/max(m_all['max_drawdown_pips'], 1e-6):<20.2f}")
    print("-" * 95)


if __name__ == "__main__":
    run_5y_audit()
