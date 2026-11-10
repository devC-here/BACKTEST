#!/usr/bin/env python3
"""
================================================================================
  5-YEAR AUDIT: ROUND 2 DISCOVERIES ACROSS 262 WEEKS (~123.2M TICKS)
  - DISC37: European Midday Ultra-Coil NY Breakout Elite
  - DISC38: Daily Floor Pivot S1/R1 Institutional Bounce Elite
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
from scratch.research_round2_alpha_lab import EuropeanMiddayUltraCoilBreakoutStrategy, DailyPivotRejectionStrategy

TRAIN_END_WEEK = 156  # 3Y In-sample vs 2Y Out-of-sample boundary


def _worker_eval(args):
    w, StratCls, kwargs = args
    ticks, _ = get_week_dataset(w)
    engine = BacktestEngine(pip_size=0.0001, use_slippage=True)
    strat = StratCls(**kwargs)
    trades = engine.run(ticks, strat)
    return w, trades


def audit_5y(StratCls, strat_name: str, **kwargs):
    weeks = [(w, StratCls, kwargs) for w in range(1, 263)]
    num_workers = min(8, os.cpu_count() or 4)
    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        results = list(ex.map(_worker_eval, weeks))

    results.sort(key=lambda x: x[0])

    all_tr = []
    train_tr = []
    test_tr = []

    for w, trades in results:
        if len(trades) > 0:
            all_tr.append(trades)
            if w <= TRAIN_END_WEEK:
                train_tr.append(trades)
            else:
                test_tr.append(trades)

    tr_all = np.concatenate(all_tr) if all_tr else np.empty(0, dtype=TRADE_DTYPE)
    tr_train = np.concatenate(train_tr) if train_tr else np.empty(0, dtype=TRADE_DTYPE)
    tr_test = np.concatenate(test_tr) if test_tr else np.empty(0, dtype=TRADE_DTYPE)

    m_5y = compute_metrics(tr_all)
    m_train = compute_metrics(tr_train)
    m_test = compute_metrics(tr_test)

    print("=" * 80)
    print(f"  5-YEAR AUDIT RESULTS: {strat_name}")
    print("=" * 80)
    print(f"3Y Train (Weeks 1-156):   {m_train['net_pips']:+8.1f}p | PF: {m_train['profit_factor']:4.2f} | WR: {m_train['win_rate_pct']:4.1f}% ({m_train['total_trades']} trades)")
    print(f"2Y Test  (Weeks 157-262): {m_test['net_pips']:+8.1f}p | PF: {m_test['profit_factor']:4.2f} | WR: {m_test['win_rate_pct']:4.1f}% ({m_test['total_trades']} trades)")
    print(f"5Y Total (Weeks 1-262):   {m_5y['net_pips']:+8.1f}p | PF: {m_5y['profit_factor']:4.2f} | WR: {m_5y['win_rate_pct']:4.1f}% ({m_5y['total_trades']} trades)")
    print(f"5Y Max Drawdown:          {m_5y['max_drawdown_pips']:8.1f} pips")
    print("=" * 80 + "\n")

    return m_5y, m_train, m_test


if __name__ == "__main__":
    t0 = time.time()
    audit_5y(EuropeanMiddayUltraCoilBreakoutStrategy, "DISC37: European Midday Ultra-Coil NY Breakout Elite", max_range_pips=15.0, target_rr=1.75)
    audit_5y(DailyPivotRejectionStrategy, "DISC38: Daily Floor Pivot S1/R1 Institutional Bounce Elite", target_rr=1.60)
    print(f"5-Year audits completed in {time.time() - t0:.1f}s.")
