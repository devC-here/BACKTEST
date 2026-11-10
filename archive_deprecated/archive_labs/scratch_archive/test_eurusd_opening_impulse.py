#!/usr/bin/env python3
"""
Test EURUSD Opening Range Impulse Expansion across Holdout and 5Y
"""

from __future__ import annotations
import sys
import os
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Any

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tick_engine.core.engine import BacktestEngine
from tick_engine.core.dataloader import get_week_dataset, get_holdout_test_weeks
from scratch.alpha_discovery_lab_v30 import OpeningRangeImpulseExpansion


def run_week(week_num: int) -> List[Dict[str, Any]]:
    pip_size = 0.0001
    engine = BacktestEngine(pip_size=pip_size, use_slippage=True)
    try:
        ticks, meta = get_week_dataset(week_num, raw_dir="data/EURUSD")
        if ticks is None or len(ticks) < 500:
            return []

        strat = OpeningRangeImpulseExpansion(
            asset="EURUSD",
            sl_pips=9.0,
            tp_pips=16.0,
            pip_size=pip_size,
            max_spread=0.6,
        )
        raw_trades = engine.run(ticks, strat)
        t_list = []
        for i in range(len(raw_trades)):
            row = raw_trades[i]
            t_list.append({
                "week": week_num,
                "pips": float(row["pips"]),
                "direction": int(row["direction"]),
                "entry_time": int(row["entry_time"]),
            })
        return t_list
    except Exception:
        return []


def main():
    test_weeks = get_holdout_test_weeks(raw_dir="data/EURUSD")
    print(f"Testing EURUSD Holdout ({len(test_weeks)} weeks)...")
    h_trades = []
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(run_week, w) for w in test_weeks]
        for f in as_completed(futures):
            h_trades.extend(f.result())

    hp = [t["pips"] for t in h_trades]
    harr = np.array(hp)
    hw = harr[harr > 0]
    hl = harr[harr < 0]
    hwr = len(hw) / len(harr) * 100.0 if len(harr) > 0 else 0.0
    hgw = float(np.sum(hw)) if len(hw) > 0 else 0.0
    hgl = float(abs(np.sum(hl))) if len(hl) > 0 else 0.0
    hpf = hgw / hgl if hgl > 0 else 0.0
    hnet = float(np.sum(harr)) if len(harr) > 0 else 0.0

    print(f"Holdout: Trades={len(harr)}, WR={hwr:.1f}%, PF={hpf:.2f}, Net={hnet:.1f}p")


if __name__ == "__main__":
    main()
