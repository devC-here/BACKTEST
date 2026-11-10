#!/usr/bin/env python3
"""
Audit Full 5-Year Performance of DISC90: USDJPY Opening Range Impulse Expansion Elite
Runs across all 300 weeks of authentic Dukascopy tick data with real floating spread + slippage.
"""

from __future__ import annotations
import sys
import os
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Any, Tuple

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tick_engine.core.engine import BacktestEngine
from tick_engine.core.dataloader import get_week_dataset
from scratch.alpha_discovery_lab_v30 import OpeningRangeImpulseExpansion

RESULTS_DIR = ROOT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def run_week(week_num: int) -> List[Dict[str, Any]]:
    pip_size = 0.01
    engine = BacktestEngine(pip_size=pip_size, use_slippage=True)
    try:
        ticks, meta = get_week_dataset(week_num, raw_dir="data/USDJPY")
        if ticks is None or len(ticks) < 500:
            return []

        strat = OpeningRangeImpulseExpansion(
            asset="USDJPY",
            sl_pips=13.0,
            tp_pips=23.5,
            pip_size=pip_size,
            max_spread=0.8,
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
                "exit_time": int(row["exit_time"]),
            })
        return t_list
    except Exception as e:
        return []


def main():
    print("=" * 80)
    print("  AUDITING DISC90: USDJPY OPENING RANGE IMPULSE EXPANSION ELITE (FULL 5-YEARS)")
    print("=" * 80)

    tasks = list(range(1, 301))
    all_trades = []
    t0 = time.time()

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(run_week, w) for w in tasks]
        for fut in as_completed(futures):
            res = fut.result()
            all_trades.extend(res)

    elapsed = time.time() - t0
    print(f"5-Year Backtest completed across 300 weeks in {elapsed:.1f}s!")

    if not all_trades:
        print("No trades generated!")
        return

    df = pd.DataFrame(all_trades)
    df.sort_values(by="entry_time", inplace=True)
    df["entry_dt"] = pd.to_datetime(df["entry_time"], unit="ms")
    df["year"] = df["entry_dt"].dt.year

    pips = df["pips"].values
    n = len(pips)
    wins = pips[pips > 0]
    losses = pips[pips < 0]
    wr = len(wins) / n * 100.0
    gw = float(np.sum(wins))
    gl = float(abs(np.sum(losses)))
    pf = gw / gl if gl > 0 else 999.0
    net = float(np.sum(pips))
    exp = net / n

    equity = np.cumsum(pips)
    peak = np.maximum.accumulate(equity)
    dd = peak - equity
    max_dd = float(np.max(dd))
    ret_dd = net / max_dd if max_dd > 0 else 999.0

    print("\n" + "=" * 80)
    print("  OVERALL 5-YEAR METRICS (2021 - 2026)")
    print("=" * 80)
    print(f"Total Trades:       {n}")
    print(f"Win Rate:           {wr:.1f}%")
    print(f"Profit Factor:      {pf:.2f}")
    print(f"Total Net Pips:     +{net:.1f} pips")
    print(f"Expectancy:         +{exp:.2f} pips/trade")
    print(f"Max Drawdown:       {max_dd:.1f} pips")
    print(f"Return / DD Ratio:  {ret_dd:.2f}x")

    print("\n" + "=" * 80)
    print("  YEAR-BY-YEAR CONSISTENCY BREAKDOWN")
    print("=" * 80)
    print(f"{'Year':<8} {'Trades':<8} {'Win Rate':<10} {'PF':<8} {'Net Pips':<12} {'Max DD':<10}")
    print("-" * 60)

    for yr, ydf in df.groupby("year"):
        yp = ydf["pips"].values
        yn = len(yp)
        yw = yp[yp > 0]
        yl = yp[yp < 0]
        ywr = len(yw) / yn * 100.0 if yn > 0 else 0.0
        ygw = float(np.sum(yw)) if len(yw) > 0 else 0.0
        ygl = float(abs(np.sum(yl))) if len(yl) > 0 else 0.0
        ypf = ygw / ygl if ygl > 0 else 999.0
        ynet = float(np.sum(yp))
        yeq = np.cumsum(yp)
        ypk = np.maximum.accumulate(yeq)
        ydd = ypk - yeq
        ymdd = float(np.max(ydd)) if len(ydd) > 0 else 0.0

        print(f"{yr:<8} {yn:<8d} {ywr:<9.1f}% {ypf:<7.2f} +{ynet:<11.1f}p {ymdd:<9.1f}p")

    csv_path = RESULTS_DIR / "disc90_usdjpy_opening_range_impulse_5y_audit.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved trade logs to: {csv_path}")


if __name__ == "__main__":
    main()
