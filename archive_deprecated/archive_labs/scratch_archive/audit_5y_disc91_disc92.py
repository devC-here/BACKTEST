#!/usr/bin/env python3
"""
Full 5-Year Audit for DISC91 (Donchian Midline Bounce) and DISC92 (Asian S/R Flip Retest) on USDJPY
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
from tick_engine.core.dataloader import get_week_dataset
from scratch.alpha_discovery_lab_v32 import SessionDonchianMidlineBounce, AsianSRFlipRetest


def run_week(args) -> Dict[str, Any]:
    model_name, week_num = args
    pip_size = 0.01
    engine = BacktestEngine(pip_size=pip_size, use_slippage=True)
    try:
        ticks, meta = get_week_dataset(week_num, raw_dir="data/USDJPY")
        if ticks is None or len(ticks) < 500:
            return {"model": model_name, "trades": []}

        if model_name == "DISC91_DONCHIAN":
            strat = SessionDonchianMidlineBounce(asset="USDJPY", sl_pips=12.0, tp_pips=24.0, pip_size=pip_size, max_spread=0.8)
        else:
            strat = AsianSRFlipRetest(asset="USDJPY", sl_pips=13.0, tp_pips=28.5, pip_size=pip_size, max_spread=0.8)

        raw_trades = engine.run(ticks, strat)
        t_list = []
        for i in range(len(raw_trades)):
            row = raw_trades[i]
            t_list.append({
                "pips": float(row["pips"]),
                "entry_time": int(row["entry_time"]),
            })
        return {"model": model_name, "trades": t_list}
    except Exception:
        return {"model": model_name, "trades": []}


def analyze_model(name: str, trades: List[Dict[str, Any]]):
    df = pd.DataFrame(trades)
    df.sort_values(by="entry_time", inplace=True)
    df["entry_dt"] = pd.to_datetime(df["entry_time"], unit="ms")
    df["year"] = df["entry_dt"].dt.year

    pips = df["pips"].values
    n = len(pips)
    wins = pips[pips > 0]
    losses = pips[pips < 0]
    wr = len(wins) / n * 100.0 if n > 0 else 0.0
    gw = float(np.sum(wins)) if len(wins) > 0 else 0.0
    gl = float(abs(np.sum(losses))) if len(losses) > 0 else 0.0
    pf = gw / gl if gl > 0 else 999.0
    net = float(np.sum(pips))
    exp = net / n if n > 0 else 0.0

    equity = np.cumsum(pips)
    peak = np.maximum.accumulate(equity)
    dd = peak - equity
    max_dd = float(np.max(dd)) if len(dd) > 0 else 0.0
    ret_dd = net / max_dd if max_dd > 0 else 999.0

    print(f"\n================================================================================")
    print(f"  {name} 5-YEAR FULL CYCLE METRICS")
    print(f"================================================================================")
    print(f"Trades: {n} | WR: {wr:.1f}% | PF: {pf:.2f} | Net: +{net:.1f}p | Exp: +{exp:.2f}p/t | MaxDD: {max_dd:.1f}p | Ret/DD: {ret_dd:.2f}x")
    print("\nYearly Breakdown:")
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
        print(f"  {yr}: {yn:<4} trades | WR: {ywr:.1f}% | PF: {ypf:.2f} | Net: +{ynet:.1f}p")


def main():
    models = ["DISC91_DONCHIAN", "DISC92_ASIAN_SR_FLIP"]
    tasks = []
    for m in models:
        for w in range(1, 301):
            tasks.append((m, w))

    print(f"Executing 5-year audits for {len(models)} champions across 300 weeks ({len(tasks)} tasks)...")
    results = {m: [] for m in models}

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(run_week, t) for t in tasks]
        for f in as_completed(futures):
            res = f.result()
            results[res["model"]].extend(res["trades"])

    for m in models:
        analyze_model(m, results[m])


if __name__ == "__main__":
    main()
