#!/usr/bin/env python3
"""
================================================================================
  CROSS-ASSET HOLDOUT VALIDATION: LAB V12 CANDIDATES (DISC62, DISC63, DISC64, DISC65)
================================================================================
Audits passing candidates from Lab V12 across both:
  - 70% In-Sample Discovery Partition
  - 30% Quarantined Out-of-Sample Holdout Partition
  - 100% Full-Cycle 5-Year History (862 weeks total across EURUSD, USDJPY, EURGBP)
================================================================================
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
from tick_engine.core.dataloader import get_week_dataset, get_discovery_weeks, get_holdout_test_weeks
from tick_engine.strategies.base import BaseStrategy
from scratch.alpha_discovery_lab_v12 import (
    M15MACDZeroCrossMomentum,
    AsianCompressionBreakout,
    CardwellRSI50RegimePullback,
    LondonCloseMeanReversionFade
)

RESULTS_DIR = ROOT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_candidate_week(task: Tuple[str, int, str, str]) -> Dict[str, Any]:
    asset, week_num, model_name, split_type = task
    pip_size = 0.01 if asset == "USDJPY" else 0.0001
    engine = BacktestEngine(pip_size=pip_size, use_slippage=True)

    ticks, meta = get_week_dataset(week_num, raw_dir=f"data/{asset}")
    if ticks is None or len(ticks) < 500:
        return {"asset": asset, "week": week_num, "strategy": model_name, "split": split_type, "trades": []}

    if model_name == "MACD_ZERO_CROSS":
        sl = 12.0 if asset == "USDJPY" else (8.0 if asset == "EURGBP" else 9.0)
        tp = 24.0 if asset == "USDJPY" else (16.0 if asset == "EURGBP" else 18.0)
        ms = 0.8 if asset in ("USDJPY", "EURGBP") else 0.6
        model = M15MACDZeroCrossMomentum(asset=asset, sl_pips=sl, tp_pips=tp, pip_size=pip_size, max_spread=ms)
    elif model_name == "ASIA_COMPRESS_BO":
        sl = 12.0 if asset == "USDJPY" else (8.0 if asset == "EURGBP" else 9.0)
        tp = 24.0 if asset == "USDJPY" else (16.0 if asset == "EURGBP" else 18.0)
        ms = 0.8 if asset in ("USDJPY", "EURGBP") else 0.6
        max_a = 22.0 if asset == "USDJPY" else (14.0 if asset == "EURGBP" else 18.0)
        model = AsianCompressionBreakout(asset=asset, sl_pips=sl, tp_pips=tp, pip_size=pip_size, max_spread=ms, max_asia_pips=max_a)
    elif model_name == "RSI_50_PULLBACK":
        sl = 12.0 if asset == "USDJPY" else (8.0 if asset == "EURGBP" else 9.0)
        tp = 24.0 if asset == "USDJPY" else (16.0 if asset == "EURGBP" else 18.0)
        ms = 0.8 if asset in ("USDJPY", "EURGBP") else 0.6
        model = CardwellRSI50RegimePullback(asset=asset, sl_pips=sl, tp_pips=tp, pip_size=pip_size, max_spread=ms)
    elif model_name == "LON_CLOSE_FADE":
        sl = 12.0 if asset == "USDJPY" else (8.0 if asset == "EURGBP" else 9.0)
        tp = 21.6 if asset == "USDJPY" else (14.4 if asset == "EURGBP" else 16.2)
        ms = 0.8 if asset in ("USDJPY", "EURGBP") else 0.6
        min_s = 55.0 if asset == "USDJPY" else (30.0 if asset == "EURGBP" else 45.0)
        model = LondonCloseMeanReversionFade(asset=asset, sl_pips=sl, tp_pips=tp, pip_size=pip_size, max_spread=ms, min_stretch_pips=min_s)
    else:
        return {"asset": asset, "week": week_num, "strategy": model_name, "split": split_type, "trades": []}

    try:
        raw_trades = engine.run(ticks, model)
        t_list = []
        for i in range(len(raw_trades)):
            row = raw_trades[i]
            t_list.append({
                "strategy": model_name,
                "asset": asset,
                "week": week_num,
                "split": split_type,
                "pips": float(row["pips"]),
                "direction": int(row["direction"])
            })
        return {"asset": asset, "week": week_num, "strategy": model_name, "split": split_type, "trades": t_list}
    except Exception:
        return {"asset": asset, "week": week_num, "strategy": model_name, "split": split_type, "trades": []}


def run_cross_validation_v12(active_models: List[str]):
    t_start = time.time()
    print("=" * 80, flush=True)
    print(f"  CROSS-ASSET HOLDOUT AUDIT FOR LAB V12 CANDIDATES: {active_models}", flush=True)
    print("=" * 80, flush=True)

    assets = ["USDJPY", "EURUSD", "EURGBP"]
    all_tasks = []

    for m_name in active_models:
        for asset in assets:
            disc_weeks = get_discovery_weeks(f"data/{asset}")
            hold_weeks = get_holdout_test_weeks(f"data/{asset}")
            for w in disc_weeks:
                all_tasks.append((asset, w, m_name, "Discovery_70"))
            for w in hold_weeks:
                all_tasks.append((asset, w, m_name, "Holdout_30"))

    print(f"[*] Queued {len(all_tasks)} week-tasks across all partitions...", flush=True)

    results_by_key: Dict[Tuple[str, str, str], List[float]] = {}
    completed = 0

    with ProcessPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(evaluate_candidate_week, t): t for t in all_tasks}
        for future in as_completed(futures):
            res = future.result()
            completed += 1
            key = (res["asset"], res["strategy"], res["split"])
            if key not in results_by_key:
                results_by_key[key] = []
            results_by_key[key].extend([t["pips"] for t in res["trades"]])

            if completed % 200 == 0 or completed == len(all_tasks):
                print(f"  [+] Audited {completed:4d}/{len(all_tasks)} tasks in {time.time() - t_start:.1f}s...", flush=True)

    # Compile Scorecard
    rows = []
    unique_pairs = sorted(list(set((k[0], k[1]) for k in results_by_key.keys())))

    for (asset, strat) in unique_pairs:
        disc_pips = results_by_key.get((asset, strat, "Discovery_70"), [])
        hold_pips = results_by_key.get((asset, strat, "Holdout_30"), [])
        all_pips = disc_pips + hold_pips

        for label, p_list in [("70% Discovery", disc_pips), ("30% Holdout", hold_pips), ("100% Full Cycle", all_pips)]:
            n = len(p_list)
            if n == 0:
                continue
            arr = np.array(p_list)
            net_pips = float(np.sum(arr))
            wins = arr[arr > 0]
            losses = -arr[arr < 0]
            wr = float(len(wins) / n * 100.0)
            sum_w = float(np.sum(wins))
            sum_l = float(np.sum(losses))
            pf = round(sum_w / sum_l, 2) if sum_l > 0 else (99.0 if sum_w > 0 else 0.0)

            equity = np.cumsum(arr)
            peak = np.maximum.accumulate(equity)
            dd = peak - equity
            max_dd = float(np.max(dd)) if len(dd) > 0 else 0.0
            ret_dd = round(net_pips / max_dd, 2) if max_dd > 0 else 0.0
            exp = round(net_pips / n, 2)

            rows.append({
                "asset": asset,
                "strategy": strat,
                "partition": label,
                "trades": n,
                "win_rate": wr,
                "profit_factor": pf,
                "net_pips": net_pips,
                "max_dd_pips": max_dd,
                "ret_dd": ret_dd,
                "expectancy": exp
            })

    df = pd.DataFrame(rows)
    out_csv = RESULTS_DIR / "lab_v12_cross_holdout_scorecard.csv"
    df.to_csv(out_csv, index=False)

    print("\n" + "=" * 110, flush=True)
    print("  LAB V12 CANDIDATES CROSS-HOLDOUT VERIFICATION SCORECARD", flush=True)
    print("=" * 110, flush=True)
    print(f"{'Asset':<8} {'Strategy':<20} {'Partition':<18} {'Trades':<8} {'WR':<8} {'PF':<8} {'Net Pips':<12} {'Max DD':<10} {'Ret/DD':<8} {'Exp':<8}")
    print("-" * 110, flush=True)
    for _, r in df.iterrows():
        print(f"{r['asset']:<8} {r['strategy']:<20} {r['partition']:<18} {int(r['trades']):<8} {r['win_rate']:>5.1f}%  {r['profit_factor']:>6.2f}  {r['net_pips']:>10.1f}p  {r['max_dd_pips']:>8.1f}p  {r['ret_dd']:>6.2f}x  {r['expectancy']:>6.2f}p")
    print("=" * 110, flush=True)
    print(f"[*] Cross-holdout audit finished in {time.time() - t_start:.2f}s.", flush=True)


if __name__ == "__main__":
    candidates = sys.argv[1:] if len(sys.argv) > 1 else ["MACD_ZERO_CROSS", "ASIA_COMPRESS_BO", "RSI_50_PULLBACK"]
    run_cross_validation_v12(candidates)
