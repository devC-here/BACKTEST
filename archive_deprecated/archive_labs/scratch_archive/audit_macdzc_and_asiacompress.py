#!/usr/bin/env python3
"""
Verification Audit for:
  1. USDJPY M15 MACD Zero-Cross Momentum (DISC62)
  2. EURGBP Asian Compression Breakout (DISC63)
Tested separately on 70% Discovery, 30% Quarantined Holdout, and Full 5-Year Cycle.
"""

from pathlib import Path
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tick_engine.core.engine import BacktestEngine
from tick_engine.core.dataloader import get_week_dataset, get_discovery_weeks, get_holdout_test_weeks
from scratch.alpha_discovery_lab_v12 import M15MACDZeroCrossMomentum, AsianCompressionBreakout

USDJPY_MULT = 0.7193
EURGBP_MULT = 1.2921

models = [
    {
        "name": "USDJPY_M15_MACD_Zero_Cross",
        "asset": "USDJPY",
        "strat": M15MACDZeroCrossMomentum("USDJPY", sl_pips=12.0, tp_pips=24.0, pip_size=0.01, max_spread=0.8),
        "pip_size": 0.01,
        "mult": USDJPY_MULT,
        "sl_pips": 12.0
    },
    {
        "name": "EURGBP_Asian_Compression_BO",
        "asset": "EURGBP",
        "strat": AsianCompressionBreakout("EURGBP", sl_pips=8.0, tp_pips=16.0, pip_size=0.0001, max_spread=0.8, max_asia_pips=14.0),
        "pip_size": 0.0001,
        "mult": EURGBP_MULT,
        "sl_pips": 8.0
    }
]

def run_task(args):
    m_idx, w, split = args
    m = models[m_idx]
    engine = BacktestEngine(pip_size=m["pip_size"], use_slippage=True)
    try:
        ticks, _ = get_week_dataset(w, raw_dir=f"data/{m['asset']}")
        if ticks is None or len(ticks) < 500:
            return m_idx, split, []
        raw_trades = engine.run(ticks, m["strat"])
        pips = [float(r["pips"]) for r in raw_trades]
        return m_idx, split, pips
    except Exception:
        return m_idx, split, []

def calc_stats(pips, mult, sl_pips):
    if not pips:
        return 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    arr = np.array(pips, dtype=np.float64)
    trades = len(arr)
    wins = arr[arr > 0]
    losses = arr[arr < 0]
    wr = len(wins) / trades * 100.0 if trades > 0 else 0.0
    gross_win = float(np.sum(wins)) if len(wins) > 0 else 0.0
    gross_loss = float(abs(np.sum(losses))) if len(losses) > 0 else 0.0
    pf = gross_win / gross_loss if gross_loss > 0 else 999.0
    net_pips = float(np.sum(arr))
    norm_pips = net_pips * mult
    net_r = net_pips / sl_pips if sl_pips > 0 else 0.0
    ann_ret = (net_r / 5.0) * 1.0  # 5-year annualized
    return trades, wr, pf, net_pips, norm_pips, net_r, ann_ret

def main():
    tasks = []
    for idx, m in enumerate(models):
        disc_w = get_discovery_weeks(raw_dir=f"data/{m['asset']}")
        hold_w = get_holdout_test_weeks(raw_dir=f"data/{m['asset']}")
        for w in disc_w:
            tasks.append((idx, w, "DISC"))
        for w in hold_w:
            tasks.append((idx, w, "HOLD"))

    res = {i: {"DISC": [], "HOLD": []} for i in range(len(models))}
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(run_task, t) for t in tasks]
        for fut in as_completed(futures):
            idx, split, pips = fut.result()
            res[idx][split].extend(pips)

    print("=" * 115)
    print(f"{'Model Name':<32} {'Partition':<12} {'Trades':<8} {'WR%':<7} {'PF':<7} {'Raw Pips':<11} {'Norm Pips':<11} {'Net R':<9} {'Annual Ret%'}")
    print("-" * 115)

    for idx, m in enumerate(models):
        d_pips = res[idx]["DISC"]
        h_pips = res[idx]["HOLD"]
        all_pips = d_pips + h_pips

        tr_d, wr_d, pf_d, np_d, norm_d, r_d, _ = calc_stats(d_pips, m["mult"], m["sl_pips"])
        tr_h, wr_h, pf_h, np_h, norm_h, r_h, _ = calc_stats(h_pips, m["mult"], m["sl_pips"])
        tr_a, wr_a, pf_a, np_a, norm_a, r_a, ann_a = calc_stats(all_pips, m["mult"], m["sl_pips"])

        # Annual return on holdout: 1.5 years
        ann_h = (r_h / 1.5) * 1.0

        print(f"{m['name']:<32} {'70% Disc':<12} {tr_d:<8} {wr_d:<7.1f} {pf_d:<7.2f} {np_d:<11.1f} {norm_d:<11.1f} {r_d:<9.1f} {(r_d/3.5):.1f}%")
        print(f"{m['name']:<32} {'30% Holdout':<12} {tr_h:<8} {wr_h:<7.1f} {pf_h:<7.2f} {np_h:<11.1f} {norm_h:<11.1f} {r_h:<9.1f} {ann_h:.1f}%")
        print(f"{m['name']:<32} {'100% 5-Year':<12} {tr_a:<8} {wr_a:<7.1f} {pf_a:<7.2f} {np_a:<11.1f} {norm_a:<11.1f} {r_a:<9.1f} {ann_a:.1f}%")
        print("-" * 115)

if __name__ == "__main__":
    main()

