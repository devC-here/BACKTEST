#!/usr/bin/env python3
"""
Formal Audit of DISC80: Double Swing Neckline Retest Elite
Testing on EURUSD, USDJPY, and EURGBP across:
  - 70% In-Sample Discovery Partition
  - 30% Quarantined Out-of-Sample Holdout Partition
  - 100% Full 5-Year Cycle
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
from tick_engine.core.dataloader import get_holdout_test_weeks, get_discovery_weeks, get_week_dataset
from strategies.new_discoveries.disc80_double_swing_neckline_retest_elite.strategy import DoubleSwingNecklineRetestElite

models = [
    {
        "name": "EURUSD_DISC80",
        "asset": "EURUSD",
        "strat": DoubleSwingNecklineRetestElite(asset="EURUSD", sl_pips=9.0, tp_pips=18.0, pip_size=0.0001, max_spread=0.6),
        "sl_pips": 9.0
    },
    {
        "name": "EURUSD_DISC80_22R",
        "asset": "EURUSD",
        "strat": DoubleSwingNecklineRetestElite(asset="EURUSD", sl_pips=8.0, tp_pips=17.6, pip_size=0.0001, max_spread=0.6),
        "sl_pips": 8.0
    },
    {
        "name": "USDJPY_DISC80",
        "asset": "USDJPY",
        "strat": DoubleSwingNecklineRetestElite(asset="USDJPY", sl_pips=12.0, tp_pips=24.0, pip_size=0.01, max_spread=0.8),
        "sl_pips": 12.0
    },
    {
        "name": "EURGBP_DISC80",
        "asset": "EURGBP",
        "strat": DoubleSwingNecklineRetestElite(asset="EURGBP", sl_pips=8.0, tp_pips=16.0, pip_size=0.0001, max_spread=0.8),
        "sl_pips": 8.0
    }
]

def run_task(args):
    m_idx, w, split = args
    m = models[m_idx]
    engine = BacktestEngine(pip_size=m["strat"].pip_size, use_slippage=True)
    try:
        ticks, _ = get_week_dataset(w, raw_dir=f"data/{m['asset']}")
        if ticks is None or len(ticks) < 500:
            return m_idx, split, []
        raw_trades = engine.run(ticks, m["strat"])
        pips = [float(r["pips"]) for r in raw_trades]
        return m_idx, split, pips
    except Exception:
        return m_idx, split, []

def calc_stats(pips, sl_pips, years):
    if not pips:
        return 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    arr = np.array(pips, dtype=np.float64)
    trades = len(arr)
    wins = arr[arr > 0]
    losses = arr[arr < 0]
    wr = len(wins) / trades * 100.0 if trades > 0 else 0.0
    gross_win = float(np.sum(wins)) if len(wins) > 0 else 0.0
    gross_loss = float(abs(np.sum(losses))) if len(losses) > 0 else 0.0
    pf = gross_win / gross_loss if gross_loss > 0 else 999.0
    net_pips = float(np.sum(arr))
    net_r = net_pips / sl_pips if sl_pips > 0 else 0.0
    ann_ret = (net_r / years) * 1.0

    equity = np.cumsum(arr)
    peak = np.maximum.accumulate(equity)
    dd = peak - equity
    max_dd = float(np.max(dd)) if len(dd) > 0 else 0.0
    ret_dd = net_pips / max_dd if max_dd > 0 else 999.0

    return trades, wr, pf, net_pips, net_r, ann_ret, max_dd, ret_dd

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

    print("=" * 125)
    print(f"{'Model Name':<20} {'Partition':<12} {'Trades':<8} {'WR%':<7} {'PF':<7} {'Net Pips':<11} {'Net R':<9} {'Annual Ret%':<13} {'Max DD(p)':<10} {'Ret/DD'}")
    print("-" * 125)

    for idx, m in enumerate(models):
        d_pips = res[idx]["DISC"]
        h_pips = res[idx]["HOLD"]
        all_pips = d_pips + h_pips

        tr_d, wr_d, pf_d, np_d, r_d, ann_d, dd_d, rdd_d = calc_stats(d_pips, m["sl_pips"], 3.5)
        tr_h, wr_h, pf_h, np_h, r_h, ann_h, dd_h, rdd_h = calc_stats(h_pips, m["sl_pips"], 1.5)
        tr_a, wr_a, pf_a, np_a, r_a, ann_a, dd_a, rdd_a = calc_stats(all_pips, m["sl_pips"], 5.0)

        print(f"{m['name']:<20} {'70% Disc':<12} {tr_d:<8} {wr_d:<7.1f} {pf_d:<7.2f} {np_d:<11.1f} {r_d:<9.1f} {ann_d:<13.1f} {dd_d:<10.1f} {rdd_d:.2f}x")
        print(f"{m['name']:<20} {'30% Holdout':<12} {tr_h:<8} {wr_h:<7.1f} {pf_h:<7.2f} {np_h:<11.1f} {r_h:<9.1f} {ann_h:<13.1f} {dd_h:<10.1f} {rdd_h:.2f}x")
        print(f"{m['name']:<20} {'100% 5-Year':<12} {tr_a:<8} {wr_a:<7.1f} {pf_a:<7.2f} {np_a:<11.1f} {r_a:<9.1f} {ann_a:<13.1f} {dd_a:<10.1f} {rdd_a:.2f}x")
        print("-" * 125)

if __name__ == "__main__":
    main()

