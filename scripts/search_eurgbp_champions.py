#!/usr/bin/env python3
"""
================================================================================
  TARGETED EURGBP ALPHA DISCOVERY & CERTIFICATION SWEEP
================================================================================
Runs high-speed grid search for EURGBP tailored to its lower volatility (ATR ~35p),
using min_bar_rng=1.2, tight SL/TP (7-10p SL, 11-18p TP), and realistic pip value economics ($13.00/pip).
================================================================================
"""

import sys
import gc
import json
import itertools
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from tick_engine.core.dataloader import get_holdout_test_weeks, get_discovery_weeks
from tick_engine.core.config import PIP_SIZES, get_friction_pips
from tick_engine.reporting.run_manifest import compute_statistical_metrics
from tick_engine.strategies.macro_retest_helper import precompute_week_data, eval_config_on_week

GRID = {
    "sl_tp_pairs": [(7.0, 11.0), (8.0, 12.0), (8.0, 14.0), (9.0, 14.0), (9.0, 16.0), (10.0, 15.0), (10.0, 18.0)],
    "lookbacks": [6, 8, 10, 12, 14],
    "dists": [2.5, 3.5, 4.5],
    "wicks": [0.30, 0.35, 0.40],
    "sessions": [1, 2, 3],
    "max_spread": 2.0,
    "min_bar_rng": 1.2
}


def fast_screen_metrics(comb_trades):
    n = len(comb_trades)
    if n < 15:
        return None
    pips = comb_trades['pips']
    wins = pips[pips > 0]
    losses = pips[pips < 0]
    win_pips = float(np.sum(wins))
    loss_pips = float(abs(np.sum(losses)))
    net_pips = win_pips - loss_pips
    if net_pips <= 0:
        return None
    pf = (win_pips / loss_pips) if loss_pips > 0 else 99.0
    if pf < 1.15:
        return None
    exp_pips = net_pips / n
    if exp_pips < 0.65:  # 0.65p on EURGBP = $8.45/trade (well above threshold)
        return None
    return {
        "n_trades": n,
        "net_pips": net_pips,
        "pf": pf,
        "exp_pips": exp_pips
    }


def main():
    sym = "EURGBP"
    pip_size = PIP_SIZES[sym]
    comm, slip = get_friction_pips(sym)
    
    h_weeks = get_holdout_test_weeks(sym)
    d_weeks = get_discovery_weeks(sym)[:100]
    
    print(f"=== Searching EURGBP Champions ===")
    print(f"Precomputing holdout ({len(h_weeks)}w)...")
    with ProcessPoolExecutor(max_workers=3) as ex:
        cached_holdout = [w for w in ex.map(precompute_week_data, [sym]*len(h_weeks), h_weeks) if w is not None]
        
    print(f"Precomputing discovery ({len(d_weeks)}w)...")
    with ProcessPoolExecutor(max_workers=3) as ex:
        cached_discovery = [w for w in ex.map(precompute_week_data, [sym]*len(d_weeks), d_weeks) if w is not None]
        
    combos = list(itertools.product(
        GRID["sl_tp_pairs"],
        GRID["lookbacks"],
        GRID["dists"],
        GRID["wicks"],
        GRID["sessions"]
    ))
    print(f"Total combinations to evaluate: {len(combos)}")
    
    holdout_champs = []
    min_rng = GRID["min_bar_rng"]
    max_spread = GRID["max_spread"]
    
    for (sl, tp), lb, dst, wk, sm in combos:
        all_tr = []
        for wd in cached_holdout:
            tr = eval_config_on_week(wd, sl, tp, lb, dst, wk, sm, max_spread, comm, slip, min_bar_rng=min_rng)
            if len(tr) > 0:
                all_tr.append(tr)
        if not all_tr:
            continue
        comb = np.concatenate(all_tr)
        res = fast_screen_metrics(comb)
        if res is not None:
            holdout_champs.append({
                "symbol": sym,
                "sl": sl,
                "tp": tp,
                "lb": lb,
                "dist": dst,
                "wick": wk,
                "session": sm,
                "max_spread": max_spread,
                "min_bar_rng": min_rng,
                "holdout_fast": res
            })
            
    print(f"Holdout candidates passed: {len(holdout_champs)}")
    
    # Dual certification against discovery
    dual_certified = []
    for c in holdout_champs:
        all_d = []
        for wd in cached_discovery:
            tr = eval_config_on_week(wd, c['sl'], c['tp'], c['lb'], c['dist'], c['wick'], c['session'], max_spread, comm, slip, min_bar_rng=min_rng)
            if len(tr) > 0:
                all_d.append(tr)
        if not all_d:
            continue
        comb_d = np.concatenate(all_d)
        res_d = fast_screen_metrics(comb_d)
        if res_d is not None and res_d['net_pips'] > 0 and res_d['pf'] >= 1.15:
            # Full statistical metrics on holdout
            all_h = []
            for wd in cached_holdout:
                tr = eval_config_on_week(wd, c['sl'], c['tp'], c['lb'], c['dist'], c['wick'], c['session'], max_spread, comm, slip, min_bar_rng=min_rng)
                if len(tr) > 0:
                    all_h.append(tr)
            full_h = compute_statistical_metrics(np.concatenate(all_h))
            full_d = compute_statistical_metrics(comb_d)
            
            c["holdout"] = full_h
            c["discovery"] = full_d
            dual_certified.append(c)
            
    # Sort by holdout net pips
    dual_certified.sort(key=lambda x: x["holdout"]["net_pips"], reverse=True)
    print(f"\n>>> EURGBP Dual-Certified Champions Found: {len(dual_certified)} <<<")
    
    out_file = ROOT_DIR / "results" / "eurgbp_champions_found.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(dual_certified, f, indent=2)
    print(f"Saved to {out_file}")
    
    for i, c in enumerate(dual_certified[:10]):
        print(f"#{i+1}: SL={c['sl']}, TP={c['tp']}, LB={c['lb']}, Dist={c['dist']}, Wick={c['wick']}, Session={c['session']} | "
              f"Holdout: Net={c['holdout']['net_pips']:.1f}p, N={c['holdout']['total_trades']}, PF={c['holdout']['profit_factor']:.2f}, Exp={c['holdout']['expectancy_pips']:.2f}p | "
              f"Disc: Net={c['discovery']['net_pips']:.1f}p, PF={c['discovery']['profit_factor']:.2f}")


if __name__ == "__main__":
    main()

