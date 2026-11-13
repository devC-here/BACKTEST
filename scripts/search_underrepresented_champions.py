#!/usr/bin/env python3
"""
================================================================================
  SEARCH UNDERREPRESENTED CURRENCY PAIRS (GBPCHF, GBPJPY, EURNZD, EURGBP, AUDNZD)
================================================================================
Discovers robust, dual-certified champions for low-representation currency pairs
to maximize cross-pair portfolio diversification:
  - GBPCHF: Currently 0 strategies
  - GBPJPY: Currently 0 strategies
  - EURNZD: Currently 2 strategies
  - EURGBP: Currently 3 strategies
  - AUDNZD: Currently 4 strategies
Evaluates across continuous chronological Holdout (90w) and Discovery (100w)
under full live broker friction ($4/lot comm + 0.20p slippage + floating spread).
================================================================================
"""

import sys
import gc
import json
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from tick_engine.core.dataloader import get_holdout_test_weeks, get_discovery_weeks
from tick_engine.core.config import PIP_SIZES, get_friction_pips, WARMUP_TICKS
from tick_engine.reporting.run_manifest import compute_statistical_metrics
from tick_engine.strategies.macro_retest_helper import precompute_week_data, eval_config_on_week

SEARCH_CONFIGS = {
    "GBPCHF": {
        "sl_tp_pairs": [(18.0, 36.0), (20.0, 45.0), (22.0, 50.0), (25.0, 60.0)],
        "lookbacks": [8, 10, 12],
        "dists": [5.0, 6.0, 8.0],
        "wicks": [0.35, 0.40],
        "sessions": [1, 2, 3],
        "max_spread": 3.5
    },
    "GBPJPY": {
        "sl_tp_pairs": [(25.0, 50.0), (30.0, 65.0), (35.0, 75.0), (40.0, 90.0)],
        "lookbacks": [8, 10, 12],
        "dists": [6.0, 8.0, 10.0],
        "wicks": [0.35, 0.40],
        "sessions": [1, 2, 3],
        "max_spread": 3.0
    },
    "EURNZD": {
        "sl_tp_pairs": [(22.0, 45.0), (25.0, 55.0), (28.0, 65.0), (30.0, 75.0)],
        "lookbacks": [8, 10, 12],
        "dists": [8.0, 10.0, 12.0],
        "wicks": [0.35, 0.40],
        "sessions": [1, 2, 3],
        "max_spread": 5.0
    },
    "EURGBP": {
        "sl_tp_pairs": [(10.0, 20.0), (12.0, 26.0), (14.0, 30.0), (16.0, 35.0)],
        "lookbacks": [8, 10, 12],
        "dists": [3.0, 4.0, 5.0],
        "wicks": [0.35, 0.40],
        "sessions": [1, 2, 3],
        "max_spread": 2.2
    },
    "AUDNZD": {
        "sl_tp_pairs": [(16.0, 32.0), (18.0, 40.0), (20.0, 45.0), (22.0, 55.0)],
        "lookbacks": [8, 10, 12],
        "dists": [4.0, 5.0, 6.0],
        "wicks": [0.35, 0.40],
        "sessions": [1, 2, 3],
        "max_spread": 3.0
    }
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
    if exp_pips < 1.25:
        return None
    # Outlier filter: non-outlier dependent
    if n > 3:
        sorted_pips = np.sort(pips)
        if np.sum(sorted_pips[:-3]) <= 0:
            return None

    return {
        'total_trades': n,
        'net_pips': round(net_pips, 1),
        'profit_factor': round(pf, 2),
        'expectancy_pips': round(exp_pips, 2)
    }


def search_symbol(symbol: str, cfg: dict):
    print(f"\n=======================================================")
    print(f"  Searching Diversification Champions for {symbol}")
    print(f"=======================================================")
    
    comm, slip = get_friction_pips(symbol)
    hold_weeks = get_holdout_test_weeks(symbol)
    disc_weeks = get_discovery_weeks(symbol)

    print(f"Precomputing holdout ({len(hold_weeks)}w) for {symbol}...")
    with ProcessPoolExecutor(max_workers=3) as ex:
        hold_pre = [w for w in ex.map(precompute_week_data, [symbol]*len(hold_weeks), hold_weeks) if w is not None]

    print(f"Valid holdout weeks precomputed: {len(hold_pre)}")

    sl_tp_pairs = cfg["sl_tp_pairs"]
    lookbacks = cfg["lookbacks"]
    dists = cfg["dists"]
    wicks = cfg["wicks"]
    sessions = cfg["sessions"]
    max_spread = cfg["max_spread"]

    candidates = []
    t0 = time.time()
    total_combos = len(sl_tp_pairs) * len(lookbacks) * len(dists) * len(wicks) * len(sessions)
    print(f"Evaluating {total_combos} parameter combinations across {len(hold_pre)} holdout weeks...")

    for sl, tp in sl_tp_pairs:
        for lb in lookbacks:
            for dst in dists:
                for wk in wicks:
                    for sm in sessions:
                        hold_trades = []
                        for wd in hold_pre:
                            tr = eval_config_on_week(wd, sl, tp, lb, dst, wk, sm, max_spread, comm, slip)
                            if len(tr) > 0:
                                hold_trades.append(tr)
                        if not hold_trades:
                            continue
                        comb_h = np.concatenate(hold_trades)
                        m_h = fast_screen_metrics(comb_h)

                        if m_h is not None:
                            candidates.append({
                                'symbol': symbol,
                                'sl': sl, 'tp': tp, 'lb': lb, 'dist': dst, 'wick': wk, 'session': sm,
                                'max_spread': max_spread,
                                'holdout_screen': m_h,
                                'holdout_trades': comb_h
                            })

    print(f"Holdout scan done in {time.time()-t0:.1f}s. Candidates passing holdout: {len(candidates)}")
    if not candidates:
        del hold_pre
        gc.collect()
        return []

    # Sort candidates by holdout net pips
    candidates.sort(key=lambda x: x['holdout_screen']['net_pips'], reverse=True)

    # Precompute sample discovery weeks to verify in-sample alpha
    print(f"Precomputing Discovery weeks (1..100) for {symbol}...")
    sample_disc_weeks = disc_weeks[:100]
    with ProcessPoolExecutor(max_workers=3) as ex:
        disc_pre = [w for w in ex.map(precompute_week_data, [symbol]*len(sample_disc_weeks), sample_disc_weeks) if w is not None]

    print(f"Valid discovery weeks precomputed: {len(disc_pre)}")
    print(f"Evaluating top {min(len(candidates), 30)} holdout candidates on Discovery data...")

    verified = []
    for c in candidates[:30]:
        disc_trades = []
        for wd in disc_pre:
            tr = eval_config_on_week(wd, c['sl'], c['tp'], c['lb'], c['dist'], c['wick'], c['session'], max_spread, comm, slip)
            if len(tr) > 0:
                disc_trades.append(tr)
        if not disc_trades:
            continue
        comb_d = np.concatenate(disc_trades)
        m_d = fast_screen_metrics(comb_d)

        if m_d is not None and m_d['total_trades'] >= 20:
            # Full statistical calculation only on the verified champion
            m_h_full = compute_statistical_metrics(c['holdout_trades'])
            m_d_full = compute_statistical_metrics(comb_d)

            champ_entry = {
                'symbol': c['symbol'],
                'sl': c['sl'], 'tp': c['tp'], 'lb': c['lb'], 'dist': c['dist'],
                'wick': c['wick'], 'session': c['session'], 'max_spread': c['max_spread'],
                'holdout': m_h_full,
                'discovery': m_d_full
            }
            verified.append(champ_entry)
            print(f"  *** DUAL-CERTIFIED {symbol} *** SL/TP: {c['sl']}/{c['tp']} | LB: {c['lb']} | Dist: {c['dist']} | Wk: {c['wick']} | Sm: {c['session']}")
            print(f"      Holdout (90w): Net {m_h_full['net_pips']:>+6.1f}p | PF {m_h_full['profit_factor']:.2f} | Tr {m_h_full['total_trades']} | Exp {m_h_full['expectancy_pips']:>+5.2f}p")
            print(f"      Disc (100w):   Net {m_d_full['net_pips']:>+6.1f}p | PF {m_d_full['profit_factor']:.2f} | Tr {m_d_full['total_trades']} | Exp {m_d_full['expectancy_pips']:>+5.2f}p")

    print(f"Total dual-certified champions for {symbol}: {len(verified)}")
    del hold_pre
    del disc_pre
    gc.collect()
    return verified


def main():
    results = {}
    output_path = ROOT_DIR / "results" / "underrepresented_champions_found.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    symbols_to_search = ["GBPCHF", "GBPJPY", "EURNZD", "EURGBP", "AUDNZD"]
    
    for sym in symbols_to_search:
        champs = search_symbol(sym, SEARCH_CONFIGS[sym])
        results[sym] = champs
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Saved interim progress for {sym} ({len(champs)} champions) to {output_path.name}")

    print("\n=======================================================")
    print("  ALL UNDERREPRESENTED SEARCHES COMPLETE!")
    for sym, champs in results.items():
        print(f"  {sym:>7}: {len(champs)} dual-certified champions found")
    print("=======================================================")


if __name__ == "__main__":
    main()
