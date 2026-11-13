#!/usr/bin/env python3
"""
================================================================================
  TARGETED MACRO & SESSION OVERLAP DISCOVERY FOR UNDERREPRESENTED PAIRS
================================================================================
Searches for new, non-redundant champions for:
- USDCAD: London/NY Overlap (already found +3431.8p!)
- GBPJPY: London Open Impulse (already found +994.5p!)
- GBPCHF: Extended Macro Donchian Retest (Lookback 14, 16)
- EURNZD: Extended Macro Donchian Retest (Lookback 14, 16)
- AUDCAD: London/NY Overlap Range Impulse (12:00-15:00 UTC)
- NZDCAD: London/NY Overlap Range Impulse (12:00-15:00 UTC)
================================================================================
"""

import sys
import gc
import json
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from tick_engine.core.dataloader import get_holdout_test_weeks, get_discovery_weeks
from tick_engine.core.config import PIP_SIZES, get_friction_pips, WARMUP_TICKS
from tick_engine.strategies.macro_retest_helper import precompute_week_data, eval_config_on_week
from scripts.hunt_diversified_cross_pair_alpha import (
    precompute_discovery_week,
    eval_expansion_archetype,
    check_anti_redundancy
)


def hunt_extended_macro_pair(symbol: str, sl_tp_grid: list, lookbacks: list, dists: list, wicks: list, sessions: list, max_spread: float):
    print("=" * 80)
    print(f"  HUNTING EXTENDED MACRO RETEST CHAMPIONS FOR {symbol}")
    print("=" * 80)

    comm, slip = get_friction_pips(symbol)
    hold_weeks = get_holdout_test_weeks(symbol)
    disc_weeks = get_discovery_weeks(symbol)[:90]

    print(f"Precomputing Discovery ({len(disc_weeks)}w)...")
    with ProcessPoolExecutor(max_workers=5) as ex:
        disc_cache = [w for w in ex.map(precompute_week_data, [symbol]*len(disc_weeks), disc_weeks) if w is not None]

    print(f"Precomputing Holdout ({len(hold_weeks)}w)...")
    with ProcessPoolExecutor(max_workers=5) as ex:
        hold_cache = [w for w in ex.map(precompute_week_data, [symbol]*len(hold_weeks), hold_weeks) if w is not None]

    found = []
    print(f"Evaluating parameter grid...")

    for sl, tp in sl_tp_grid:
        for lb in lookbacks:
            for dst in dists:
                for wk in wicks:
                    for sess in sessions:
                        # 1. Screen Discovery
                        tr_disc = []
                        for w in disc_cache:
                            t = eval_config_on_week(w, sl, tp, lb, dst, wk, sess, max_spread, comm, slip)
                            if len(t) > 0: tr_disc.append(t)
                        if not tr_disc: continue
                        comb_d = np.concatenate(tr_disc)
                        if len(comb_d) < 20: continue
                        p_d = comb_d['pips']
                        w_d = float(np.sum(p_d[p_d > 0]))
                        l_d = float(abs(np.sum(p_d[p_d < 0])))
                        pf_d = w_d / l_d if l_d > 0 else 99.0
                        net_d = w_d - l_d
                        if pf_d < 1.25 or net_d <= 0: continue

                        # 2. Evaluate Quarantined Holdout
                        tr_hold = []
                        for w in hold_cache:
                            t = eval_config_on_week(w, sl, tp, lb, dst, wk, sess, max_spread, comm, slip)
                            if len(t) > 0: tr_hold.append(t)
                        if not tr_hold: continue
                        comb_h = np.concatenate(tr_hold)
                        if len(comb_h) < 15: continue
                        p_h = comb_h['pips']
                        w_h = float(np.sum(p_h[p_h > 0]))
                        l_h = float(abs(np.sum(p_h[p_h < 0])))
                        pf_h = w_h / l_h if l_h > 0 else 99.0
                        net_h = w_h - l_h
                        exp_h = net_h / len(comb_h)

                        # Outlier check
                        p_sorted = np.sort(p_h)
                        p_ex3 = float(np.sum(p_sorted[:-3]))

                        # Anti-redundancy check
                        is_distinct, msg = check_anti_redundancy(symbol, comb_h, max_overlap_pct=50.0, max_corr=0.50)

                        if pf_h >= 1.20 and net_h > 0 and exp_h >= 2.0 and p_ex3 > 0 and is_distinct:
                            sess_name = "Asian+London" if sess == 1 else ("London+NY" if sess == 2 else "All-Day")
                            found.append({
                                "symbol": symbol,
                                "sl_pips": sl,
                                "tp_pips": tp,
                                "lookback_h4": lb,
                                "dist_pips": dst,
                                "min_wick": wk,
                                "session_mode": sess,
                                "session_name": sess_name,
                                "max_spread": max_spread,
                                "disc_trades": len(comb_d),
                                "disc_pips": round(net_d, 1),
                                "disc_pf": round(pf_d, 2),
                                "hold_trades": len(comb_h),
                                "hold_pips": round(net_h, 1),
                                "hold_pf": round(pf_h, 2),
                                "hold_exp": round(exp_h, 2),
                                "hold_pips_ex3": round(p_ex3, 1)
                            })
                            print(f"  FOUND: SL {sl:.0f}/TP {tp:.0f} | LB {lb} | {sess_name:<12} | Hold PnL: {net_h:>+7.1f}p | Hold PF: {pf_h:.2f} | Trades: {len(comb_h)}")

    print(f"Total non-redundant champions found for {symbol}: {len(found)}")
    return found


def hunt_overlap_impulse_pair(symbol: str, sl_tp_grid: list):
    print("=" * 80)
    print(f"  HUNTING LONDON/NY OVERLAP IMPULSE FOR {symbol}")
    print("=" * 80)
    comm, slip = get_friction_pips(symbol)
    hold_weeks = get_holdout_test_weeks(symbol)
    disc_weeks = get_discovery_weeks(symbol)[::3]

    print(f"Precomputing Discovery ({len(disc_weeks)}w)...")
    with ProcessPoolExecutor(max_workers=5) as ex:
        disc_cache = [w for w in ex.map(precompute_discovery_week, [symbol]*len(disc_weeks), disc_weeks) if w is not None]

    print(f"Precomputing Holdout ({len(hold_weeks)}w)...")
    with ProcessPoolExecutor(max_workers=5) as ex:
        hold_cache = [w for w in ex.map(precompute_discovery_week, [symbol]*len(hold_weeks), hold_weeks) if w is not None]

    found = []
    # 12:00 to 14:30 UTC
    for sl, tp in sl_tp_grid:
        for atr_m in [1.10, 1.25]:
            for br in [0.50, 0.55]:
                # Discovery
                tr_d = [eval_expansion_archetype(w, 12, 14, sl, tp, atr_m, br, 2.5, comm, slip) for w in disc_cache]
                tr_d = [t for t in tr_d if len(t) > 0]
                if not tr_d: continue
                comb_d = np.concatenate(tr_d)
                if len(comb_d) < 20: continue
                p_d = comb_d['pips']
                w_d, l_d = float(np.sum(p_d[p_d > 0])), float(abs(np.sum(p_d[p_d < 0])))
                pf_d = w_d / l_d if l_d > 0 else 99.0
                net_d = w_d - l_d
                if pf_d < 1.25 or net_d <= 0: continue

                # Holdout
                tr_h = [eval_expansion_archetype(w, 12, 14, sl, tp, atr_m, br, 2.5, comm, slip) for w in hold_cache]
                tr_h = [t for t in tr_h if len(t) > 0]
                if not tr_h: continue
                comb_h = np.concatenate(tr_h)
                if len(comb_h) < 15: continue
                p_h = comb_h['pips']
                w_h, l_h = float(np.sum(p_h[p_h > 0])), float(abs(np.sum(p_h[p_h < 0])))
                pf_h = w_h / l_h if l_h > 0 else 99.0
                net_h = w_h - l_h
                exp_h = net_h / len(comb_h)

                p_sorted = np.sort(p_h)
                p_ex3 = float(np.sum(p_sorted[:-3]))

                is_distinct, msg = check_anti_redundancy(symbol, comb_h, max_overlap_pct=50.0, max_corr=0.50)

                if pf_h >= 1.20 and net_h > 0 and exp_h >= 1.50 and p_ex3 > 0 and is_distinct:
                    found.append({
                        "symbol": symbol,
                        "archetype": "London_NY_Overlap_Impulse",
                        "sl_pips": sl,
                        "tp_pips": tp,
                        "atr_mult": atr_m,
                        "body_ratio": br,
                        "max_spread": 2.5,
                        "start_hr": 12,
                        "end_hr": 14,
                        "disc_trades": len(comb_d),
                        "disc_pips": round(net_d, 1),
                        "disc_pf": round(pf_d, 2),
                        "hold_trades": len(comb_h),
                        "hold_pips": round(net_h, 1),
                        "hold_pf": round(pf_h, 2),
                        "hold_exp": round(exp_h, 2),
                        "hold_pips_ex3": round(p_ex3, 1)
                    })
                    print(f"  FOUND: SL {sl:.0f}/TP {tp:.0f} | ATR {atr_m} | Hold PnL: {net_h:>+7.1f}p | Hold PF: {pf_h:.2f} | Trades: {len(comb_h)}")

    print(f"Total non-redundant champions found for {symbol}: {len(found)}")
    return found


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="GBPCHF")
    args = parser.parse_args()

    sym = args.symbol.upper()
    if sym == "GBPCHF":
        res = hunt_extended_macro_pair(
            sym,
            sl_tp_grid=[(20.0, 48.0), (22.0, 52.0), (24.0, 58.0), (25.0, 60.0)],
            lookbacks=[14, 16],
            dists=[6.0, 8.0, 10.0],
            wicks=[0.35, 0.40],
            sessions=[1, 2],
            max_spread=3.5
        )
    elif sym == "EURNZD":
        res = hunt_extended_macro_pair(
            sym,
            sl_tp_grid=[(25.0, 60.0), (28.0, 68.0), (30.0, 75.0), (32.0, 80.0)],
            lookbacks=[14, 16],
            dists=[8.0, 10.0, 12.0],
            wicks=[0.35, 0.40],
            sessions=[1, 2],
            max_spread=5.0
        )
    elif sym in ["AUDCAD", "NZDCAD"]:
        res = hunt_overlap_impulse_pair(
            sym,
            sl_tp_grid=[(14.0, 28.0), (16.0, 32.0), (18.0, 38.0), (20.0, 42.0)]
        )
    else:
        res = []

    if res:
        out = ROOT_DIR / "results" / f"new_macro_{sym.lower()}.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
        print(f"Saved results to {out}")

