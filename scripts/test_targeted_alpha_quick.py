#!/usr/bin/env python3
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from tick_engine.core.config import PIP_SIZES, get_friction_pips
from tick_engine.core.dataloader import get_complete_split_weeks
from scripts.hunt_diversified_cross_pair_alpha import (
    precompute_discovery_week,
    eval_expansion_archetype,
    check_anti_redundancy
)

def test_quick(symbol: str, start_hr: int, end_hr: int, sl_tp_list: list, max_spread: float):
    comm, slip = get_friction_pips(symbol)
    hold_weeks = get_complete_split_weeks(symbol, "holdout_test_30")
    disc_weeks = get_complete_split_weeks(symbol, "discovery_70")[::3]

    with ProcessPoolExecutor(max_workers=6) as ex:
        disc_cache = [w for w in ex.map(precompute_discovery_week, [symbol]*len(disc_weeks), disc_weeks) if w is not None]
        hold_cache = [w for w in ex.map(precompute_discovery_week, [symbol]*len(hold_weeks), hold_weeks) if w is not None]

    for sl, tp in sl_tp_list:
        for atr_m in [1.10, 1.25]:
            for br in [0.50, 0.55]:
                # Discovery
                tr_d = [eval_expansion_archetype(w, start_hr, end_hr, sl, tp, atr_m, br, max_spread, comm, slip) for w in disc_cache]
                tr_d = [t for t in tr_d if len(t) > 0]
                if not tr_d: continue
                comb_d = np.concatenate(tr_d)
                if len(comb_d) < 20: continue
                p_d = comb_d['pips']
                pf_d = float(np.sum(p_d[p_d > 0])) / float(abs(np.sum(p_d[p_d < 0]))) if np.any(p_d < 0) else 99.0
                net_d = float(np.sum(p_d))
                if pf_d < 1.20 or net_d <= 0: continue

                # Holdout
                tr_h = [eval_expansion_archetype(w, start_hr, end_hr, sl, tp, atr_m, br, max_spread, comm, slip) for w in hold_cache]
                tr_h = [t for t in tr_h if len(t) > 0]
                if not tr_h: continue
                comb_h = np.concatenate(tr_h)
                if len(comb_h) < 15: continue
                p_h = comb_h['pips']
                pf_h = float(np.sum(p_h[p_h > 0])) / float(abs(np.sum(p_h[p_h < 0]))) if np.any(p_h < 0) else 99.0
                net_h = float(np.sum(p_h))
                exp_h = net_h / len(comb_h)
                p_ex3 = float(np.sum(np.sort(p_h)[:-3]))
                is_distinct, msg = check_anti_redundancy(symbol, comb_h, max_overlap_pct=50.0, max_corr=0.50)

                if pf_h >= 1.20 and net_h > 0 and exp_h >= 2.0 and p_ex3 > 0 and is_distinct:
                    print(f"[{symbol}] WINNER: Hours {start_hr}-{end_hr} | SL {sl}/TP {tp} | ATR {atr_m} | Disc PF: {pf_d:.2f} | Hold PnL: {net_h:>+7.1f}p | Hold PF: {pf_h:.2f} | Trades: {len(comb_h)} | Exp: {exp_h:.2f}p")

if __name__ == "__main__":
    # Test EURNZD London/NY overlap
    print("Testing EURNZD 12-15...")
    test_quick("EURNZD", 12, 14, [(25.0, 55.0), (30.0, 70.0), (35.0, 80.0)], max_spread=4.5)
    # Test EURGBP Asian/London pre-market
    print("Testing EURGBP 06-08...")
    test_quick("EURGBP", 6, 8, [(8.0, 14.0), (9.0, 16.0), (10.0, 18.0)], max_spread=1.8)

