import json
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from tick_engine.core.dataloader import get_holdout_test_weeks
from tick_engine.core.config import PIP_SIZES, get_friction_pips
from scripts.search_clean_macro_champions import precompute_week_data, eval_config_on_week
from scripts.promote_clean_champions import calc_overlap

def get_trades(symbol, c, hold_pre):
    comm, slip = get_friction_pips(symbol)
    max_spread = 2.5 if symbol != 'EURNZD' else 5.0
    all_tr = []
    for wd in hold_pre:
        tr = eval_config_on_week(wd, c['sl'], c['tp'], c['lb'], c['dist'], c['wick'], c['session'], max_spread, comm, slip)
        if len(tr) > 0:
            all_tr.append(tr)
    return np.concatenate(all_tr) if all_tr else np.array([])

def select_diverse(symbol, candidates, target_count, ex):
    hold_weeks = get_holdout_test_weeks(symbol)
    hold_pre = [w for w in ex.map(precompute_week_data, [symbol]*len(hold_weeks), hold_weeks) if w is not None]

    # Pre-calculate trades for all candidates
    cand_trades = []
    for c in candidates:
        tr = get_trades(symbol, c, hold_pre)
        cand_trades.append((c, tr))

    # Sort by holdout net pips
    cand_trades.sort(key=lambda x: x[0]['holdout']['net_pips'], reverse=True)

    selected = []
    for c, tr in cand_trades:
        if len(selected) == 0:
            selected.append((c, tr))
            print(f"  [{symbol}] Selected 1/{target_count}: SL/TP {c['sl']}/{c['tp']} | LB {c['lb']} | Dist {c['dist']} | Wk {c['wick']} | Sm {c['session']} | Net {c['holdout']['net_pips']:>+6.1f}p | PF {c['holdout']['profit_factor']:.2f}")
            continue

        # Check overlap with all already selected
        max_ov = 0.0
        for s_c, s_tr in selected:
            ov = calc_overlap(tr, s_tr, window_ms=15*60*1000)
            if ov > max_ov:
                max_ov = ov

        if max_ov < 0.35:
            selected.append((c, tr))
            print(f"  [{symbol}] Selected {len(selected)}/{target_count}: SL/TP {c['sl']}/{c['tp']} | LB {c['lb']} | Dist {c['dist']} | Wk {c['wick']} | Sm {c['session']} | Net {c['holdout']['net_pips']:>+6.1f}p | PF {c['holdout']['profit_factor']:.2f} (Max Overlap: {max_ov*100:.1f}%)")
            if len(selected) == target_count:
                break

    # If not enough with strict 0.35, allow next lowest overlap
    if len(selected) < target_count:
        for c, tr in cand_trades:
            if any(c == s[0] for s in selected):
                continue
            max_ov = max(calc_overlap(tr, s[1], window_ms=15*60*1000) for s in selected)
            if max_ov < 0.55:
                selected.append((c, tr))
                print(f"  [{symbol}] Selected (relaxed) {len(selected)}/{target_count}: SL/TP {c['sl']}/{c['tp']} | LB {c['lb']} | Dist {c['dist']} | Wk {c['wick']} | Sm {c['session']} | Net {c['holdout']['net_pips']:>+6.1f}p | PF {c['holdout']['profit_factor']:.2f} (Max Overlap: {max_ov*100:.1f}%)")
                if len(selected) == target_count:
                    break

    return [s[0] for s in selected]

def main():
    data = json.load(open('results/clean_macro_champions_found.json'))
    
    # Target distribution:
    # AUDCAD: 6
    # NZDCAD: 6
    # USDCAD: 5
    # EURNZD: 2
    # Total: 19
    targets = {
        'AUDCAD': 6,
        'NZDCAD': 6,
        'USDCAD': 5,
        'EURNZD': 2
    }

    final_19 = {}
    with ProcessPoolExecutor(max_workers=4) as ex:
        for sym, count in targets.items():
            print(f"\nSelecting {count} diverse champions for {sym}...")
            cands = data[sym]
            sel = select_diverse(sym, cands, count, ex)
            final_19[sym] = sel
            print(f"Successfully selected {len(sel)} champions for {sym}")

    with open('results/selected_19_champions.json', 'w') as f:
        json.dump(final_19, f, indent=2, default=str)
    print("\nSaved selections to results/selected_19_champions.json")

if __name__ == '__main__':
    main()
