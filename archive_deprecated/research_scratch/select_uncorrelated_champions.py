import json
from pathlib import Path
import numpy as np

def select_top_uncorrelated(candidates, max_select=5, max_overlap_thresh=0.35):
    # Sort candidates by Holdout Profit Factor * Net Expectancy
    sorted_cands = sorted(
        candidates,
        key=lambda x: (x['holdout']['profit_factor'] * x['holdout']['expectancy_pips']),
        reverse=True
    )
    
    selected = []
    for cand in sorted_cands:
        if len(selected) >= max_select:
            break
            
        # We want diversity in lookback, SL/TP, and session mode
        conflict = False
        for s in selected:
            # Check parameter closeness
            same_session = (s['session'] == cand['session'])
            same_sltp = (s['sl'] == cand['sl'] and s['tp'] == cand['tp'])
            same_lb = (s['lb'] == cand['lb'])
            if same_session and same_sltp and same_lb:
                conflict = True
                break
                
        if not conflict:
            selected.append(cand)
            
    return selected

if __name__ == "__main__":
    with open("results/clean_macro_champions_found.json") as f:
        data = json.load(f)
        
    for sym in ["AUDCAD", "NZDCAD"]:
        cands = data.get(sym, [])
        top = select_top_uncorrelated(cands, max_select=5)
        print(f"\n================ Top 5 Diverse Champions for {sym} ================")
        for i, c in enumerate(top, 1):
            h = c['holdout']
            d = c['discovery']
            print(f"[{i}] SL/TP: {c['sl']}/{c['tp']} | LB: {c['lb']} | Dist: {c['dist']} | Wick: {c['wick']} | Sm: {c['session']}")
            print(f"    Holdout: Net {h['net_pips']:>+6.1f}p | PF {h['profit_factor']:.2f} | Tr: {h['total_trades']:>2} | Exp: {h['expectancy_pips']:>+5.2f}p")
            print(f"    Disc:    Net {d['net_pips']:>+6.1f}p | PF {d['profit_factor']:.2f} | Tr: {d['total_trades']:>2} | Exp: {d['expectancy_pips']:>+5.2f}p")
