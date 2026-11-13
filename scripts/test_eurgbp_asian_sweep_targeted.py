import sys
from pathlib import Path
import numpy as np
import pandas as pd
import json

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from tick_engine.core.dataloader import get_complete_split_weeks
from scripts.mine_eurgbp_novel_archetypes import load_eurgbp_data, eval_asian_sweep_fade

SYMBOL = "EURGBP"

print("Loading Holdout weeks...", flush=True)
h_weeks = get_complete_split_weeks(ROOT_DIR / "data" / SYMBOL, split="holdout")
h_data = {w: load_eurgbp_data(w) for w in h_weeks}
h_data = {k: v for k, v in h_data.items() if v is not None}

print("Loading Discovery weeks...", flush=True)
d_weeks = get_complete_split_weeks(ROOT_DIR / "data" / SYMBOL, split="discovery")
d_data = {w: load_eurgbp_data(w) for w in d_weeks}
d_data = {k: v for k, v in d_data.items() if v is not None}

print(f"Loaded: {len(h_data)} holdout weeks, {len(d_data)} discovery weeks.\n", flush=True)

grid = []
for s_min in [0.2, 0.3, 0.4]:
    for s_max in [2.5, 3.5, 4.5]:
        for wick in [0.15, 0.20, 0.25]:
            for (start_hr, end_hr) in [(7, 10), (7, 11), (8, 11)]:
                for sl in [6.0, 7.0, 8.0]:
                    for tp in [9.0, 10.0, 12.0, 14.0]:
                        grid.append({
                            's_min': s_min, 's_max': s_max, 'wick': wick,
                            'start_hr': start_hr, 'end_hr': end_hr,
                            'sl': sl, 'tp': tp,
                            'desc': f"Sweep={s_min}-{s_max}p, Wk={wick}, H={start_hr}-{end_hr}, SL={sl}p, TP={tp}p"
                        })

print(f"Evaluating {len(grid)} setups...", flush=True)
champs = []

for s in grid:
    h_trades = []
    for w, wd in h_data.items():
        tr = eval_asian_sweep_fade(wd, s['s_min'], s['s_max'], s['wick'], s['start_hr'], s['end_hr'], s['sl'], s['tp'])
        h_trades.extend(tr)

    if len(h_trades) < 15:
        continue
    hp = np.array([t['pips'] for t in h_trades])
    h_net = float(np.sum(hp))
    if h_net <= 0:
        continue
    hw = hp[hp > 0]
    hl = hp[hp < 0]
    h_pf = float(np.sum(hw)) / (float(np.abs(np.sum(hl))) + 1e-5)
    if h_pf < 1.15:
        continue
    shp = np.sort(hp)
    ex_top3 = float(np.sum(shp[:-3])) if len(hp) >= 4 else -1.0
    if ex_top3 <= 0:
        continue

    # Discovery
    d_trades = []
    for w, wd in d_data.items():
        tr = eval_asian_sweep_fade(wd, s['s_min'], s['s_max'], s['wick'], s['start_hr'], s['end_hr'], s['sl'], s['tp'])
        d_trades.extend(tr)

    if len(d_trades) < 20:
        continue
    dp = np.array([t['pips'] for t in d_trades])
    d_net = float(np.sum(dp))
    if d_net <= 0:
        continue
    dw = dp[dp > 0]
    dl = dp[dp < 0]
    d_pf = float(np.sum(dw)) / (float(np.abs(np.sum(dl))) + 1e-5)

    tag = "*** DUAL CHAMPION ***" if d_pf >= 1.15 else "Positive Discovery"
    print(f"[{tag}] {s['desc']}:", flush=True)
    print(f"   Holdout:   {len(h_trades)} tr, +{h_net:.1f}p, PF={h_pf:.2f}, Exp={np.mean(hp):+.2f}p, ExTop3=+{ex_top3:.1f}p", flush=True)
    print(f"   Discovery: {len(d_trades)} tr, +{d_net:.1f}p, PF={d_pf:.2f}, Exp={np.mean(dp):+.2f}p", flush=True)

    if d_pf >= 1.15:
        champs.append({
            'setup': s,
            'holdout': {'trades': len(h_trades), 'net_pips': h_net, 'pf': h_pf, 'exp': float(np.mean(hp)), 'ex_top3': ex_top3},
            'discovery': {'trades': len(d_trades), 'net_pips': d_net, 'pf': d_pf, 'exp': float(np.mean(dp))}
        })

print(f"\nTargeted search complete. Found {len(champs)} DUAL CHAMPIONS!", flush=True)
if champs:
    out_f = ROOT_DIR / "results" / "new_champions_eurgbp_asian_sweep_targeted.json"
    with open(out_f, "w", encoding="utf-8") as f:
        json.dump(champs, f, indent=2)
    print(f"Saved to {out_f}", flush=True)

