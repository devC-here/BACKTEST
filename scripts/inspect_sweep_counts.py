import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from tick_engine.core.dataloader import get_complete_split_weeks
from scripts.mine_eurgbp_novel_archetypes import load_eurgbp_data, eval_asian_sweep_fade, eval_pivot_s1r1_reversal, eval_london_fix_reversion

SYMBOL = "EURGBP"
h_weeks = get_complete_split_weeks(ROOT_DIR / "data" / SYMBOL, split="holdout")
h_data = {w: load_eurgbp_data(w) for w in h_weeks[:20]}
h_data = {k: v for k, v in h_data.items() if v is not None}

print("Testing first 20 holdout weeks...")
for s_min in [0.3, 0.5, 1.0]:
    for s_max in [3.0, 5.0]:
        for wick in [0.20, 0.30]:
            tr = []
            for w, wd in h_data.items():
                t = eval_asian_sweep_fade(wd, s_min, s_max, wick, 7, 11, 8.0, 12.0)
                tr.extend(t)
            if len(tr) > 0:
                pips = np.array([x['pips'] for x in tr])
                print(f"Sweep {s_min}-{s_max}p, wk {wick}: {len(tr)} trades, net {np.sum(pips):.1f}p")

