#!/usr/bin/env python3
"""
Inspect Discovery Performance for the 18 EURGBP Holdout Champions
"""
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from tick_engine.core.dataloader import get_complete_split_weeks
from scripts.hunt_eurgbp_trend_and_orb_alpha import load_week_data, eval_h1_pullback

disc_weeks = get_complete_split_weeks(Path('data/EURGBP'), split='discovery')
print(f"Loading {len(disc_weeks)} discovery weeks...", flush=True)

disc_data = {}
for w in disc_weeks:
    wd = load_week_data(w)
    if wd is not None:
        disc_data[w] = wd
print(f"Loaded {len(disc_data)} discovery weeks into RAM.", flush=True)

# Grid of the 18 holdout champions:
# Candidates 1-18 from the run log
candidates = [
    {"desc": "Candidate 1: Touch=1.5, Wk=0.35, SL=8.0, TP=16.0", "touch": 1.5, "wick": 0.35, "sl": 8.0, "tp": 16.0},
    {"desc": "Candidate 2: Touch=1.5, Wk=0.35, SL=8.0, TP=18.0", "touch": 1.5, "wick": 0.35, "sl": 8.0, "tp": 18.0},
    {"desc": "Candidate 3: Touch=1.5, Wk=0.35, SL=9.0, TP=16.0", "touch": 1.5, "wick": 0.35, "sl": 9.0, "tp": 16.0},
    {"desc": "Candidate 4: Touch=1.5, Wk=0.35, SL=10.0, TP=16.0", "touch": 1.5, "wick": 0.35, "sl": 10.0, "tp": 16.0},
    {"desc": "Candidate 5: Touch=2.5, Wk=0.35, SL=8.0, TP=16.0", "touch": 2.5, "wick": 0.35, "sl": 8.0, "tp": 16.0},
    {"desc": "Candidate 6: Touch=2.5, Wk=0.35, SL=8.0, TP=20.0", "touch": 2.5, "wick": 0.35, "sl": 8.0, "tp": 20.0},
    {"desc": "Candidate 7: Touch=2.5, Wk=0.40, SL=8.0, TP=16.0", "touch": 2.5, "wick": 0.40, "sl": 8.0, "tp": 16.0},
    {"desc": "Candidate 8: Touch=2.5, Wk=0.40, SL=8.0, TP=18.0", "touch": 2.5, "wick": 0.40, "sl": 8.0, "tp": 18.0},
    {"desc": "Candidate 9: Touch=2.5, Wk=0.40, SL=8.0, TP=20.0", "touch": 2.5, "wick": 0.40, "sl": 8.0, "tp": 20.0},
    {"desc": "Candidate 10: Touch=2.5, Wk=0.40, SL=8.0, TP=22.0", "touch": 2.5, "wick": 0.40, "sl": 8.0, "tp": 22.0},
    {"desc": "Candidate 11: Touch=2.5, Wk=0.40, SL=9.0, TP=16.0", "touch": 2.5, "wick": 0.40, "sl": 9.0, "tp": 16.0},
    {"desc": "Candidate 12: Touch=2.5, Wk=0.40, SL=9.0, TP=20.0", "touch": 2.5, "wick": 0.40, "sl": 9.0, "tp": 20.0},
    {"desc": "Candidate 13: Touch=3.5, Wk=0.40, SL=8.0, TP=16.0", "touch": 3.5, "wick": 0.40, "sl": 8.0, "tp": 16.0},
    {"desc": "Candidate 14: Touch=3.5, Wk=0.40, SL=8.0, TP=18.0", "touch": 3.5, "wick": 0.40, "sl": 8.0, "tp": 18.0},
    {"desc": "Candidate 15: Touch=3.5, Wk=0.40, SL=9.0, TP=16.0", "touch": 3.5, "wick": 0.40, "sl": 9.0, "tp": 16.0},
    {"desc": "Candidate 16: Touch=3.5, Wk=0.40, SL=9.0, TP=18.0", "touch": 3.5, "wick": 0.40, "sl": 9.0, "tp": 18.0},
    {"desc": "Candidate 17: Touch=3.5, Wk=0.40, SL=9.0, TP=20.0", "touch": 3.5, "wick": 0.40, "sl": 9.0, "tp": 20.0},
    {"desc": "Candidate 18: Touch=3.5, Wk=0.40, SL=9.0, TP=22.0", "touch": 3.5, "wick": 0.40, "sl": 9.0, "tp": 22.0},
]

print("\n--- RESULTS ACROSS FULL 196 DISCOVERY WEEKS ---", flush=True)
for c in candidates:
    all_trades = []
    for w, wd in disc_data.items():
        tr = eval_h1_pullback(wd, c["touch"], c["wick"], c["sl"], c["tp"], 0.8)
        all_trades.extend(tr)
    
    pips = np.array([t['pips'] for t in all_trades], dtype=np.float64)
    net_pips = float(np.sum(pips))
    wins = pips[pips > 0]
    losses = pips[pips < 0]
    sum_w = float(np.sum(wins)) if len(wins) > 0 else 0.0
    sum_l = float(np.abs(np.sum(losses))) if len(losses) > 0 else 1e-6
    pf = sum_w / sum_l
    exp = float(np.mean(pips)) if len(pips) > 0 else 0.0
    
    print(f"{c['desc']}: {len(all_trades)} trades, Net={net_pips:+.1f}p, PF={pf:.2f}, Exp={exp:+.2f}p", flush=True)

