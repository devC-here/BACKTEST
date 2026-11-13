import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from tick_engine.core.dataloader import get_complete_split_weeks
from scripts.refine_eurgbp_pivot_pullback import load_eurgbp_pivot_slope_data, eval_pivot_trend_slope

SYMBOL = "EURGBP"

# Load existing fleet trades for EURGBP
disc107_csv = list((ROOT_DIR / "results" / "certified_strategy_manifests").glob("*disc107*trades.csv"))[0]
df_107 = pd.read_csv(disc107_csv)

disc172_csv = ROOT_DIR / "results" / "certified_strategy_manifests" / "EURGBP__disc172_eurgbp_inside_day_breakout_elite_trades.csv"
df_172 = pd.read_csv(disc172_csv)

disc174_csv = ROOT_DIR / "results" / "certified_strategy_manifests" / "EURGBP__disc174_eurgbp_h1_pullback_elite_trades.csv"
df_174 = pd.read_csv(disc174_csv)

print("Loading Holdout weeks...", flush=True)
h_weeks = get_complete_split_weeks(ROOT_DIR / "data" / SYMBOL, split="holdout")
h_data = {w: load_eurgbp_pivot_slope_data(w) for w in h_weeks}
h_data = {k: v for k, v in h_data.items() if v is not None}

# Candidate: Touch=[+2.0, -4.0]p, Wk=0.25, H=8-14, SL=7.0p, TP=14.0p
all_dfs = []
for w, wd in h_data.items():
    tr = eval_pivot_trend_slope(wd, 2.0, 4.0, 0.25, True, 8, 14, 7.0, 14.0)
    if len(tr) > 0:
        all_dfs.append(pd.DataFrame(tr))

cand_df = pd.concat(all_dfs, ignore_index=True)
cand_ts = set(cand_df['entry_time'].values)

print(f"\nCandidate Holdout Trades: {len(cand_df)}, Net Pips: {cand_df['pips'].sum():.1f}")

# Overlaps
cand_df['day'] = pd.to_datetime(cand_df['entry_time'], unit='ms', utc=True).dt.date
c_daily = cand_df.groupby('day')['pips'].sum()

for name, df in [('disc107', df_107), ('disc172', df_172), ('disc174', df_174)]:
    ex_ts = set(df['entry_time'].values)
    common = cand_ts.intersection(ex_ts)
    overlap_pct = (len(common) / len(cand_ts)) * 100.0
    print(f"Overlap vs {name}: {overlap_pct:.1f}% ({len(common)}/{len(cand_ts)} trades)")

    # Correlation
    df['day'] = pd.to_datetime(df['entry_time'], unit='ms', utc=True).dt.date
    e_daily = df.groupby('day')['pips'].sum()
    all_days = sorted(list(set(c_daily.index).union(set(e_daily.index))))
    cs = [c_daily.get(d, 0.0) for d in all_days]
    es = [e_daily.get(d, 0.0) for d in all_days]
    corr = np.corrcoef(cs, es)[0, 1] if len(all_days) > 5 else 0.0
    print(f"Daily Correlation vs {name}: {corr:.3f}")
