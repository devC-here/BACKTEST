import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from tick_engine.core.dataloader import get_week_dataset, get_complete_split_weeks
from tick_engine.core.config import PIP_SIZES, get_friction_pips, WARMUP_TICKS
from scripts.refine_eurgbp_h1_pullback import load_h1_data, eval_setup

SYMBOL = "EURGBP"

print("Loading Holdout weeks...", flush=True)
h_weeks = get_complete_split_weeks(ROOT_DIR / "data" / SYMBOL, split="holdout")
h_data = {w: load_h1_data(w) for w in h_weeks}
h_data = {k: v for k, v in h_data.items() if v is not None}

print("Running candidate evaluations...", flush=True)

def get_trades_df(cand, data_dict):
    all_tr = []
    for w, wd in data_dict.items():
        tr = eval_setup(wd, cand['start_hr'], cand['end_hr'], cand['touch'], cand['wick'], cand['use_slope'], cand['sl'], cand['tp'])
        if len(tr) > 0:
            df = pd.DataFrame(tr)
            df['week'] = w
            all_tr.append(df)
    return pd.concat(all_tr, ignore_index=True) if all_tr else pd.DataFrame()

cand_A = {'start_hr': 8, 'end_hr': 15, 'touch': 0.8, 'wick': 0.35, 'use_slope': True, 'sl': 7.0, 'tp': 14.0, 'name': 'Cand_A_0.8p_7_14'}
cand_B = {'start_hr': 8, 'end_hr': 14, 'touch': 1.5, 'wick': 0.35, 'use_slope': True, 'sl': 9.0, 'tp': 16.0, 'name': 'Cand_B_1.5p_9_16'}

df_A = get_trades_df(cand_A, h_data)
df_B = get_trades_df(cand_B, h_data)

print(f"Cand A holdout trades: {len(df_A)}, Net Pips: {df_A['pips'].sum():.1f}", flush=True)
print(f"Cand B holdout trades: {len(df_B)}, Net Pips: {df_B['pips'].sum():.1f}", flush=True)
print(f"Cand A columns: {df_A.columns.tolist()}", flush=True)

# Check timestamps of entry
ts_col = 'entry_time' if 'entry_time' in df_A.columns else 'entry_timestamp'
ts_A = set(df_A[ts_col].values)
ts_B = set(df_B[ts_col].values)

common = ts_A.intersection(ts_B)
print(f"Exact timestamp overlaps: {len(common)}", flush=True)
print(f"Cand A unique: {len(ts_A - common)}", flush=True)
print(f"Cand B unique: {len(ts_B - common)}", flush=True)
print(f"Overlap ratio: {len(common) / min(len(ts_A), len(ts_B)):.1%}", flush=True)

# Check correlation of weekly returns
all_weeks = sorted(list(h_data.keys()))
pips_A_weekly = df_A.groupby('week')['pips'].sum().reindex(all_weeks, fill_value=0.0)
pips_B_weekly = df_B.groupby('week')['pips'].sum().reindex(all_weeks, fill_value=0.0)
corr = np.corrcoef(pips_A_weekly, pips_B_weekly)[0, 1]
print(f"Weekly Return Correlation between A and B: {corr:.3f}", flush=True)
