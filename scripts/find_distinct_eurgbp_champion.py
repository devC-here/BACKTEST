import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from tick_engine.core.dataloader import get_complete_split_weeks
from scripts.refine_eurgbp_h1_pullback import load_h1_data, eval_setup

SYMBOL = "EURGBP"

# Load disc174 trades
disc174_csv = ROOT_DIR / "results" / "certified_strategy_manifests" / "EURGBP__disc174_eurgbp_h1_pullback_elite_trades.csv"
df_174 = pd.read_csv(disc174_csv)
ts_174 = set(df_174['entry_time'].values)
df_174['day'] = pd.to_datetime(df_174['entry_time'], unit='ms', utc=True).dt.date
pips_174_daily = df_174.groupby('day')['pips'].sum()

# Load existing fleet trades for EURGBP
disc107_csv = list((ROOT_DIR / "results" / "certified_strategy_manifests").glob("*disc107*trades.csv"))[0]
df_107 = pd.read_csv(disc107_csv)
ts_107 = set(df_107['entry_time'].values)

disc172_csv = ROOT_DIR / "results" / "certified_strategy_manifests" / "EURGBP__disc172_eurgbp_inside_day_breakout_elite_trades.csv"
df_172 = pd.read_csv(disc172_csv)
ts_172 = set(df_172['entry_time'].values)

data = json.load(open(ROOT_DIR / "results" / "new_champions_eurgbp_h1_pullback.json"))
print(f"Scanning {len(data)} dual champions for anti-redundancy vs disc174, disc107, disc172...", flush=True)

print("Loading Holdout weeks...", flush=True)
h_weeks = get_complete_split_weeks(ROOT_DIR / "data" / SYMBOL, split="holdout")
h_data = {w: load_h1_data(w) for w in h_weeks}
h_data = {k: v for k, v in h_data.items() if v is not None}

results = []

for idx, c in enumerate(data):
    s = c['setup']
    all_tr = []
    for w, wd in h_data.items():
        tr = eval_setup(wd, s['start_hr'], s['end_hr'], s['touch'], s['wick'], s['use_slope'], s['sl'], s['tp'])
        if len(tr) > 0:
            df = pd.DataFrame(tr)
            all_tr.append(df)
    if not all_tr:
        continue
    cand_df = pd.concat(all_tr, ignore_index=True)
    cand_ts = set(cand_df['entry_time'].values)

    # Overlap with disc174
    common_174 = cand_ts.intersection(ts_174)
    overlap_174 = len(common_174) / max(len(cand_ts), 1)

    # Overlap with disc107 and disc172
    overlap_107 = len(cand_ts.intersection(ts_107)) / max(len(cand_ts), 1)
    overlap_172 = len(cand_ts.intersection(ts_172)) / max(len(cand_ts), 1)

    # Correlation with disc174
    cand_df['day'] = pd.to_datetime(cand_df['entry_time'], unit='ms', utc=True).dt.date
    cand_daily = cand_df.groupby('day')['pips'].sum()
    all_days = sorted(list(set(pips_174_daily.index).union(set(cand_daily.index))))
    c_series = [pips_174_daily.get(d, 0.0) for d in all_days]
    e_series = [cand_daily.get(d, 0.0) for d in all_days]
    corr_174 = np.corrcoef(c_series, e_series)[0, 1] if len(all_days) > 5 else 0.0
    if np.isnan(corr_174): corr_174 = 0.0

    results.append({
        'setup': s,
        'holdout': c['holdout'],
        'discovery': c['discovery'],
        'overlap_174_pct': round(overlap_174 * 100, 1),
        'overlap_107_pct': round(overlap_107 * 100, 1),
        'overlap_172_pct': round(overlap_172 * 100, 1),
        'corr_174': round(float(corr_174), 3),
        'n_trades': len(cand_df)
    })

# Sort by overlap ascending
sorted_by_overlap = sorted(results, key=lambda x: x['overlap_174_pct'])
print("\n--- TOP 10 LOWEST OVERLAP CANDIDATES VS DISC174 ---", flush=True)
for r in sorted_by_overlap[:10]:
    s = r['setup']
    h = r['holdout']
    d = r['discovery']
    print(f"Overlap vs 174: {r['overlap_174_pct']}% | Corr: {r['corr_174']} | Overlap vs 107: {r['overlap_107_pct']}% | Overlap vs 172: {r['overlap_172_pct']}% | {s['desc']}", flush=True)
    print(f"   Holdout: {h['trades']}tr, +{h['net_pips']:.1f}p, PF={h['pf']:.2f} | Disc: {d['trades']}tr, +{d['net_pips']:.1f}p, PF={d['pf']:.2f}\n", flush=True)

out_f = ROOT_DIR / "results" / "eurgbp_champions_overlap_analysis.json"
with open(out_f, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved analysis to {out_f}", flush=True)
