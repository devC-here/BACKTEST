#!/usr/bin/env python3
"""
Scan all scorecards and archive folder to find promising previous strategies
with positive pips and latent alpha that can be optimized to reach economic viability (PF >= 1.25).
"""

import sys
from pathlib import Path
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT_DIR / "results"
ARCHIVE_DIR = ROOT_DIR / "strategies" / "archive_non_viable"

promising_candidates = []

# 1. Scan scorecards
for p in sorted(RESULTS_DIR.glob("*scorecard*.csv")):
    try:
        df = pd.read_csv(p)
        cols = {c.lower(): c for c in df.columns}
        
        # Check standard columns
        pips_col = cols.get("net_pips") or cols.get("pips") or cols.get("net")
        pf_col = cols.get("profit_factor") or cols.get("pf")
        trades_col = cols.get("trades") or cols.get("total_trades")
        asset_col = cols.get("asset") or cols.get("pair")
        model_col = cols.get("model") or cols.get("strategy") or cols.get("name")
        
        if pips_col and pf_col and model_col:
            for _, row in df.iterrows():
                try:
                    pips = float(row[pips_col])
                    pf = float(row[pf_col])
                    trades = int(row[trades_col]) if trades_col and pd.notna(row[trades_col]) else 0
                    asset = str(row[asset_col]) if asset_col and pd.notna(row[asset_col]) else "UNKNOWN"
                    model = str(row[model_col])
                    
                    if pips >= 100.0 and 1.05 <= pf <= 1.25 and trades >= 30:
                        promising_candidates.append({
                            "source": p.name,
                            "asset": asset,
                            "model": model,
                            "trades": trades,
                            "net_pips": pips,
                            "profit_factor": pf
                        })
                except Exception:
                    continue
    except Exception:
        continue

df_promising = pd.DataFrame(promising_candidates)
if not df_promising.empty:
    df_promising = df_promising.sort_values(by="net_pips", ascending=False).drop_duplicates(subset=["asset", "model"])
    print(f"Found {len(df_promising)} promising candidate architectures from scorecards:\n")
    print(df_promising.to_string(index=False))
else:
    print("No matching candidates found in scorecards.")

# 2. List strategies in archive_non_viable
print("\n" + "=" * 80)
print(f"ARCHIVED STRATEGIES IN: {ARCHIVE_DIR}")
print("=" * 80)
if ARCHIVE_DIR.exists():
    archived_dirs = [d.name for d in ARCHIVE_DIR.iterdir() if d.is_dir()]
    print(f"Total archived folders: {len(archived_dirs)}")
    for d in sorted(archived_dirs)[:30]:
        print(f"  - {d}")

