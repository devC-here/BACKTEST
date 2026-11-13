#!/usr/bin/env python3
"""
================================================================================
  CLUSTERING & REDUNDANCY PRUNING ANALYSIS
================================================================================
Evaluates redundant clusters within each currency pair using graph connected
components / hierarchical clustering:
- Overlap threshold >= 75% OR Daily Return Correlation >= 0.75
- In each redundant cluster, selects the single best performing champion
  (Highest Score = (Net Pips / Max DD) * Profit Factor * sqrt(Trades))
- Prunes the inferior duplicates
================================================================================
"""

import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
FLEET_MANIFEST = ROOT_DIR / "certified_production_fleet" / "CERTIFIED_FLEET_MANIFEST.json"
MANIFEST_DIR = ROOT_DIR / "results" / "certified_strategy_manifests"

with open(FLEET_MANIFEST, "r", encoding="utf-8") as f:
    fleet_data = json.load(f)

strategies = fleet_data["strategies"]

strat_trades = {}
strat_manifests = {}
strat_scores = {}

for s in strategies:
    sym = s["symbol"]
    name = s["name"]
    key = f"{sym}__{name}"

    m_candidates = [
        MANIFEST_DIR / f"{sym}__{name}_run_manifest.json",
        MANIFEST_DIR / f"{sym.lower()}_{name}_run_manifest.json",
        MANIFEST_DIR / f"{name}_run_manifest.json"
    ]
    m_file = next((f for f in m_candidates if f.exists()), None)
    if not m_file:
        continue
    with open(m_file, encoding="utf-8") as f:
        m = json.load(f)
    strat_manifests[key] = m

    stats = m["statistical_metrics"]
    pips = stats.get("net_pips", 0.0)
    pf = stats.get("profit_factor", 0.0)
    dd = abs(stats.get("max_drawdown_pips", 1.0))
    n_trades = stats.get("total_trades", 1)
    exp = stats.get("expectancy_pips", 0.0)
    # Institutional quality score
    score = (pips / max(dd, 5.0)) * pf * np.sqrt(n_trades)
    strat_scores[key] = {
        "pips": pips,
        "pf": pf,
        "dd": dd,
        "trades": n_trades,
        "exp": exp,
        "score": score
    }

    csv_candidates = [
        MANIFEST_DIR / f"{sym}__{name}_trades.csv",
        MANIFEST_DIR / f"{sym.lower()}_{name}_trades.csv",
        MANIFEST_DIR / f"{name}_trades.csv"
    ]
    csv_file = next((f for f in csv_candidates if f.exists()), None)
    if csv_file:
        df = pd.read_csv(csv_file)
        strat_trades[key] = df
    else:
        tl = m.get("trade_log", [])
        strat_trades[key] = pd.DataFrame(tl) if tl else pd.DataFrame()

# Group by symbol
by_sym = defaultdict(list)
for key in strat_manifests:
    sym = key.split("__")[0]
    by_sym[sym].append(key)

all_redundant_to_prune = []
all_champions_to_keep = []

print("=" * 105)
print(f"{'SYMBOL':<8} | {'CURRENT':<8} | {'REDUNDANT IDENTIFIED':<45} | {'NEW COUNT'}")
print("=" * 105)

for sym, keys in sorted(by_sym.items()):
    n = len(keys)
    # Build adjacency matrix of redundancy
    is_redundant_with = defaultdict(set)

    for i in range(n):
        for j in range(i + 1, n):
            k1 = keys[i]
            k2 = keys[j]
            df1 = strat_trades.get(k1, pd.DataFrame())
            df2 = strat_trades.get(k2, pd.DataFrame())
            if len(df1) == 0 or len(df2) == 0:
                continue

            times1 = df1["entry_time"].values
            times2 = df2["entry_time"].values
            dir1 = df1["direction"].values if "direction" in df1 else np.ones(len(df1))
            dir2 = df2["direction"].values if "direction" in df2 else np.ones(len(df2))

            matches = 0
            tolerance_ms = 30 * 60 * 1000
            for idx1, t1 in enumerate(times1):
                d1 = dir1[idx1]
                diffs = np.abs(times2 - t1)
                if np.any((diffs <= tolerance_ms) & (dir2 == d1)):
                    matches += 1

            overlap_pct_1 = (matches / len(times1)) * 100.0 if len(times1) > 0 else 0
            overlap_pct_2 = (matches / len(times2)) * 100.0 if len(times2) > 0 else 0
            max_overlap = max(overlap_pct_1, overlap_pct_2)

            # Daily PnL correlation
            df1_c = df1.copy()
            df2_c = df2.copy()
            df1_c["day"] = pd.to_datetime(df1_c["entry_time"], unit="ms", utc=True).dt.date
            df2_c["day"] = pd.to_datetime(df2_c["entry_time"], unit="ms", utc=True).dt.date
            daily1 = df1_c.groupby("day")["pips"].sum()
            daily2 = df2_c.groupby("day")["pips"].sum()
            all_days = sorted(list(set(daily1.index).union(set(daily2.index))))
            d1_series = [daily1.get(d, 0.0) for d in all_days]
            d2_series = [daily2.get(d, 0.0) for d in all_days]
            corr = np.corrcoef(d1_series, d2_series)[0, 1] if len(all_days) > 5 else 0.0
            if np.isnan(corr):
                corr = 0.0

            # Redundancy criteria: Overlap >= 75% OR (Overlap >= 60% AND Corr >= 0.75) OR Corr >= 0.85
            if max_overlap >= 75.0 or (max_overlap >= 60.0 and corr >= 0.75) or corr >= 0.85:
                is_redundant_with[k1].add(k2)
                is_redundant_with[k2].add(k1)

    # Find clusters of redundancy using connected components
    visited = set()
    clusters = []
    for k in keys:
        if k not in visited:
            cluster = []
            queue = [k]
            visited.add(k)
            while queue:
                curr = queue.pop(0)
                cluster.append(curr)
                for neighbor in is_redundant_with[curr]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            clusters.append(cluster)

    # For each cluster, pick the champion with highest score, prune others
    keep_for_sym = []
    prune_for_sym = []

    for cluster in clusters:
        if len(cluster) == 1:
            keep_for_sym.append(cluster[0])
        else:
            # Sort cluster by score descending
            cluster.sort(key=lambda x: strat_scores[x]["score"], reverse=True)
            champ = cluster[0]
            keep_for_sym.append(champ)
            for duplicate in cluster[1:]:
                prune_for_sym.append(duplicate)

    all_redundant_to_prune.extend(prune_for_sym)
    all_champions_to_keep.extend(keep_for_sym)

    pruned_names = ", ".join([p.split("__")[1] for p in prune_for_sym]) if prune_for_sym else "None (All Unique)"
    if len(pruned_names) > 42:
        pruned_names = pruned_names[:39] + "..."
    print(f"{sym:<8} | {n:<8} | {pruned_names:<45} | {len(keep_for_sym)}")

print("=" * 105)
print(f"Total current strategies:     {len(strategies)}")
print(f"Total redundant to prune:     {len(all_redundant_to_prune)}")
print(f"Remaining non-redundant fleet: {len(all_champions_to_keep)}")
print("=" * 105)

