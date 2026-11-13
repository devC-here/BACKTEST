#!/usr/bin/env python3
"""
================================================================================
  SYSTEMATIC REDUNDANCY PRUNING ENGINE
================================================================================
Tests different redundancy pruning thresholds (e.g. overlap >= 90%, >= 85%, >= 80%)
to find the optimal balance between high diversification and zero duplicate clutter,
targeting the user's desired ~72 strategy capacity.
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
strat_meta = {}

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
    stats = m["statistical_metrics"]
    strat_meta[key] = {
        "symbol": sym,
        "name": name,
        "category": s.get("category", ""),
        "pips": stats["net_pips"],
        "pf": stats["profit_factor"],
        "trades": stats["total_trades"],
        "dd": abs(stats["max_drawdown_pips"]),
        "exp": stats["expectancy_pips"],
        "score": (stats["net_pips"] / max(abs(stats["max_drawdown_pips"]), 5.0)) * stats["profit_factor"] * np.sqrt(stats["total_trades"])
    }

    csv_candidates = [
        MANIFEST_DIR / f"{sym}__{name}_trades.csv",
        MANIFEST_DIR / f"{sym.lower()}_{name}_trades.csv",
        MANIFEST_DIR / f"{name}_trades.csv"
    ]
    csv_file = next((f for f in csv_candidates if f.exists()), None)
    if csv_file:
        strat_trades[key] = pd.read_csv(csv_file)
    else:
        tl = m.get("trade_log", [])
        strat_trades[key] = pd.DataFrame(tl) if tl else pd.DataFrame()

# Precompute pairwise overlap and correlation within each symbol
by_sym = defaultdict(list)
for k in strat_meta:
    sym = k.split("__")[0]
    by_sym[sym].append(k)

pairwise_matrix = {}
for sym, keys in by_sym.items():
    n = len(keys)
    for i in range(n):
        for j in range(i + 1, n):
            k1, k2 = keys[i], keys[j]
            df1, df2 = strat_trades[k1], strat_trades[k2]
            if len(df1) == 0 or len(df2) == 0:
                continue
            times1, times2 = df1["entry_time"].values, df2["entry_time"].values
            dir1 = df1["direction"].values if "direction" in df1 else np.ones(len(df1))
            dir2 = df2["direction"].values if "direction" in df2 else np.ones(len(df2))

            tolerance_ms = 30 * 60 * 1000
            matches = sum(1 for idx, t1 in enumerate(times1) if np.any((np.abs(times2 - t1) <= tolerance_ms) & (dir2 == dir1[idx])))
            overlap_pct = (matches / min(len(times1), len(times2))) * 100.0

            df1_c, df2_c = df1.copy(), df2.copy()
            df1_c["day"] = pd.to_datetime(df1_c["entry_time"], unit="ms", utc=True).dt.date
            df2_c["day"] = pd.to_datetime(df2_c["entry_time"], unit="ms", utc=True).dt.date
            daily1, daily2 = df1_c.groupby("day")["pips"].sum(), df2_c.groupby("day")["pips"].sum()
            all_days = sorted(list(set(daily1.index).union(set(daily2.index))))
            d1_s = [daily1.get(d, 0.0) for d in all_days]
            d2_s = [daily2.get(d, 0.0) for d in all_days]
            corr = np.corrcoef(d1_s, d2_s)[0, 1] if len(all_days) > 5 else 0.0
            if np.isnan(corr): corr = 0.0

            pairwise_matrix[(k1, k2)] = (overlap_pct, corr)
            pairwise_matrix[(k2, k1)] = (overlap_pct, corr)


def test_pruning_policy(min_overlap: float, min_corr: float):
    """Greedily prunes redundant strategies within each symbol."""
    survivors = set(strat_meta.keys())
    pruned = []

    # Sort candidates by score ascending (lowest quality evaluated first for redundancy pruning)
    for sym, keys in sorted(by_sym.items()):
        # Sort keys from lowest score to highest score
        sorted_keys = sorted(keys, key=lambda k: strat_meta[k]["score"])
        for k_low in sorted_keys:
            if k_low not in survivors:
                continue
            # Check if there is a higher-scoring survivor that has high overlap & correlation with k_low
            higher_keys = [k for k in keys if k in survivors and k != k_low and strat_meta[k]["score"] > strat_meta[k_low]["score"]]
            for k_high in higher_keys:
                overlap, corr = pairwise_matrix.get((k_low, k_high), (0.0, 0.0))
                if overlap >= min_overlap and corr >= min_corr:
                    survivors.remove(k_low)
                    pruned.append((k_low, k_high, overlap, corr))
                    break

    return survivors, pruned

print("=" * 90)
print(f"{'OVERLAP THRESH':<15} | {'CORR THRESH':<12} | {'PRUNED':<8} | {'REMAINING FLEET':<18} | {'TARGET FIT'}")
print("=" * 90)

for o in [95.0, 90.0, 85.0, 80.0, 75.0]:
    for c in [0.85, 0.80, 0.75, 0.70]:
        surv, pr = test_pruning_policy(o, c)
        fit = "<-- EXACT TARGET (72)" if len(surv) == 72 else f"({len(surv)})"
        print(f"{o:>13.1f}% | {c:>10.2f} | {len(pr):>6} | {len(surv):>14} strats | {fit}")

print("=" * 90)

