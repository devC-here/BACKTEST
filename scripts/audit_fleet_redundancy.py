#!/usr/bin/env python3
"""
================================================================================
  FLEET REDUNDANCY & CLUSTERING AUDIT
================================================================================
Analyzes all 85 certified strategies across the 10 currency pairs to detect:
1. Trade Execution Overlap (Jaccard similarity of trade entry timestamps & intervals)
2. Daily / Weekly PnL Correlation among strategies within the same symbol
3. Parameter & Archetype Duplication
4. Performance Dominance (if Strategy A and B overlap > 60%, but A has higher PF,
   higher Expectancy, lower DD, then B is strictly redundant)
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
print(f"Loaded {len(strategies)} strategies from fleet manifest.")

# Load trades and manifests for all strategies
strat_trades = {}
strat_manifests = {}

for s in strategies:
    sym = s["symbol"]
    name = s["name"]
    key = f"{sym}__{name}"

    # Manifest
    m_candidates = [
        MANIFEST_DIR / f"{sym}__{name}_run_manifest.json",
        MANIFEST_DIR / f"{sym.lower()}_{name}_run_manifest.json",
        MANIFEST_DIR / f"{name}_run_manifest.json"
    ]
    m_file = next((f for f in m_candidates if f.exists()), None)
    if not m_file:
        print(f"Warning: Manifest missing for {key}")
        continue
    with open(m_file, encoding="utf-8") as f:
        m = json.load(f)
    strat_manifests[key] = m

    # Trades CSV
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
        # Fall back to trade_log in manifest
        tl = m.get("trade_log", [])
        if tl:
            df = pd.DataFrame(tl)
            strat_trades[key] = df
        else:
            strat_trades[key] = pd.DataFrame()

print(f"Loaded data for {len(strat_manifests)} strategies.")

# Group by symbol
by_sym = defaultdict(list)
for key in strat_manifests:
    sym = key.split("__")[0]
    by_sym[sym].append(key)

redundancy_pairs = []

print("\n" + "=" * 100)
print(f"{'SYMBOL':<8} | {'STRATEGY A':<22} | {'STRATEGY B':<22} | {'OVERLAP %':<10} | {'CORR':<7} | {'DECISION'}")
print("=" * 100)

for sym, keys in sorted(by_sym.items()):
    n_strats = len(keys)
    if n_strats <= 1:
        continue

    for i in range(n_strats):
        for j in range(i + 1, n_strats):
            k1 = keys[i]
            k2 = keys[j]
            df1 = strat_trades.get(k1, pd.DataFrame())
            df2 = strat_trades.get(k2, pd.DataFrame())

            if len(df1) == 0 or len(df2) == 0:
                continue

            # Trade time alignment: Match trades entering within 30 minutes of each other and same direction
            times1 = df1["entry_time"].values
            times2 = df2["entry_time"].values
            dir1 = df1["direction"].values if "direction" in df1 else np.ones(len(df1))
            dir2 = df2["direction"].values if "direction" in df2 else np.ones(len(df2))

            # Count overlapping trades (within 30 mins = 1,800,000 ms, same direction)
            matches = 0
            tolerance_ms = 30 * 60 * 1000

            for idx1, t1 in enumerate(times1):
                d1 = dir1[idx1]
                # check if any in times2 within tolerance and same dir
                diffs = np.abs(times2 - t1)
                matching_indices = np.where((diffs <= tolerance_ms) & (dir2 == d1))[0]
                if len(matching_indices) > 0:
                    matches += 1

            overlap_pct_1 = (matches / len(times1)) * 100.0 if len(times1) > 0 else 0
            overlap_pct_2 = (matches / len(times2)) * 100.0 if len(times2) > 0 else 0
            max_overlap = max(overlap_pct_1, overlap_pct_2)
            min_overlap = min(overlap_pct_1, overlap_pct_2)
            avg_overlap = (overlap_pct_1 + overlap_pct_2) / 2.0

            # Daily / Weekly PnL correlation
            # Map trades to date (UTC day)
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

            # Flag as redundant if high overlap (> 50%) or very high correlation (> 0.70)
            if max_overlap >= 40.0 or corr >= 0.65:
                # Compare performance to see which one dominates
                m1 = strat_manifests[k1]
                m2 = strat_manifests[k2]
                s1_pips = m1["statistical_metrics"]["net_pips"]
                s2_pips = m2["statistical_metrics"]["net_pips"]
                s1_pf = m1["statistical_metrics"]["profit_factor"]
                s2_pf = m2["statistical_metrics"]["profit_factor"]
                s1_dd = m1["statistical_metrics"]["max_drawdown_pips"]
                s2_dd = m2["statistical_metrics"]["max_drawdown_pips"]
                s1_exp = m1["statistical_metrics"]["expectancy_pips"]
                s2_exp = m2["statistical_metrics"]["expectancy_pips"]

                # Scoring dominance
                # Higher pips, higher PF, higher expectancy, lower DD
                score1 = (s1_pips / max(abs(s1_dd), 1)) * s1_pf
                score2 = (s2_pips / max(abs(s2_dd), 1)) * s2_pf

                superior = k1 if score1 >= score2 else k2
                redundant = k2 if score1 >= score2 else k1

                redundancy_pairs.append({
                    "symbol": sym,
                    "strat_a": k1,
                    "strat_b": k2,
                    "overlap_a": round(overlap_pct_1, 1),
                    "overlap_b": round(overlap_pct_2, 1),
                    "max_overlap": round(max_overlap, 1),
                    "corr": round(corr, 2),
                    "superior": superior,
                    "redundant": redundant,
                    "score_a": round(score1, 1),
                    "score_b": round(score2, 1)
                })

                decision = f"Keep {superior.split('__')[1]} (drop {redundant.split('__')[1]})"
                print(f"{sym:<8} | {k1.split('__')[1]:<22} | {k2.split('__')[1]:<22} | {max_overlap:>8.1f}% | {corr:>6.2f} | {decision}")

print("=" * 100)
print(f"Total potential redundancy pairs found: {len(redundancy_pairs)}")

