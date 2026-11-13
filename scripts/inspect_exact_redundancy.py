#!/usr/bin/env python3
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
    strat_meta[key] = {
        "pips": m["statistical_metrics"]["net_pips"],
        "pf": m["statistical_metrics"]["profit_factor"],
        "trades": m["statistical_metrics"]["total_trades"],
        "dd": m["statistical_metrics"]["max_drawdown_pips"],
        "exp": m["statistical_metrics"]["expectancy_pips"],
        "cat": s.get("category", "")
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

by_sym = defaultdict(list)
for k in strat_meta:
    sym = k.split("__")[0]
    by_sym[sym].append(k)

print(f"{'SYM':<7} | {'STRAT A':<30} | {'STRAT B':<30} | {'OVERLAP %':<9} | {'CORR':<6} | {'PIPS A':<8} | {'PIPS B':<8} | {'PF A':<5} | {'PF B':<5}")
print("-" * 125)

high_redundancy = []
for sym, keys in sorted(by_sym.items()):
    n = len(keys)
    for i in range(n):
        for j in range(i + 1, n):
            k1 = keys[i]
            k2 = keys[j]
            df1 = strat_trades[k1]
            df2 = strat_trades[k2]
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

            overlap_pct = (matches / min(len(times1), len(times2))) * 100.0

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

            m1 = strat_meta[k1]
            m2 = strat_meta[k2]

            # High redundancy threshold: Overlap >= 80% and Corr >= 0.70
            if overlap_pct >= 80.0 and corr >= 0.65:
                high_redundancy.append({
                    "sym": sym,
                    "k1": k1,
                    "k2": k2,
                    "overlap": overlap_pct,
                    "corr": corr,
                    "m1": m1,
                    "m2": m2
                })
                print(f"{sym:<7} | {k1.split('__')[1]:<30} | {k2.split('__')[1]:<30} | {overlap_pct:>7.1f}% | {corr:>5.2f} | {m1['pips']:>7.1f}p | {m2['pips']:>7.1f}p | {m1['pf']:>4.2f} | {m2['pf']:>4.2f}")

print("-" * 125)
print(f"Total highly redundant pairs (>=80% overlap & >=0.65 corr): {len(high_redundancy)}")

