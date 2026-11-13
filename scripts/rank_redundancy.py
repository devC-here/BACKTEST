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
    stats = m["statistical_metrics"]
    pips = stats["net_pips"]
    pf = stats["profit_factor"]
    dd = abs(stats["max_drawdown_pips"])
    n_trades = stats["total_trades"]
    exp = stats["expectancy_pips"]
    score = (pips / max(dd, 5.0)) * pf * np.sqrt(n_trades)
    strat_meta[key] = {
        "symbol": sym,
        "name": name,
        "category": s.get("category", ""),
        "pips": pips,
        "pf": pf,
        "trades": n_trades,
        "dd": dd,
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
        strat_trades[key] = pd.read_csv(csv_file)
    else:
        tl = m.get("trade_log", [])
        strat_trades[key] = pd.DataFrame(tl) if tl else pd.DataFrame()

by_sym = defaultdict(list)
for k in strat_meta:
    sym = k.split("__")[0]
    by_sym[sym].append(k)

redundancy_list = []
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

            if overlap_pct >= 80.0 and corr >= 0.70:
                # Decide which one is superior
                m1, m2 = strat_meta[k1], strat_meta[k2]
                if m1["score"] >= m2["score"]:
                    champ = k1
                    duplicate = k2
                else:
                    champ = k2
                    duplicate = k1

                r_score = (overlap_pct / 100.0) * corr
                redundancy_list.append({
                    "symbol": sym,
                    "champion": champ,
                    "duplicate": duplicate,
                    "overlap": overlap_pct,
                    "corr": corr,
                    "r_score": r_score,
                    "champ_pips": strat_meta[champ]["pips"],
                    "champ_pf": strat_meta[champ]["pf"],
                    "dup_pips": strat_meta[duplicate]["pips"],
                    "dup_pf": strat_meta[duplicate]["pf"],
                })

# Sort redundancy list descending by r_score
redundancy_list.sort(key=lambda x: x["r_score"], reverse=True)

print("=" * 115)
print(f"{'RANK':<5} | {'SYM':<7} | {'CHAMPION (KEEP)':<25} | {'DUPLICATE (PRUNE)':<25} | {'OVERLAP':<8} | {'CORR':<6} | {'R-SCORE'}")
print("=" * 115)

for idx, r in enumerate(redundancy_list[:30], 1):
    c_name = r["champion"].split("__")[1]
    d_name = r["duplicate"].split("__")[1]
    print(f"{idx:<5} | {r['symbol']:<7} | {c_name:<25} | {d_name:<25} | {r['overlap']:>6.1f}% | {r['corr']:>5.2f} | {r['r_score']:>6.3f}")

print("=" * 115)

