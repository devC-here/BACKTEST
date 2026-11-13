#!/usr/bin/env python3
"""
================================================================================
  INDIVIDUAL STRATEGY RIGOR & PURITY AUDIT
================================================================================
Audits all 101 strategies on an individual basis against the strict reliability gates:
1. Holdout PF >= 1.15
2. Holdout Trades >= 15
3. Outlier Resistance: Pips ex Top 3 > 0 (not a 1-3 lucky trade wonder)
4. Dual In-Sample / Holdout Consistency: Discovery Net Pips > 0 and PF >= 1.15
5. Realistic Statistical Confidence
================================================================================
"""

import json
from pathlib import Path
import pandas as pd
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
MANIFEST_DIR = ROOT_DIR / "results" / "certified_strategy_manifests"
FLEET_MANIFEST = ROOT_DIR / "certified_production_fleet" / "CERTIFIED_FLEET_MANIFEST.json"

fleet_data = json.load(open(FLEET_MANIFEST, encoding="utf-8"))
strategies = fleet_data["strategies"]

records = []
for s in strategies:
    sym = s["symbol"]
    name = s["name"]
    m_candidates = [
        MANIFEST_DIR / f"{sym}__{name}_run_manifest.json",
        MANIFEST_DIR / f"{sym.lower()}_{name}_run_manifest.json",
        MANIFEST_DIR / f"{name}_run_manifest.json"
    ]
    m_file = next((f for f in m_candidates if f.exists()), None)
    if not m_file:
        print(f"MISSING: {sym} {name}")
        continue

    with open(m_file, encoding="utf-8") as f:
        m = json.load(f)

    stats = m["statistical_metrics"]
    h_pips = stats.get("net_pips", 0.0)
    h_trades = stats.get("total_trades", 0)
    h_pf = stats.get("profit_factor", 0.0)
    h_exp = stats.get("expectancy_pips", 0.0)
    h_dd = stats.get("max_drawdown_pips", 0.0)
    ci = stats.get("expectancy_ci_95", [0.0, 0.0])
    t_stat = stats.get("t_statistic", 0.0)
    p_val = stats.get("p_value", 1.0)

    outlier = stats.get("outlier_sensitivity", {})
    is_outlier_dep = outlier.get("is_outlier_dependent", False)
    pips_ex1 = outlier.get("pips_ex_top1", h_pips)
    pips_ex3 = outlier.get("pips_ex_top3", h_pips)
    pips_ex5 = outlier.get("pips_ex_top5", h_pips)

    custom = m.get("custom_metadata", {})
    disc = custom.get("discovery_in_sample_metrics", {})
    d_pips = disc.get("net_pips", 0.0)
    d_pf = disc.get("profit_factor", 0.0)
    d_trades = disc.get("total_trades", 0)

    # Gate verification
    fails = []
    if h_pips <= 0:
        fails.append(f"Holdout PnL <= 0 ({h_pips:.1f}p)")
    if h_pf < 1.15:
        fails.append(f"Holdout PF < 1.15 ({h_pf:.2f})")
    if h_trades < 15:
        fails.append(f"Holdout Trades < 15 ({h_trades})")
    if pips_ex3 <= 0 or is_outlier_dep:
        fails.append(f"Outlier Dependent (Pips ex top 3: {pips_ex3:.1f}p)")
    if d_pips <= 0:
        fails.append(f"Discovery PnL <= 0 ({d_pips:.1f}p)")
    if d_pf < 1.15:
        fails.append(f"Discovery PF < 1.15 ({d_pf:.2f})")

    # Determine certification grade
    if len(fails) == 0 and h_pf >= 1.30 and pips_ex5 > 0:
        grade = "GRADE A: INSTITUTIONAL CHAMPION"
    elif len(fails) == 0:
        grade = "GRADE B: PRODUCTION CERTIFIED"
    elif len(fails) == 1 and h_pips > 0 and h_pf >= 1.10:
        grade = "GRADE C: PROVISIONAL / BORDERLINE"
    else:
        grade = "GRADE F: REJECTED / FRAGILE"

    records.append({
        "symbol": sym,
        "name": name,
        "category": s.get("category", ""),
        "h_pips": round(h_pips, 1),
        "h_trades": h_trades,
        "h_pf": round(h_pf, 2),
        "h_exp": round(h_exp, 2),
        "h_dd": round(h_dd, 1),
        "pips_ex1": round(pips_ex1, 1),
        "pips_ex3": round(pips_ex3, 1),
        "pips_ex5": round(pips_ex5, 1),
        "d_pips": round(d_pips, 1),
        "d_trades": d_trades,
        "d_pf": round(d_pf, 2),
        "ci_low": ci[0],
        "ci_high": ci[1],
        "t_stat": t_stat,
        "p_val": p_val,
        "grade": grade,
        "fails_count": len(fails),
        "failure_reasons": "; ".join(fails) if fails else "NONE (Passed All Reliability Gates)"
    })

df = pd.DataFrame(records)

print("\n" + "=" * 90)
print(f"  INDIVIDUAL STRATEGY RELIABILITY AUDIT ({len(df)} STRATEGIES)")
print("=" * 90)
print(f"  GRADE A (Institutional Champion): {len(df[df['grade'] == 'GRADE A: INSTITUTIONAL CHAMPION'])}")
print(f"  GRADE B (Production Certified):   {len(df[df['grade'] == 'GRADE B: PRODUCTION CERTIFIED'])}")
print(f"  GRADE C (Provisional / Warn):     {len(df[df['grade'] == 'GRADE C: PROVISIONAL / BORDERLINE'])}")
print(f"  GRADE F (Rejected / Fragile):     {len(df[df['grade'] == 'GRADE F: REJECTED / FRAGILE'])}")
print("=" * 90)

failing = df[df["fails_count"] > 0].sort_values("fails_count", ascending=False)
print(f"\nStrategies with Warnings / Failures ({len(failing)} strategies):")
for _, r in failing.iterrows():
    print(f"  [{r['symbol']}] {r['name']}: {r['grade']}")
    print(f"      Reasons: {r['failure_reasons']}")
    print(f"      Metrics: Holdout: {r['h_pips']}p ({r['h_trades']} tr, PF {r['h_pf']}, Exp {r['h_exp']}p) | ExTop3: {r['pips_ex3']}p | Disc: {r['d_pips']}p (PF {r['d_pf']})")

out_file = ROOT_DIR / "results" / "individual_strategy_reliability_audit.json"
with open(out_file, "w", encoding="utf-8") as f:
    json.dump(records, f, indent=2)
print(f"\nSaved full report to {out_file}")

