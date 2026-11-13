#!/usr/bin/env python3
"""
================================================================================
  COMPREHENSIVE INDIVIDUAL STRATEGY RIGOR & PURITY AUDIT
================================================================================
Evaluates each of the 78 strategies in the certified production fleet
completely independently. No aggregate hiding, no portfolio masking.

Criteria for Independent Viability:
1. Holdout Total Trades >= 15 (Statistical significance & sample size)
2. Holdout Profit Factor >= 1.15 (Edge exceeds broker friction + noise)
3. Holdout Net Pips > 0 (Strict positive profitability)
4. Outlier Resistance: Net Pips ex Top 3 trades > 0 (Not a 1-3 trade anomaly)
5. Discovery Consistency: Discovery PF >= 1.15 and Discovery Net Pips > 0
================================================================================
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
FLEET_MANIFEST = ROOT_DIR / "certified_production_fleet" / "CERTIFIED_FLEET_MANIFEST.json"
MANIFEST_DIR = ROOT_DIR / "results" / "certified_strategy_manifests"
OUTPUT_REPORT = ROOT_DIR / "results" / "individual_strategy_independent_audit.json"

def main():
    with open(FLEET_MANIFEST, "r", encoding="utf-8") as f:
        fleet_data = json.load(f)

    strategies = fleet_data.get("strategies", [])
    print(f"Loaded {len(strategies)} strategies from fleet manifest.")

    records = []

    for s in strategies:
        sym = s["symbol"]
        name = s["name"]
        cat = s.get("category", "")
        rel_path = s.get("relative_path", "")

        # Locate trade CSV
        mf = s.get("manifest_file")
        csv_file = None
        if mf:
            c = ROOT_DIR / Path(mf).with_name(Path(mf).name.replace("_run_manifest.json", "_trades.csv"))
            if c.exists():
                csv_file = c
        
        if not csv_file or not csv_file.exists():
            candidates = list(MANIFEST_DIR.glob(f"*{sym}*{name}*trades.csv"))
            if not candidates:
                candidates = list(MANIFEST_DIR.glob(f"*{name}*trades.csv"))
            if candidates:
                csv_file = candidates[0]

        if not csv_file or not csv_file.exists():
            print(f"WARNING: Trade CSV not found for {sym} {name} (path: {rel_path})")
            continue

        df = pd.read_csv(csv_file)
        if len(df) == 0:
            print(f"WARNING: Trade CSV empty for {sym} {name}")
            continue

        pip_col = "pnl_pips" if "pnl_pips" in df.columns else ("pips" if "pips" in df.columns else None)
        if not pip_col:
            print(f"WARNING: No pips column found in {csv_file}")
            continue

        pips = df[pip_col].values
        n_trades = len(pips)
        net_pips = float(np.sum(pips))
        wins = pips[pips > 0]
        losses = pips[pips < 0]
        gross_win = float(np.sum(wins))
        gross_loss = float(abs(np.sum(losses)))
        pf = (gross_win / gross_loss) if gross_loss > 0 else 999.0
        exp = net_pips / n_trades if n_trades > 0 else 0.0
        wr = (len(wins) / n_trades * 100.0) if n_trades > 0 else 0.0

        # Outlier metrics
        sorted_pips = np.sort(pips)[::-1]
        top1_pip = float(sorted_pips[0]) if n_trades > 0 else 0.0
        top3_pips = float(np.sum(sorted_pips[:min(3, n_trades)]))
        pips_ex_top1 = float(np.sum(sorted_pips[1:])) if n_trades > 1 else 0.0
        pips_ex_top3 = float(np.sum(sorted_pips[3:])) if n_trades > 3 else 0.0
        pips_ex_top5 = float(np.sum(sorted_pips[5:])) if n_trades > 5 else 0.0

        is_outlier_dependent = (pips_ex_top3 <= 0.0) or (net_pips > 0 and top1_pip / net_pips > 0.50 and n_trades < 30)

        # Discovery metrics from manifest
        disc_m = s.get("discovery_metrics", {})
        d_pips = disc_m.get("net_pips", 0.0)
        d_pf = disc_m.get("profit_factor", 0.0)
        d_trades = disc_m.get("total_trades", 0)

        # Independent gate checks
        flags = []
        if n_trades < 15:
            flags.append(f"Low Sample Size ({n_trades} trades < 15)")
        if pf < 1.15:
            flags.append(f"Sub-par Profit Factor ({pf:.2f} < 1.15)")
        if net_pips <= 0:
            flags.append(f"Non-positive Net Pips ({net_pips:.1f}p <= 0)")
        if pips_ex_top3 <= 0.0:
            flags.append(f"Outlier Dependent (Pips ex top 3: {pips_ex_top3:.1f}p <= 0)")
        if d_trades > 0 and d_pf < 1.15:
            flags.append(f"Weak Discovery PF ({d_pf:.2f} < 1.15)")
        if d_trades > 0 and d_pips <= 0:
            flags.append(f"Negative Discovery PnL ({d_pips:.1f}p <= 0)")

        # Grade
        if len(flags) == 0 and pf >= 1.30 and pips_ex_top5 > 0 and n_trades >= 30:
            grade = "GRADE A: INSTITUTIONAL CHAMPION"
        elif len(flags) == 0:
            grade = "GRADE B: PRODUCTION CERTIFIED"
        elif len(flags) == 1 and net_pips > 0 and pf >= 1.10:
            grade = "GRADE C: BORDERLINE / CONDITIONAL"
        else:
            grade = "GRADE F: REJECTED / FRAGILE"

        records.append({
            "symbol": sym,
            "name": name,
            "category": cat,
            "relative_path": rel_path,
            "csv_file": str(csv_file),
            "holdout_trades": n_trades,
            "holdout_net_pips": round(net_pips, 1),
            "holdout_pf": round(pf, 2),
            "holdout_expectancy": round(exp, 2),
            "holdout_win_rate": round(wr, 1),
            "top1_pip": round(top1_pip, 1),
            "top3_pips": round(top3_pips, 1),
            "pips_ex_top1": round(pips_ex_top1, 1),
            "pips_ex_top3": round(pips_ex_top3, 1),
            "pips_ex_top5": round(pips_ex_top5, 1),
            "is_outlier_dependent": is_outlier_dependent,
            "discovery_trades": d_trades,
            "discovery_net_pips": round(d_pips, 1),
            "discovery_pf": round(d_pf, 2),
            "grade": grade,
            "flags_count": len(flags),
            "flags": flags,
            "status": "PASS" if len(flags) == 0 else "FAIL"
        })

    with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    df_res = pd.DataFrame(records)
    print("\n" + "=" * 80)
    print("  FLEET INDIVIDUAL-LEVEL RELIABILITY SCORECARD")
    print("=" * 80)
    print(f"Total Models Audited:       {len(df_res)}")
    print(f"PASS (All Gates Satisfied): {len(df_res[df_res['status'] == 'PASS'])}")
    print(f"FAIL (Flagged for Pruning): {len(df_res[df_res['status'] == 'FAIL'])}")
    print(f"   Grade A (Champions):     {len(df_res[df_res['grade'].str.startswith('GRADE A')])}")
    print(f"   Grade B (Certified):     {len(df_res[df_res['grade'].str.startswith('GRADE B')])}")
    print(f"   Grade C (Borderline):    {len(df_res[df_res['grade'].str.startswith('GRADE C')])}")
    print(f"   Grade F (Fragile/Reject):{len(df_res[df_res['grade'].str.startswith('GRADE F')])}")
    print("=" * 80)

    failing = df_res[df_res["status"] == "FAIL"].sort_values("flags_count", ascending=False)
    print("\nDETAILED BREAKDOWN OF FLAGGED STRATEGIES:")
    print("-" * 80)
    for _, r in failing.iterrows():
        print(f"[{r['symbol']}] {r['name']} ({r['category']}) -> {r['grade']}")
        print(f"   Holdout:   {r['holdout_net_pips']}p | {r['holdout_trades']} tr | PF {r['holdout_pf']} | Exp {r['holdout_expectancy']}p | ExTop3: {r['pips_ex_top3']}p")
        print(f"   Discovery: {r['discovery_net_pips']}p | {r['discovery_trades']} tr | PF {r['discovery_pf']}")
        print(f"   Defects:   {'; '.join(r['flags'])}")
        print("-" * 80)

if __name__ == "__main__":
    main()
