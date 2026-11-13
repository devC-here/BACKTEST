#!/usr/bin/env python3
"""
================================================================================
  GENERATE PRISTINE 63-STRATEGY PRODUCTION FLEET MANIFEST & AUDIT
================================================================================
Filters out the 15 flagged/fragile/undersampled strategies from the 78 fleet,
leaving only the 63 strategies that independently satisfy all institutional reliability gates:
1. Holdout Total Trades >= 15
2. Holdout Profit Factor >= 1.15
3. Holdout Outlier Resistance: Net Pips ex Top 3 > 0
4. Discovery Consistency: Discovery Net Pips > 0
================================================================================
"""

import json
import shutil
from pathlib import Path
import pandas as pd
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
FLEET_DIR = ROOT_DIR / "certified_production_fleet"
FLEET_MANIFEST = FLEET_DIR / "CERTIFIED_FLEET_MANIFEST.json"
AUDIT_REPORT = ROOT_DIR / "results" / "individual_strategy_independent_audit.json"
ARCHIVE_DIR = ROOT_DIR / "archive_deprecated" / "pruned_low_edge_undersampled"

def main():
    with open(FLEET_MANIFEST, "r", encoding="utf-8") as f:
        fleet_data = json.load(f)

    with open(AUDIT_REPORT, "r", encoding="utf-8") as f:
        audit_records = json.load(f)

    audit_df = pd.DataFrame(audit_records)

    # Passing mask
    pass_mask = (
        (audit_df['holdout_trades'] >= 15) &
        (audit_df['holdout_pf'] >= 1.15) &
        (audit_df['pips_ex_top3'] > 0.0) &
        (audit_df['discovery_net_pips'] > 0.0)
    )

    passing_df = audit_df[pass_mask]
    failing_df = audit_df[~pass_mask]

    passing_keys = set(zip(passing_df['symbol'], passing_df['name']))
    failing_keys = set(zip(failing_df['symbol'], failing_df['name']))

    print(f"Total fleet: {len(audit_df)}")
    print(f"Passing:     {len(passing_keys)}")
    print(f"Failing:     {len(failing_keys)}")

    # Archive the failing strategies
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    archived_count = 0
    for _, r in failing_df.iterrows():
        rel = r.get("relative_path")
        if rel:
            source_folder = (ROOT_DIR / rel).parent
            if source_folder.exists():
                dest = ARCHIVE_DIR / source_folder.name
                if not dest.exists():
                    shutil.copytree(source_folder, dest)
                    archived_count += 1
                # Remove from certified_production_fleet
                # (keep on disk in archive)
                shutil.rmtree(source_folder, ignore_errors=True)

    print(f"Archived {archived_count} strategy folders to {ARCHIVE_DIR}")

    # Build purified strategies list
    purified_strategies = []
    total_holdout_pips = 0.0
    total_holdout_trades = 0

    for s in fleet_data["strategies"]:
        key = (s["symbol"], s["name"])
        if key in passing_keys:
            purified_strategies.append(s)
            hm = s.get("holdout_metrics", {})
            total_holdout_pips += hm.get("net_pips", 0.0)
            total_holdout_trades += hm.get("total_trades", 0)

    # Update manifest
    fleet_data["manifest_version"] = "6.0.0_pristine_63_elite"
    fleet_data["fleet_version"] = "6.0.0_pristine_63_elite"
    fleet_data["total_strategies"] = len(purified_strategies)
    fleet_data["strategy_count"] = len(purified_strategies)
    fleet_data["strategies"] = purified_strategies
    fleet_data["portfolio_totals"] = {
        "total_strategies": len(purified_strategies),
        "holdout_net_pips": round(total_holdout_pips, 1),
        "holdout_total_trades": total_holdout_trades,
        "holdout_expectancy_pips": round(total_holdout_pips / total_holdout_trades, 2) if total_holdout_trades > 0 else 0.0,
        "pruned_strategies_count": len(failing_keys),
        "pruning_criteria": "Holdout Trades >= 15, Holdout PF >= 1.15, Pips ex Top 3 > 0, Discovery Pips > 0"
    }

    with open(FLEET_MANIFEST, "w", encoding="utf-8") as f:
        json.dump(fleet_data, f, indent=2)

    print(f"\nSuccessfully wrote pristine manifest {FLEET_MANIFEST} with {len(purified_strategies)} strategies.")

if __name__ == "__main__":
    main()

