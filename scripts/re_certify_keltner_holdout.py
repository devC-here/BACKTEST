#!/usr/bin/env python3
"""
Re-certifies disc166, disc167, and disc168 on Quarantined Holdout (Weeks 211..300)
to ensure primary manifests and trade CSVs represent the official holdout partition.
"""

import sys
import json
import shutil
import hashlib
from pathlib import Path
from collections import defaultdict
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from scripts.certify_strategy import certify_strategy

FLEET_DIR = ROOT_DIR / "certified_production_fleet"
FLEET_MANIFEST = FLEET_DIR / "CERTIFIED_FLEET_MANIFEST.json"
MANIFEST_DIR = ROOT_DIR / "results" / "certified_strategy_manifests"

TARGETS = [
    {
        "strat_id": "disc166",
        "symbol": "GBPCHF",
        "category": "gbpchf_champions",
        "folder": "disc166_gbpchf_h1_keltner_squeeze_elite",
        "disc_pips": 1225.0, "disc_trades": 539, "disc_pf": 1.20, "disc_exp": 2.27
    },
    {
        "strat_id": "disc167",
        "symbol": "EURNZD",
        "category": "eurnzd_champions",
        "folder": "disc167_eurnzd_frankfurt_london_keltner_elite",
        "disc_pips": 85.6, "disc_trades": 221, "disc_pf": 1.02, "disc_exp": 0.39
    },
    {
        "strat_id": "disc168",
        "symbol": "EURNZD",
        "category": "eurnzd_champions",
        "folder": "disc168_eurnzd_ny_overlap_keltner_runner_elite",
        "disc_pips": 405.5, "disc_trades": 177, "disc_pf": 1.10, "disc_exp": 2.29
    }
]

def main():
    print("=" * 80)
    print("  RE-CERTIFYING KELTNER CHAMPIONS ON QUARANTINED HOLDOUT (WEEKS 211..300)")
    print("=" * 80)

    holdout_results = {}

    for t in TARGETS:
        sid = t["strat_id"]
        sym = t["symbol"]
        folder = t["folder"]
        cat = t["category"]
        strat_file = FLEET_DIR / cat / folder / "strategy.py"

        print(f"\nCertifying {sym} {sid} on Quarantined Holdout...")
        manifest = certify_strategy(
            strategy_file=strat_file,
            symbol=sym,
            mode="holdout",
            run_stress=True,
            run_walkforward=False
        )

        h_stats = manifest["statistical_metrics"]
        holdout_results[sid] = h_stats

        # Update canonical alias files
        orig_m = MANIFEST_DIR / f"{sym}__{folder}_run_manifest.json"
        orig_t = MANIFEST_DIR / f"{sym}__{folder}_trades.csv"
        alias_m = MANIFEST_DIR / f"{sym}__{sid}_run_manifest.json"
        alias_t = MANIFEST_DIR / f"{sym}__{sid}_trades.csv"

        if orig_m.exists():
            shutil.copyfile(orig_m, alias_m)
        if orig_t.exists():
            shutil.copyfile(orig_t, alias_t)

        print(f"  Holdout PnL: +{h_stats['net_pips']:.1f}p | PF: {h_stats['profit_factor']:.2f} | Trades: {h_stats['total_trades']}")

    # Now update CERTIFIED_FLEET_MANIFEST.json
    with open(FLEET_MANIFEST, "r", encoding="utf-8") as f:
        fleet_data = json.load(f)

    for s in fleet_data["strategies"]:
        sid = s["name"]
        if sid in holdout_results:
            h = holdout_results[sid]
            matching_t = next(t for t in TARGETS if t["strat_id"] == sid)
            s["holdout_metrics"] = {
                "net_pips": round(float(h.get("net_pips", 0.0)), 1),
                "total_trades": int(h.get("total_trades", 0)),
                "profit_factor": round(float(h.get("profit_factor", 0.0)), 2),
                "expectancy_pips": round(float(h.get("expectancy_pips", 0.0)), 2),
                "max_drawdown_pips": round(float(h.get("max_drawdown_pips", 0.0)), 1),
                "win_rate_pct": round(float(h.get("win_rate_pct", 0.0)), 1),
                "is_statistically_significant": bool(h.get("is_statistically_significant", True))
            }
            s["discovery_metrics"] = {
                "net_pips": matching_t["disc_pips"],
                "total_trades": matching_t["disc_trades"],
                "profit_factor": matching_t["disc_pf"],
                "expectancy_pips": matching_t["disc_exp"]
            }

    strategies = fleet_data["strategies"]
    holdout_pips = sum(s.get("holdout_metrics", {}).get("net_pips", 0.0) for s in strategies)
    holdout_trades = sum(s.get("holdout_metrics", {}).get("total_trades", 0) for s in strategies)
    avg_pf = np.mean([s.get("holdout_metrics", {}).get("profit_factor", 1.0) for s in strategies])
    avg_exp = np.mean([s.get("holdout_metrics", {}).get("expectancy_pips", 0.0) for s in strategies])

    pair_counts = defaultdict(int)
    for s in strategies:
        pair_counts[s["symbol"]] += 1

    fleet_data["fleet_version"] = "5.0.0_expanded_78_elite"
    fleet_data["last_updated"] = "2026-11-11T22:30:00Z"
    fleet_data["total_strategies"] = len(strategies)
    fleet_data["fleet_summary"] = {
        "total_strategies": len(strategies),
        "total_pairs": len(pair_counts),
        "total_holdout_pips": round(holdout_pips, 1),
        "total_holdout_trades": holdout_trades,
        "average_profit_factor": round(float(avg_pf), 2),
        "average_expectancy_pips": round(float(avg_exp), 2),
        "pair_breakdown": dict(sorted(pair_counts.items()))
    }

    with open(FLEET_MANIFEST, "w", encoding="utf-8") as f:
        json.dump(fleet_data, f, indent=2)

    print("\n" + "=" * 80)
    print(f"  FLEET CERTIFIED: {len(strategies)} STRATEGIES")
    print(f"  Total Holdout Net Pips: +{holdout_pips:.1f}p across {holdout_trades:,} trades")
    print(f"  Average Profit Factor:  {avg_pf:.2f}")
    print(f"  Average Expectancy:     +{avg_exp:.2f} pips/trade")
    print("  Pair Breakdown:")
    for sym, count in sorted(pair_counts.items()):
        print(f"    - {sym}: {count} strategies")
    print("=" * 80)

if __name__ == "__main__":
    main()

