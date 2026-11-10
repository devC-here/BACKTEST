#!/usr/bin/env python3
"""
================================================================================
  DATASET PARTITIONING: 70/30 RANDOM SPLIT FOR ALL ASSETS
================================================================================
Enforces the strict quantitative discipline:
  - 70% of randomly selected weeks -> 'discovery_70' (Used for research, analysis,
    alpha finding, feature exploration, and parameter tuning).
  - 30% of randomly selected weeks -> 'holdout_test_30' (Strictly QUARANTINED for
    final out-of-sample forward testing, never touched during edge discovery).

Assets Partitioned:
  1. EURUSD: 262 weeks -> 183 Discovery (69.8%), 79 Holdout Test (30.2%)
  2. EURGBP: 157 weeks -> 110 Discovery (70.1%), 47 Holdout Test (29.9%)
  3. AUDNZD:  53 weeks ->  37 Discovery (69.8%), 16 Holdout Test (30.2%)

Seed: 42 (Deterministic, reproducible, non-overlapping).

Actions:
  - Updates `manifest_weekly_sets.json` in each asset folder with `"split"` and `"split_ratio"`.
  - Creates `split_70_30.json` in each asset folder.
  - Creates central registry `data_splits/master_70_30_splits.json`.
================================================================================
"""

from __future__ import annotations
import sys
import json
import random
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
WEEKLY_DATA_DIR = ROOT_DIR / "weekly_data"
SPLITS_DIR = ROOT_DIR / "data_splits"
SPLITS_DIR.mkdir(parents=True, exist_ok=True)

SYMBOLS = ["EURUSD", "EURGBP", "AUDNZD"]
SEED = 42


def partition_symbol(symbol: str):
    sym_dir = WEEKLY_DATA_DIR / symbol
    manifest_path = sym_dir / "manifest_weekly_sets.json"
    if not manifest_path.exists():
        print(f"[!] Warning: manifest not found at {manifest_path}")
        return

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # Sort items by week number
    manifest.sort(key=lambda x: x["week"])
    total_weeks = len(manifest)
    weeks = [item["week"] for item in manifest]

    # Deterministic shuffle
    rng = random.Random(SEED)
    shuffled_weeks = weeks.copy()
    rng.shuffle(shuffled_weeks)

    n_discovery = int(round(total_weeks * 0.70))
    discovery_weeks = sorted(shuffled_weeks[:n_discovery])
    test_weeks = sorted(shuffled_weeks[n_discovery:])

    discovery_set = set(discovery_weeks)
    test_set = set(test_weeks)

    assert len(discovery_set.intersection(test_set)) == 0, "Error: Overlapping weeks detected!"
    assert len(discovery_set) + len(test_set) == total_weeks, "Error: Week count mismatch!"

    # Calculate tick totals
    disc_ticks = 0
    test_ticks = 0

    for item in manifest:
        w = item["week"]
        if w in discovery_set:
            item["split"] = "discovery_70"
            item["split_category"] = "discovery"
            item["split_role"] = "Analysis, alpha discovery, and calibration"
            disc_ticks += item.get("tick_count", 0)
        else:
            item["split"] = "holdout_test_30"
            item["split_category"] = "holdout_test"
            item["split_role"] = "Quarantined out-of-sample test only"
            test_ticks += item.get("tick_count", 0)

    # Save updated manifest
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # Save symbol-specific split file
    symbol_split_data = {
        "symbol": symbol,
        "seed": SEED,
        "total_weeks": total_weeks,
        "discovery_70": {
            "count": len(discovery_weeks),
            "percentage": round(len(discovery_weeks) / total_weeks * 100, 2),
            "total_ticks": disc_ticks,
            "weeks": discovery_weeks
        },
        "holdout_test_30": {
            "count": len(test_weeks),
            "percentage": round(len(test_weeks) / total_weeks * 100, 2),
            "total_ticks": test_ticks,
            "weeks": test_weeks
        }
    }

    sym_split_path = sym_dir / "split_70_30.json"
    with open(sym_split_path, "w", encoding="utf-8") as f:
        json.dump(symbol_split_data, f, indent=2)

    print(f"[+] {symbol:<7}: Total {total_weeks:>3} weeks | Discovery (70%): {len(discovery_weeks):>3} weeks ({disc_ticks:,} ticks) | Holdout Test (30%): {len(test_weeks):>3} weeks ({test_ticks:,} ticks)")
    return symbol_split_data


def main():
    print("=" * 100)
    print("   DATASET 70:30 RANDOM WEEK PARTITIONING (DISCOVERY VS STRICT HOLDOUT TEST)   ".center(100))
    print("=" * 100)
    print(f"Random Seed: {SEED} (Deterministic)\n")

    master_splits = {}
    for sym in SYMBOLS:
        res = partition_symbol(sym)
        if res:
            master_splits[sym] = res

    # Save central registry
    master_path = SPLITS_DIR / "master_70_30_splits.json"
    with open(master_path, "w", encoding="utf-8") as f:
        json.dump(master_splits, f, indent=2)

    print("\n" + "=" * 100)
    print(f"[+] Central registry successfully saved to: {master_path}")
    print("=" * 100)


if __name__ == "__main__":
    main()
