import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import datetime
from tick_engine.core.dataloader import (
    get_discovery_weeks,
    get_holdout_test_weeks,
    get_week_dataset,
    load_manifest,
)

SPLITS_FILE = Path("data_splits/master_70_30_splits.json")
with open(SPLITS_FILE, "r", encoding="utf-8") as f:
    registry = json.load(f)

print("=" * 85)
print("  FLEET-WIDE TICK DATASET & 70/30 SPLIT VERIFICATION AUDIT")
print("=" * 85)
print(f"Total Registered Assets: {registry['total_assets']}")
print(f"Total Fleet Weeks:       {registry['total_fleet_weeks']:,}")
print(f"Total Fleet Ticks:       {registry['total_fleet_ticks']:,}")
print("-" * 85)
print(f"{'Asset':<14} | {'Weeks':<5} | {'Disc (70%)':<10} | {'Test (30%)':<10} | {'Total Ticks':<15} | {'Dataloader Check'}")
print("-" * 85)

for sym, data in registry["assets"].items():
    tot_weeks = data["total_weeks"]
    disc_weeks = data["discovery_70"]["count"]
    test_weeks = data["holdout_test_30"]["count"]
    tot_ticks = data["discovery_70"]["total_ticks"] + data["holdout_test_30"]["total_ticks"]

    # Verify dataloader load for week 1
    status = "OK"
    try:
        ticks, meta = get_week_dataset(1, raw_dir=f"data/{sym}")
        if len(ticks) == 0:
            status = "ERR: Empty W1"
        else:
            status = f"W1: {len(ticks):,} ticks"
    except Exception as e:
        status = f"ERR: {e}"

    print(f"{sym:<14} | {tot_weeks:<5} | {disc_weeks:<10} | {test_weeks:<10} | {tot_ticks:>15,} | {status}")

print("=" * 85)

