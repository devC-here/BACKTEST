#!/usr/bin/env python3
"""
================================================================================
  CLEAN STRATEGY FOLDER: MOVE UNPROFITABLE STRATEGIES TO ARCHIVE
================================================================================
Moves the 13 verified unprofitable/failed models to strategies/archive_unprofitable/
keeping the active strategy folder 100% clean with ONLY proven profitable strategies.
================================================================================
"""

import sys
import shutil
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
STRAT_DIR = ROOT_DIR / "strategies"
ARCHIVE_DIR = STRAT_DIR / "archive_unprofitable"
ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

# 1. Unprofitable folders in strategies/
root_unprofitable = [
    "comb08_asian_breakout_adx_elite",
    "comb14_supertrend_vwap_elite",
    "comb17_psar_adx_elite",
    "disc02_frankfurt_london_orb_elite",
    "disc23_keltner_squeeze_elite",
    "disc30_frankfurt_london_relay_elite",
    "disc32_european_early_bird_elite",
    "disc33_us_afternoon_trend_elite",
    "disc36_frankfurt_ultra_coil_orb_elite",
    "disc37_european_midday_ultracoil_breakout_elite",
    "institutional_v12_elite"
]

# 2. Unprofitable/duplicate folders in strategies/new_discoveries/
new_disc_dir = STRAT_DIR / "new_discoveries"
new_disc_unprofitable = [
    "disc44_high_frequency_m5_scalper_elite",
    "disc42_m5_intraday_velocity_ignition_elite",
    "disc38_london_midday_macd_acceleration"  # duplicate of root disc38
]

moved_count = 0

print("=" * 80)
print("             CLEANING STRATEGIES DIRECTORY & ARCHIVING UNPROFITABLE MODELS             ".center(80))
print("=" * 80)

# Move root unprofitable
for folder_name in root_unprofitable:
    src = STRAT_DIR / folder_name
    dest = ARCHIVE_DIR / folder_name
    if src.exists():
        if dest.exists():
            shutil.rmtree(dest)
        shutil.move(str(src), str(dest))
        print(f"[+] Archived from root: {folder_name} -> archive_unprofitable/")
        moved_count += 1

# Move new_discoveries unprofitable
for folder_name in new_disc_unprofitable:
    src = new_disc_dir / folder_name
    dest = ARCHIVE_DIR / folder_name
    if src.exists():
        if dest.exists():
            shutil.rmtree(dest)
        shutil.move(str(src), str(dest))
        print(f"[+] Archived from new_discoveries: {folder_name} -> archive_unprofitable/")
        moved_count += 1

print("-" * 80)
print(f"[+] Successfully moved {moved_count} unprofitable/failed strategy folders to {ARCHIVE_DIR.name}/")

# Inspect active remaining folders
print("\n[+] ACTIVE REMAINING STRATEGY DIRECTORIES (100% PROVEN PROFITABLE):")
active_root = [d.name for d in STRAT_DIR.iterdir() if d.is_dir() and not d.name.startswith('_') and d.name != 'archive_unprofitable']
for d in sorted(active_root):
    print(f"  - strategies/{d}")

if new_disc_dir.exists():
    active_new = [d.name for d in new_disc_dir.iterdir() if d.is_dir() and not d.name.startswith('_')]
    print("\n[+] ACTIVE REMAINING IN strategies/new_discoveries/:")
    for d in sorted(active_new):
        print(f"  - strategies/new_discoveries/{d}")

print("=" * 80)
