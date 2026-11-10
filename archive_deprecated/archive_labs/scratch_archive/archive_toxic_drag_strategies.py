#!/usr/bin/env python3
"""
================================================================================
  ARCHIVE TOXIC DRAG & LOW PIP GAIN STRATEGIES
================================================================================
Moves the 7 identified low-expectancy / high-drawdown strategies to
strategies/archive_unprofitable/:
  1. COMB28: Unified Squeeze Ensemble Elite (+60.6p, 343.9p DD, Ret/DD 0.18)
  2. DISC34: Friday Weekly Reversal Elite (+35.2p, 149.5p DD, Ret/DD 0.24)
  3. DISC31: London Lunch Trap Fade Elite (+80.4p, 329.3p DD, Ret/DD 0.24)
  4. S01_M15: M15 Displacement Retest Elite (+165.0p, 382.2p DD, Ret/DD 0.43)
  5. COMB26: Asian Tight Coil Breakout Elite (+264.9p, 306.6p DD, Ret/DD 0.86)
  6. INST05: Market Structure Shift Elite (+422.6p, 385.7p DD, Ret/DD 1.10)
  7. S34_HA: Heikin Ashi Momentum Elite (+505.3p, 328.1p DD, Ret/DD 1.54, Sidelined)
================================================================================
"""

import sys
import shutil
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
STRAT_DIR = ROOT_DIR / "strategies"
ARCHIVE_DIR = STRAT_DIR / "archive_unprofitable"
ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

drag_folders = [
    "comb28_unified_squeeze_ensemble_elite",
    "disc34_friday_weekly_reversal_elite",
    "disc31_london_lunch_trap_fade_elite",
    "s01_m15_displacement_retest_elite",
    "comb26_asian_tight_coil_elite",
    "inst05_market_structure_shift_elite",
    "s34_heikin_ashi_v2_elite"
]

print("=" * 80)
print("             ARCHIVING TOXIC DRAG & HIGH-DRAWDOWN STRATEGIES             ".center(80))
print("=" * 80)

moved_count = 0
for folder_name in drag_folders:
    src = STRAT_DIR / folder_name
    dest = ARCHIVE_DIR / folder_name
    if src.exists():
        if dest.exists():
            shutil.rmtree(dest)
        shutil.move(str(src), str(dest))
        print(f"[+] Moved to archive: {folder_name}")
        moved_count += 1
    else:
        print(f"[-] Not found (already moved): {folder_name}")

print("-" * 80)
print(f"[+] Successfully archived {moved_count} toxic drag folders into {ARCHIVE_DIR.name}/")

# Inspect active remaining folders
print("\n[+] PRISTINE ACTIVE ROSTER (ELITE HIGH-ALPHA ENGINES ONLY):")
active_root = [d.name for d in STRAT_DIR.iterdir() if d.is_dir() and not d.name.startswith('_') and d.name != 'archive_unprofitable']
for d in sorted(active_root):
    print(f"  - strategies/{d}")

new_disc_dir = STRAT_DIR / "new_discoveries"
if new_disc_dir.exists():
    active_new = [d.name for d in new_disc_dir.iterdir() if d.is_dir() and not d.name.startswith('_')]
    print("\n[+] ELITE MULTI-TIMEFRAME DISCOVERIES in strategies/new_discoveries/:")
    for d in sorted(active_new):
        print(f"  - strategies/new_discoveries/{d}")

print("=" * 80)
