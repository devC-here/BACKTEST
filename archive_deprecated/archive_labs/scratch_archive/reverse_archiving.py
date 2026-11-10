#!/usr/bin/env python3
"""
================================================================================
  REVERSE ARCHIVING: RESTORE ALL STRATEGIES TO PRODUCTION FLEETS
================================================================================
Restores all models that were moved to archive_unprofitable/ back to:
  - strategies/eurusd_production_fleet/
  - strategies/eurgbp_fleet/
  - strategies/
================================================================================
"""

import shutil
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
STRAT_DIR = ROOT_DIR / "strategies"
EURUSD_FLEET = STRAT_DIR / "eurusd_production_fleet"
EURGBP_FLEET = STRAT_DIR / "eurgbp_fleet"
ARCHIVE_DIR = STRAT_DIR / "archive_unprofitable"

# 1. Restore EURUSD Fleet models
eurusd_models = [
    "high_frequency_m15_scalper_elite",
    "s66_bhanushali_44ma_elite",
    "s14_inside_bar_m30_elite",
    "s10_dual_ema_pullback_elite",
    "disc39_stochastic_deep_pullback_elite",
    "comb12_bollinger_squeeze_elite",
    "comb27_squeeze_supertrend_hybrid_elite"
]

for name in eurusd_models:
    archived_dir = ARCHIVE_DIR / f"eurusd_{name}"
    target_dir = EURUSD_FLEET / name
    if archived_dir.exists():
        if target_dir.exists():
            shutil.rmtree(target_dir)
        shutil.move(str(archived_dir), str(target_dir))
        print(f"[+] Restored to EURUSD Fleet: {name}")

# 2. Restore EURGBP Fleet models
eurgbp_models = [
    "disc43_intraday_precision_h1_m5_elite"
]

for name in eurgbp_models:
    archived_dir = ARCHIVE_DIR / f"eurgbp_{name}"
    target_dir = EURGBP_FLEET / name
    if archived_dir.exists():
        if target_dir.exists():
            shutil.rmtree(target_dir)
        shutil.move(str(archived_dir), str(target_dir))
        print(f"[+] Restored to EURGBP Fleet: {name}")

# 3. Restore root models
root_models = [
    "comb12_bollinger_squeeze_elite",
    "comb27_squeeze_supertrend_hybrid_elite",
    "high_frequency_m15_scalper_elite",
    "s10_dual_ema_pullback_elite",
    "s14_inside_bar_m30_elite",
    "s66_bhanushali_44ma_elite"
]

for name in root_models:
    archived_dir = ARCHIVE_DIR / f"root_{name}"
    target_dir = STRAT_DIR / name
    if archived_dir.exists():
        if target_dir.exists():
            shutil.rmtree(target_dir)
        shutil.move(str(archived_dir), str(target_dir))
        print(f"[+] Restored to root strategies/: {name}")

print("\n[+] All models successfully restored!")
