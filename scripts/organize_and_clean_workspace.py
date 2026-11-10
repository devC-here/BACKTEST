#!/usr/bin/env python3
"""
Cleans and organizes the BACKTEST root directory:
Moves deprecated search templates, legacy discovery labs, and old experimental fleets
into archive_deprecated/, leaving the root spotless and focused on the verified
certified_production_fleet and tick_engine verification platform.
"""

import shutil
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
ARCHIVE_DIR = ROOT_DIR / "archive_deprecated"
ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

# Deprecated folders to isolate
deprecated_targets = [
    "discovery_labs",
    "alpha_searcher",
    "legacy_labs",
    "archive_labs",
    "archive_versions",
    "scratch",
    "v 4.5",
    "strategies"
]

print("=" * 80)
print("  ORGANIZING AND CLEANING WORKSPACE: ISOLATING DEPRECATED TEMPLATES")
print("=" * 80)

for target_name in deprecated_targets:
    target_path = ROOT_DIR / target_name
    if target_path.exists():
        dest_path = ARCHIVE_DIR / target_name
        if dest_path.exists():
            shutil.rmtree(dest_path)
        shutil.move(str(target_path), str(dest_path))
        print(f"Moved: {target_name} -> archive_deprecated/{target_name}")

print("\nCleaning complete! Root directory is pristine and starting from scratch.")

