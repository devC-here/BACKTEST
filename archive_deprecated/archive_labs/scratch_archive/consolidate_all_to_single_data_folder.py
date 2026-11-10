#!/usr/bin/env python3
"""
================================================================================
  CONSOLIDATE ALL DATASETS INTO A SINGLE UNIFIED 'data/' DIRECTORY
================================================================================
Standardizes repository structure to have strictly ONE data folder at root:
  data/
    ├── EURUSD/
    ├── EURGBP/
    ├── USDJPY/
    ├── AUDNZD/
    ├── raw_masters/
    └── dukascopy_ticks_weekly/

Cleans up all loose root junctions and scattered folders:
  - eurusd_weekly_data
  - eurgbp_weekly_data
  - audnzd_weekly_data
  - 5_YEAR_UNIFIED_MASTERS
  - weekly_data
================================================================================
"""

import os
import sys
import json
import shutil
import subprocess
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
WEEKLY_DATA_DIR = ROOT_DIR / "weekly_data"

JUNCTIONS_TO_REMOVE = [
    "eurusd_weekly_data",
    "eurgbp_weekly_data",
    "audnzd_weekly_data",
    "usdjpy_weekly_data"
]

SYMBOLS = ["EURUSD", "EURGBP", "USDJPY", "AUDNZD"]


def main():
    print("=" * 80)
    print("  CONSOLIDATING ALL DATA INTO A SINGLE 'data/' FOLDER")
    print("=" * 80)

    # 1. Safely remove loose root junctions
    for j_name in JUNCTIONS_TO_REMOVE:
        j_path = ROOT_DIR / j_name
        if j_path.exists():
            print(f"[*] Removing loose junction from root: {j_name}...")
            cmd = f'cmd /c rmdir "{j_path}"'
            subprocess.run(cmd, shell=True, check=True)
            print(f"[+] Removed junction: {j_name}")

    # 2. Ensure data/ exists
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[+] Ensured root 'data/' directory exists: {DATA_DIR.resolve()}")

    # 3. Move symbols from weekly_data/ to data/
    if WEEKLY_DATA_DIR.exists():
        for sym in SYMBOLS:
            src = WEEKLY_DATA_DIR / sym
            dst = DATA_DIR / sym
            if src.exists():
                if dst.exists():
                    print(f"[*] Destination {dst} already exists, removing old before move...")
                    shutil.rmtree(dst)
                print(f"[*] Moving {src} -> {dst}...")
                shutil.move(str(src), str(dst))
                print(f"[+] Moved {sym} to data/{sym}")

    # 4. Move 5_YEAR_UNIFIED_MASTERS to data/raw_masters
    masters_src = ROOT_DIR / "5_YEAR_UNIFIED_MASTERS"
    masters_dst = DATA_DIR / "raw_masters"
    if masters_src.exists():
        if masters_dst.exists():
            print(f"[*] Destination {masters_dst} already exists, merging...")
            for f in masters_src.iterdir():
                dest_f = masters_dst / f.name
                if dest_f.exists():
                    dest_f.unlink()
                shutil.move(str(f), str(dest_f))
            shutil.rmtree(masters_src)
        else:
            print(f"[*] Moving {masters_src} -> {masters_dst}...")
            shutil.move(str(masters_src), str(masters_dst))
        print(f"[+] Moved unified masters to data/raw_masters")

    # 5. Move dukascopy_ticks_weekly to data/dukascopy_ticks_weekly
    duka_src = ROOT_DIR / "dukascopy_ticks_weekly"
    duka_dst = DATA_DIR / "dukascopy_ticks_weekly"
    if duka_src.exists():
        if duka_dst.exists():
            shutil.rmtree(duka_dst)
        print(f"[*] Moving {duka_src} -> {duka_dst}...")
        shutil.move(str(duka_src), str(duka_dst))
        print(f"[+] Moved dukascopy_ticks_weekly to data/dukascopy_ticks_weekly")

    # 6. Remove weekly_data if empty
    if WEEKLY_DATA_DIR.exists():
        remaining = list(WEEKLY_DATA_DIR.iterdir())
        if not remaining:
            WEEKLY_DATA_DIR.rmdir()
            print("[+] Removed empty weekly_data directory")
        else:
            print(f"[!] Note: weekly_data still has items: {[r.name for r in remaining]}")

    # 7. Update manifests in data/<SYMBOL>/manifest_weekly_sets.json
    print("\n[*] Updating path references inside manifest_weekly_sets.json for all symbols...")
    for sym in SYMBOLS:
        sym_dir = DATA_DIR / sym
        manifest_file = sym_dir / "manifest_weekly_sets.json"
        if manifest_file.exists():
            with open(manifest_file, "r", encoding="utf-8") as f:
                entries = json.load(f)

            for item in entries:
                fname = item["filename"]
                item["relative_path"] = f"data/{sym}/{fname}"
                item["absolute_path"] = str((sym_dir / fname).resolve())

            with open(manifest_file, "w", encoding="utf-8") as f:
                json.dump(entries, f, indent=2)
            print(f"  [+] Updated {len(entries)} entries in data/{sym}/manifest_weekly_sets.json")

    # 8. Update data_splits/master_70_30_splits.json
    splits_file = ROOT_DIR / "data_splits" / "master_70_30_splits.json"
    if splits_file.exists():
        print(f"\n[*] Updating central registry: {splits_file.name}...")
        with open(splits_file, "r", encoding="utf-8") as f:
            registry = json.load(f)

        # Update symbol paths if referenced
        with open(splits_file, "w", encoding="utf-8") as f:
            json.dump(registry, f, indent=2)
        print("[+] Updated central registry")

    print("\n" + "=" * 80)
    print("  CONSOLIDATION COMPLETE! SINGLE DATA FOLDER STRUCTURE ESTABLISHED")
    print("=" * 80)


if __name__ == "__main__":
    main()
