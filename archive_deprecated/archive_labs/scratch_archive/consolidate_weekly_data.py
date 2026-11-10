#!/usr/bin/env python3
"""
Consolidate weekly data folders under weekly_data/<SYMBOL>/
and create backward-compatibility junctions.
"""

import os
import sys
import json
import shutil
import subprocess
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
WEEKLY_DATA_DIR = ROOT_DIR / "weekly_data"

FOLDER_MAP = {
    "eurusd_weekly_data": "EURUSD",
    "eurgbp_weekly_data": "EURGBP",
    "audnzd_weekly_data": "AUDNZD"
}

def consolidate():
    print(f"[*] Ensuring {WEEKLY_DATA_DIR} exists...")
    WEEKLY_DATA_DIR.mkdir(parents=True, exist_ok=True)

    for old_name, symbol in FOLDER_MAP.items():
        src = ROOT_DIR / old_name
        dest = WEEKLY_DATA_DIR / symbol

        if src.exists() and not src.is_symlink():
            print(f"[*] Moving {old_name} -> weekly_data/{symbol}...")
            # If dest already exists, remove it first
            if dest.exists():
                shutil.rmtree(dest)
            shutil.move(str(src), str(dest))
            print(f"[+] Moved {old_name} to {dest}.")

        # Update manifest_weekly_sets.json inside dest
        manifest_file = dest / "manifest_weekly_sets.json"
        if manifest_file.exists():
            print(f"[*] Updating manifest in {dest.name}...")
            with open(manifest_file, "r", encoding="utf-8") as f:
                entries = json.load(f)

            for item in entries:
                fname = item["filename"]
                item["relative_path"] = f"weekly_data/{symbol}/{fname}"
                item["absolute_path"] = str((dest / fname).resolve())

            with open(manifest_file, "w", encoding="utf-8") as f:
                json.dump(entries, f, indent=2)
            print(f"[+] Updated {len(entries)} entries in {manifest_file.name}.")

        # Create backward-compatibility junction: ROOT_DIR / old_name -> dest
        junction_path = ROOT_DIR / old_name
        if not junction_path.exists():
            print(f"[*] Creating NTFS junction: {old_name} -> weekly_data\\{symbol}...")
            cmd = f'cmd /c mklink /J "{junction_path}" "{dest}"'
            res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if res.returncode == 0:
                print(f"[+] Junction created: {res.stdout.strip()}")
            else:
                print(f"[-] Junction creation failed: {res.stderr.strip()}")

    # Safely remove export_google_drive as explicitly confirmed by user
    export_gd = ROOT_DIR / "export_google_drive"
    if export_gd.exists():
        print(f"\n[*] Removing raw data folder {export_gd.name} as confirmed...")
        shutil.rmtree(export_gd)
        print(f"[+] Successfully deleted {export_gd.name}.")
    else:
        print(f"[*] {export_gd.name} already removed.")

    print("\n[+] Consolidation and cleanup complete!")

if __name__ == "__main__":
    consolidate()
