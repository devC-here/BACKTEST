#!/usr/bin/env python3
"""
Pre-caches all 300 AUDNZD weekly CSVs to contiguous .npy binary files
in tick_engine/data/processed_audnzd/week_NN.npy for ultra-high-speed mmap backtesting.
"""

import os
import sys
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pyarrow.csv as pcsv

ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT_DIR / "data" / "AUDNZD"
OUT_DIR = ROOT_DIR / "tick_engine" / "data" / "processed_audnzd"

def convert_single_week(w: int):
    candidates = list(RAW_DIR.glob(f"week_{w:02d}_*.csv"))
    if not candidates:
        return w, 0, "Not found"
    csv_file = candidates[0]
    target_npy = OUT_DIR / f"week_{w:02d}.npy"

    if target_npy.exists():
        return w, 0, "Already exists"

    table = pcsv.read_csv(csv_file)
    ts = table["timestamp"].to_numpy()
    bid = table["bidPrice"].to_numpy()
    ask = table["askPrice"].to_numpy()

    arr = np.empty(len(ts), dtype=[("timestamp", np.int64), ("bid", np.float64), ("ask", np.float64)])
    arr["timestamp"] = ts
    arr["bid"] = bid
    arr["ask"] = ask

    np.save(target_npy, arr)
    return w, len(ts), "OK"

def main():
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_files = list(RAW_DIR.glob("week_*.csv"))
    total = len(csv_files)
    print(f"[*] Pre-caching {total} AUDNZD weekly CSVs to {OUT_DIR}...")

    max_workers = min(os.cpu_count() or 4, 12)
    completed = 0
    total_ticks = 0

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(convert_single_week, w): w for w in range(1, total + 1)}
        for fut in as_completed(futures):
            w, n_ticks, status = fut.result()
            completed += 1
            total_ticks += n_ticks
            if completed % 50 == 0 or completed == total:
                elapsed = time.time() - t0
                print(f"  --> Progress: {completed}/{total} weeks cached ({completed/max(elapsed,0.01):.1f} w/s)...")

    print(f"[+] All {completed} weeks cached ({total_ticks:,} ticks) in {time.time()-t0:.1f}s!")

if __name__ == "__main__":
    main()

