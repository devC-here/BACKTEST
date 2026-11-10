#!/usr/bin/env python3
"""
================================================================================
  SLICE GOOGLE DRIVE EXPORT PARQUETS INTO STANDARDIZED WEEKLY TICK DATASETS
================================================================================
Processes:
  - EURGBP (2021, 2022, 2024 -> ~65.89M ticks)
  - AUDNZD (2024 -> ~17.66M ticks)

Output:
  - eurgbp_weekly_data/
      week_01_YYYY-MM-DD_to_YYYY-MM-DD.csv ...
      manifest_weekly_sets.json
  - audnzd_weekly_data/
      week_01_YYYY-MM-DD_to_YYYY-MM-DD.csv ...
      manifest_weekly_sets.json

Standard Header:
  timestamp,askPrice,bidPrice,askVolume,bidVolume
================================================================================
"""

from __future__ import annotations
import sys
import os
import time
import json
import shutil
import datetime
from pathlib import Path
import pyarrow.parquet as pq
import pyarrow.csv as pcsv
import pandas as pd
import numpy as np

sys.dont_write_bytecode = True

ROOT_DIR = Path(__file__).resolve().parent.parent
EXPORT_DIR = ROOT_DIR / "export_google_drive"


def slice_symbol_parquets(symbol: str, parquet_files: list[Path], output_dir: Path):
    t0 = time.time()
    print("=" * 80)
    print(f"  PROCESSING {symbol} ({len(parquet_files)} parquet files)")
    print("=" * 80)

    if output_dir.exists():
        print(f"[*] Removing previous output directory {output_dir.name}...")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_entries = []
    global_week_idx = 1
    total_symbol_ticks = 0

    # Process each year file sequentially to guarantee minimal RAM usage
    for pf in sorted(parquet_files):
        pf_size_mb = pf.stat().st_size / 1e6
        print(f"\n[*] Ingesting {pf.name} ({pf_size_mb:.1f} MB)...")
        
        t_load = time.time()
        table = pq.read_table(pf, columns=["timestamp_ms", "ask", "bid", "ask_volume", "bid_volume"])
        n_rows = table.num_rows
        print(f"[+] Loaded {n_rows:,} rows in {time.time() - t_load:.2f}s.")

        # Rename columns to standard repository naming
        table = table.rename_columns(["timestamp", "askPrice", "bidPrice", "askVolume", "bidVolume"])
        ts_arr = table["timestamp"].to_numpy()

        # Determine date range of this year
        min_dt = datetime.datetime.fromtimestamp(ts_arr[0] / 1000, tz=datetime.timezone.utc)
        max_dt = datetime.datetime.fromtimestamp(ts_arr[-1] / 1000, tz=datetime.timezone.utc)
        print(f"    Date range: {min_dt.strftime('%Y-%m-%d')} to {max_dt.strftime('%Y-%m-%d')}")

        # Build Monday-Friday schedule for this year
        # Find first Monday (or Sunday night) of this dataset
        start_monday = (min_dt - datetime.timedelta(days=min_dt.weekday())).date()
        end_date = max_dt.date()

        curr = start_monday
        year_weeks = 0
        year_ticks = 0

        while curr <= end_date:
            mon = curr
            fri = mon + datetime.timedelta(days=4)

            # Sunday 20:00 UTC (market open) to Friday 23:59:59.999 UTC
            sun_start_dt = datetime.datetime(mon.year, mon.month, mon.day, 0, 0, 0, tzinfo=datetime.timezone.utc) - datetime.timedelta(hours=4)
            fri_end_dt = datetime.datetime(fri.year, fri.month, fri.day, 23, 59, 59, 999000, tzinfo=datetime.timezone.utc)

            s_ms = int(sun_start_dt.timestamp() * 1000)
            e_ms = int(fri_end_dt.timestamp() * 1000)

            # Binary search bounds
            idx_start = int(np.searchsorted(ts_arr, s_ms, side="left"))
            idx_end = int(np.searchsorted(ts_arr, e_ms, side="right"))
            count = idx_end - idx_start

            if count > 0:
                filename = f"week_{global_week_idx:02d}_{mon.isoformat()}_to_{fri.isoformat()}.csv"
                out_csv = output_dir / filename

                # Zero-copy slice and high-speed C++ PyArrow CSV write
                sub_tbl = table.slice(idx_start, count)
                pcsv.write_csv(sub_tbl, str(out_csv))
                size_mb = out_csv.stat().st_size / (1024 * 1024)

                first_utc = datetime.datetime.fromtimestamp(ts_arr[idx_start] / 1000, tz=datetime.timezone.utc).strftime("%Y-%m-%d %H:%M")
                last_utc = datetime.datetime.fromtimestamp(ts_arr[idx_end - 1] / 1000, tz=datetime.timezone.utc).strftime("%Y-%m-%d %H:%M")

                manifest_entries.append({
                    "week": global_week_idx,
                    "filename": filename,
                    "relative_path": f"{output_dir.name}/{filename}",
                    "absolute_path": str(out_csv.resolve()),
                    "start_date": mon.isoformat(),
                    "end_date": fri.isoformat(),
                    "start_day": "Mon",
                    "end_day": "Fri",
                    "first_tick_utc": first_utc,
                    "last_tick_utc": last_utc,
                    "tick_count": count,
                    "size_mb": round(size_mb, 2)
                })

                year_weeks += 1
                year_ticks += count
                global_week_idx += 1

            curr += datetime.timedelta(days=7)

        total_symbol_ticks += year_ticks
        print(f"    Sliced {year_weeks} weeks ({year_ticks:,} ticks) from {pf.name}.")

    # Write manifest_weekly_sets.json
    manifest_path = output_dir / "manifest_weekly_sets.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_entries, f, indent=2)

    dt_total = time.time() - t0
    print("\n" + "-" * 60)
    print(f"SUMMARY FOR {symbol}:")
    print(f"  * Total Weeks Created: {len(manifest_entries)}")
    print(f"  * Total Ticks Written: {total_symbol_ticks:,}")
    print(f"  * Manifest:            {manifest_path.name}")
    print(f"  * Time Elapsed:        {dt_total:.1f}s")
    print("-" * 60)


def main():
    t_start = time.time()

    # 1. EURGBP
    eurgbp_files = sorted(list((EXPORT_DIR / "EURGBP").glob("*.parquet")))
    if eurgbp_files:
        slice_symbol_parquets("EURGBP", eurgbp_files, ROOT_DIR / "eurgbp_weekly_data")

    # 2. AUDNZD
    audnzd_files = sorted(list((EXPORT_DIR / "AUDNZD").glob("*.parquet")))
    if audnzd_files:
        slice_symbol_parquets("AUDNZD", audnzd_files, ROOT_DIR / "audnzd_weekly_data")

    print(f"\n[+] ALL PARQUET EXPORTS CONVERTED AND SLICED IN {time.time() - t_start:.1f}s TOTAL!")


if __name__ == "__main__":
    main()
