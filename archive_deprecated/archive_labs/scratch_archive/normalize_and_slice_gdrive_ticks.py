#!/usr/bin/env python3
"""
================================================================================
  NORMALIZE & SLICE GOOGLE DRIVE EXPORT TICKS (EURGBP & AUDNZD)
================================================================================
1. Reads raw parquet exports from export_google_drive (EURGBP & AUDNZD)
2. Normalizes schema to standard Dukascopy format matching eurusd_weekly_data:
     Columns: timestamp,askPrice,bidPrice,askVolume,bidVolume
     timestamp: Unix epoch in milliseconds (int64)
     askPrice, bidPrice: float (5 decimal places)
     askVolume, bidVolume: float (2 decimal places)
3. Ensures strict monotonic chronological ordering and deduplication.
4. Partitions into strict calendar trading weeks:
     Filename: week_{idx:02d}_{start_date}_to_{end_date}.csv
5. Generates manifest_weekly_sets.json with full tick counts, sizes, and date spans.
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
import polars as pl
import pandas as pd
import numpy as np

sys.dont_write_bytecode = True

ROOT_DIR = Path(__file__).resolve().parent.parent
EXPORT_DIR = ROOT_DIR / "export_google_drive"
HEADER = "timestamp,askPrice,bidPrice,askVolume,bidVolume\n"


def process_symbol(symbol: str, parquet_files: list[Path], output_dir: Path):
    print("=" * 80)
    print(f"  PROCESSING {symbol}: {len(parquet_files)} parquet files")
    print("=" * 80)
    
    if output_dir.exists():
        print(f"[*] Cleaning output directory {output_dir.name}...")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load and normalize all parquets for this symbol
    dfs = []
    total_raw_rows = 0
    t0 = time.time()
    
    for pf in sorted(parquet_files):
        print(f"[*] Scanning {pf.name} ({pf.stat().st_size / 1e6:.1f} MB)...")
        # Load via Polars for ultra-fast ingestion
        df = pl.read_parquet(pf).select([
            pl.col("timestamp_ms").cast(pl.Int64).alias("timestamp"),
            pl.col("ask").cast(pl.Float64).alias("askPrice"),
            pl.col("bid").cast(pl.Float64).alias("bidPrice"),
            pl.col("ask_volume").cast(pl.Float64).alias("askVolume"),
            pl.col("bid_volume").cast(pl.Float64).alias("bidVolume"),
        ])
        total_raw_rows += len(df)
        dfs.append(df)
        
    print(f"[*] Combining datasets (total raw ticks: {total_raw_rows:,})...")
    combined_df = pl.concat(dfs)
    
    # Sort by timestamp ascending and deduplicate
    print("[*] Sorting and verifying data integrity...")
    combined_df = combined_df.sort("timestamp").unique(subset=["timestamp", "askPrice", "bidPrice"], keep="first")
    n_clean_rows = len(combined_df)
    print(f"[+] Clean, deduplicated ticks: {n_clean_rows:,} (deduped {total_raw_rows - n_clean_rows:,} ticks)")
    
    # 2. Extract min and max dates
    min_ms = combined_df["timestamp"].min()
    max_ms = combined_df["timestamp"].max()
    min_dt = datetime.datetime.fromtimestamp(min_ms / 1000, tz=datetime.timezone.utc)
    max_dt = datetime.datetime.fromtimestamp(max_ms / 1000, tz=datetime.timezone.utc)
    print(f"[*] Complete span: {min_dt.strftime('%Y-%m-%d %H:%M:%S')} UTC to {max_dt.strftime('%Y-%m-%d %H:%M:%S')} UTC")
    
    # 3. Build weekly calendar segments
    # Find the Monday of min_dt's week
    start_monday = (min_dt - datetime.timedelta(days=min_dt.weekday())).date()
    end_date = max_dt.date()
    
    schedule = []
    curr = start_monday
    week_idx = 1
    
    while curr <= end_date:
        mon = curr
        fri = mon + datetime.timedelta(days=4)
        
        # Include Sunday open (Sunday 20:00 UTC) to Friday 23:59:59.999 UTC
        # Sunday 20:00 UTC prior to Monday
        sun_start_dt = datetime.datetime(mon.year, mon.month, mon.day, 0, 0, 0, tzinfo=datetime.timezone.utc) - datetime.timedelta(hours=4)
        fri_end_dt = datetime.datetime(fri.year, fri.month, fri.day, 23, 59, 59, 999000, tzinfo=datetime.timezone.utc)
        
        start_ms = int(sun_start_dt.timestamp() * 1000)
        end_ms = int(fri_end_dt.timestamp() * 1000)
        
        filename = f"week_{week_idx:02d}_{mon.isoformat()}_to_{fri.isoformat()}.csv"
        schedule.append({
            "week": week_idx,
            "filename": filename,
            "mon_date": mon.isoformat(),
            "fri_date": fri.isoformat(),
            "start_ms": start_ms,
            "end_ms": end_ms
        })
        
        curr += datetime.timedelta(days=7)
        week_idx += 1
        
    print(f"[*] Generated {len(schedule)} weekly calendar slots. Slicing into standard CSVs...")
    
    manifest_entries = []
    sliced_weeks = 0
    total_written_ticks = 0
    
    # Convert Polars DataFrame to Pandas/NumPy for fast slicing
    ts_arr = combined_df["timestamp"].to_numpy()
    ask_arr = combined_df["askPrice"].to_numpy()
    bid_arr = combined_df["bidPrice"].to_numpy()
    ask_vol_arr = combined_df["askVolume"].to_numpy()
    bid_vol_arr = combined_df["bidVolume"].to_numpy()
    
    for sched in schedule:
        w_num = sched["week"]
        s_ms = sched["start_ms"]
        e_ms = sched["end_ms"]
        
        # Binary search slice boundaries
        idx_start = int(np.searchsorted(ts_arr, s_ms, side="left"))
        idx_end = int(np.searchsorted(ts_arr, e_ms, side="right"))
        
        cnt = idx_end - idx_start
        if cnt == 0:
            # Empty week (e.g. gap year 2023 for EURGBP)
            continue
            
        sliced_weeks += 1
        out_csv = output_dir / sched["filename"]
        
        # Build DataFrame and write CSV
        sub_df = pd.DataFrame({
            "timestamp": ts_arr[idx_start:idx_end],
            "askPrice": np.round(ask_arr[idx_start:idx_end], 5),
            "bidPrice": np.round(bid_arr[idx_start:idx_end], 5),
            "askVolume": np.round(ask_vol_arr[idx_start:idx_end], 2),
            "bidVolume": np.round(bid_vol_arr[idx_start:idx_end], 2),
        })
        
        sub_df.to_csv(out_csv, index=False)
        total_written_ticks += cnt
        
        size_mb = out_csv.stat().st_size / (1024 * 1024)
        first_utc = datetime.datetime.fromtimestamp(ts_arr[idx_start] / 1000, tz=datetime.timezone.utc).strftime("%Y-%m-%d %H:%M")
        last_utc = datetime.datetime.fromtimestamp(ts_arr[idx_end - 1] / 1000, tz=datetime.timezone.utc).strftime("%Y-%m-%d %H:%M")
        
        manifest_entries.append({
            "week": w_num,
            "filename": sched["filename"],
            "relative_path": f"{output_dir.name}/{sched['filename']}",
            "absolute_path": str(out_csv.resolve()),
            "start_date": sched["mon_date"],
            "end_date": sched["fri_date"],
            "start_day": "Mon",
            "end_day": "Fri",
            "first_tick_utc": first_utc,
            "last_tick_utc": last_utc,
            "tick_count": cnt,
            "size_mb": round(size_mb, 2)
        })
        
        if sliced_weeks % 10 == 0 or sliced_weeks == 1:
            print(f"  -> Week {w_num:03d}: {sched['filename']} | {cnt:,} ticks | {size_mb:.2f} MB")
            
    # Write manifest_weekly_sets.json
    manifest_path = output_dir / "manifest_weekly_sets.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_entries, f, indent=2)
        
    total_dt = time.time() - t0
    print(f"\n[+] {symbol} COMPLETE:")
    print(f"  * Generated Weeks:   {sliced_weeks} weekly files")
    print(f"  * Written Ticks:     {total_written_ticks:,}")
    print(f"  * Output Directory:  {output_dir.resolve()}")
    print(f"  * Manifest File:     {manifest_path.name}")
    print(f"  * Elapsed Time:      {total_dt:.1f}s")


def main():
    t_global = time.time()
    
    # 1. Process EURGBP
    eurgbp_files = list((EXPORT_DIR / "EURGBP").glob("*.parquet"))
    if eurgbp_files:
        process_symbol("EURGBP", eurgbp_files, ROOT_DIR / "eurgbp_weekly_data")
        
    # 2. Process AUDNZD
    audnzd_files = list((EXPORT_DIR / "AUDNZD").glob("*.parquet"))
    if audnzd_files:
        process_symbol("AUDNZD", audnzd_files, ROOT_DIR / "audnzd_weekly_data")
        
    print("\n" + "=" * 80)
    print(f"  ALL SYMBOLS SUCCESSFULLY NORMALIZED AND PARTITIONED IN {time.time() - t_global:.1f}s")
    print("=" * 80)


if __name__ == "__main__":
    main()
