#!/usr/bin/env python3
"""
================================================================================
  ULTRA-FAST STREAMING SLICER: 5-YEAR UNIFIED MASTERS (EURGBP & USDJPY)
================================================================================
Ingests unified 5-year master parquets:
  - 5_YEAR_UNIFIED_MASTERS/eurgbp_2021_2026_master.parquet (85.8M ticks)
  - 5_YEAR_UNIFIED_MASTERS/usdjpy_2021_2026_master.parquet (192.3M ticks)

Actions:
  1. Removes earlier parts for matching symbols (e.g. existing EURGBP 157 weeks).
  2. Slices ticks into strict weekly CSV datasets (weekly_data/<SYMBOL>/week_NN_YYYY-MM-DD_to_YYYY-MM-DD.csv)
     using zero-copy row-group streaming (low memory footprint, C++ PyArrow CSV writer).
  3. Generates complete manifest_weekly_sets.json for each symbol.
  4. Applies the institutional 70/30 deterministic random partitioning (Seed 42).
  5. Generates split_70_30.json in each symbol directory.
  6. Creates NTFS directory junctions (eurgbp_weekly_data, usdjpy_weekly_data).
  7. Updates central registry data_splits/master_70_30_splits.json.
================================================================================
"""

from __future__ import annotations
import sys
import os
import time
import json
import random
import shutil
import subprocess
import datetime
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.csv as pcsv

sys.dont_write_bytecode = True

ROOT_DIR = Path(__file__).resolve().parent.parent
MASTERS_DIR = ROOT_DIR / "5_YEAR_UNIFIED_MASTERS"
WEEKLY_DATA_DIR = ROOT_DIR / "weekly_data"
SPLITS_DIR = ROOT_DIR / "data_splits"

SEED = 42

TARGETS = [
    {
        "symbol": "EURGBP",
        "parquet_file": MASTERS_DIR / "eurgbp_2021_2026_master.parquet",
        "junction_name": "eurgbp_weekly_data",
        "remove_earlier": True
    },
    {
        "symbol": "USDJPY",
        "parquet_file": MASTERS_DIR / "usdjpy_2021_2026_master.parquet",
        "junction_name": "usdjpy_weekly_data",
        "remove_earlier": True
    }
]


def build_schedule_from_bounds(first_ms: int, last_ms: int):
    """
    Builds the sequence of trading weeks from first tick to last tick.
    Week start: Sunday 20:00:00.000 UTC (market open).
    Week end:   Friday 23:59:59.999 UTC (market close).
    """
    first_dt = datetime.datetime.fromtimestamp(first_ms / 1000, tz=datetime.timezone.utc)
    last_dt = datetime.datetime.fromtimestamp(last_ms / 1000, tz=datetime.timezone.utc)

    # If first tick is on Sunday, Monday is the next day (+1)
    if first_dt.weekday() == 6:  # Sunday
        start_mon = (first_dt + datetime.timedelta(days=1)).date()
    else:
        start_mon = (first_dt - datetime.timedelta(days=first_dt.weekday())).date()

    end_date = last_dt.date()

    schedule = []
    curr = start_mon
    while curr <= end_date:
        mon = curr
        fri = mon + datetime.timedelta(days=4)

        # Sunday 20:00:00 UTC (4 hours before Monday 00:00 UTC)
        sun_start_dt = datetime.datetime(mon.year, mon.month, mon.day, 0, 0, 0, tzinfo=datetime.timezone.utc) - datetime.timedelta(hours=4)
        fri_end_dt = datetime.datetime(fri.year, fri.month, fri.day, 23, 59, 59, 999000, tzinfo=datetime.timezone.utc)

        schedule.append({
            "mon": mon,
            "fri": fri,
            "start_ms": int(sun_start_dt.timestamp() * 1000),
            "end_ms": int(fri_end_dt.timestamp() * 1000)
        })
        curr += datetime.timedelta(days=7)

    return schedule


def slice_master_parquet(symbol: str, parquet_path: Path, output_dir: Path):
    t_start = time.time()
    print("\n" + "=" * 80)
    print(f"  SLICING MASTER 5-YEAR PARQUET: {symbol}")
    print("=" * 80)
    print(f"Source: {parquet_path.resolve()} ({parquet_path.stat().st_size / (1024**2):.1f} MB)")
    print(f"Target: {output_dir.resolve()}")

    if not parquet_path.exists():
        print(f"[!] Error: Source file not found: {parquet_path}")
        return None

    # 1. Clean previous data
    if output_dir.exists():
        print(f"[*] Removing existing earlier parts in {output_dir.name}...")
        # Clear files inside without deleting directory to preserve potential junctions
        for item in output_dir.iterdir():
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)
    else:
        output_dir.mkdir(parents=True, exist_ok=True)

    pf = pq.ParquetFile(parquet_path)
    total_rows = pf.metadata.num_rows
    num_rg = pf.num_row_groups
    print(f"[*] Parquet metadata: {total_rows:,} rows across {num_rg} row groups.")

    col_idx = pf.schema.names.index("timestamp_ms")
    first_ms = pf.metadata.row_group(0).column(col_idx).statistics.min
    last_ms = pf.metadata.row_group(num_rg - 1).column(col_idx).statistics.max

    schedule = build_schedule_from_bounds(first_ms, last_ms)
    n_sched = len(schedule)
    print(f"[*] Built weekly schedule: {n_sched} potential calendar weeks.")

    # State tracking
    sched_idx = 0
    active_week_num = 1
    manifest_entries = []
    
    # State for the currently open week
    cur_week_info = None
    cur_out_file = None
    cur_tick_count = 0
    cur_first_tick_ms = None
    cur_last_tick_ms = None

    def close_current_week():
        nonlocal cur_week_info, cur_out_file, cur_tick_count, cur_first_tick_ms, cur_last_tick_ms, active_week_num
        if cur_out_file is not None and cur_tick_count > 0:
            size_mb = cur_out_file.stat().st_size / (1024 * 1024) if cur_out_file.exists() else 0.0
            first_utc = datetime.datetime.fromtimestamp(cur_first_tick_ms / 1000, tz=datetime.timezone.utc).strftime("%Y-%m-%d %H:%M")
            last_utc = datetime.datetime.fromtimestamp(cur_last_tick_ms / 1000, tz=datetime.timezone.utc).strftime("%Y-%m-%d %H:%M")

            entry = {
                "week": active_week_num,
                "filename": cur_out_file.name,
                "relative_path": f"weekly_data/{symbol}/{cur_out_file.name}",
                "absolute_path": str(cur_out_file.resolve()),
                "start_date": cur_week_info["mon"].isoformat(),
                "end_date": cur_week_info["fri"].isoformat(),
                "start_day": "Mon",
                "end_day": "Fri",
                "first_tick_utc": first_utc,
                "last_tick_utc": last_utc,
                "tick_count": cur_tick_count,
                "size_mb": round(size_mb, 2)
            }
            manifest_entries.append(entry)
            
            if active_week_num % 25 == 0 or active_week_num == 1:
                print(f"  [+] Week {active_week_num:03d}: {cur_tick_count:,} ticks ({size_mb:.1f} MB) -> {cur_out_file.name}")
                
            active_week_num += 1

        cur_week_info = None
        cur_out_file = None
        cur_tick_count = 0
        cur_first_tick_ms = None
        cur_last_tick_ms = None

    total_ticks_written = 0

    # Stream through row groups
    for rg_idx in range(num_rg):
        tbl = pf.read_row_group(
            rg_idx,
            columns=["timestamp_ms", "ask", "bid", "ask_volume", "bid_volume"]
        )
        tbl = tbl.rename_columns(["timestamp", "askPrice", "bidPrice", "askVolume", "bidVolume"])
        
        ts_arr = tbl["timestamp"].to_numpy()
        rg_min_ms = ts_arr[0]
        rg_max_ms = ts_arr[-1]

        # Advance schedule if this row group is past previous weeks
        while sched_idx < n_sched and schedule[sched_idx]["end_ms"] < rg_min_ms:
            close_current_week()
            sched_idx += 1

        if sched_idx >= n_sched:
            break

        # Process all weeks that intersect with this row group
        while sched_idx < n_sched and schedule[sched_idx]["start_ms"] <= rg_max_ms:
            w_info = schedule[sched_idx]
            s_ms = w_info["start_ms"]
            e_ms = w_info["end_ms"]

            # Locate slice in current table
            i_start = int(np.searchsorted(ts_arr, s_ms, side="left"))
            i_end = int(np.searchsorted(ts_arr, e_ms, side="right"))
            slice_count = i_end - i_start

            if slice_count > 0:
                sub_tbl = tbl.slice(i_start, slice_count)
                
                if cur_out_file is None:
                    # Initialize new week file
                    cur_week_info = w_info
                    filename = f"week_{active_week_num:02d}_{w_info['mon'].isoformat()}_to_{w_info['fri'].isoformat()}.csv"
                    cur_out_file = output_dir / filename
                    
                    with open(cur_out_file, "wb") as f_out:
                        pcsv.write_csv(sub_tbl, f_out, write_options=pcsv.WriteOptions(include_header=True))
                    cur_first_tick_ms = ts_arr[i_start]
                else:
                    # Append to existing open week file
                    with open(cur_out_file, "ab") as f_out:
                        pcsv.write_csv(sub_tbl, f_out, write_options=pcsv.WriteOptions(include_header=False))

                cur_last_tick_ms = ts_arr[i_end - 1]
                cur_tick_count += slice_count
                total_ticks_written += slice_count

            # If this row group reaches or passes the end of this week, close it
            if rg_max_ms >= e_ms:
                close_current_week()
                sched_idx += 1
            else:
                # The week extends into subsequent row groups
                break

        if (rg_idx + 1) % 50 == 0 or rg_idx + 1 == num_rg:
            elapsed = time.time() - t_start
            rate = total_ticks_written / max(elapsed, 0.001)
            print(f"  --> Progress: Row Group {rg_idx+1:03d}/{num_rg} | {total_ticks_written:,} ticks written ({rate:,.0f} ticks/s)...")

    # Close any remaining week
    close_current_week()

    # 2. Write manifest_weekly_sets.json
    manifest_path = output_dir / "manifest_weekly_sets.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_entries, f, indent=2)

    elapsed_total = time.time() - t_start
    print("-" * 60)
    print(f"SLICING COMPLETE FOR {symbol}:")
    print(f"  * Total Active Weeks Sliced: {len(manifest_entries)}")
    print(f"  * Total Ticks Written:       {total_ticks_written:,}")
    print(f"  * Output Directory:          {output_dir.name}")
    print(f"  * Total Elapsed Time:        {elapsed_total:.1f}s")
    print("-" * 60)

    return manifest_entries


def partition_symbol_70_30(symbol: str, manifest: list[dict], output_dir: Path):
    """Applies the deterministic 70/30 split standard (Seed 42)."""
    print(f"[*] Applying 70/30 deterministic partition on {symbol} (Seed {SEED})...")
    total_weeks = len(manifest)
    weeks = [item["week"] for item in manifest]

    rng = random.Random(SEED)
    shuffled_weeks = weeks.copy()
    rng.shuffle(shuffled_weeks)

    n_discovery = int(round(total_weeks * 0.70))
    discovery_weeks = sorted(shuffled_weeks[:n_discovery])
    test_weeks = sorted(shuffled_weeks[n_discovery:])

    discovery_set = set(discovery_weeks)
    test_set = set(test_weeks)

    disc_ticks = 0
    test_ticks = 0

    for item in manifest:
        w = item["week"]
        if w in discovery_set:
            item["split"] = "discovery_70"
            item["split_category"] = "discovery"
            item["split_role"] = "Analysis, alpha discovery, and calibration"
            disc_ticks += item.get("tick_count", 0)
        else:
            item["split"] = "holdout_test_30"
            item["split_category"] = "holdout_test"
            item["split_role"] = "Quarantined out-of-sample test only"
            test_ticks += item.get("tick_count", 0)

    # Re-save manifest with split metadata
    manifest_path = output_dir / "manifest_weekly_sets.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # Save split_70_30.json
    split_info = {
        "symbol": symbol,
        "seed": SEED,
        "total_weeks": total_weeks,
        "discovery_70": {
            "count": len(discovery_weeks),
            "percentage": round(len(discovery_weeks) / total_weeks * 100, 2),
            "total_ticks": disc_ticks,
            "weeks": discovery_weeks
        },
        "holdout_test_30": {
            "count": len(test_weeks),
            "percentage": round(len(test_weeks) / total_weeks * 100, 2),
            "total_ticks": test_ticks,
            "weeks": test_weeks
        }
    }

    split_file = output_dir / "split_70_30.json"
    with open(split_file, "w", encoding="utf-8") as f:
        json.dump(split_info, f, indent=2)

    print(f"  [+] {symbol}: {total_weeks} weeks -> {len(discovery_weeks)} Discovery ({disc_ticks:,} ticks) | {len(test_weeks)} Holdout ({test_ticks:,} ticks)")
    return split_info


def ensure_junction(junction_name: str, target_dir: Path):
    """Creates NTFS directory junction for backward compatibility."""
    junction_path = ROOT_DIR / junction_name
    if not junction_path.exists():
        print(f"[*] Creating NTFS junction: {junction_name} -> {target_dir}...")
        cmd = f'cmd /c mklink /J "{junction_path}" "{target_dir}"'
        res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        print(f"    {res.stdout.strip()}")
    else:
        print(f"[*] Junction {junction_name} already exists.")


def update_master_registry():
    """Consolidates all asset splits into master_70_30_splits.json."""
    print("\n[*] Updating central registry: data_splits/master_70_30_splits.json...")
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    
    symbols = ["EURUSD", "EURGBP", "USDJPY", "AUDNZD"]
    master_summary = {
        "standard": "70/30 Randomized Week Partitioning",
        "policy": "Discovery partition strictly for research; holdout partition strictly quarantined for out-of-sample forward verification",
        "seed": SEED,
        "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "assets": {}
    }

    total_fleet_weeks = 0
    total_fleet_disc_weeks = 0
    total_fleet_test_weeks = 0
    total_fleet_ticks = 0
    total_fleet_disc_ticks = 0
    total_fleet_test_ticks = 0

    for sym in symbols:
        split_file = WEEKLY_DATA_DIR / sym / "split_70_30.json"
        if split_file.exists():
            with open(split_file, "r", encoding="utf-8") as f:
                s_data = json.load(f)
            master_summary["assets"][sym] = s_data
            
            total_fleet_weeks += s_data["total_weeks"]
            total_fleet_disc_weeks += s_data["discovery_70"]["count"]
            total_fleet_test_weeks += s_data["holdout_test_30"]["count"]
            total_fleet_ticks += s_data["discovery_70"]["total_ticks"] + s_data["holdout_test_30"]["total_ticks"]
            total_fleet_disc_ticks += s_data["discovery_70"]["total_ticks"]
            total_fleet_test_ticks += s_data["holdout_test_30"]["total_ticks"]

    master_summary["fleet_totals"] = {
        "total_weeks": total_fleet_weeks,
        "discovery_weeks": total_fleet_disc_weeks,
        "holdout_test_weeks": total_fleet_test_weeks,
        "total_ticks": total_fleet_ticks,
        "discovery_ticks": total_fleet_disc_ticks,
        "holdout_test_ticks": total_fleet_test_ticks
    }

    master_file = SPLITS_DIR / "master_70_30_splits.json"
    with open(master_file, "w", encoding="utf-8") as f:
        json.dump(master_summary, f, indent=2)
    print(f"[+] Master registry saved to {master_file.resolve()}.")
    print(f"    Fleet Total: {total_fleet_weeks} weeks, {total_fleet_ticks:,} ticks across {len(master_summary['assets'])} symbols.")


def main():
    t_global = time.time()
    print("=" * 80)
    print("  EXECUTING 5-YEAR UNIFIED MASTERS INGESTION & ARRANGEMENT PIPELINE")
    print("=" * 80)

    WEEKLY_DATA_DIR.mkdir(parents=True, exist_ok=True)

    for item in TARGETS:
        symbol = item["symbol"]
        parquet_file = item["parquet_file"]
        junction_name = item["junction_name"]
        output_dir = WEEKLY_DATA_DIR / symbol

        manifest = slice_master_parquet(symbol, parquet_file, output_dir)
        if manifest:
            partition_symbol_70_30(symbol, manifest, output_dir)
            ensure_junction(junction_name, output_dir)

    update_master_registry()

    print("\n" + "=" * 80)
    print(f"  ALL 5-YEAR MASTERS INGESTED, ARRANGED, AND PARTITIONED IN {time.time() - t_global:.1f}s!")
    print("=" * 80)


if __name__ == "__main__":
    main()
