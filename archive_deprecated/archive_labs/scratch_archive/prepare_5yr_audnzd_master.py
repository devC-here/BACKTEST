#!/usr/bin/env python3
"""
================================================================================
  ULTRA-FAST STREAMING SLICER: 5-YEAR UNIFIED MASTER (AUDNZD)
================================================================================
Ingests unified 5-year master parquet:
  - 5_YEAR_UNIFIED_MASTERS/audnzd_2021_2026_master.parquet (139.0M ticks / 300 weeks)

Pipeline:
  1. Safely archives legacy 53-week AUDNZD data to data/legacy_archive/AUDNZD_53w_2024.
  2. Slices ticks into strict weekly CSV datasets (data/AUDNZD/week_NN_YYYY-MM-DD_to_YYYY-MM-DD.csv)
     using zero-copy row-group streaming (low memory footprint, C++ PyArrow CSV writer).
  3. Generates complete manifest_weekly_sets.json for AUDNZD.
  4. Applies institutional 70/30 deterministic random partitioning (Seed 42).
  5. Generates split_70_30.json in data/AUDNZD/.
  6. Creates NTFS directory junction audnzd_weekly_data -> data/AUDNZD.
  7. Updates central registry data_splits/master_70_30_splits.json with full multi-asset fleet totals.
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
DATA_DIR = ROOT_DIR / "data"
SPLITS_DIR = ROOT_DIR / "data_splits"

SEED = 42

AUDNZD_PARQUET = MASTERS_DIR / "audnzd_2021_2026_master.parquet"
OUTPUT_DIR = DATA_DIR / "AUDNZD"
JUNCTION_NAME = "audnzd_weekly_data"


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


def archive_legacy_data():
    """Safely moves existing 53-week AUDNZD data to legacy archive."""
    if OUTPUT_DIR.exists():
        legacy_archive = DATA_DIR / "legacy_archive" / "AUDNZD_53w_2024"
        legacy_archive.parent.mkdir(parents=True, exist_ok=True)
        if not legacy_archive.exists():
            print(f"[*] Archiving existing legacy AUDNZD data to {legacy_archive}...")
            shutil.copytree(OUTPUT_DIR, legacy_archive)
            print("    Legacy data successfully archived.")
        
        # Purge files in OUTPUT_DIR
        print(f"[*] Clearing existing files in {OUTPUT_DIR.resolve()} for 5-year master ingestion...")
        for item in OUTPUT_DIR.iterdir():
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)
    else:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def slice_audnzd_master():
    t_start = time.time()
    print("\n" + "=" * 80)
    print("  SLICING MASTER 5-YEAR PARQUET: AUDNZD")
    print("=" * 80)
    print(f"Source: {AUDNZD_PARQUET.resolve()} ({AUDNZD_PARQUET.stat().st_size / (1024**2):.1f} MB)")
    print(f"Target: {OUTPUT_DIR.resolve()}")

    if not AUDNZD_PARQUET.exists():
        print(f"[!] Error: Source file not found: {AUDNZD_PARQUET}")
        return None

    archive_legacy_data()

    pf = pq.ParquetFile(AUDNZD_PARQUET)
    total_rows = pf.metadata.num_rows
    num_rg = pf.num_row_groups
    print(f"[*] Parquet metadata: {total_rows:,} rows across {num_rg} row groups.")

    col_idx = pf.schema.names.index("timestamp_ms")
    first_ms = pf.metadata.row_group(0).column(col_idx).statistics.min
    last_ms = pf.metadata.row_group(num_rg - 1).column(col_idx).statistics.max

    schedule = build_schedule_from_bounds(first_ms, last_ms)
    n_sched = len(schedule)
    print(f"[*] Built weekly schedule: {n_sched} potential calendar weeks.")

    sched_idx = 0
    active_week_num = 1
    manifest_entries = []

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
                "relative_path": f"data/AUDNZD/{cur_out_file.name}",
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
                    cur_week_info = w_info
                    filename = f"week_{active_week_num:02d}_{w_info['mon'].isoformat()}_to_{w_info['fri'].isoformat()}.csv"
                    cur_out_file = OUTPUT_DIR / filename

                    with open(cur_out_file, "wb") as f_out:
                        pcsv.write_csv(sub_tbl, f_out, write_options=pcsv.WriteOptions(include_header=True))
                    cur_first_tick_ms = ts_arr[i_start]
                else:
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
                break

        if (rg_idx + 1) % 30 == 0 or rg_idx + 1 == num_rg:
            elapsed = time.time() - t_start
            rate = total_ticks_written / max(elapsed, 0.001)
            print(f"  --> Progress: Row Group {rg_idx+1:03d}/{num_rg} | {total_ticks_written:,} ticks written ({rate:,.0f} ticks/s)...")

    # Close any remaining week
    close_current_week()

    # Write manifest_weekly_sets.json
    manifest_path = OUTPUT_DIR / "manifest_weekly_sets.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_entries, f, indent=2)

    elapsed_total = time.time() - t_start
    print("-" * 60)
    print(f"SLICING COMPLETE FOR AUDNZD:")
    print(f"  * Total Active Weeks Sliced: {len(manifest_entries)}")
    print(f"  * Total Ticks Written:       {total_ticks_written:,}")
    print(f"  * Output Directory:          {OUTPUT_DIR.name}")
    print(f"  * Total Elapsed Time:        {elapsed_total:.1f}s")
    print("-" * 60)

    return manifest_entries


def partition_audnzd_70_30(manifest: list[dict]):
    """Applies the deterministic 70/30 split standard (Seed 42)."""
    print(f"\n[*] Applying 70/30 deterministic partition on AUDNZD (Seed {SEED})...")
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
    manifest_path = OUTPUT_DIR / "manifest_weekly_sets.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # Save split_70_30.json
    split_info = {
        "symbol": "AUDNZD",
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

    split_file = OUTPUT_DIR / "split_70_30.json"
    with open(split_file, "w", encoding="utf-8") as f:
        json.dump(split_info, f, indent=2)

    print(f"  [+] AUDNZD: {total_weeks} weeks -> {len(discovery_weeks)} Discovery ({disc_ticks:,} ticks) | {len(test_weeks)} Holdout ({test_ticks:,} ticks)")
    return split_info


def ensure_junction():
    """Creates NTFS directory junction audnzd_weekly_data -> data/AUDNZD."""
    junction_path = ROOT_DIR / JUNCTION_NAME
    if not junction_path.exists():
        print(f"[*] Creating NTFS junction: {JUNCTION_NAME} -> {OUTPUT_DIR}...")
        cmd = f'cmd /c mklink /J "{junction_path}" "{OUTPUT_DIR}"'
        res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        print(f"    {res.stdout.strip()}")
    else:
        print(f"[*] Junction {JUNCTION_NAME} already exists.")


def update_master_registry():
    """Consolidates all asset splits into data_splits/master_70_30_splits.json."""
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
        split_file = DATA_DIR / sym / "split_70_30.json"
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
    t0 = time.time()
    manifest = slice_audnzd_master()
    if manifest:
        partition_audnzd_70_30(manifest)
        ensure_junction()
        update_master_registry()
    print(f"\n[✓] ALL OPERATIONS COMPLETED IN {time.time() - t0:.1f} SECONDS!")


if __name__ == "__main__":
    main()

