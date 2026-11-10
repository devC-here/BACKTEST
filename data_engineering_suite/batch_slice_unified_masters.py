#!/usr/bin/env python3
"""
================================================================================
  BATCH SLICER & INGESTION ENGINE FOR 5-YEAR UNIFIED MASTERS
================================================================================
Slices unified 5-year master parquets from 5_YEAR_UNIFIED_MASTERS into:
  - data/<SYMBOL>/week_NN_YYYY-MM-DD_to_YYYY-MM-DD.csv
  - data/<SYMBOL>/manifest_weekly_sets.json
  - data/<SYMBOL>/split_70_30.json (Seed 42)
  - data_splits/master_70_30_splits.json

Standardizes schema to:
  [timestamp, askPrice, bidPrice, askVolume, bidVolume]

Supports all 13 symbols:
  EURUSD, USDJPY, EURGBP, AUDNZD, AUDUSD, AUDCAD, USDCAD,
  AUDJPY, GBPUSD, USDCHF, EURJPY, GBPJPY, USATECHIDXUSD
================================================================================
"""

from __future__ import annotations
import sys
import os
import time
import json
import random
import shutil
import datetime
import argparse
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.csv as pcsv

ROOT_DIR = Path(__file__).resolve().parent.parent
MASTERS_DIR = ROOT_DIR / "5_YEAR_UNIFIED_MASTERS"
DATA_DIR = ROOT_DIR / "data"
SPLITS_DIR = ROOT_DIR / "data_splits"

SEED = 42

ALL_SYMBOLS = [
    "EURUSD",
    "USDJPY",
    "EURGBP",
    "AUDNZD",
    "AUDUSD",
    "AUDCAD",
    "USDCAD",
    "AUDJPY",
    "GBPUSD",
    "USDCHF",
    "EURJPY",
    "GBPJPY",
    "USATECHIDXUSD",
    "EURNZD",
    "GBPCHF",
    "NZDCAD",
]


def build_schedule_from_bounds(first_ms: int, last_ms: int):
    """
    Builds the sequence of trading weeks from first tick to last tick.
    Week start: Sunday 20:00:00.000 UTC (market open).
    Week end:   Friday 23:59:59.999 UTC (market close).
    """
    first_dt = datetime.datetime.fromtimestamp(first_ms / 1000, tz=datetime.timezone.utc)
    last_dt = datetime.datetime.fromtimestamp(last_ms / 1000, tz=datetime.timezone.utc)

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


def normalize_arrow_table(tbl: pa.Table) -> pa.Table:
    cols = tbl.column_names

    # timestamp
    if "timestamp_ms" in cols:
        ts_col = tbl["timestamp_ms"]
    elif "timestamp" in cols:
        ts_col = tbl["timestamp"]
    else:
        raise ValueError("Missing timestamp column")

    # ask
    if "ask" in cols:
        ask_col = tbl["ask"]
    elif "askPrice" in cols:
        ask_col = tbl["askPrice"]
    else:
        raise ValueError("Missing ask column")

    # bid
    if "bid" in cols:
        bid_col = tbl["bid"]
    elif "bidPrice" in cols:
        bid_col = tbl["bidPrice"]
    else:
        raise ValueError("Missing bid column")

    # ask volume
    if "ask_volume" in cols:
        ask_vol = tbl["ask_volume"]
    elif "askVolume" in cols:
        ask_vol = tbl["askVolume"]
    else:
        ask_vol = pa.repeat(1.0, len(tbl))

    # bid volume
    if "bid_volume" in cols:
        bid_vol = tbl["bid_volume"]
    elif "bidVolume" in cols:
        bid_vol = tbl["bidVolume"]
    else:
        bid_vol = pa.repeat(1.0, len(tbl))

    return pa.Table.from_arrays(
        [ts_col, ask_col, bid_col, ask_vol, bid_vol],
        names=["timestamp", "askPrice", "bidPrice", "askVolume", "bidVolume"]
    )


def partition_symbol_70_30(symbol: str, manifest: list[dict], output_dir: Path):
    """Applies deterministic 70/30 split standard (Seed 42)."""
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

    # Save updated manifest
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

    return split_info


def slice_symbol_master(symbol: str, source_dir: Optional[Path] = None, replace_existing: bool = True):
    t_start = time.time()
    symbol = symbol.upper()

    master_file = None
    if source_dir:
        master_file = Path(source_dir) / f"{symbol.lower()}_2021_2026_master.parquet"
    else:
        candidates = [
            ROOT_DIR / "FOREX_TICKS_GBPCHF_EURNZD_NZDCAD" / f"{symbol.lower()}_2021_2026_master.parquet",
            MASTERS_DIR / f"{symbol.lower()}_2021_2026_master.parquet",
        ]
        for c in candidates:
            if c.exists():
                master_file = c
                break
        if master_file is None:
            master_file = MASTERS_DIR / f"{symbol.lower()}_2021_2026_master.parquet"

    print("\n" + "=" * 80)
    print(f"  SLICING MASTER 5-YEAR PARQUET: {symbol}")
    print("=" * 80)

    if not master_file.exists():
        print(f"[!] Error: Master parquet not found: {master_file}")
        return None

    file_size_mb = master_file.stat().st_size / (1024**2)
    output_dir = DATA_DIR / symbol
    print(f"Source: {master_file} ({file_size_mb:.1f} MB)")
    print(f"Target: {output_dir}")

    # Clean existing weekly data if replace_existing is True
    if output_dir.exists():
        if replace_existing:
            print(f"[*] Removing earlier data in {output_dir.name}...")
            for item in output_dir.iterdir():
                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    shutil.rmtree(item)
    else:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Clean stale .npy cache in tick_engine if replaced
    npy_cache = ROOT_DIR / "tick_engine" / "data" / (f"processed_{symbol.lower()}" if symbol != "EURUSD" else "processed")
    if npy_cache.exists():
        print(f"[*] Cleaning stale .npy binary cache in {npy_cache.name}...")
        for item in npy_cache.glob("week_*.npy"):
            try:
                item.unlink()
            except Exception:
                pass

    pf = pq.ParquetFile(master_file)
    total_rows = pf.metadata.num_rows
    num_rg = pf.num_row_groups
    print(f"[*] Metadata: {total_rows:,} rows across {num_rg} row groups.")

    col_idx = pf.schema.names.index("timestamp_ms")
    first_ms = pf.metadata.row_group(0).column(col_idx).statistics.min
    last_ms = pf.metadata.row_group(num_rg - 1).column(col_idx).statistics.max

    schedule = build_schedule_from_bounds(first_ms, last_ms)
    n_sched = len(schedule)
    print(f"[*] Weekly schedule: {n_sched} potential calendar weeks.")

    sched_idx = 0
    active_week_num = 1
    manifest_entries = []

    cur_week_info = None
    cur_out_file = None
    cur_f_out = None
    cur_tick_count = 0
    cur_first_tick_ms = None
    cur_last_tick_ms = None

    def close_current_week():
        nonlocal cur_week_info, cur_out_file, cur_f_out, cur_tick_count, cur_first_tick_ms, cur_last_tick_ms, active_week_num
        if cur_f_out is not None:
            try:
                cur_f_out.close()
            except Exception:
                pass
            cur_f_out = None

        if cur_out_file is not None and cur_tick_count > 0:
            size_mb = cur_out_file.stat().st_size / (1024 * 1024) if cur_out_file.exists() else 0.0
            first_utc = datetime.datetime.fromtimestamp(cur_first_tick_ms / 1000, tz=datetime.timezone.utc).strftime("%Y-%m-%d %H:%M")
            last_utc = datetime.datetime.fromtimestamp(cur_last_tick_ms / 1000, tz=datetime.timezone.utc).strftime("%Y-%m-%d %H:%M")

            entry = {
                "week": active_week_num,
                "filename": cur_out_file.name,
                "relative_path": f"data/{symbol}/{cur_out_file.name}",
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

            if active_week_num % 50 == 0 or active_week_num == 1:
                print(f"  [+] Week {active_week_num:03d}: {cur_tick_count:,} ticks ({size_mb:.1f} MB) -> {cur_out_file.name}", flush=True)

            active_week_num += 1

        cur_week_info = None
        cur_out_file = None
        cur_f_out = None
        cur_tick_count = 0
        cur_first_tick_ms = None
        cur_last_tick_ms = None

    total_ticks_written = 0

    # Stream through row groups
    for rg_idx in range(num_rg):
        raw_tbl = pf.read_row_group(rg_idx)
        tbl = normalize_arrow_table(raw_tbl)

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
                    # New week file
                    cur_week_info = w_info
                    filename = f"week_{active_week_num:02d}_{w_info['mon'].isoformat()}_to_{w_info['fri'].isoformat()}.csv"
                    cur_out_file = output_dir / filename
                    cur_f_out = open(cur_out_file, "wb")
                    pcsv.write_csv(sub_tbl, cur_f_out, write_options=pcsv.WriteOptions(include_header=True))
                    cur_first_tick_ms = int(ts_arr[i_start])
                else:
                    # Append to current open week file
                    pcsv.write_csv(sub_tbl, cur_f_out, write_options=pcsv.WriteOptions(include_header=False))

                cur_last_tick_ms = int(ts_arr[i_end - 1])
                cur_tick_count += slice_count
                total_ticks_written += slice_count

            # If this row group reaches or passes the end of this week, close it
            if rg_max_ms >= e_ms:
                close_current_week()
                sched_idx += 1
            else:
                break

        if (rg_idx + 1) % 100 == 0 or rg_idx + 1 == num_rg:
            elapsed = time.time() - t_start
            rate = total_ticks_written / max(elapsed, 0.001)
            print(f"  --> Progress: Row Group {rg_idx+1:04d}/{num_rg} | {total_ticks_written:,} ticks written ({rate:,.0f} ticks/s)...", flush=True)

    # Close trailing week
    close_current_week()

    # Save manifest
    manifest_path = output_dir / "manifest_weekly_sets.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_entries, f, indent=2)

    # Apply 70/30 deterministic random partition
    split_info = partition_symbol_70_30(symbol, manifest_entries, output_dir)

    elapsed_total = time.time() - t_start
    print("-" * 60)
    print(f"SLICING & PARTITIONING COMPLETE FOR {symbol}:")
    print(f"  * Total Active Weeks Sliced: {len(manifest_entries)}")
    print(f"  * 70% Discovery Weeks:       {split_info['discovery_70']['count']} ({split_info['discovery_70']['total_ticks']:,} ticks)")
    print(f"  * 30% Holdout Weeks:         {split_info['holdout_test_30']['count']} ({split_info['holdout_test_30']['total_ticks']:,} ticks)")
    print(f"  * Total Ticks Written:       {total_ticks_written:,}")
    print(f"  * Elapsed Time:              {elapsed_total:.1f}s")
    print("-" * 60)

    return split_info


def update_master_registry():
    """Consolidates all asset splits into data_splits/master_70_30_splits.json."""
    print("\n[*] Updating central registry: data_splits/master_70_30_splits.json...")
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    master_summary = {
        "standard": "70/30 Randomized Week Partitioning (Seed 42)",
        "policy": "Discovery partition strictly for research; holdout partition strictly quarantined for out-of-sample forward verification",
        "seed": SEED,
        "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "total_assets": 0,
        "total_fleet_weeks": 0,
        "total_fleet_ticks": 0,
        "assets": {}
    }

    total_fleet_weeks = 0
    total_fleet_disc_weeks = 0
    total_fleet_test_weeks = 0
    total_fleet_ticks = 0

    # Dynamically detect all active asset datasets
    active_syms = sorted([d.name for d in DATA_DIR.iterdir() if d.is_dir() and (d / "split_70_30.json").exists()])

    for sym in active_syms:
        split_file = DATA_DIR / sym / "split_70_30.json"
        if split_file.exists():
            with open(split_file, "r", encoding="utf-8") as f:
                s_data = json.load(f)
            master_summary["assets"][sym] = s_data

            total_fleet_weeks += s_data["total_weeks"]
            total_fleet_disc_weeks += s_data["discovery_70"]["count"]
            total_fleet_test_weeks += s_data["holdout_test_30"]["count"]
            total_ticks = s_data["discovery_70"]["total_ticks"] + s_data["holdout_test_30"]["total_ticks"]
            total_fleet_ticks += total_ticks

    master_summary["total_assets"] = len(master_summary["assets"])
    master_summary["total_fleet_weeks"] = total_fleet_weeks
    master_summary["total_fleet_ticks"] = total_fleet_ticks

    out_file = SPLITS_DIR / "master_70_30_splits.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(master_summary, f, indent=2)

    print(f"[+] Registry updated: {master_summary['total_assets']} assets registered | {total_fleet_weeks:,} total weeks | {total_fleet_ticks:,} total ticks.")
    return master_summary


def main():
    parser = argparse.ArgumentParser(description="Batch Slicer & Ingestion Engine for 5-Year Unified Masters")
    parser.add_argument("--symbol", type=str, help="Specific symbol to slice (e.g. EURUSD, AUDUSD).")
    parser.add_argument("--symbols", type=str, help="Comma-separated symbols to slice (e.g. EURUSD,USDJPY,AUDNZD).")
    parser.add_argument("--source-dir", type=str, default=None, help="Directory containing master parquets.")
    parser.add_argument("--all", action="store_true", help="Slice all 13 symbols sequentially")
    parser.add_argument("--skip-existing", action="store_true", help="Skip symbols whose data/<SYMBOL> already has 300+ weeks")

    args = parser.parse_args()

    symbols_to_process = []
    if args.symbol:
        symbols_to_process = [args.symbol.upper()]
    elif args.symbols:
        symbols_to_process = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    elif args.all or not args.symbol:
        symbols_to_process = ALL_SYMBOLS

    src_dir = Path(args.source_dir) if args.source_dir else None

    print("=" * 80)
    print(f"  BATCH SLICER & INGESTION ENGINE: {len(symbols_to_process)} TARGET SYMBOLS")
    print("=" * 80)

    start_all = time.time()
    for idx, sym in enumerate(symbols_to_process, 1):
        if args.skip_existing:
            manifest_file = DATA_DIR / sym / "manifest_weekly_sets.json"
            if manifest_file.exists():
                with open(manifest_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if len(data) >= 300:
                    print(f"Skipping {sym} (already has {len(data)} sliced weeks).")
                    continue

        print(f"\n>>> [{idx}/{len(symbols_to_process)}] Processing {sym}...")
        slice_symbol_master(sym, source_dir=src_dir, replace_existing=True)

    update_master_registry()
    print("\n" + "=" * 80)
    print(f"  ALL SYMBOLS PROCESSED IN {time.time() - start_all:.1f}s!")
    print("=" * 80)


if __name__ == "__main__":
    main()
