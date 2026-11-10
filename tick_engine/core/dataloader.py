"""
High-Speed Data Ingestion and Loading Pipeline for Tick Data.
Converts CSV tick files to contiguous .npy binary files or .parquet,
and provides memory-mapped zero-copy loading for large datasets.
"""

from __future__ import annotations
import os
import glob
from pathlib import Path
from typing import Optional, Union, Sequence
import numpy as np

# Structured tick array dtype as specified in architecture
TICK_DTYPE = np.dtype([
    ('timestamp', np.int64),    # Unix epoch (milliseconds or nanoseconds)
    ('bid', np.float64),        # Current bid price
    ('ask', np.float64)         # Current ask price
])


def _resolve_columns(col_names: Sequence[str]) -> tuple[str, str, str]:
    """
    Resolve timestamp, bid, and ask column names from CSV headers.
    Handles case-insensitive variations commonly found in FX datasets.
    """
    lower_map = {name.lower().replace(" ", "").replace("_", ""): name for name in col_names}

    # Timestamp candidates
    time_candidates = ["timestamp", "time", "datetime", "date", "timems", "epoch"]
    time_col = next((lower_map[c] for c in time_candidates if c in lower_map), None)

    # Bid candidates
    bid_candidates = ["bidprice", "bid", "b", "closebid", "pricebid"]
    bid_col = next((lower_map[c] for c in bid_candidates if c in lower_map), None)

    # Ask candidates
    ask_candidates = ["askprice", "ask", "a", "closeask", "priceask"]
    ask_col = next((lower_map[c] for c in ask_candidates if c in lower_map), None)

    if not time_col or not bid_col or not ask_col:
        raise ValueError(
            f"Could not automatically identify tick columns in {list(col_names)}. "
            f"Found: timestamp={time_col}, bid={bid_col}, ask={ask_col}."
        )

    return time_col, bid_col, ask_col


def parse_dukascopy_csv(
    csv_path: Union[str, Path],
    npy_path: Optional[Union[str, Path]] = None,
    overwrite: bool = False
) -> Path:
    """
    Parses a Dukascopy tick CSV specifically with standard columns:
    [Time, Ask, Bid, AskVolume, BidVolume].
    Note: In Dukascopy tick data, Ask comes BEFORE Bid.
    Parses timestamps to int64 Unix milliseconds (UTC).
    Converts and saves contiguous structured NumPy array to .npy with dtype:
    [('timestamp', 'i8'), ('bid', 'f8'), ('ask', 'f8')].
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Source Dukascopy CSV not found: {csv_path}")

    if npy_path is None:
        npy_path = csv_path.with_suffix('.npy')
    else:
        npy_path = Path(npy_path)

    if npy_path.exists() and not overwrite:
        return npy_path

    npy_path.parent.mkdir(parents=True, exist_ok=True)

    import pandas as pd
    # Check if header exists
    with open(csv_path, 'r', encoding='utf-8', errors='ignore') as f:
        first_line = f.readline().strip()
    has_header = any(c.isalpha() for c in first_line)

    if has_header:
        df = pd.read_csv(csv_path)
        col_names = [c.strip() for c in df.columns]
        time_col, bid_col, ask_col = _resolve_columns(col_names)
    else:
        # Standard Dukascopy headerless order: 0=Time, 1=Ask, 2=Bid, 3=AskVol, 4=BidVol
        df = pd.read_csv(csv_path, header=None)
        time_col, ask_col, bid_col = 0, 1, 2

    # Parse timestamps to int64 Unix milliseconds (UTC)
    if pd.api.types.is_numeric_dtype(df[time_col]):
        val = df[time_col].to_numpy(dtype=np.int64)
        if len(val) > 0 and val[0] < 100_000_000_000:       # seconds -> ms
            ts_arr = val * 1000
        elif len(val) > 0 and val[0] > 100_000_000_000_000: # nanoseconds -> ms
            ts_arr = val // 1_000_000
        else:                                               # already milliseconds
            ts_arr = val
    else:
        # Datetime string -> UTC Unix milliseconds
        dt_s = pd.to_datetime(df[time_col], utc=True).dt.tz_localize(None)
        ts_arr = dt_s.astype('datetime64[ms]').to_numpy().view(np.int64)

    bid_arr = df[bid_col].to_numpy(dtype=np.float64)
    ask_arr = df[ask_col].to_numpy(dtype=np.float64)

    n = len(ts_arr)
    ticks = np.empty(n, dtype=TICK_DTYPE)
    ticks['timestamp'] = ts_arr
    ticks['bid'] = bid_arr
    ticks['ask'] = ask_arr

    # Save contiguous array
    np.save(str(npy_path), ticks)
    return npy_path


def csv_to_npy(
    csv_path: Union[str, Path],
    npy_path: Optional[Union[str, Path]] = None,
    overwrite: bool = False
) -> Path:
    """
    Ingests a raw tick CSV file and converts it into a contiguous binary .npy
    structured array adhering to TICK_DTYPE.

    Uses PyArrow for high-speed multi-threaded parsing if available,
    falling back to NumPy/Pandas.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Source CSV not found: {csv_path}")

    if npy_path is None:
        npy_path = csv_path.with_suffix('.npy')
    else:
        npy_path = Path(npy_path)

    if npy_path.exists() and not overwrite:
        return npy_path

    npy_path.parent.mkdir(parents=True, exist_ok=True)

    # Fast parsing via pyarrow
    try:
        import pyarrow.csv as pv
        table = pv.read_csv(str(csv_path))
        col_names = table.column_names
        time_col, bid_col, ask_col = _resolve_columns(col_names)

        ts_chunk = table[time_col].to_numpy()
        # Handle timestamp parsing (integers, floats, or ISO strings)
        if ts_chunk.dtype.kind in ('i', 'u'):
            ts_arr = ts_chunk.astype(np.int64)
        elif ts_chunk.dtype.kind == 'f':
            ts_arr = ts_chunk.astype(np.int64)
        else:
            # Try parsing timestamp strings to datetime64[ms]
            import pandas as pd
            ts_arr = pd.to_datetime(table[time_col].to_pandas()).astype('int64').to_numpy()

        bid_arr = table[bid_col].to_numpy().astype(np.float64)
        ask_arr = table[ask_col].to_numpy().astype(np.float64)

        n = len(ts_arr)
        ticks = np.empty(n, dtype=TICK_DTYPE)
        ticks['timestamp'] = ts_arr
        ticks['bid'] = bid_arr
        ticks['ask'] = ask_arr

    except ImportError:
        # Fallback to pandas
        import pandas as pd
        df = pd.read_csv(csv_path)
        time_col, bid_col, ask_col = _resolve_columns(df.columns)

        if not pd.api.types.is_numeric_dtype(df[time_col]):
            dt_s = pd.to_datetime(df[time_col], utc=True).dt.tz_localize(None)
            ts_arr = dt_s.astype('datetime64[ms]').to_numpy().view(np.int64)
        else:
            val = df[time_col].to_numpy(dtype=np.int64)
            if len(val) > 0 and val[0] < 100_000_000_000:
                ts_arr = val * 1000
            elif len(val) > 0 and val[0] > 100_000_000_000_000:
                ts_arr = val // 1_000_000
            else:
                ts_arr = val

        n = len(df)
        ticks = np.empty(n, dtype=TICK_DTYPE)
        ticks['timestamp'] = ts_arr
        ticks['bid'] = df[bid_col].to_numpy(dtype=np.float64)
        ticks['ask'] = df[ask_col].to_numpy(dtype=np.float64)

    # Save contiguous binary array
    np.save(str(npy_path), ticks)
    return npy_path


def csv_to_parquet(
    csv_path: Union[str, Path],
    parquet_path: Optional[Union[str, Path]] = None,
    overwrite: bool = False
) -> Path:
    """
    Ingests a raw tick CSV and converts it to a standard Parquet file.
    """
    csv_path = Path(csv_path)
    if parquet_path is None:
        parquet_path = csv_path.with_suffix('.parquet')
    else:
        parquet_path = Path(parquet_path)

    if parquet_path.exists() and not overwrite:
        return parquet_path

    parquet_path.parent.mkdir(parents=True, exist_ok=True)

    import pyarrow.csv as pv
    import pyarrow.parquet as pq

    table = pv.read_csv(str(csv_path))
    time_col, bid_col, ask_col = _resolve_columns(table.column_names)
    table_subset = table.select([time_col, bid_col, ask_col])
    table_renamed = table_subset.rename_columns(["timestamp", "bid", "ask"])
    pq.write_table(table_renamed, str(parquet_path), compression='zstd')
    return parquet_path


def load_ticks(file_path: Union[str, Path], mmap: bool = True) -> np.ndarray:
    """
    Loads tick data from binary format (.npy or .parquet).

    If mmap is True and file is .npy, loads via memory-mapping (np.load(..., mmap_mode='r')),
    ensuring multi-gigabyte files do not exhaust RAM and provide zero-copy access.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Tick file not found: {file_path}")

    suffix = file_path.suffix.lower()

    if suffix == '.npy':
        mode = 'r' if mmap else None
        ticks = np.load(str(file_path), mmap_mode=mode)
        # Verify dtype compatibility
        if ticks.dtype != TICK_DTYPE:
            ticks = ticks.astype(TICK_DTYPE)
        return ticks

    elif suffix in ('.parquet', '.pq'):
        import pyarrow.parquet as pq
        table = pq.read_table(str(file_path), columns=['timestamp', 'bid', 'ask'])
        n = len(table)
        ticks = np.empty(n, dtype=TICK_DTYPE)
        ticks['timestamp'] = table['timestamp'].to_numpy().astype(np.int64)
        ticks['bid'] = table['bid'].to_numpy().astype(np.float64)
        ticks['ask'] = table['ask'].to_numpy().astype(np.float64)
        return ticks

    elif suffix == '.csv':
        # Auto-convert CSV on the fly and load
        npy_cache = file_path.with_suffix('.npy')
        csv_to_npy(file_path, npy_cache)
        return load_ticks(npy_cache, mmap=mmap)

    else:
        raise ValueError(f"Unsupported tick file format: {suffix}. Must be .npy, .parquet, or .csv")


def batch_convert_csv_to_npy(
    raw_dir: Union[str, Path],
    processed_dir: Union[str, Path],
    pattern: str = "*.csv",
    overwrite: bool = False
) -> list[Path]:
    """
    Batch converts all CSV files in raw_dir into .npy structured files in processed_dir.
    """
    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(list(raw_dir.glob(pattern)))
    converted_files = []

    for csv_file in csv_files:
        dest_npy = processed_dir / csv_file.with_suffix('.npy').name
        out_path = csv_to_npy(csv_file, dest_npy, overwrite=overwrite)
        converted_files.append(out_path)

    return converted_files


def _resolve_raw_dir(raw_dir: Optional[Union[str, Path]]) -> Path:
    """Resolve raw directory path supporting data/<SYMBOL>, weekly_data/<SYMBOL>, or symbol strings."""
    if raw_dir is None:
        p = Path("data/EURUSD")
        if p.exists():
            return p
        return Path("weekly_data/EURUSD")
    p = Path(raw_dir)
    if p.exists():
        return p
    # Check if it's a symbol like 'EURUSD', 'EURGBP', 'USDJPY', 'AUDNZD', etc.
    cand = Path("data") / str(raw_dir).upper()
    if cand.exists():
        return cand
    cand_lower = Path("data") / str(raw_dir).lower()
    if cand_lower.exists():
        return cand_lower
    cand_weekly = Path("weekly_data") / str(raw_dir).upper()
    if cand_weekly.exists():
        return cand_weekly
    return p


def load_manifest(
    raw_dir: Optional[Union[str, Path]] = None,
    split: Optional[str] = None
) -> dict[int, dict]:
    """
    Loads manifest metadata for weekly tick datasets.
    Optional split filtering:
      - 'discovery', 'train', or '70': Returns strictly the 70% discovery weeks.
      - 'test', 'holdout', or '30': Returns strictly the 30% quarantined holdout test weeks.
      - None: Returns all weeks.
    Returns mapping of week_num -> metadata dictionary.
    """
    import json
    raw_dir = _resolve_raw_dir(raw_dir)

    manifest_file = raw_dir / "manifest_weekly_sets.json"
    if manifest_file.exists():
        with open(manifest_file, "r", encoding="utf-8") as f:
            data = json.load(f)

            if split is not None:
                s_lower = str(split).lower()
                if any(x in s_lower for x in ["disc", "train", "70", "calib", "in_sample"]):
                    data = [item for item in data if item.get("split") in ["discovery_70", "discovery", "in_sample_calibration_70"]]
                elif any(x in s_lower for x in ["test", "hold", "30", "oos", "out_of_sample"]):
                    data = [item for item in data if item.get("split") in ["holdout_test_30", "holdout_30", "holdout", "out_of_sample_holdout_30"]]

            return {item["week"]: item for item in data}

    # Fallback: scan directory
    manifest = {}
    csv_files = sorted(list(raw_dir.glob("week_*.csv")))
    for path in csv_files:
        parts = path.stem.split("_")
        if len(parts) >= 2 and parts[1].isdigit():
            w = int(parts[1])
            manifest[w] = {
                "week": w,
                "filename": path.name,
                "relative_path": str(path),
                "absolute_path": str(path.resolve()),
                "tick_count": 0,
            }
    return manifest


def get_split_weeks(
    raw_dir: Optional[Union[str, Path]] = None,
    split: str = "discovery"
) -> list[int]:
    """
    Returns sorted list of week numbers for the requested split ('discovery' or 'test').
    """
    manifest = load_manifest(raw_dir=raw_dir, split=split)
    return sorted(list(manifest.keys()))


def is_complete_trading_week(
    item: dict,
    min_hours: float = 80.0,
    min_ticks: int = 15000
) -> bool:
    """
    Institutional Data Completeness Gate:
    Validates whether a week contains a full trading window.
    Rejects severely truncated/partial weeks (e.g. 23-hour or 30-hour fragments).
    """
    first_utc = item.get("first_tick_utc")
    last_utc = item.get("last_tick_utc")
    tick_count = item.get("tick_count", 0)

    if tick_count < min_ticks:
        return False

    if first_utc and last_utc:
        try:
            import pandas as pd
            dt1 = pd.to_datetime(first_utc)
            dt2 = pd.to_datetime(last_utc)
            span_hours = (dt2 - dt1).total_seconds() / 3600.0
            if span_hours < min_hours:
                return False
        except Exception:
            pass

    return True


def get_complete_split_weeks(
    raw_dir: Optional[Union[str, Path]] = None,
    split: str = "discovery",
    min_hours: float = 80.0,
    min_ticks: int = 15000
) -> list[int]:
    """
    Returns sorted list of week numbers for the requested split, guaranteed to pass
    the institutional data completeness gate (minimum span hours and tick threshold).
    """
    manifest = load_manifest(raw_dir=raw_dir, split=split)
    valid_weeks = []
    for w, item in manifest.items():
        if is_complete_trading_week(item, min_hours=min_hours, min_ticks=min_ticks):
            valid_weeks.append(w)
    return sorted(valid_weeks)


def get_discovery_weeks(raw_dir: Optional[Union[str, Path]] = None) -> list[int]:
    """
    Returns the 70% randomized discovery weeks used strictly for analysis,
    feature engineering, and edge discovery.
    """
    return get_split_weeks(raw_dir=raw_dir, split="discovery")


def get_holdout_test_weeks(raw_dir: Optional[Union[str, Path]] = None) -> list[int]:
    """
    Returns the 30% quarantined holdout test weeks, strictly forbidden from
    analysis or edge finding, reserved exclusively for out-of-sample forward testing.
    """
    return get_split_weeks(raw_dir=raw_dir, split="test")


def get_chronological_split_weeks(
    symbol: str,
    split: str = "in_sample"
) -> list[int]:
    """
    Returns sorted list of week numbers under strict chronological walk-forward partitioning.
    'in_sample' (first 70% continuous weeks) or 'out_of_sample' (last 30% continuous forward weeks).
    """
    import json
    root = Path(__file__).resolve().parent.parent.parent
    chrono_file = root / "data_splits" / "master_chronological_splits.json"
    if not chrono_file.exists():
        # Fallback to standard 70/30 chronological calculation
        return list(range(1, 211)) if split in {"in_sample", "discovery"} else list(range(211, 301))

    with open(chrono_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    s = symbol.upper()
    asset_data = data.get("assets", {}).get(s, {})
    if not asset_data:
        return list(range(1, 211)) if split in {"in_sample", "discovery"} else list(range(211, 301))

    if split in {"in_sample", "discovery", "calibration"}:
        return asset_data.get("in_sample_calibration_70", {}).get("weeks", [])
    else:
        return asset_data.get("out_of_sample_holdout_30", {}).get("weeks", [])


def get_week_dataset(
    week: int,
    raw_dir: Optional[Union[str, Path]] = None,
    processed_dir: Optional[Union[str, Path]] = None,
    auto_convert: bool = True,
    warmup_prev_week_ticks: int = 0
) -> tuple[np.ndarray, dict]:
    """
    Retrieves memory-mapped ticks for a given week number.
    Auto-converts raw CSV to .npy if not already cached.
    Isolates binary caches per symbol to prevent collisions.
    If warmup_prev_week_ticks > 0 and week > 1, prepends tail ticks from prior week
    to eliminate cold-start indicator distortion.
    """
    raw_dir = _resolve_raw_dir(raw_dir)

    if processed_dir is None:
        sym = raw_dir.name.upper().replace("_WEEKLY_DATA", "")
        if sym == "EURUSD":
            processed_dir = Path("tick_engine/data/processed")
        else:
            processed_dir = Path(f"tick_engine/data/processed_{sym.lower()}")
    else:
        processed_dir = Path(processed_dir)

    processed_dir.mkdir(parents=True, exist_ok=True)
    target_npy = processed_dir / f"week_{week:02d}.npy"

    manifest = load_manifest(raw_dir)
    meta = manifest.get(week, {"week": week, "tick_count": 0})

    if target_npy.exists():
        ticks = load_ticks(target_npy, mmap=True)
    else:
        # Locate raw CSV
        candidates = list(raw_dir.glob(f"week_{week:02d}_*.csv"))
        if not candidates:
            raise FileNotFoundError(f"Could not find raw weekly CSV for week {week} in {raw_dir}")

        csv_file = candidates[0]
        if auto_convert:
            csv_to_npy(csv_file, target_npy)
            ticks = load_ticks(target_npy, mmap=True)
        else:
            raise FileNotFoundError(f"Processed .npy not found for week {week} and auto_convert is False")

    if warmup_prev_week_ticks > 0 and week > 1:
        prev_npy = processed_dir / f"week_{week-1:02d}.npy"
        if prev_npy.exists():
            prev_ticks = load_ticks(prev_npy, mmap=True)
            warmup_slice = prev_ticks[-warmup_prev_week_ticks:] if len(prev_ticks) > warmup_prev_week_ticks else prev_ticks
            ticks = np.concatenate([warmup_slice, ticks])

    return ticks, meta

