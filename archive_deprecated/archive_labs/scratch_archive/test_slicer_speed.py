#!/usr/bin/env python3
import sys
import time
from pathlib import Path
import pyarrow.parquet as pq
import polars as pl

ROOT_DIR = Path(__file__).resolve().parent.parent
export_dir = ROOT_DIR / "export_google_drive"
sample_file = export_dir / "AUDNZD" / "audnzd_2024.parquet"

print(f"Benchmarking Polars scan on {sample_file.name} ({sample_file.stat().st_size / 1e6:.1f} MB)...")
t0 = time.time()
# Polars scan_parquet is lazy and ultra-fast
q = pl.scan_parquet(str(sample_file)).select([
    pl.col("timestamp_ms").alias("timestamp"),
    pl.col("ask").alias("askPrice"),
    pl.col("bid").alias("bidPrice"),
    pl.col("ask_volume").alias("askVolume"),
    pl.col("bid_volume").alias("bidVolume"),
])
df = q.collect()
dt = time.time() - t0
print(f"Polars read {len(df):,} rows in {dt:.2f}s!")
print("Sample rows:")
print(df.head(3))
