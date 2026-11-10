#!/usr/bin/env python3
import time
from pathlib import Path
import pyarrow.parquet as pq

ROOT_DIR = Path(__file__).resolve().parent.parent
pf = ROOT_DIR / "export_google_drive" / "AUDNZD" / "audnzd_2024.parquet"

t0 = time.time()
table = pq.read_table(pf, columns=["timestamp_ms", "ask", "bid", "ask_volume", "bid_volume"])
print(f"PyArrow loaded {table.num_rows:,} rows in {time.time() - t0:.2f}s!")
print(f"Memory: {table.nbytes / 1e6:.1f} MB")
