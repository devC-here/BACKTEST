#!/usr/bin/env python3
import sys
from pathlib import Path
import pandas as pd
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
export_dir = ROOT_DIR / "export_google_drive"

print("Inspecting export_google_drive...")
for sym in ["AUDNZD", "EURGBP"]:
    sym_dir = export_dir / sym
    files = sorted(list(sym_dir.glob("*.parquet")))
    print(f"\n==================================================")
    print(f"Symbol: {sym} | Found {len(files)} parquet files")
    print(f"==================================================")
    for f in files:
        df = pd.read_parquet(f)
        cnt = len(df)
        print(f"\nFile: {f.name}")
        print(f"  Rows: {cnt:,}")
        print(f"  Columns: {list(df.columns)}")
        print(f"  Dtypes:\n{df.dtypes}")
        
        # Check timestamps
        ts_col = 'datetime' if 'datetime' in df.columns else 'timestamp'
        t_min = df[ts_col].min()
        t_max = df[ts_col].max()
        print(f"  Range: {t_min} -> {t_max}")
        
        # Check if timestamp_ms is present
        if 'timestamp_ms' in df.columns:
            print(f"  timestamp_ms range: {df['timestamp_ms'].min()} -> {df['timestamp_ms'].max()}")
            is_sorted = df['timestamp_ms'].is_monotonic_increasing
            print(f"  Is monotonic sorted: {is_sorted}")
            
        print(f"  Head(2):\n{df.head(2)}")
        print(f"  Tail(2):\n{df.tail(2)}")
