#!/usr/bin/env python3
import sys
from pathlib import Path
import pandas as pd
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
data_dir = ROOT_DIR / "dukascopy_ticks_weekly"

print("Scanning dukascopy_ticks_weekly...")
symbols = [d.name for d in data_dir.iterdir() if d.is_dir()]
print(f"Symbols found: {symbols}")

for sym in symbols:
    sym_dir = data_dir / sym
    files = sorted(list(sym_dir.glob("*.parquet")))
    print(f"\n========================================================")
    print(f"Symbol: {sym} | Total parquet files: {len(files)}")
    print(f"========================================================")
    
    if not files:
        continue
        
    # Inspect first normal-sized file and first small file
    sizes = [(f, f.stat().st_size) for f in files]
    sizes.sort(key=lambda x: x[1])
    
    smallest_f, smallest_sz = sizes[0]
    largest_f, largest_sz = sizes[-1]
    
    print(f"Smallest file: {smallest_f.name} ({smallest_sz:,} bytes)")
    print(f"Largest file:  {largest_f.name} ({largest_sz:,} bytes)")
    
    # Read sample
    df_sample = pd.read_parquet(largest_f)
    print(f"\nSample Schema ({largest_f.name}):")
    print(df_sample.info())
    print("\nHead(3):")
    print(df_sample.head(3))
    print("\nTail(3):")
    print(df_sample.tail(3))
    
    # Check smallest file
    df_small = pd.read_parquet(smallest_f)
    print(f"\nSmallest file rows: {len(df_small)} rows")
    if len(df_small) > 0:
        print(df_small.head(2))
        print("Time range in small file:", df_small.iloc[0].get('timestamp', df_small.index[0]), "to", df_small.iloc[-1].get('timestamp', df_small.index[-1]))

    # Analyze week continuity
    # Files are named like SYM_YYYY_Www.parquet
    weeks_by_year = {}
    for f in files:
        parts = f.stem.split('_')
        # e.g. ['AUDNZD', '2021', 'W03']
        if len(parts) >= 3:
            yr = parts[1]
            wk = int(parts[2].replace('W', ''))
            weeks_by_year.setdefault(yr, []).append((wk, f.stat().st_size))
            
    print(f"\nWeek distribution by year for {sym}:")
    for yr in sorted(weeks_by_year.keys()):
        wks = [w for w, sz in weeks_by_year[yr]]
        normal_wks = [w for w, sz in weeks_by_year[yr] if sz > 500_000]
        tiny_wks = [w for w, sz in weeks_by_year[yr] if sz <= 500_000]
        print(f"  {yr}: Total files={len(wks)} (Normal >500KB: {len(normal_wks)}, Tiny <=500KB: {len(tiny_wks)}) | Weeks: min={min(wks)}, max={max(wks)}")
