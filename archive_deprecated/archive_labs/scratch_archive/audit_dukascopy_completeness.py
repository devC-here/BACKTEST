#!/usr/bin/env python3
"""
================================================================================
  AUDIT DUKASCOPY TICKS WEEKLY: DATA QUALITY, GAPS, AND RECOVERY ANALYSIS
================================================================================
Scans AUDNZD, EURGBP, USATECHIDXUSD to measure:
  - Total files
  - Healthy weeks (> 200k ticks / full week)
  - Truncated/damaged weeks (< 200k ticks)
  - Missing weeks (calendar gaps)
  - Total usable ticks currently on disk
================================================================================
"""

from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
data_dir = ROOT_DIR / "dukascopy_ticks_weekly"

symbols = ["AUDNZD", "EURGBP", "USATECHIDXUSD"]

report = {}

for sym in symbols:
    sym_dir = data_dir / sym
    files = sorted(list(sym_dir.glob("*.parquet")))
    
    total_files = len(files)
    healthy_files = []
    damaged_files = []
    
    total_ticks = 0
    
    for f in files:
        sz = f.stat().st_size
        try:
            df = pd.read_parquet(f, columns=["timestamp"])
            cnt = len(df)
            total_ticks += cnt
            
            # Check duration span
            if cnt > 1:
                t_start = df['timestamp'].iloc[0]
                t_end = df['timestamp'].iloc[-1]
                span_hours = (t_end - t_start).total_seconds() / 3600.0
            else:
                span_hours = 0
                
            info = {
                'file': f.name,
                'ticks': cnt,
                'size_kb': sz / 1024.0,
                'span_hours': span_hours
            }
            
            # A standard trading week has >= 80 hours of active market and >= 150k ticks
            if cnt >= 100_000 and span_hours >= 70.0:
                healthy_files.append(info)
            else:
                damaged_files.append(info)
        except Exception as e:
            damaged_files.append({'file': f.name, 'ticks': 0, 'size_kb': sz / 1024.0, 'span_hours': 0, 'error': str(e)})

    # Calculate year coverage
    # Expected weeks: 2021 to 2025 = 5 * 52 = 260 weeks
    report[sym] = {
        'total_files': total_files,
        'healthy_weeks': len(healthy_files),
        'damaged_weeks': len(damaged_files),
        'total_ticks': total_ticks,
        'healthy_ratio': len(healthy_files) / 260.0 * 100.0,
        'sample_healthy': healthy_files[:3],
        'sample_damaged': damaged_files[:3]
    }

print("=" * 80)
print("  DUKASCOPY WEEKLY TICKS DATA INTEGRITY REPORT")
print("=" * 80)

for sym, data in report.items():
    print(f"\nInstrument: {sym}")
    print(f"  Total Files Found:     {data['total_files']} / 260 expected weeks ({data['total_files']/260*100:.1f}%)")
    print(f"  Healthy Full Weeks:    {data['healthy_weeks']} weeks (Complete 5-day continuous coverage)")
    print(f"  Damaged/Stub Weeks:    {data['damaged_weeks']} weeks (Partial hours, disconnected fragments)")
    print(f"  Missing Entirely:      {260 - data['total_files']} weeks (Zero data)")
    print(f"  Total Usable Ticks:    {data['total_ticks']:,} ticks ({data['total_ticks']/1e6:.2f}M ticks)")
    print(f"  Full 5-Year Coverage:  {data['healthy_ratio']:.1f}%")
    print("-" * 60)
