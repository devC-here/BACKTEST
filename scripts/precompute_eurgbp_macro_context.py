#!/usr/bin/env python3
"""
================================================================================
  PRECOMPUTE CONTINUOUS EURGBP MACRO CONTEXT (DAILY & WEEKLY BARS)
================================================================================
Extracts continuous daily and weekly OHLC bars across all 300 weeks of EURGBP:
  - Daily: Date, Open, High, Low, Close, Range (pips)
  - Weekly: WeekNum, Open, High, Low, Close, Range (pips)
Saves to `data/EURGBP/macro_bars.json` for lightning-fast lookup during alpha mining.
================================================================================
"""

import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from tick_engine.core.dataloader import get_week_dataset
from tick_engine.core.config import PIP_SIZES
from data_engineering_suite.bars import build_time_bars

SYMBOL = "EURGBP"
PIP_SIZE = PIP_SIZES.get(SYMBOL, 0.0001)
RAW_DIR = ROOT_DIR / "data" / SYMBOL

def main():
    print(f"Extracting daily and weekly bars for {SYMBOL} across Weeks 1..300...", flush=True)
    all_m15 = []

    for w in range(1, 301):
        try:
            ticks, _ = get_week_dataset(week=w, raw_dir=RAW_DIR, warmup_prev_week_ticks=0)
        except Exception:
            continue
        if len(ticks) < 1000:
            continue
        m15 = build_time_bars(ticks, timeframe_min=15, pip_size=PIP_SIZE)
        if len(m15) > 0:
            m15['week_num'] = w
            all_m15.append(m15)
        if w % 50 == 0:
            print(f"  Processed week {w}/300...", flush=True)

    full_m15 = pd.concat(all_m15).sort_index()
    # Remove any duplicate index timestamps
    full_m15 = full_m15[~full_m15.index.duplicated(keep='first')]

    # Daily aggregation
    daily = full_m15.resample('1D').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last'
    }).dropna()
    daily['range_pips'] = (daily['high'] - daily['low']) / PIP_SIZE

    daily_dict = {}
    for idx, row in daily.iterrows():
        d_str = idx.strftime('%Y-%m-%d')
        daily_dict[d_str] = {
            'date': d_str,
            'open': float(row['open']),
            'high': float(row['high']),
            'low': float(row['low']),
            'close': float(row['close']),
            'range_pips': float(row['range_pips'])
        }

    # Weekly aggregation
    weekly = full_m15.resample('1W-FRI').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last'
    }).dropna()
    weekly['range_pips'] = (weekly['high'] - weekly['low']) / PIP_SIZE

    weekly_dict = {}
    for idx, row in weekly.iterrows():
        w_str = idx.strftime('%Y-%m-%d')
        weekly_dict[w_str] = {
            'end_date': w_str,
            'open': float(row['open']),
            'high': float(row['high']),
            'low': float(row['low']),
            'close': float(row['close']),
            'range_pips': float(row['range_pips'])
        }

    out_file = RAW_DIR / "macro_bars.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            'daily': daily_dict,
            'weekly': weekly_dict,
            'total_daily_bars': len(daily_dict),
            'total_weekly_bars': len(weekly_dict)
        }, f, indent=2)

    print(f"Precomputation complete! Saved {len(daily_dict)} daily bars and {len(weekly_dict)} weekly bars to {out_file}", flush=True)

if __name__ == "__main__":
    main()

