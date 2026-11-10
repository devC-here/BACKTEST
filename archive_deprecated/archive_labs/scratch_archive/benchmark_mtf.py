#!/usr/bin/env python3
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tick_engine.core.dataloader import get_week_dataset
from data_engineering_suite.bars import build_time_bars

def test():
    t0 = time.time()
    all_h4 = []
    all_h1 = []
    all_m5 = []
    
    print("Loading 10 weeks to benchmark multi-timeframe aggregation...")
    for w in range(1, 11):
        ticks, _ = get_week_dataset(w)
        if len(ticks) < 1000:
            continue
        h4 = build_time_bars(ticks, timeframe_min=240)
        h1 = build_time_bars(ticks, timeframe_min=60)
        m5 = build_time_bars(ticks, timeframe_min=5)
        all_h4.append(h4)
        all_h1.append(h1)
        all_m5.append(m5)
        
    df_h4 = pd.concat(all_h4)
    df_h1 = pd.concat(all_h1)
    df_m5 = pd.concat(all_m5)
    
    dt = time.time() - t0
    print(f"Loaded 10 weeks in {dt:.2f}s!")
    print(f"H4 bars: {len(df_h4)}, H1 bars: {len(df_h1)}, M5 bars: {len(df_m5)}")

if __name__ == '__main__':
    test()
