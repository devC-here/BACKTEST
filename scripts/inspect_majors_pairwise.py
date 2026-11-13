#!/usr/bin/env python3
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
from scripts.inspect_exact_redundancy import strat_trades, strat_meta, by_sym

for target_sym in ['EURUSD', 'USDJPY']:
    keys = by_sym[target_sym]
    print(f"\n=== {target_sym} PAIRWISE (keys: {len(keys)}) ===")
    for i in range(len(keys)):
        for j in range(i+1, len(keys)):
            k1, k2 = keys[i], keys[j]
            df1, df2 = strat_trades[k1], strat_trades[k2]
            if len(df1) == 0 or len(df2) == 0:
                continue
            times1, times2 = df1['entry_time'].values, df2['entry_time'].values
            dir1, dir2 = df1['direction'].values, df2['direction'].values
            matches = sum(1 for idx, t1 in enumerate(times1) if np.any((np.abs(times2 - t1) <= 1800000) & (dir2 == dir1[idx])))
            overlap = (matches / min(len(times1), len(times2))) * 100.0
            
            df1_c, df2_c = df1.copy(), df2.copy()
            df1_c['day'] = pd.to_datetime(df1_c['entry_time'], unit='ms', utc=True).dt.date
            df2_c['day'] = pd.to_datetime(df2_c['entry_time'], unit='ms', utc=True).dt.date
            daily1, daily2 = df1_c.groupby('day')['pips'].sum(), df2_c.groupby('day')['pips'].sum()
            all_days = sorted(list(set(daily1.index).union(set(daily2.index))))
            d1_s = [daily1.get(d, 0.0) for d in all_days]
            d2_s = [daily2.get(d, 0.0) for d in all_days]
            corr = np.corrcoef(d1_s, d2_s)[0, 1] if len(all_days) > 5 else 0.0
            if overlap >= 40.0 or corr >= 0.40:
                name1 = k1.split('__')[1]
                name2 = k2.split('__')[1]
                p1, p2 = strat_meta[k1]['pips'], strat_meta[k2]['pips']
                pf1, pf2 = strat_meta[k1]['pf'], strat_meta[k2]['pf']
                print(f"  {name1:<40} vs {name2:<40} | Overlap: {overlap:>5.1f}% | Corr: {corr:>5.2f} | PnL: {p1:>6.1f}p vs {p2:>6.1f}p | PF: {pf1:.2f} vs {pf2:.2f}")
