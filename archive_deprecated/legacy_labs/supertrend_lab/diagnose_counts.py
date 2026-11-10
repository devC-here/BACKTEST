"""
Diagnostic script to inspect Supertrend and Strong Candle occurrences in detail.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from supertrend_lab.supertrend_engine import (
    resample_ticks_to_candles,
    calculate_supertrend,
    identify_strong_candles
)
from tick_engine.core.dataloader import get_week_dataset

def diagnose_week(week_num: int = 1):
    ticks, meta = get_week_dataset(week_num)
    candles = resample_ticks_to_candles(ticks, timeframe="5min")
    n_c = len(candles)

    st_line, trend, atr = calculate_supertrend(candles, atr_period=10, multiplier=3.0)
    strong_bull, strong_bear = identify_strong_candles(candles, atr_values=atr, min_body_ratio=0.55, min_atr_factor=0.50)

    # Count flips
    trend_flips_bull = 0
    trend_flips_bear = 0
    flip_with_strong_bull = 0
    flip_with_strong_bear = 0

    total_bull_candles = 0
    total_bear_candles = 0
    strong_bull_candles = 0
    strong_bear_candles = 0

    st_bull_state_candles = 0
    st_bear_state_candles = 0

    st_bull_and_strong_bull = 0
    st_bear_and_strong_bear = 0

    for i in range(1, n_c):
        prev_t = trend[i - 1]
        cur_t = trend[i]

        if cur_t == 1:
            st_bull_state_candles += 1
            if strong_bull[i]:
                st_bull_and_strong_bull += 1
        elif cur_t == -1:
            st_bear_state_candles += 1
            if strong_bear[i]:
                st_bear_and_strong_bear += 1

        if prev_t == -1 and cur_t == 1:
            trend_flips_bull += 1
            if strong_bull[i]:
                flip_with_strong_bull += 1

        if prev_t == 1 and cur_t == -1:
            trend_flips_bear += 1
            if strong_bear[i]:
                flip_with_strong_bear += 1

        if candles['close'].iloc[i] > candles['open'].iloc[i]:
            total_bull_candles += 1
        else:
            total_bear_candles += 1

        if strong_bull[i]:
            strong_bull_candles += 1
        if strong_bear[i]:
            strong_bear_candles += 1

    print(f"=== DIAGNOSTIC REPORT FOR WEEK {week_num} (M5 CANDLES) ===")
    print(f"Total M5 Candles in Week              : {n_c}")
    print(f"Supertrend Flips from Bear to Bull    : {trend_flips_bull}")
    print(f"Supertrend Flips from Bull to Bear    : {trend_flips_bear}")
    print(f"Total Supertrend Flips                : {trend_flips_bull + trend_flips_bear}")
    print(f"---")
    print(f"Flips where the flip candle was STRONG BULL : {flip_with_strong_bull}")
    print(f"Flips where the flip candle was STRONG BEAR : {flip_with_strong_bear}")
    print(f"Total Flips with Strong Candle             : {flip_with_strong_bull + flip_with_strong_bear}")
    print(f"---")
    print(f"Candles in Bullish Supertrend State   : {st_bull_state_candles}")
    print(f"Candles in Bearish Supertrend State   : {st_bear_state_candles}")
    print(f"---")
    print(f"Total Strong Bull Candles (any state) : {strong_bull_candles}")
    print(f"Total Strong Bear Candles (any state) : {strong_bear_candles}")
    print(f"Strong Bull Candles while ST is Bull  : {st_bull_and_strong_bull}")
    print(f"Strong Bear Candles while ST is Bear  : {st_bear_and_strong_bear}")
    print(f"TOTAL (ST Trending + Strong Candle)   : {st_bull_and_strong_bull + st_bear_and_strong_bear}")

if __name__ == "__main__":
    diagnose_week(1)
    print("\n")
    diagnose_week(2)
