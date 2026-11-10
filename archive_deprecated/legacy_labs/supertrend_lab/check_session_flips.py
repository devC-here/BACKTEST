"""
Inspect flip counts specifically during session hours vs 24 hours.
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

def check_session_flips(week_num: int = 1):
    ticks, _ = get_week_dataset(week_num)
    candles = resample_ticks_to_candles(ticks, timeframe="5min")
    n_c = len(candles)

    st_line, trend, atr = calculate_supertrend(candles, atr_period=10, multiplier=3.0)
    strong_bull, strong_bear = identify_strong_candles(candles, atr_values=atr, min_body_ratio=0.55, min_atr_factor=0.50)

    hours = candles.index.hour.to_numpy()

    session_flips = 0
    session_flips_strong = 0
    session_strong_candles_in_trend = 0

    all_flips = 0
    all_flips_strong = 0
    all_strong_candles_in_trend = 0

    for i in range(1, n_c):
        prev_t = trend[i - 1]
        cur_t = trend[i]
        h = hours[i]
        in_sess = (7 <= h < 18)

        is_flip = (prev_t != cur_t)
        is_flip_strong = (prev_t == -1 and cur_t == 1 and strong_bull[i]) or (prev_t == 1 and cur_t == -1 and strong_bear[i])
        is_strong_in_trend = (cur_t == 1 and strong_bull[i]) or (cur_t == -1 and strong_bear[i])

        if is_flip:
            all_flips += 1
            if in_sess:
                session_flips += 1

        if is_flip_strong:
            all_flips_strong += 1
            if in_sess:
                session_flips_strong += 1

        if is_strong_in_trend:
            all_strong_candles_in_trend += 1
            if in_sess:
                session_strong_candles_in_trend += 1

    print(f"WEEK {week_num}:")
    print(f"  24h Total Flips                    : {all_flips}")
    print(f"  24h Flips with Strong Candle       : {all_flips_strong}")
    print(f"  24h Strong Candles in Trend        : {all_strong_candles_in_trend}")
    print(f"  Session (7-18h) Flips              : {session_flips}")
    print(f"  Session Flips with Strong Candle   : {session_flips_strong}")
    print(f"  Session Strong Candles in Trend    : {session_strong_candles_in_trend}")

if __name__ == "__main__":
    for w in [1, 2, 3]:
        check_session_flips(w)
