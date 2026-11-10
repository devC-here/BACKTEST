#!/usr/bin/env python3
"""
================================================================================
  IN-SAMPLE CALIBRATION & HOLDOUT VERIFICATION: GBPCHF ASIAN SWEEP
================================================================================
Calibrates parameters strictly on In-Sample Discovery (Weeks 1..210),
locks the best candidate, and validates on Quarantined Holdout (Weeks 211..300).
================================================================================
"""

import sys
import json
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from tick_engine.core.engine import run_matching_loop
from tick_engine.core.dataloader import get_week_dataset, get_complete_split_weeks
from alpha_searcher.config import PIP_SIZES, get_friction_pips

SYMBOL = "GBPCHF"
PIP_SIZE = PIP_SIZES.get(SYMBOL, 0.0001)
COMM_PIPS, SLIP_PIPS = get_friction_pips(SYMBOL)
WARMUP_TICKS = 100000

DISC_WEEKS = get_complete_split_weeks(SYMBOL, "discovery_70")
HOLD_WEEKS = get_complete_split_weeks(SYMBOL, "holdout_test_30")


def eval_week(week_num: int, delta_pips: float, sl_pips: float, tp_pips: float, min_range: float = 15.0, max_range: float = 70.0):
    raw_dir = ROOT_DIR / "data" / SYMBOL
    try:
        ticks, _ = get_week_dataset(week=week_num, raw_dir=raw_dir, warmup_prev_week_ticks=WARMUP_TICKS)
    except Exception:
        return np.empty(0)

    if len(ticks) <= WARMUP_TICKS + 500:
        return np.empty(0)

    # Build M15 bars
    ts = pd.to_datetime(ticks['timestamp'], unit='ms')
    mid = 0.5 * (ticks['bid'] + ticks['ask'])
    spread = (ticks['ask'] - ticks['bid']) / PIP_SIZE
    df = pd.DataFrame({'mid': mid, 'spread': spread}, index=ts)
    m15 = df.resample('15min').agg({'mid': ['first', 'max', 'min', 'last'], 'spread': 'mean'}).dropna()
    m15.columns = ['open', 'high', 'low', 'close', 'spread_pips']

    hours = m15.index.hour.values
    dates = m15.index.date
    c = m15['close'].values
    h = m15['high'].values
    l = m15['low'].values
    o = m15['open'].values
    spr = m15['spread_pips'].values
    time_ms = m15.index.astype('datetime64[ms]').view('int64')
    dur_ms = 15 * 60 * 1000
    tick_ts = ticks['timestamp']

    n = len(ticks)
    signals = np.zeros(n, dtype=np.int8)

    curr_date = None
    asian_high = -1e9
    asian_low = 1e9
    traded_today = False

    for i in range(len(m15)):
        dt = dates[i]
        hr = hours[i]

        if dt != curr_date:
            curr_date = dt
            asian_high = -1e9
            asian_low = 1e9
            traded_today = False

        if 0 <= hr < 7:
            if h[i] > asian_high: asian_high = h[i]
            if l[i] < asian_low: asian_low = l[i]
            continue

        if 7 <= hr <= 13 and not traded_today and asian_high > asian_low:
            asian_rng = (asian_high - asian_low) / PIP_SIZE
            if not (min_range <= asian_rng <= max_range):
                continue
            if spr[i] > 2.5:
                continue

            sig = 0
            if h[i] >= asian_high + delta_pips * PIP_SIZE and c[i] < asian_high and c[i] < o[i]:
                sig = -1  # Short
            elif l[i] <= asian_low - delta_pips * PIP_SIZE and c[i] > asian_low and c[i] > o[i]:
                sig = 1   # Long

            if sig != 0:
                idx = int(np.searchsorted(tick_ts, time_ms[i] + dur_ms))
                if idx < n and signals[idx] == 0:
                    signals[idx] = sig
                    traded_today = True

    min_entry_ts = ticks['timestamp'][WARMUP_TICKS]

    trades = run_matching_loop(
        ticks=ticks,
        signals=signals,
        pip_size=PIP_SIZE,
        sl_pips=sl_pips,
        tp_pips=tp_pips,
        use_slippage=True,
        close_at_end=True,
        commission_pips=COMM_PIPS,
        entry_slippage_pips=SLIP_PIPS,
        min_entry_timestamp=min_entry_ts,
        enforce_subsequent_tick=True
    )
    return trades


def run_grid_search():
    print("=" * 80)
    print("  CALIBRATING GBPCHF ASIAN SWEEP: STRICT IN-SAMPLE (WEEKS 1..210)")
    print("=" * 80)

    # Grid parameters
    delta_grid = [6.0, 8.0, 10.0, 12.0]
    sl_grid = [16.0, 20.0, 25.0]
    tp_grid = [35.0, 45.0, 55.0]

    # Pre-select representative subset of in-sample weeks for fast screening (e.g. 50 diverse weeks)
    screen_weeks = DISC_WEEKS[::3]  # 70 diverse weeks across 4 years

    best_pips = -9999
    best_params = None

    with ProcessPoolExecutor(max_workers=8) as executor:
        for delta in delta_grid:
            for sl in sl_grid:
                for tp in tp_grid:
                    futures = [executor.submit(eval_week, w, delta, sl, tp) for w in screen_weeks]
                    all_tr = []
                    for f in as_completed(futures):
                        tr = f.result()
                        if len(tr) > 0:
                            all_tr.append(tr)
                    
                    comb = np.concatenate(all_tr) if all_tr else np.empty(0)
                    if len(comb) < 15:
                        continue
                    pips = float(np.sum(comb['pips']))
                    wins = float(np.sum(comb['pips'][comb['pips'] > 0]))
                    losses = float(np.abs(np.sum(comb['pips'][comb['pips'] < 0])))
                    pf = round(wins / losses, 2) if losses > 0 else 0.0
                    wr = round(float(np.sum(comb['pips'] > 0)) / len(comb) * 100, 1)

                    print(f"Delta: {delta:>4.1f}p | SL: {sl:>4.1f}p | TP: {tp:>4.1f}p | Screen Trades: {len(comb):>3} | Net: {pips:>+6.1f}p | PF: {pf:>4.2f} | WR: {wr:>4.1f}%")

                    if pips > best_pips and pf > 1.15:
                        best_pips = pips
                        best_params = (delta, sl, tp)

    print("\n" + "=" * 80)
    print(f"BEST IN-SAMPLE PARAMETERS: Delta={best_params[0]}p, SL={best_params[1]}p, TP={best_params[2]}p (Screen Pips: {best_pips:+.1f}p)")
    print("=" * 80)

    # NOW TEST ON UNTOUCHED HOLDOUT (WEEKS 211..300)
    print("\n" + "=" * 80)
    print("  TESTING LOCKED PARAMETERS ON QUARANTINED HOLDOUT (WEEKS 211..300)")
    print("=" * 80)

    delta_b, sl_b, tp_b = best_params
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(eval_week, w, delta_b, sl_b, tp_b) for w in HOLD_WEEKS]
        hold_tr = []
        for f in as_completed(futures):
            tr = f.result()
            if len(tr) > 0:
                hold_tr.append(tr)

    comb_hold = np.concatenate(hold_tr) if hold_tr else np.empty(0)
    if len(comb_hold) > 0:
        h_pips = float(np.sum(comb_hold['pips']))
        h_wins = float(np.sum(comb_hold['pips'][comb_hold['pips'] > 0]))
        h_losses = float(np.abs(np.sum(comb_hold['pips'][comb_hold['pips'] < 0])))
        h_pf = round(h_wins / h_losses, 2) if h_losses > 0 else 0.0
        h_wr = round(float(np.sum(comb_hold['pips'] > 0)) / len(comb_hold) * 100, 1)
        print(f"HOLDOUT TRADES:     {len(comb_hold)}")
        print(f"HOLDOUT NET PIPS:   {h_pips:+.1f} pips")
        print(f"HOLDOUT PF:         {h_pf:.2f}")
        print(f"HOLDOUT WIN RATE:   {h_wr:.1f}%")
        print("=" * 80)
    else:
        print("Zero trades in holdout!")


if __name__ == "__main__":
    run_grid_search()

