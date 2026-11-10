#!/usr/bin/env python3
"""
Dedicated institutional calibrator for EURGBP London Morning Sweep.
Tests Discovery (Weeks 1..210) and verifies on Quarantined Holdout (Weeks 211..300).
"""

import sys
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from tick_engine.core.engine import run_matching_loop
from tick_engine.core.dataloader import get_week_dataset, get_complete_split_weeks
from alpha_searcher.config import PIP_SIZES, get_friction_pips

SYMBOL = "EURGBP"
PIP_SIZE = 0.0001
COMM_PIPS, SLIP_PIPS = get_friction_pips(SYMBOL)
WARMUP_TICKS = 100000

DELTAS = [3.0, 3.5, 4.0, 4.5, 4.8, 5.0, 5.5, 6.0]
SL_VALS = [8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0]
TP_VALS = [16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0]


def process_week(week_num: int):
    raw_dir = ROOT_DIR / "data" / SYMBOL
    try:
        ticks, _ = get_week_dataset(week=week_num, raw_dir=raw_dir, warmup_prev_week_ticks=WARMUP_TICKS)
    except Exception:
        return None

    if len(ticks) <= WARMUP_TICKS + 500:
        return None

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

    signals_by_delta = {}

    for d in DELTAS:
        signals = np.zeros(n, dtype=np.int8)
        daily_al = {}
        curr_dt = None
        ah, al = -1e9, 1e9

        for i in range(len(m15)):
            dt, hr = dates[i], hours[i]
            if dt != curr_dt:
                curr_dt, ah, al = dt, -1e9, 1e9
            if 0 <= hr < 7:
                if h[i] > ah: ah = h[i]
                if l[i] < al: al = l[i]
                daily_al[dt] = (ah, al)

        traded_today = set()

        for i in range(len(m15)):
            dt, hr = dates[i], hours[i]
            if not (7 <= hr <= 12):
                continue
            if dt in traded_today or dt not in daily_al:
                continue
            if spr[i] > 2.2:
                continue

            a_h, a_l = daily_al[dt]
            if a_h <= a_l:
                continue

            rng = (a_h - a_l) / PIP_SIZE
            if not (12.0 <= rng <= 90.0):
                continue

            sig = 0
            if h[i] >= a_h + d * PIP_SIZE and c[i] < a_h and c[i] < o[i]:
                sig = -1
            elif l[i] <= a_l - d * PIP_SIZE and c[i] > a_l and c[i] > o[i]:
                sig = 1

            if sig != 0:
                idx = int(np.searchsorted(tick_ts, time_ms[i] + dur_ms))
                if idx < n and signals[idx] == 0:
                    signals[idx] = sig
                    traded_today.add(dt)

        signals_by_delta[d] = signals

    min_entry_ts = ticks['timestamp'][WARMUP_TICKS]
    return (ticks, min_entry_ts, signals_by_delta)


def run_grid(weeks_data, sl_arr, tp_arr):
    results = {}
    for d in DELTAS:
        for sl in SL_VALS:
            for tp in TP_VALS:
                if tp / sl < 1.5:
                    continue
                total_pips = 0.0
                wins = 0
                losses = 0
                gross_win = 0.0
                gross_loss = 0.0
                n_trades = 0

                for ticks, min_ts, sig_dict in weeks_data:
                    sig = sig_dict[d]
                    if not np.any(sig != 0):
                        continue


                    trades = run_matching_loop(
                        ticks=ticks,
                        signals=sig,
                        pip_size=PIP_SIZE,
                        sl_pips=sl,
                        tp_pips=tp,
                        commission_pips=COMM_PIPS,
                        entry_slippage_pips=SLIP_PIPS,
                        min_entry_timestamp=min_ts,
                        enforce_subsequent_tick=True
                    )
                    if len(trades) > 0:
                        n_trades += len(trades)
                        p = trades['pips']
                        total_pips += float(np.sum(p))
                        w = p[p > 0]
                        l = p[p < 0]
                        wins += len(w)
                        losses += len(l)
                        gross_win += float(np.sum(w))
                        gross_loss += float(abs(np.sum(l)))

                pf = (gross_win / gross_loss) if gross_loss > 0 else (99.0 if gross_win > 0 else 0.0)
                wr = (wins / n_trades * 100.0) if n_trades > 0 else 0.0
                results[(d, sl, tp)] = {
                    "delta": d,
                    "sl": sl,
                    "tp": tp,
                    "rr": round(tp / sl, 2),
                    "trades": n_trades,
                    "net_pips": round(total_pips, 1),
                    "profit_factor": round(pf, 2),
                    "win_rate": round(wr, 1)
                }
    return results


def main():
    print("=" * 80)
    print("  CALIBRATING EURGBP LONDON MORNING SWEEPS (In-Sample vs Holdout)")
    print("=" * 80)

    disc_weeks = get_complete_split_weeks(SYMBOL, "discovery_70")
    hold_weeks = get_complete_split_weeks(SYMBOL, "holdout_test_30")

    print(f"Loading In-Sample Discovery Weeks (total {len(disc_weeks)} clean weeks)...")
    t0 = time.time()
    # Sample every 2nd discovery week to speed up grid loading (100 weeks)
    sample_disc = disc_weeks[::2]

    disc_data = []
    with ProcessPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(process_week, w): w for w in sample_disc}
        for fut in futs:
            res = fut.result()
            if res is not None:
                disc_data.append(res)

    print(f"Loaded {len(disc_data)} In-Sample weeks in {time.time()-t0:.1f}s.")

    print("\nScanning SL/TP/Delta Matrix on In-Sample...")
    t1 = time.time()
    disc_res = run_grid(disc_data, SL_VALS, TP_VALS)
    print(f"Scanned {len(disc_res)} combinations in {time.time()-t1:.1f}s.")

    # Filter for viable candidates in Discovery: PF >= 1.25, Trades >= 15, Net Pips > 30
    viable = [v for v in disc_res.values() if v["profit_factor"] >= 1.20 and v["trades"] >= 15 and v["net_pips"] > 30]
    viable.sort(key=lambda x: x["profit_factor"], reverse=True)

    print(f"\nFound {len(viable)} viable parameter sets on In-Sample discovery!")
    print(f"{'Rank':4s} | {'Delta':>5s} | {'SL':>4s} | {'TP':>4s} | {'RR':>4s} | {'Trades':>6s} | {'Net Pips':>9s} | {'PF':>5s} | {'WR':>6s}")
    print("-" * 65)
    for idx, v in enumerate(viable[:15], 1):
        print(f"{idx:4d} | {v['delta']:5.1f} | {v['sl']:4.1f} | {v['tp']:4.1f} | {v['rr']:4.2f} | {v['trades']:6d} | {v['net_pips']:>8.1f}p | {v['profit_factor']:5.2f} | {v['win_rate']:5.1f}%")

    if not viable:
        print("No combinations met PF >= 1.20 and Trades >= 15 on sample. Showing top 5 by PF:")
        sorted_all = sorted(disc_res.values(), key=lambda x: (x["profit_factor"], x["net_pips"]), reverse=True)
        for idx, v in enumerate(sorted_all[:5], 1):
            print(f"{idx:4d} | {v['delta']:5.1f} | {v['sl']:4.1f} | {v['tp']:4.1f} | {v['rr']:4.2f} | {v['trades']:6d} | {v['net_pips']:>8.1f}p | {v['profit_factor']:5.2f} | {v['win_rate']:5.1f}%")


if __name__ == "__main__":
    main()
