#!/usr/bin/env python3
"""
================================================================================
  ULTRA-FAST MULTI-PAIR ASIAN & NY SWEEP CALIBRATOR
================================================================================
Loads each week's tick data once into memory per process, generates signals for
candidate sweep deltas, and scans SL/TP matrices via compiled Numba loop in <10s.
Evaluates In-Sample Discovery (Weeks 1..210) to select parameters,
then independently validates on Quarantined Holdout (Weeks 211..300).
================================================================================
"""

import sys
import json
import time
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from tick_engine.core.engine import run_matching_loop
from tick_engine.core.dataloader import get_week_dataset, get_complete_split_weeks
from tick_engine.core.config import PIP_SIZES, get_friction_pips

WARMUP_TICKS = 100000


def _process_week_deltas(
    symbol: str,
    week_num: int,
    session_type: str, # "asian" or "ny"
    deltas: list[float],
    min_range: float = 12.0,
    max_range: float = 80.0
):
    """Loads tick stream once, builds bars, and returns (ticks, dict of signals per delta)."""
    raw_dir = ROOT_DIR / "data" / symbol
    pip_size = PIP_SIZES.get(symbol, 0.0001)

    try:
        ticks, _ = get_week_dataset(week=week_num, raw_dir=raw_dir, warmup_prev_week_ticks=WARMUP_TICKS)
    except Exception:
        return None

    if len(ticks) <= WARMUP_TICKS + 500:
        return None

    # Resample M15 bars
    ts = pd.to_datetime(ticks['timestamp'], unit='ms')
    mid = 0.5 * (ticks['bid'] + ticks['ask'])
    spread = (ticks['ask'] - ticks['bid']) / pip_size
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

    for d in deltas:
        signals = np.zeros(n, dtype=np.int8)
        curr_date = None
        s_high, s_low = -1e9, 1e9
        traded_today = False

        for i in range(len(m15)):
            dt = dates[i]
            hr = hours[i]

            if dt != curr_date:
                curr_date = dt
                s_high, s_low = -1e9, 1e9
                traded_today = False

            # Session ranges
            if session_type == "asian":
                range_hr = (0 <= hr < 7)
                trade_hr = (7 <= hr <= 14)
                max_spr = 2.8
            else: # NY sweep
                range_hr = (0 <= hr < 12)
                trade_hr = (12 <= hr <= 18)
                max_spr = 2.2

            if range_hr:
                if h[i] > s_high: s_high = h[i]
                if l[i] < s_low: s_low = l[i]
                continue

            if trade_hr and not traded_today and s_high > s_low:
                rng = (s_high - s_low) / pip_size
                if not (min_range <= rng <= max_range):
                    continue
                if spr[i] > max_spr:
                    continue

                sig = 0
                if h[i] >= s_high + d * pip_size and c[i] < s_high and c[i] < o[i]:
                    sig = -1  # Short fade
                elif l[i] <= s_low - d * pip_size and c[i] > s_low and c[i] > o[i]:
                    sig = 1   # Long fade

                if sig != 0:
                    idx = int(np.searchsorted(tick_ts, time_ms[i] + dur_ms))
                    if idx < n and signals[idx] == 0:
                        signals[idx] = sig
                        traded_today = True

        signals_by_delta[d] = signals

    min_entry_ts = ticks['timestamp'][WARMUP_TICKS]
    return (ticks, min_entry_ts, signals_by_delta)


def calibrate_symbol(symbol: str, session_type: str = "asian"):
    pip_size = PIP_SIZES.get(symbol, 0.0001)
    comm_pips, slip_pips = get_friction_pips(symbol)

    disc_weeks = get_complete_split_weeks(symbol, "discovery_70")
    hold_weeks = get_complete_split_weeks(symbol, "holdout_test_30")

    # Select representative in-sample weeks (e.g. 70 weeks across 4 years)
    screen_weeks = disc_weeks[::3]

    if symbol == "EURGBP":
        deltas = [3.0, 3.8, 4.5, 4.8, 5.5]
        sl_vals = [8.0, 10.0, 11.0, 12.0]
        tp_vals = [18.0, 22.0, 25.0, 28.0]
        min_rng, max_rng = 10.0, 60.0
    elif symbol in ["EURUSD", "NZDCAD"]:
        deltas = [3.5, 4.5, 5.5, 7.0]
        sl_vals = [10.0, 12.0, 14.0, 16.0]
        tp_vals = [20.0, 25.0, 30.0, 35.0]
        min_rng, max_rng = 12.0, 70.0
    elif symbol == "USDCAD":
        deltas = [5.0, 6.5, 8.0, 9.5]
        sl_vals = [12.0, 15.0, 18.0, 22.0]
        tp_vals = [28.0, 35.0, 45.0, 55.0]
        min_rng, max_rng = 15.0, 80.0
    else:
        deltas = [6.0, 8.0, 10.0, 12.0, 14.0]
        sl_vals = [14.0, 18.0, 22.0, 26.0]
        tp_vals = [30.0, 40.0, 50.0, 60.0]
        min_rng, max_rng = 15.0, 100.0
    tp_vals = [30.0, 40.0, 50.0, 60.0]

    print("=" * 90)
    print(f"  CALIBRATING {symbol} ({session_type.upper()} SWEEP): SCREENING {len(screen_weeks)} IN-SAMPLE WEEKS")
    print(f"  Friction: {comm_pips:.2f}p Commission + {slip_pips:.2f}p Slippage + Floating Spread")
    print("=" * 90)

    t0 = time.time()
    week_data_list = []

    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(_process_week_deltas, symbol, w, session_type, deltas, min_rng, max_rng) for w in screen_weeks]
        for f in as_completed(futures):
            res = f.result()
            if res is not None:
                week_data_list.append(res)

    print(f"Loaded and signal-mapped {len(week_data_list)} weeks in {time.time() - t0:.1f}s.")

    # Now evaluate all 5 x 4 x 4 = 80 combinations in memory!
    best_pf = -1
    best_pips = -9999
    best_cfg = None

    for d in deltas:
        for sl in sl_vals:
            for tp in tp_vals:
                trades_all = []
                for ticks, min_ts, sigs_dict in week_data_list:
                    sig = sigs_dict[d]
                    if not np.any(sig != 0):
                        continue
                    tr = run_matching_loop(
                        ticks=ticks,
                        signals=sig,
                        pip_size=pip_size,
                        sl_pips=sl,
                        tp_pips=tp,
                        use_slippage=True,
                        close_at_end=True,
                        commission_pips=comm_pips,
                        entry_slippage_pips=slip_pips,
                        min_entry_timestamp=min_ts,
                        enforce_subsequent_tick=True
                    )
                    if len(tr) > 0:
                        trades_all.append(tr)

                comb = np.concatenate(trades_all) if trades_all else np.empty(0)
                if len(comb) < 15:
                    continue
                pips = float(np.sum(comb['pips']))
                wins = float(np.sum(comb['pips'][comb['pips'] > 0]))
                losses = float(np.abs(np.sum(comb['pips'][comb['pips'] < 0])))
                pf = round(wins / losses, 2) if losses > 0 else 0.0
                wr = round(float(np.sum(comb['pips'] > 0)) / len(comb) * 100, 1)

                if pf > 1.15 and pips > 0:
                    print(f"  Delta: {d:>4.1f}p | SL: {sl:>4.1f}p | TP: {tp:>4.1f}p | Trades: {len(comb):>3} | Net: {pips:>+6.1f}p | PF: {pf:>4.2f} | WR: {wr:>4.1f}%")

                if pips > best_pips and pf > 1.15:
                    best_pips = pips
                    best_pf = pf
                    best_cfg = (d, sl, tp)

    if not best_cfg:
        print("No parameter combination passed the in-sample hurdle (PF > 1.15, Net > 0)!")
        return None

    print("\n" + "=" * 90)
    print(f"  OPTIMAL IN-SAMPLE CONFIG: Delta={best_cfg[0]}p, SL={best_cfg[1]}p, TP={best_cfg[2]}p (PF {best_pf:.2f}, Net {best_pips:+.1f}p)")
    print("=" * 90)

    # NOW TEST ON FULL QUARANTINED HOLDOUT (WEEKS 211..300)
    print("\n" + "=" * 90)
    print(f"  VERIFYING OPTIMAL CONFIG ON UNTOUCHED HOLDOUT: {len(hold_weeks)} WEEKS (WEEKS 211..300)")
    print("=" * 90)

    d_opt, sl_opt, tp_opt = best_cfg
    t1 = time.time()
    hold_trades_all = []

    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(_process_week_deltas, symbol, w, session_type, [d_opt], min_rng, max_rng) for w in hold_weeks]
        for f in as_completed(futures):
            res = f.result()
            if res is not None:
                ticks, min_ts, sigs_dict = res
                sig = sigs_dict[d_opt]
                if np.any(sig != 0):
                    tr = run_matching_loop(
                        ticks=ticks,
                        signals=sig,
                        pip_size=pip_size,
                        sl_pips=sl_opt,
                        tp_pips=tp_opt,
                        use_slippage=True,
                        close_at_end=True,
                        commission_pips=comm_pips,
                        entry_slippage_pips=slip_pips,
                        min_entry_timestamp=min_ts,
                        enforce_subsequent_tick=True
                    )
                    if len(tr) > 0:
                        hold_trades_all.append(tr)

    comb_hold = np.concatenate(hold_trades_all) if hold_trades_all else np.empty(0)
    if len(comb_hold) > 0:
        h_pips = float(np.sum(comb_hold['pips']))
        h_wins = float(np.sum(comb_hold['pips'][comb_hold['pips'] > 0]))
        h_losses = float(np.abs(np.sum(comb_hold['pips'][comb_hold['pips'] < 0])))
        h_pf = round(h_wins / h_losses, 2) if h_losses > 0 else 0.0
        h_wr = round(float(np.sum(comb_hold['pips'] > 0)) / len(comb_hold) * 100, 1)

        print(f"  HOLDOUT TRADES:     {len(comb_hold)}")
        print(f"  HOLDOUT NET PIPS:   {h_pips:+.1f} pips")
        print(f"  HOLDOUT PF:         {h_pf:.2f}")
        print(f"  HOLDOUT WIN RATE:   {h_wr:.1f}%")
        print("=" * 90)
        return {
            "symbol": symbol,
            "session": session_type,
            "params": best_cfg,
            "holdout_trades": len(comb_hold),
            "holdout_pips": h_pips,
            "holdout_pf": h_pf,
            "holdout_wr": h_wr
        }
    else:
        print("Zero trades in holdout!")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, default="GBPCHF")
    parser.add_argument("--session", type=str, choices=["asian", "ny"], default="asian")
    args = parser.parse_args()

    calibrate_symbol(args.symbol, args.session)
