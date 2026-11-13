#!/usr/bin/env python3
"""
================================================================================
  ASIAN HIGH/LOW RETEST REJECTION LAB FOR CROSS PAIRS
================================================================================
Tests the Asian Session High/Low Breakout & Retest Rejection archetype on:
  - EURGBP
  - AUDCAD
  - NZDCAD

Architecture:
  - Asian Reference Range established between 00:00 - 06:30 UTC.
  - Active Execution Window: London (07:00 - 11:30 UTC) or NY (12:00 - 16:00 UTC).
  - Breakout & Retest Mechanism:
      * Break Asian level by >= 1.5 - 2.5 pips.
      * Retest level within tolerance with rejection wick >= 35% - 40%.
      * Candle closes back in trend direction.
  - Macro Filter: H4 20-EMA trend agreement.
  - Strict subsequent-tick causal execution (T_fill > T_signal) and ECN friction.
  - Anti-redundancy check against all existing fleet models in symbol.
================================================================================
"""

import sys
import gc
import json
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from tick_engine.core.engine import run_matching_loop
from tick_engine.core.dataloader import get_week_dataset, get_complete_split_weeks
from tick_engine.core.config import PIP_SIZES, get_friction_pips, WARMUP_TICKS
from data_engineering_suite.bars import build_time_bars
from scripts.hunt_diversified_cross_pair_alpha import check_anti_redundancy


def precompute_asian_week(symbol: str, week_num: int):
    pip_size = PIP_SIZES.get(symbol, 0.0001)
    raw_dir = ROOT_DIR / "data" / symbol
    try:
        ticks, _ = get_week_dataset(week=week_num, raw_dir=raw_dir, warmup_prev_week_ticks=WARMUP_TICKS)
    except Exception:
        return None
    if len(ticks) <= WARMUP_TICKS + 500:
        return None

    m15 = build_time_bars(ticks, timeframe_min=15, pip_size=pip_size)
    h4 = build_time_bars(ticks, timeframe_min=240, pip_size=pip_size)
    if len(m15) < 35 or len(h4) < 10:
        return None

    h4_c = pd.Series(h4['close'].values)
    h4_ema20 = h4_c.ewm(span=20, adjust=False).mean().values
    h4_bias = np.where(h4_c.values > h4_ema20, 1, -1)
    h4_df = pd.DataFrame({'h4_bias': h4_bias}, index=h4.index)

    m15_df = pd.merge_asof(
        m15.sort_index(),
        h4_df.sort_index(),
        left_index=True,
        right_index=True,
        direction='backward'
    )

    c = m15_df['close'].values.astype(np.float64)
    h = m15_df['high'].values.astype(np.float64)
    l = m15_df['low'].values.astype(np.float64)
    o = m15_df['open'].values.astype(np.float64)
    spread = m15_df['spread_pips'].values.astype(np.float64)
    h4_trend = m15_df['h4_bias'].fillna(0).values.astype(np.int8)
    hours = m15_df.index.hour.values.astype(np.int16)
    minutes = m15_df.index.minute.values.astype(np.int16)
    dates = m15_df.index.date
    time_ms = m15_df.index.astype('datetime64[ms]').view('int64')
    dur_ms = 15 * 60 * 1000
    tick_ts = ticks['timestamp']
    n_ticks = len(ticks)

    exec_indices = np.searchsorted(tick_ts, time_ms + dur_ms).astype(np.int32)

    # Compute Asian reference range for each day (00:00 - 06:30 UTC)
    asian_ranges = {}
    for d in np.unique(dates):
        mask = (dates == d) & (hours >= 0) & ((hours < 6) | ((hours == 6) & (minutes <= 30)))
        if np.sum(mask) >= 6:
            asian_ranges[d] = (float(np.max(h[mask])), float(np.min(l[mask])))

    return {
        'ticks': ticks,
        'n_ticks': n_ticks,
        'pip_size': pip_size,
        'min_entry_ts': ticks['timestamp'][WARMUP_TICKS],
        'c': c, 'h': h, 'l': l, 'o': o,
        'spread': spread,
        'h4_trend': h4_trend,
        'hours': hours,
        'minutes': minutes,
        'dates': dates,
        'exec_indices': exec_indices,
        'asian_ranges': asian_ranges,
        'n_bars': len(m15_df),
        'signals_buf': np.zeros(n_ticks, dtype=np.int8),
        'sl_buf': np.zeros(n_ticks, dtype=np.float64),
        'tp_buf': np.zeros(n_ticks, dtype=np.float64)
    }


def eval_asian_week(week_data, sl_pips, tp_pips, start_hr, end_hr, min_wick, tol_pips, max_spread, comm, slip):
    n_bars = week_data['n_bars']
    n_ticks = week_data['n_ticks']
    pip_size = week_data['pip_size']
    c, h, l, o = week_data['c'], week_data['h'], week_data['l'], week_data['o']
    spread = week_data['spread']
    h4_trend = week_data['h4_trend']
    hours = week_data['hours']
    dates = week_data['dates']
    exec_indices = week_data['exec_indices']
    asian_ranges = week_data['asian_ranges']
    signals_buf = week_data['signals_buf']
    sl_buf = week_data['sl_buf']
    tp_buf = week_data['tp_buf']

    last_trade_date = None
    fired_indices = []

    broke_hi_map = defaultdict(bool)
    broke_lo_map = defaultdict(bool)

    tol = tol_pips * pip_size
    break_buf = 2.0 * pip_size

    for i in range(15, n_bars):
        dt = dates[i]
        hr = hours[i]

        if dt not in asian_ranges:
            continue

        asia_hi, asia_lo = asian_ranges[dt]
        asia_rng = asia_hi - asia_lo
        if asia_rng < 6.0 * pip_size:
            continue

        if h[i] > asia_hi + break_buf:
            broke_hi_map[dt] = True
        if l[i] < asia_lo - break_buf:
            broke_lo_map[dt] = True

        if not (start_hr <= hr <= end_hr):
            continue
        if dt == last_trade_date:
            continue
        if spread[i] > max_spread:
            continue

        macro = h4_trend[i]
        if macro == 0:
            continue

        bar_rng = h[i] - l[i]
        if bar_rng < 2.0 * pip_size:
            continue

        sig = 0
        # Bullish retest of broken Asian High
        if macro == 1 and broke_hi_map[dt]:
            tested_level = (l[i] <= asia_hi + tol) and (h[i] >= asia_hi - tol)
            closed_above = c[i] > asia_hi
            lower_wick = (min(o[i], c[i]) - l[i]) / max(bar_rng, 1e-9)
            if tested_level and closed_above and lower_wick >= min_wick:
                sig = 1

        # Bearish retest of broken Asian Low
        elif macro == -1 and broke_lo_map[dt]:
            tested_level = (h[i] >= asia_lo - tol) and (l[i] <= asia_lo + tol)
            closed_below = c[i] < asia_lo
            upper_wick = (h[i] - max(o[i], c[i])) / max(bar_rng, 1e-9)
            if tested_level and closed_below and upper_wick >= min_wick:
                sig = -1

        if sig != 0:
            idx = exec_indices[i]
            if idx < n_ticks and signals_buf[idx] == 0:
                signals_buf[idx] = sig
                sl_buf[idx] = sl_pips
                tp_buf[idx] = tp_pips
                fired_indices.append(idx)
                last_trade_date = dt

    if not fired_indices:
        return np.empty(0)

    tr = run_matching_loop(
        ticks=week_data['ticks'],
        signals=signals_buf,
        pip_size=pip_size,
        sl_pips=sl_buf,
        tp_pips=tp_buf,
        use_slippage=True,
        close_at_end=True,
        commission_pips=comm,
        entry_slippage_pips=slip,
        min_entry_timestamp=week_data['min_entry_ts'],
        enforce_subsequent_tick=True
    )

    for idx in fired_indices:
        signals_buf[idx] = 0
        sl_buf[idx] = 0.0
        tp_buf[idx] = 0.0

    return tr


def hunt_asian_retest_for_symbol(symbol: str, sl_tp_grid: list, sessions: list, max_spread: float):
    pip_size = PIP_SIZES.get(symbol, 0.0001)
    comm, slip = get_friction_pips(symbol)

    disc_weeks = get_complete_split_weeks(symbol, "discovery_70")
    hold_weeks = get_complete_split_weeks(symbol, "holdout_test_30")
    sample_disc_weeks = disc_weeks[::3]

    print("=" * 80, flush=True)
    print(f"  HUNTING ASIAN RETEST ALPHA FOR {symbol}", flush=True)
    print(f"  Discovery Sample: {len(sample_disc_weeks)}w | Quarantined Holdout: {len(hold_weeks)}w", flush=True)
    print(f"  Friction: {comm:.2f}p Commission + {slip:.2f}p Slippage + Floating Spreads", flush=True)
    print("=" * 80, flush=True)

    t0 = time.time()
    print("Precomputing Discovery cache...", flush=True)
    with ProcessPoolExecutor(max_workers=6) as ex:
        disc_cache = [w for w in ex.map(precompute_asian_week, [symbol]*len(sample_disc_weeks), sample_disc_weeks) if w is not None]
    print(f"Discovery cache built ({len(disc_cache)}w) in {time.time()-t0:.1f}s.", flush=True)

    candidates = []

    for start_hr, end_hr, sess_name in sessions:
        for sl, tp in sl_tp_grid:
            for min_w in [0.35, 0.40]:
                for tol in [2.0, 3.0]:
                    trades_list = []
                    for w in disc_cache:
                        tr = eval_asian_week(w, sl, tp, start_hr, end_hr, min_w, tol, max_spread, comm, slip)
                        if len(tr) > 0:
                            trades_list.append(tr)
                    if not trades_list:
                        continue
                    comb = np.concatenate(trades_list)
                    if len(comb) < 15:
                        continue

                    pips = comb['pips']
                    wins = float(np.sum(pips[pips > 0]))
                    losses = float(abs(np.sum(pips[pips < 0])))
                    pf = (wins / losses) if losses > 0 else 99.0
                    net = wins - losses

                    if pf >= 1.25 and net > 0:
                        candidates.append({
                            "symbol": symbol,
                            "session_name": sess_name,
                            "start_hr": start_hr,
                            "end_hr": end_hr,
                            "sl_pips": sl,
                            "tp_pips": tp,
                            "min_wick": min_w,
                            "tol_pips": tol,
                            "max_spread": max_spread,
                            "disc_trades": len(comb),
                            "disc_pips": round(net, 1),
                            "disc_pf": round(pf, 2),
                            "disc_exp": round(net / len(comb), 2)
                        })

    print(f"Discovery screening complete. Candidates: {len(candidates)}", flush=True)
    if not candidates:
        return []

    candidates.sort(key=lambda x: x["disc_pf"] * x["disc_pips"], reverse=True)
    top_candidates = candidates[:15]

    print(f"Precomputing Quarantined Holdout cache ({len(hold_weeks)}w)...", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=6) as ex:
        hold_cache = [w for w in ex.map(precompute_asian_week, [symbol]*len(hold_weeks), hold_weeks) if w is not None]
    print(f"Holdout cache built ({len(hold_cache)}w) in {time.time()-t0:.1f}s.", flush=True)

    certified = []
    print("\n" + "=" * 90, flush=True)
    print(f"{'SESSION':<20} | {'TOL':<4} | {'SL/TP':<10} | {'HOLD PIPS':<10} | {'HOLD PF':<8} | {'TRADES':<7} | {'EXP':<7} | {'STATUS'}", flush=True)
    print("=" * 90, flush=True)

    for c in top_candidates:
        trades_list = []
        for w in hold_cache:
            tr = eval_asian_week(
                w, c["sl_pips"], c["tp_pips"],
                c["start_hr"], c["end_hr"], c["min_wick"], c["tol_pips"], c["max_spread"], comm, slip
            )
            if len(tr) > 0:
                trades_list.append(tr)
        if not trades_list:
            continue
        comb_h = np.concatenate(trades_list)
        if len(comb_h) < 15:
            continue

        pips = comb_h['pips']
        wins = float(np.sum(pips[pips > 0]))
        losses = float(abs(np.sum(pips[pips < 0])))
        pf = (wins / losses) if losses > 0 else 99.0
        net = wins - losses
        exp = net / len(comb_h)

        pips_sorted = np.sort(pips)
        pips_ex3 = float(np.sum(pips_sorted[:-3]))

        is_distinct, anti_red_msg = check_anti_redundancy(symbol, comb_h, max_overlap_pct=50.0, max_corr=0.50)

        sl_tp_str = f"{c['sl_pips']:.0f}/{c['tp_pips']:.0f}"
        if pf >= 1.20 and net > 0 and exp >= 1.50 and pips_ex3 > 0 and is_distinct:
            status = "CERTIFIED"
            c["hold_trades"] = len(comb_h)
            c["hold_pips"] = round(net, 1)
            c["hold_pf"] = round(pf, 2)
            c["hold_exp"] = round(exp, 2)
            c["hold_pips_ex3"] = round(pips_ex3, 1)
            c["trades_array"] = comb_h
            certified.append(c)
        else:
            reason = "Failed Gates" if (pf < 1.20 or net <= 0 or exp < 1.50) else ("Outlier Dependent" if pips_ex3 <= 0 else anti_red_msg)
            status = f"REJECTED ({reason})"

        print(f"{c['session_name']:<20} | {c['tol_pips']:<4.1f} | {sl_tp_str:<10} | {net:>+8.1f}p | {pf:>6.2f} | {len(comb_h):>6} | {exp:>5.2f}p | {status}", flush=True)

    print("=" * 90, flush=True)
    print(f"Symbol {symbol}: {len(certified)} non-redundant Asian Retest champions certified!", flush=True)
    return certified


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="EURGBP")
    args = parser.parse_args()

    sym = args.symbol.upper()

    if sym == "EURGBP":
        sl_tp = [(8.0, 16.0), (9.0, 18.0), (10.0, 20.0), (11.0, 24.0)]
        sessions = [
            (7, 11, "London_Morning"),
            (12, 16, "NY_Overlap"),
            (7, 16, "London_and_NY")
        ]
        max_spread = 1.6
    elif sym == "AUDCAD":
        sl_tp = [(12.0, 24.0), (14.0, 30.0), (16.0, 35.0), (18.0, 40.0)]
        sessions = [
            (7, 11, "London_Morning"),
            (12, 16, "NY_Overlap"),
            (7, 16, "London_and_NY")
        ]
        max_spread = 2.8
    else: # NZDCAD
        sl_tp = [(12.0, 24.0), (14.0, 30.0), (16.0, 35.0), (18.0, 40.0)]
        sessions = [
            (7, 11, "London_Morning"),
            (12, 16, "NY_Overlap"),
            (7, 16, "London_and_NY")
        ]
        max_spread = 3.0

    champs = hunt_asian_retest_for_symbol(sym, sl_tp, sessions, max_spread)
    if champs:
        out_path = ROOT_DIR / "results" / f"asian_retest_champions_{sym.lower()}.json"
        serializable = [{k: v for k, v in c.items() if k != "trades_array"} for c in champs]
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2)
        print(f"Saved {len(serializable)} champions to {out_path}", flush=True)

