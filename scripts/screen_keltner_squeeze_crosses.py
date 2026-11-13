#!/usr/bin/env python3
"""
================================================================================
  MULTI-PAIR KELTNER SQUEEZE BREAKOUT EXPLOSION LAB
================================================================================
Explores the John Carter TTM Volatility Squeeze & Expansion archetype across:
  - GBPCHF
  - EURNZD
  - AUDCAD
  - NZDCAD
  - EURGBP

Architecture:
  - H1 Timeframe: Bollinger Bands (20, 2.0 SD) vs Keltner Channels (20 EMA, mult*ATR).
  - Squeeze Fired: Bollinger bands expand back outside Keltner Channel.
  - Direction: H1 Close vs H1 EMA20.
  - Tactical M15 bar confirming expansion during specified session window.
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


def precompute_keltner_week(symbol: str, week_num: int):
    pip_size = PIP_SIZES.get(symbol, 0.0001)
    raw_dir = ROOT_DIR / "data" / symbol
    try:
        ticks, _ = get_week_dataset(week=week_num, raw_dir=raw_dir, warmup_prev_week_ticks=WARMUP_TICKS)
    except Exception:
        return None
    if len(ticks) <= WARMUP_TICKS + 500:
        return None

    m15 = build_time_bars(ticks, timeframe_min=15, pip_size=pip_size)
    h1 = build_time_bars(ticks, timeframe_min=60, pip_size=pip_size)
    if len(m15) < 30 or len(h1) < 25:
        return None

    h1_c = h1['close'].values
    h1_h = h1['high'].values
    h1_l = h1['low'].values

    c_s = pd.Series(h1_c)
    h1_ema20 = c_s.ewm(span=20, adjust=False).mean().values
    h1_std20 = c_s.rolling(20).std().values
    bb_upper = h1_ema20 + 2.0 * h1_std20
    bb_lower = h1_ema20 - 2.0 * h1_std20

    tr = np.maximum(h1_h - h1_l, np.maximum(np.abs(h1_h - np.roll(h1_c, 1)), np.abs(h1_l - np.roll(h1_c, 1))))
    tr[0] = h1_h[0] - h1_l[0]
    atr20 = pd.Series(tr).rolling(20).mean().bfill().values

    # Precalculate for different kc_mult
    squeeze_data = {}
    for mult in [1.2, 1.5, 1.8]:
        kc_upper = h1_ema20 + mult * atr20
        kc_lower = h1_ema20 - mult * atr20
        sqz_on = (bb_lower > kc_lower) & (bb_upper < kc_upper)
        sqz_fired = pd.Series(sqz_on).shift(1).fillna(False).values & (~sqz_on)
        h1_mom = np.where(h1_c > h1_ema20, 1, np.where(h1_c < h1_ema20, -1, 0))

        h1_df = pd.DataFrame({'sqz_fire': sqz_fired, 'h1_mom': h1_mom}, index=h1.index)
        m15_merged = pd.merge_asof(
            m15.sort_index(),
            h1_df.sort_index(),
            left_index=True,
            right_index=True,
            direction='backward'
        )
        squeeze_data[mult] = {
            'sqz_fire': m15_merged['sqz_fire'].fillna(False).values,
            'h1_mom': m15_merged['h1_mom'].fillna(0).values.astype(np.int8)
        }

    c = m15['close'].values.astype(np.float64)
    o = m15['open'].values.astype(np.float64)
    h = m15['high'].values.astype(np.float64)
    l = m15['low'].values.astype(np.float64)
    spread = m15['spread_pips'].values.astype(np.float64)
    hours = m15.index.hour.values.astype(np.int16)
    dates = m15.index.date
    time_ms = m15.index.astype('datetime64[ms]').view('int64')
    dur_ms = 15 * 60 * 1000
    tick_ts = ticks['timestamp']
    n_ticks = len(ticks)

    exec_indices = np.searchsorted(tick_ts, time_ms + dur_ms).astype(np.int32)

    return {
        'ticks': ticks,
        'n_ticks': n_ticks,
        'pip_size': pip_size,
        'min_entry_ts': ticks['timestamp'][WARMUP_TICKS],
        'squeeze_data': squeeze_data,
        'c': c, 'o': o, 'h': h, 'l': l,
        'spread': spread,
        'hours': hours,
        'dates': dates,
        'exec_indices': exec_indices,
        'n_bars': len(m15),
        'signals_buf': np.zeros(n_ticks, dtype=np.int8),
        'sl_buf': np.zeros(n_ticks, dtype=np.float64),
        'tp_buf': np.zeros(n_ticks, dtype=np.float64)
    }


def eval_keltner_week(week_data, kc_mult, sl_pips, tp_pips, start_hr, end_hr, max_spread, comm, slip):
    n_bars = week_data['n_bars']
    n_ticks = week_data['n_ticks']
    pip_size = week_data['pip_size']
    hours = week_data['hours']
    dates = week_data['dates']
    c, o = week_data['c'], week_data['o']
    spread = week_data['spread']
    exec_indices = week_data['exec_indices']
    signals_buf = week_data['signals_buf']
    sl_buf = week_data['sl_buf']
    tp_buf = week_data['tp_buf']

    sqz_fire = week_data['squeeze_data'][kc_mult]['sqz_fire']
    h1_mom = week_data['squeeze_data'][kc_mult]['h1_mom']

    last_trade_date = None
    fired_indices = []

    for i in range(15, n_bars):
        hr = hours[i]
        if start_hr <= end_hr:
            in_window = (start_hr <= hr <= end_hr)
        else:
            in_window = (hr >= start_hr or hr <= end_hr)
        if not in_window:
            continue

        dt = dates[i]
        if dt == last_trade_date:
            continue
        if spread[i] > max_spread:
            continue

        if not sqz_fire[i]:
            continue
        mom = h1_mom[i]
        if mom == 0:
            continue

        sig = 0
        if mom == 1 and c[i] > o[i]:
            sig = 1
        elif mom == -1 and c[i] < o[i]:
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


def hunt_keltner_for_symbol(symbol: str, sl_tp_grid: list, sessions: list, max_spread: float):
    pip_size = PIP_SIZES.get(symbol, 0.0001)
    comm, slip = get_friction_pips(symbol)

    disc_weeks = get_complete_split_weeks(symbol, "discovery_70")
    hold_weeks = get_complete_split_weeks(symbol, "holdout_test_30")
    sample_disc_weeks = disc_weeks[::3]

    print("=" * 80, flush=True)
    print(f"  HUNTING KELTNER SQUEEZE EXPLOSION FOR {symbol}", flush=True)
    print(f"  Discovery Sample: {len(sample_disc_weeks)}w | Quarantined Holdout: {len(hold_weeks)}w", flush=True)
    print(f"  Friction: {comm:.2f}p Commission + {slip:.2f}p Slippage + Floating Spreads", flush=True)
    print("=" * 80, flush=True)

    t0 = time.time()
    print("Precomputing Discovery cache...", flush=True)
    with ProcessPoolExecutor(max_workers=6) as ex:
        disc_cache = [w for w in ex.map(precompute_keltner_week, [symbol]*len(sample_disc_weeks), sample_disc_weeks) if w is not None]
    print(f"Discovery cache built ({len(disc_cache)}w) in {time.time()-t0:.1f}s.", flush=True)

    candidates = []

    for kc_m in [1.2, 1.5, 1.8]:
        for start_hr, end_hr, sess_name in sessions:
            for sl, tp in sl_tp_grid:
                trades_list = []
                for w in disc_cache:
                    tr = eval_keltner_week(w, kc_m, sl, tp, start_hr, end_hr, max_spread, comm, slip)
                    if len(tr) > 0:
                        trades_list.append(tr)
                if not trades_list:
                    continue
                comb = np.concatenate(trades_list)
                if len(comb) < 20:
                    continue

                pips = comb['pips']
                wins = float(np.sum(pips[pips > 0]))
                losses = float(abs(np.sum(pips[pips < 0])))
                pf = (wins / losses) if losses > 0 else 99.0
                net = wins - losses

                if pf >= 1.25 and net > 0:
                    candidates.append({
                        "symbol": symbol,
                        "kc_mult": kc_m,
                        "session_name": sess_name,
                        "start_hr": start_hr,
                        "end_hr": end_hr,
                        "sl_pips": sl,
                        "tp_pips": tp,
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
        hold_cache = [w for w in ex.map(precompute_keltner_week, [symbol]*len(hold_weeks), hold_weeks) if w is not None]
    print(f"Holdout cache built ({len(hold_cache)}w) in {time.time()-t0:.1f}s.", flush=True)

    certified = []
    print("\n" + "=" * 90, flush=True)
    print(f"{'SESSION':<20} | {'KC':<4} | {'SL/TP':<10} | {'HOLD PIPS':<10} | {'HOLD PF':<8} | {'TRADES':<7} | {'EXP':<7} | {'STATUS'}", flush=True)
    print("=" * 90, flush=True)

    for c in top_candidates:
        trades_list = []
        for w in hold_cache:
            tr = eval_keltner_week(
                w, c["kc_mult"], c["sl_pips"], c["tp_pips"],
                c["start_hr"], c["end_hr"], c["max_spread"], comm, slip
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

        print(f"{c['session_name']:<20} | {c['kc_mult']:<4.1f} | {sl_tp_str:<10} | {net:>+8.1f}p | {pf:>6.2f} | {len(comb_h):>6} | {exp:>5.2f}p | {status}", flush=True)

    print("=" * 90, flush=True)
    print(f"Symbol {symbol}: {len(certified)} non-redundant Keltner Squeeze champions certified!", flush=True)
    return certified


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="GBPCHF")
    args = parser.parse_args()

    sym = args.symbol.upper()

    if sym == "GBPCHF":
        sl_tp = [(16.0, 32.0), (18.0, 38.0), (20.0, 42.0), (22.0, 50.0), (25.0, 55.0)]
        sessions = [
            (7, 16, "London_and_NY"),
            (7, 11, "London_Morning"),
            (12, 16, "NY_Overlap"),
            (0, 6, "Asian_Tokyo")
        ]
        max_spread = 3.2
    elif sym == "EURNZD":
        sl_tp = [(24.0, 50.0), (28.0, 60.0), (32.0, 70.0), (35.0, 80.0), (40.0, 90.0)]
        sessions = [
            (6, 17, "European_and_NY"),
            (6, 10, "Frankfurt_London_Open"),
            (12, 16, "NY_Overlap"),
            (21, 2, "Asian_Wellington")
        ]
        max_spread = 5.0
    elif sym == "AUDCAD":
        sl_tp = [(12.0, 24.0), (14.0, 30.0), (16.0, 35.0), (18.0, 40.0), (20.0, 45.0)]
        sessions = [
            (7, 16, "London_and_NY"),
            (12, 16, "NY_Overlap"),
            (21, 2, "Sydney_Tokyo"),
            (7, 10, "London_Morning")
        ]
        max_spread = 2.8
    elif sym == "NZDCAD":
        sl_tp = [(12.0, 24.0), (14.0, 30.0), (16.0, 35.0), (18.0, 40.0), (20.0, 45.0)]
        sessions = [
            (7, 16, "London_and_NY"),
            (12, 16, "NY_Overlap"),
            (21, 2, "Wellington_Tokyo"),
            (7, 10, "London_Morning")
        ]
        max_spread = 3.0
    else: # EURGBP
        sl_tp = [(8.0, 16.0), (9.0, 18.0), (10.0, 20.0), (11.0, 22.0), (12.0, 25.0)]
        sessions = [
            (7, 16, "London_and_NY"),
            (6, 10, "Frankfurt_London_Open"),
            (12, 16, "NY_Overlap")
        ]
        max_spread = 1.6

    champs = hunt_keltner_for_symbol(sym, sl_tp, sessions, max_spread)
    if champs:
        out_path = ROOT_DIR / "results" / f"keltner_champions_{sym.lower()}.json"
        serializable = [{k: v for k, v in c.items() if k != "trades_array"} for c in champs]
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2)
        print(f"Saved {len(serializable)} champions to {out_path}", flush=True)

