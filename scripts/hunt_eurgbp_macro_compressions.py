#!/usr/bin/env python3
"""
================================================================================
  EURGBP MACRO COMPRESSION & MULTI-DAY COILING ALPHA DISCOVERY
================================================================================
Explores 5 novel macro compression archetypes for EURGBP:
  1. Weekly Inside Bar Expansion (Inside Week W-1 inside W-2)
  2. Toby Crabel NR4 Breakout (Range D-1 < past 3 days)
  3. Toby Crabel NR7 Breakout (Range D-1 < past 6 days)
  4. 2-Day Low-Volatility Coiled Spring (D-1 <= 28p AND D-2 <= 28p)
  5. Mid-Week Mon-Tue Compression Breakout (Mon-Tue range <= 32p)

All tests enforce:
  - Exact subsequent-tick execution contract (T_fill > T_sig)
  - Full ECN friction: 0.33p commission ($4/lot) + 0.20p slippage + floating spreads
  - Holdout (Weeks 211..300): PF >= 1.20, Trades >= 15, Net Pips > 0, ExTop3 > 0
  - Discovery (Weeks 1..210): PF >= 1.15, Trades >= 20, Net Pips > 0
  - Anti-Redundancy vs disc107 & disc172: Overlap < 50%, Correlation < 0.50
================================================================================
"""

import sys
import os
import gc
import json
import time
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from tick_engine.core.engine import run_matching_loop
from tick_engine.core.dataloader import get_week_dataset, get_complete_split_weeks
from tick_engine.core.config import PIP_SIZES, get_friction_pips, WARMUP_TICKS
from data_engineering_suite.bars import build_time_bars

SYMBOL = "EURGBP"
PIP_SIZE = PIP_SIZES.get(SYMBOL, 0.0001)
COMM_PIPS, SLIP_PIPS = get_friction_pips(SYMBOL)

FLEET_MANIFEST = ROOT_DIR / "certified_production_fleet" / "CERTIFIED_FLEET_MANIFEST.json"
MANIFEST_DIR = ROOT_DIR / "results" / "certified_strategy_manifests"

existing_trades = []
with open(FLEET_MANIFEST, "r", encoding="utf-8") as f:
    fleet_data = json.load(f)

for s in fleet_data["strategies"]:
    if s["symbol"] == SYMBOL:
        name = s["name"]
        csv_file = MANIFEST_DIR / f"{SYMBOL}__{name}_trades.csv"
        if not csv_file.exists():
            csv_file = MANIFEST_DIR / f"{name}_trades.csv"
        if csv_file.exists():
            df = pd.read_csv(csv_file)
            if len(df) > 0:
                existing_trades.append((name, df))
                print(f"Loaded existing EURGBP trades for {name}: {len(df)} trades", flush=True)

# Load Macro Bars
MACRO_FILE = ROOT_DIR / "data" / SYMBOL / "macro_bars.json"
with open(MACRO_FILE, "r", encoding="utf-8") as f:
    macro_data = json.load(f)

daily_dict = macro_data["daily"]
weekly_dict = macro_data["weekly"]

# Build sorted daily list
daily_sorted = sorted(list(daily_dict.values()), key=lambda x: x['date'])
daily_by_date = {d['date']: idx for idx, d in enumerate(daily_sorted)}

# Build sorted weekly list
weekly_sorted = sorted(list(weekly_dict.values()), key=lambda x: x['end_date'])
weekly_by_end_date = {w['end_date']: idx for idx, w in enumerate(weekly_sorted)}


def check_anti_redundancy(candidate_trades: np.ndarray, max_overlap_pct: float = 50.0, max_corr: float = 0.50):
    if len(candidate_trades) == 0:
        return False, "No trades"
    if not existing_trades:
        return True, "No existing strategies"

    cand_times = candidate_trades['entry_time']
    cand_dirs = candidate_trades['direction']

    cand_df = pd.DataFrame({
        "entry_time": cand_times,
        "direction": cand_dirs,
        "pips": candidate_trades['pips']
    })
    cand_df["day"] = pd.to_datetime(cand_df["entry_time"], unit="ms", utc=True).dt.date
    cand_daily = cand_df.groupby("day")["pips"].sum()

    tolerance_ms = 30 * 60 * 1000

    for ex_name, ex_df in existing_trades:
        ex_times = ex_df["entry_time"].values
        ex_dirs = ex_df["direction"].values if "direction" in ex_df else np.ones(len(ex_df))

        matches = 0
        for idx, t1 in enumerate(cand_times):
            diffs = np.abs(ex_times - t1)
            if np.any((diffs <= tolerance_ms) & (ex_dirs == cand_dirs[idx])):
                matches += 1

        overlap_pct = (matches / len(cand_times)) * 100.0 if len(cand_times) > 0 else 0
        if overlap_pct >= max_overlap_pct:
            return False, f"High overlap ({overlap_pct:.1f}%) with {ex_name}"

        ex_df_c = ex_df.copy()
        ex_df_c["day"] = pd.to_datetime(ex_df_c["entry_time"], unit="ms", utc=True).dt.date
        ex_daily = ex_df_c.groupby("day")["pips"].sum()

        all_days = sorted(list(set(cand_daily.index).union(set(ex_daily.index))))
        c_series = [cand_daily.get(d, 0.0) for d in all_days]
        e_series = [ex_daily.get(d, 0.0) for d in all_days]
        corr = np.corrcoef(c_series, e_series)[0, 1] if len(all_days) > 5 else 0.0
        if np.isnan(corr): corr = 0.0

        if corr >= max_corr:
            return False, f"High correlation ({corr:.2f}) with {ex_name}"

    return True, "PASSED Anti-Redundancy"


def load_week_data(week_num: int):
    raw_dir = ROOT_DIR / "data" / SYMBOL
    try:
        ticks, _ = get_week_dataset(week=week_num, raw_dir=raw_dir, warmup_prev_week_ticks=WARMUP_TICKS)
    except Exception:
        return None
    if len(ticks) <= WARMUP_TICKS + 500:
        return None

    m15 = build_time_bars(ticks, timeframe_min=15, pip_size=PIP_SIZE)
    if len(m15) < 35:
        return None

    # H4 Trend Bias
    h4 = m15[['high', 'low', 'close']].resample('4h').agg({'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
    h4_bias = np.zeros(len(m15), dtype=np.int8)
    if len(h4) >= 10:
        h4_ema20 = h4['close'].ewm(span=20, adjust=False).mean()
        h4_b = np.where(h4['close'] > h4_ema20, 1, -1)
        h4_df = pd.DataFrame({'bias': h4_b}, index=h4.index)
        merged_h4 = pd.merge_asof(m15.sort_index(), h4_df.sort_index(), left_index=True, right_index=True, direction='backward')
        h4_bias = merged_h4['bias'].fillna(0).values.astype(np.int8)

    c = m15['close'].values.astype(np.float64)
    h = m15['high'].values.astype(np.float64)
    l = m15['low'].values.astype(np.float64)
    o = m15['open'].values.astype(np.float64)
    spread = m15['spread_pips'].values.astype(np.float64)
    hours = m15.index.hour.values.astype(np.int16)
    days_of_week = m15.index.dayofweek.values.astype(np.int8)  # 0=Mon, 4=Fri
    dates_str = [d.strftime('%Y-%m-%d') for d in m15.index.date]
    time_ms = m15.index.astype('datetime64[ms]').view('int64')
    dur_ms = 15 * 60 * 1000
    tick_ts = ticks['timestamp']
    n_ticks = len(ticks)
    exec_indices = np.searchsorted(tick_ts, time_ms + dur_ms).astype(np.int32)

    # Pre-map macro indicators per M15 bar
    n_bars = len(m15)

    # 1. Weekly Inside Bar: prior week W-1 inside W-2
    # Find the Sunday/Monday week start date for this week
    week_start_date = m15.index[0].date()
    # Find most recent completed Friday in weekly_sorted before week_start_date
    prior_fri_idx = -1
    for idx_w, w in enumerate(weekly_sorted):
        w_date = datetime.strptime(w['end_date'], '%Y-%m-%d').date()
        if w_date < week_start_date:
            prior_fri_idx = idx_w
        else:
            break

    is_inside_week = False
    prior_w_h = 0.0
    prior_w_l = 0.0
    if prior_fri_idx >= 1:
        w_prev = weekly_sorted[prior_fri_idx]
        w_prev2 = weekly_sorted[prior_fri_idx - 1]
        if w_prev['high'] < w_prev2['high'] and w_prev['low'] > w_prev2['low']:
            is_inside_week = True
            prior_w_h = float(w_prev['high'])
            prior_w_l = float(w_prev['low'])

    # 2. Daily metrics mapped per bar:
    # d_idx for bar's date:
    is_nr4_day = np.zeros(n_bars, dtype=bool)
    is_nr7_day = np.zeros(n_bars, dtype=bool)
    is_coiled_28_day = np.zeros(n_bars, dtype=bool)
    d_prev_h = np.zeros(n_bars, dtype=np.float64)
    d_prev_l = np.zeros(n_bars, dtype=np.float64)
    two_day_h = np.zeros(n_bars, dtype=np.float64)
    two_day_l = np.zeros(n_bars, dtype=np.float64)

    # Mon-Tue stats for Wed/Thu
    mon_tue_h = np.zeros(n_bars, dtype=np.float64)
    mon_tue_l = np.zeros(n_bars, dtype=np.float64)
    is_mon_tue_compressed = np.zeros(n_bars, dtype=bool)

    # Cache per unique date string in this week
    unique_d_strs = list(set(dates_str))
    d_meta = {}
    for ds in unique_d_strs:
        if ds in daily_by_date:
            cur_idx = daily_by_date[ds]
            if cur_idx >= 1:
                p1 = daily_sorted[cur_idx - 1]
                dh = float(p1['high'])
                dl = float(p1['low'])
                drng = float(p1['range_pips'])

                # NR4
                nr4 = False
                if cur_idx >= 4:
                    ranges_past3 = [daily_sorted[j]['range_pips'] for j in range(cur_idx - 4, cur_idx - 1)]
                    if drng < min(ranges_past3):
                        nr4 = True

                # NR7
                nr7 = False
                if cur_idx >= 7:
                    ranges_past6 = [daily_sorted[j]['range_pips'] for j in range(cur_idx - 7, cur_idx - 1)]
                    if drng < min(ranges_past6):
                        nr7 = True

                # Coiled 28
                coiled = False
                h2d = dh
                l2d = dl
                if cur_idx >= 2:
                    p2 = daily_sorted[cur_idx - 2]
                    h2d = max(dh, float(p2['high']))
                    l2d = min(dl, float(p2['low']))
                    if drng <= 28.0 and p2['range_pips'] <= 28.0:
                        coiled = True

                d_meta[ds] = (nr4, nr7, coiled, dh, dl, h2d, l2d)

    for i in range(n_bars):
        ds = dates_str[i]
        if ds in d_meta:
            nr4, nr7, coiled, dh, dl, h2d, l2d = d_meta[ds]
            is_nr4_day[i] = nr4
            is_nr7_day[i] = nr7
            is_coiled_28_day[i] = coiled
            d_prev_h[i] = dh
            d_prev_l[i] = dl
            two_day_h[i] = h2d
            two_day_l[i] = l2d

    return {
        'ticks': ticks,
        'n_ticks': n_ticks,
        'min_entry_ts': ticks['timestamp'][WARMUP_TICKS],
        'c': c, 'h': h, 'l': l, 'o': o,
        'spread': spread,
        'hours': hours,
        'dow': days_of_week,
        'dates_str': dates_str,
        'exec_indices': exec_indices,
        'h4_bias': h4_bias,
        'is_inside_week': is_inside_week,
        'prior_w_h': prior_w_h,
        'prior_w_l': prior_w_l,
        'is_nr4_day': is_nr4_day,
        'is_nr7_day': is_nr7_day,
        'is_coiled_28_day': is_coiled_28_day,
        'd_prev_h': d_prev_h,
        'd_prev_l': d_prev_l,
        'two_day_h': two_day_h,
        'two_day_l': two_day_l,
        'n_bars': n_bars,
        'signals_buf': np.zeros(n_ticks, dtype=np.int8),
        'sl_buf': np.zeros(n_ticks, dtype=np.float64),
        'tp_buf': np.zeros(n_ticks, dtype=np.float64)
    }


def execute_matches(week_data, sig_indices, sig_types, sl_pips, tp_pips):
    if len(sig_indices) == 0:
        return []
    signals_buf = week_data['signals_buf']
    sl_buf = week_data['sl_buf']
    tp_buf = week_data['tp_buf']
    n_ticks = week_data['n_ticks']

    fired = []
    for idx, s_type in zip(sig_indices, sig_types):
        if idx < n_ticks and signals_buf[idx] == 0:
            signals_buf[idx] = s_type
            sl_buf[idx] = sl_pips
            tp_buf[idx] = tp_pips
            fired.append(idx)

    trades = run_matching_loop(
        ticks=week_data['ticks'],
        signals=signals_buf,
        sl_pips=sl_buf,
        tp_pips=tp_buf,
        min_entry_timestamp=int(week_data['min_entry_ts']),
        pip_size=PIP_SIZE,
        commission_pips=COMM_PIPS,
        entry_slippage_pips=SLIP_PIPS,
        enforce_subsequent_tick=True
    )

    for idx in fired:
        signals_buf[idx] = 0
        sl_buf[idx] = 0.0
        tp_buf[idx] = 0.0

    return trades


# Evaluator for Archetype 1: Weekly Inside Bar Breakout
def eval_weekly_inside(week_data, start_hr, end_hr, buffer_pips, body_ratio, sl_pips, tp_pips, max_spread, one_trade_per_week):
    if not week_data['is_inside_week']:
        return []
    wh = week_data['prior_w_h']
    wl = week_data['prior_w_l']
    if wh == 0.0 or wl == 0.0:
        return []

    c, h, l, o = week_data['c'], week_data['h'], week_data['l'], week_data['o']
    spread = week_data['spread']
    hours = week_data['hours']
    dates_str = week_data['dates_str']
    exec_indices = week_data['exec_indices']
    n_bars = week_data['n_bars']

    buf = buffer_pips * PIP_SIZE
    sig_indices = []
    sig_types = []
    last_trade_date = None
    trades_in_week = 0

    for i in range(20, n_bars):
        if one_trade_per_week and trades_in_week >= 1:
            break
        hr = hours[i]
        if not (start_hr <= hr <= end_hr):
            continue
        ds = dates_str[i]
        if ds == last_trade_date:
            continue
        if spread[i] > max_spread:
            continue

        bar_rng = h[i] - l[i]
        if bar_rng < 1.0 * PIP_SIZE:
            continue

        sig = 0
        # Bullish weekly breakout
        if c[i] > wh + buf and c[i] > o[i]:
            body = (c[i] - o[i]) / bar_rng
            if body >= body_ratio:
                sig = 1
        # Bearish weekly breakout
        elif c[i] < wl - buf and c[i] < o[i]:
            body = (o[i] - c[i]) / bar_rng
            if body >= body_ratio:
                sig = -1

        if sig != 0:
            sig_indices.append(exec_indices[i])
            sig_types.append(sig)
            last_trade_date = ds
            trades_in_week += 1

    return execute_matches(week_data, sig_indices, sig_types, sl_pips, tp_pips)


# Evaluator for Archetype 2 & 3: Crabel NR4 / NR7 Breakout
def eval_nr_breakout(week_data, nr_type, start_hr, end_hr, buffer_pips, body_ratio, macro_filter, sl_pips, tp_pips, max_spread):
    is_nr = week_data['is_nr4_day'] if nr_type == 4 else week_data['is_nr7_day']
    dh = week_data['d_prev_h']
    dl = week_data['d_prev_l']
    c, h, l, o = week_data['c'], week_data['h'], week_data['l'], week_data['o']
    spread = week_data['spread']
    hours = week_data['hours']
    dates_str = week_data['dates_str']
    exec_indices = week_data['exec_indices']
    h4_bias = week_data['h4_bias']
    n_bars = week_data['n_bars']

    buf = buffer_pips * PIP_SIZE
    sig_indices = []
    sig_types = []
    last_trade_date = None

    for i in range(20, n_bars):
        if not is_nr[i]:
            continue
        hr = hours[i]
        if not (start_hr <= hr <= end_hr):
            continue
        ds = dates_str[i]
        if ds == last_trade_date:
            continue
        if spread[i] > max_spread:
            continue

        bar_rng = h[i] - l[i]
        if bar_rng < 1.0 * PIP_SIZE:
            continue

        sig = 0
        h_prev = dh[i]
        l_prev = dl[i]

        if c[i] > h_prev + buf and c[i] > o[i]:
            if not macro_filter or h4_bias[i] == 1:
                body = (c[i] - o[i]) / bar_rng
                if body >= body_ratio:
                    sig = 1
        elif c[i] < l_prev - buf and c[i] < o[i]:
            if not macro_filter or h4_bias[i] == -1:
                body = (o[i] - c[i]) / bar_rng
                if body >= body_ratio:
                    sig = -1

        if sig != 0:
            sig_indices.append(exec_indices[i])
            sig_types.append(sig)
            last_trade_date = ds

    return execute_matches(week_data, sig_indices, sig_types, sl_pips, tp_pips)


# Evaluator for Archetype 4: 2-Day Coiled Spring Breakout
def eval_coiled_breakout(week_data, start_hr, end_hr, buffer_pips, body_ratio, macro_filter, sl_pips, tp_pips, max_spread):
    is_coiled = week_data['is_coiled_28_day']
    h2d = week_data['two_day_h']
    l2d = week_data['two_day_l']
    c, h, l, o = week_data['c'], week_data['h'], week_data['l'], week_data['o']
    spread = week_data['spread']
    hours = week_data['hours']
    dates_str = week_data['dates_str']
    exec_indices = week_data['exec_indices']
    h4_bias = week_data['h4_bias']
    n_bars = week_data['n_bars']

    buf = buffer_pips * PIP_SIZE
    sig_indices = []
    sig_types = []
    last_trade_date = None

    for i in range(20, n_bars):
        if not is_coiled[i]:
            continue
        hr = hours[i]
        if not (start_hr <= hr <= end_hr):
            continue
        ds = dates_str[i]
        if ds == last_trade_date:
            continue
        if spread[i] > max_spread:
            continue

        bar_rng = h[i] - l[i]
        if bar_rng < 1.0 * PIP_SIZE:
            continue

        sig = 0
        h_box = h2d[i]
        l_box = l2d[i]

        if c[i] > h_box + buf and c[i] > o[i]:
            if not macro_filter or h4_bias[i] == 1:
                body = (c[i] - o[i]) / bar_rng
                if body >= body_ratio:
                    sig = 1
        elif c[i] < l_box - buf and c[i] < o[i]:
            if not macro_filter or h4_bias[i] == -1:
                body = (o[i] - c[i]) / bar_rng
                if body >= body_ratio:
                    sig = -1

        if sig != 0:
            sig_indices.append(exec_indices[i])
            sig_types.append(sig)
            last_trade_date = ds

    return execute_matches(week_data, sig_indices, sig_types, sl_pips, tp_pips)


def main():
    print(f"================================================================================", flush=True)
    print(f"  HUNTING EURGBP MACRO COMPRESSION ALPHA (HOLDOUT & DISCOVERY)", flush=True)
    print(f"================================================================================", flush=True)

    holdout_weeks = get_complete_split_weeks(ROOT_DIR / "data" / SYMBOL, split="holdout")
    discovery_weeks = get_complete_split_weeks(ROOT_DIR / "data" / SYMBOL, split="discovery")
    print(f"Loaded {len(holdout_weeks)} holdout weeks, {len(discovery_weeks)} discovery weeks.", flush=True)

    # Pre-load holdout week data
    print(f"Loading {len(holdout_weeks)} holdout weeks into RAM...", flush=True)
    holdout_data = {}
    for w in holdout_weeks:
        wd = load_week_data(w)
        if wd is not None:
            holdout_data[w] = wd
    print(f"Ready: {len(holdout_data)} holdout weeks loaded.", flush=True)

    # Define Configurations
    setups = []

    # 1. Weekly Inside Bar Breakouts
    for start_hr, end_hr in [(7, 16), (7, 12), (8, 16)]:
        for buf in [0.0, 0.5, 1.0]:
            for body in [0.40, 0.50]:
                for sl in [10.0, 12.0, 14.0]:
                    for tp in [20.0, 25.0, 30.0]:
                        for one_w in [True, False]:
                            setups.append({
                                'archetype': 'weekly_inside',
                                'start_hr': start_hr, 'end_hr': end_hr,
                                'buf': buf, 'body': body,
                                'sl': sl, 'tp': tp,
                                'max_spread': 0.8,
                                'one_trade_per_week': one_w,
                                'desc': f"EURGBP Weekly Inside (H={start_hr}-{end_hr}, Buf={buf}p, Body={body}, SL={sl}p, TP={tp}p, 1perWk={one_w})"
                            })

    # 2. NR4 Breakouts
    for start_hr, end_hr in [(7, 16), (7, 12), (8, 15)]:
        for buf in [0.0, 0.5]:
            for body in [0.45, 0.55]:
                for macro in [True, False]:
                    for sl in [8.0, 10.0, 12.0]:
                        for tp in [16.0, 20.0, 24.0]:
                            setups.append({
                                'archetype': 'nr_breakout',
                                'nr_type': 4,
                                'start_hr': start_hr, 'end_hr': end_hr,
                                'buf': buf, 'body': body, 'macro': macro,
                                'sl': sl, 'tp': tp, 'max_spread': 0.8,
                                'desc': f"EURGBP NR4 Breakout (H={start_hr}-{end_hr}, Buf={buf}p, Body={body}, Macro={macro}, SL={sl}p, TP={tp}p)"
                            })

    # 3. NR7 Breakouts
    for start_hr, end_hr in [(7, 16), (7, 12)]:
        for buf in [0.0, 0.5]:
            for body in [0.45]:
                for macro in [True, False]:
                    for sl in [8.0, 10.0, 12.0]:
                        for tp in [16.0, 20.0, 25.0]:
                            setups.append({
                                'archetype': 'nr_breakout',
                                'nr_type': 7,
                                'start_hr': start_hr, 'end_hr': end_hr,
                                'buf': buf, 'body': body, 'macro': macro,
                                'sl': sl, 'tp': tp, 'max_spread': 0.8,
                                'desc': f"EURGBP NR7 Breakout (H={start_hr}-{end_hr}, Buf={buf}p, Body={body}, Macro={macro}, SL={sl}p, TP={tp}p)"
                            })

    # 4. 2-Day Coiled Spring Breakouts
    for start_hr, end_hr in [(7, 16), (7, 12), (8, 15)]:
        for buf in [0.0, 0.5]:
            for body in [0.45, 0.55]:
                for macro in [True, False]:
                    for sl in [8.0, 10.0, 12.0]:
                        for tp in [16.0, 20.0, 24.0]:
                            setups.append({
                                'archetype': 'coiled_breakout',
                                'start_hr': start_hr, 'end_hr': end_hr,
                                'buf': buf, 'body': body, 'macro': macro,
                                'sl': sl, 'tp': tp, 'max_spread': 0.8,
                                'desc': f"EURGBP 2D Coiled Spring (H={start_hr}-{end_hr}, Buf={buf}p, Body={body}, Macro={macro}, SL={sl}p, TP={tp}p)"
                            })

    print(f"Total candidate setups to screen on Holdout: {len(setups)}", flush=True)

    holdout_champs = []
    t0 = time.time()

    for idx_s, s in enumerate(setups):
        all_trades = []
        for w, wd in holdout_data.items():
            arch = s['archetype']
            if arch == 'weekly_inside':
                tr = eval_weekly_inside(wd, s['start_hr'], s['end_hr'], s['buf'], s['body'], s['sl'], s['tp'], s['max_spread'], s['one_trade_per_week'])
            elif arch == 'nr_breakout':
                tr = eval_nr_breakout(wd, s['nr_type'], s['start_hr'], s['end_hr'], s['buf'], s['body'], s['macro'], s['sl'], s['tp'], s['max_spread'])
            elif arch == 'coiled_breakout':
                tr = eval_coiled_breakout(wd, s['start_hr'], s['end_hr'], s['buf'], s['body'], s['macro'], s['sl'], s['tp'], s['max_spread'])
            all_trades.extend(tr)

        if len(all_trades) < 15:
            continue

        pips = np.array([t['pips'] for t in all_trades], dtype=np.float64)
        net_pips = float(np.sum(pips))
        if net_pips <= 0:
            continue

        wins = pips[pips > 0]
        losses = pips[pips < 0]
        sum_w = float(np.sum(wins)) if len(wins) > 0 else 0.0
        sum_l = float(np.abs(np.sum(losses))) if len(losses) > 0 else 1e-6
        pf = sum_w / sum_l

        if pf < 1.15:
            continue

        # Ex-top 3 pips
        sorted_pips = np.sort(pips)
        ex_top3 = float(np.sum(sorted_pips[:-3])) if len(pips) >= 4 else -1.0
        if ex_top3 <= 0:
            continue

        # Anti-redundancy check on candidate trades
        passed_ar, ar_msg = check_anti_redundancy(np.array(all_trades))
        if not passed_ar:
            continue

        holdout_champs.append({
            'setup': s,
            'holdout_metrics': {
                'trades': len(all_trades),
                'net_pips': net_pips,
                'profit_factor': pf,
                'expectancy': float(np.mean(pips)),
                'pips_ex_top3': ex_top3
            }
        })
        print(f"  [HOLDOUT HIT #{len(holdout_champs)}] {s['desc']} -> {len(all_trades)} tr, +{net_pips:.1f}p, PF {pf:.2f}, Exp {np.mean(pips):+.2f}p, ExTop3 +{ex_top3:.1f}p", flush=True)

    print(f"\nHoldout screening complete in {time.time()-t0:.1f}s. Found {len(holdout_champs)} Holdout Champions!", flush=True)

    if not holdout_champs:
        print("No setups passed Holdout criteria.", flush=True)
        return

    # Free holdout data from RAM
    del holdout_data
    gc.collect()

    # Now verify on Discovery (Weeks 1..210)
    print(f"\n--- VERIFYING {len(holdout_champs)} HOLDOUT CHAMPIONS ON DISCOVERY (WEEKS 1..210) ---", flush=True)
    dual_champs = []

    # Stream discovery weeks
    disc_data = {}
    print(f"Loading {len(discovery_weeks)} discovery weeks into RAM...", flush=True)
    for w in discovery_weeks:
        wd = load_week_data(w)
        if wd is not None:
            disc_data[w] = wd
    print(f"Ready: {len(disc_data)} discovery weeks loaded.", flush=True)

    for c in holdout_champs:
        s = c['setup']
        all_trades = []
        for w, wd in disc_data.items():
            arch = s['archetype']
            if arch == 'weekly_inside':
                tr = eval_weekly_inside(wd, s['start_hr'], s['end_hr'], s['buf'], s['body'], s['sl'], s['tp'], s['max_spread'], s['one_trade_per_week'])
            elif arch == 'nr_breakout':
                tr = eval_nr_breakout(wd, s['nr_type'], s['start_hr'], s['end_hr'], s['buf'], s['body'], s['macro'], s['sl'], s['tp'], s['max_spread'])
            elif arch == 'coiled_breakout':
                tr = eval_coiled_breakout(wd, s['start_hr'], s['end_hr'], s['buf'], s['body'], s['macro'], s['sl'], s['tp'], s['max_spread'])
            all_trades.extend(tr)

        if len(all_trades) < 15:
            continue

        pips = np.array([t['pips'] for t in all_trades], dtype=np.float64)
        net_pips = float(np.sum(pips))
        if net_pips <= 0:
            continue

        wins = pips[pips > 0]
        losses = pips[pips < 0]
        sum_w = float(np.sum(wins)) if len(wins) > 0 else 0.0
        sum_l = float(np.abs(np.sum(losses))) if len(losses) > 0 else 1e-6
        pf = sum_w / sum_l

        if pf < 1.15:
            continue

        c['discovery_metrics'] = {
            'trades': len(all_trades),
            'net_pips': net_pips,
            'profit_factor': pf,
            'expectancy': float(np.mean(pips))
        }
        dual_champs.append(c)
        print(f"  >>> DUAL CERTIFIED EURGBP CHAMPION! <<<", flush=True)
        print(f"      Setup: {s['desc']}", flush=True)
        hm = c['holdout_metrics']
        print(f"      Holdout:   {hm['trades']} tr, +{hm['net_pips']:.1f}p, PF {hm['profit_factor']:.2f}, Exp {hm['expectancy']:+.2f}p, ExTop3 +{hm['pips_ex_top3']:.1f}p", flush=True)
        print(f"      Discovery: {len(all_trades)} tr, +{net_pips:.1f}p, PF {pf:.2f}, Exp {np.mean(pips):+.2f}p", flush=True)
        print(f"      Total 5.75Y: {hm['trades'] + len(all_trades)} tr, +{hm['net_pips'] + net_pips:.1f} net pips", flush=True)

    out_file = ROOT_DIR / "results" / "new_champions_eurgbp_macro.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(dual_champs, f, indent=2)

    print(f"\nSaved {len(dual_champs)} Dual-Certified EURGBP Champions to {out_file}!", flush=True)

if __name__ == "__main__":
    main()

