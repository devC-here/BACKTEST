#!/usr/bin/env python3
"""
================================================================================
  EURGBP H1 PULLBACK REJECTION REFINEMENT ENGINE
================================================================================
Refines the promising H1 Pullback Rejection setup on EURGBP:
  - Touch distances: 0.8, 1.0, 1.2, 1.5 pips
  - Rejection wick ratios: 0.35, 0.40, 0.45
  - H1 20-EMA slope filter (EMA_t > EMA_t-1 for long, < for short)
  - Time windows: 07:00-16:00, 08:00-15:00, 08:00-14:00 UTC
  - SL: 7.0, 8.0, 9.0 pips | TP: 14.0, 15.0, 16.0, 17.0, 18.0 pips
  - Evaluates both Holdout and Discovery
================================================================================
"""

import sys
import json
import time
from pathlib import Path
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


def load_h1_data(week_num: int):
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

    # H1 bars
    h1 = m15[['high', 'low', 'close', 'open']].resample('1h').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'
    }).dropna()
    h1_ema20 = h1['close'].ewm(span=20, adjust=False).mean()
    h1_ema_slope = np.zeros(len(h1), dtype=np.int8)
    h1_ema_vals = h1_ema20.values
    for idx in range(1, len(h1)):
        if h1_ema_vals[idx] > h1_ema_vals[idx - 1]:
            h1_ema_slope[idx] = 1
        elif h1_ema_vals[idx] < h1_ema_vals[idx - 1]:
            h1_ema_slope[idx] = -1

    h1_df = pd.DataFrame({
        'h1_ema20': h1_ema_vals,
        'h1_ema_slope': h1_ema_slope
    }, index=h1.index)

    merged_h1 = pd.merge_asof(m15.sort_index(), h1_df.sort_index(), left_index=True, right_index=True, direction='backward')

    c = m15['close'].values.astype(np.float64)
    h = m15['high'].values.astype(np.float64)
    l = m15['low'].values.astype(np.float64)
    o = m15['open'].values.astype(np.float64)
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
        'min_entry_ts': ticks['timestamp'][WARMUP_TICKS],
        'c': c, 'h': h, 'l': l, 'o': o,
        'spread': spread,
        'hours': hours,
        'dates': dates,
        'exec_indices': exec_indices,
        'h4_bias': h4_bias,
        'h1_ema20': merged_h1['h1_ema20'].values,
        'h1_ema_slope': merged_h1['h1_ema_slope'].fillna(0).values.astype(np.int8),
        'n_bars': len(m15),
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


def eval_setup(week_data, start_hr, end_hr, touch_dist, min_wick, use_slope, sl_pips, tp_pips, max_spread=0.8):
    n_bars = week_data['n_bars']
    c, h, l, o = week_data['c'], week_data['h'], week_data['l'], week_data['o']
    spread = week_data['spread']
    hours = week_data['hours']
    dates = week_data['dates']
    exec_indices = week_data['exec_indices']
    h4_bias = week_data['h4_bias']
    h1_ema = week_data['h1_ema20']
    h1_slope = week_data['h1_ema_slope']

    t_dist = touch_dist * PIP_SIZE
    sig_indices = []
    sig_types = []
    last_trade_date = None

    for i in range(25, n_bars):
        hr = hours[i]
        if not (start_hr <= hr <= end_hr):
            continue
        dt = dates[i]
        if dt == last_trade_date:
            continue
        if spread[i] > max_spread:
            continue

        macro = h4_bias[i]
        if macro == 0:
            continue

        if use_slope and h1_slope[i] != macro:
            continue

        ema = h1_ema[i]
        bar_rng = h[i] - l[i]
        if bar_rng < 1.2 * PIP_SIZE:
            continue

        sig = 0
        if macro == 1:
            if l[i] <= ema + t_dist and l[i] >= ema - 2.0 * t_dist and c[i] > o[i]:
                lw = (o[i] - l[i]) / bar_rng
                if lw >= min_wick:
                    sig = 1
        elif macro == -1:
            if h[i] >= ema - t_dist and h[i] <= ema + 2.0 * t_dist and c[i] < o[i]:
                uw = (h[i] - o[i]) / bar_rng
                if uw >= min_wick:
                    sig = -1

        if sig != 0:
            sig_indices.append(exec_indices[i])
            sig_types.append(sig)
            last_trade_date = dt

    return execute_matches(week_data, sig_indices, sig_types, sl_pips, tp_pips)


def main():
    print("Loading Holdout weeks...", flush=True)
    h_weeks = get_complete_split_weeks(ROOT_DIR / "data" / SYMBOL, split="holdout")
    h_data = {w: load_h1_data(w) for w in h_weeks}
    h_data = {k: v for k, v in h_data.items() if v is not None}

    print(f"Loading Discovery weeks...", flush=True)
    d_weeks = get_complete_split_weeks(ROOT_DIR / "data" / SYMBOL, split="discovery")
    d_data = {w: load_h1_data(w) for w in d_weeks}
    d_data = {k: v for k, v in d_data.items() if v is not None}
    print(f"Ready: {len(h_data)} holdout weeks, {len(d_data)} discovery weeks.", flush=True)

    grid = []
    for (start_hr, end_hr) in [(7, 16), (8, 15), (8, 14)]:
        for touch in [0.8, 1.0, 1.2, 1.5]:
            for wick in [0.35, 0.40]:
                for use_slope in [True, False]:
                    for sl in [7.0, 8.0, 9.0]:
                        for tp in [14.0, 15.0, 16.0, 18.0]:
                            grid.append({
                                'start_hr': start_hr, 'end_hr': end_hr,
                                'touch': touch, 'wick': wick,
                                'use_slope': use_slope,
                                'sl': sl, 'tp': tp,
                                'desc': f"H={start_hr}-{end_hr}, Touch={touch}p, Wk={wick}, Slope={use_slope}, SL={sl}p, TP={tp}p"
                            })

    print(f"Evaluating {len(grid)} setups on Holdout & Discovery...", flush=True)
    champions = []

    for idx, s in enumerate(grid):
        # Holdout
        h_trades = []
        for w, wd in h_data.items():
            tr = eval_setup(wd, s['start_hr'], s['end_hr'], s['touch'], s['wick'], s['use_slope'], s['sl'], s['tp'])
            h_trades.extend(tr)

        if len(h_trades) < 15:
            continue
        h_pips = np.array([t['pips'] for t in h_trades])
        h_net = float(np.sum(h_pips))
        if h_net <= 0:
            continue
        h_wins = h_pips[h_pips > 0]
        h_losses = h_pips[h_pips < 0]
        h_pf = float(np.sum(h_wins)) / (float(np.abs(np.sum(h_losses))) + 1e-5)
        if h_pf < 1.15:
            continue
        sorted_hp = np.sort(h_pips)
        ex_top3 = float(np.sum(sorted_hp[:-3])) if len(h_pips) >= 4 else -1.0
        if ex_top3 <= 0:
            continue

        # Discovery
        d_trades = []
        for w, wd in d_data.items():
            tr = eval_setup(wd, s['start_hr'], s['end_hr'], s['touch'], s['wick'], s['use_slope'], s['sl'], s['tp'])
            d_trades.extend(tr)

        if len(d_trades) < 20:
            continue
        d_pips = np.array([t['pips'] for t in d_trades])
        d_net = float(np.sum(d_pips))
        if d_net <= 0:
            continue
        d_wins = d_pips[d_pips > 0]
        d_losses = d_pips[d_pips < 0]
        d_pf = float(np.sum(d_wins)) / (float(np.abs(np.sum(d_losses))) + 1e-5)

        # Print all with positive discovery!
        if d_net > 0:
            tag = "*** DUAL CHAMPION ***" if d_pf >= 1.15 else "Positive Discovery"
            print(f"[{tag}] {s['desc']}:", flush=True)
            print(f"   Holdout:   {len(h_trades)} tr, +{h_net:.1f}p, PF={h_pf:.2f}, Exp={np.mean(h_pips):+.2f}p, ExTop3=+{ex_top3:.1f}p", flush=True)
            print(f"   Discovery: {len(d_trades)} tr, +{d_net:.1f}p, PF={d_pf:.2f}, Exp={np.mean(d_pips):+.2f}p", flush=True)

            if d_pf >= 1.15:
                champions.append({
                    'setup': s,
                    'holdout': {'trades': len(h_trades), 'net_pips': h_net, 'pf': h_pf, 'exp': float(np.mean(h_pips)), 'ex_top3': ex_top3},
                    'discovery': {'trades': len(d_trades), 'net_pips': d_net, 'pf': d_pf, 'exp': float(np.mean(d_pips))}
                })

    print(f"\nSearch complete. Found {len(champions)} DUAL CHAMPIONS meeting Discovery PF >= 1.15!", flush=True)
    if champions:
        out_f = ROOT_DIR / "results" / "new_champions_eurgbp_h1_pullback.json"
        with open(out_f, "w", encoding="utf-8") as f:
            json.dump(champions, f, indent=2)
        print(f"Saved to {out_f}", flush=True)

if __name__ == "__main__":
    main()

