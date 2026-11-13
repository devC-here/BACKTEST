#!/usr/bin/env python3
"""
================================================================================
  MACRO STRUCTURE RETEST SHARED HELPER (ZERO-ALLOCATION HIGH-SPEED ENGINE)
================================================================================
Modular bar precomputations and matching evaluation for Macro Retest models.
Strict subsequent-tick causal execution (T_fill > T_signal) and live friction.
================================================================================
"""

from pathlib import Path
import numpy as np
import pandas as pd

from tick_engine.core.engine import run_matching_loop
from tick_engine.core.dataloader import get_week_dataset
from tick_engine.core.config import PIP_SIZES, WARMUP_TICKS
from data_engineering_suite.bars import build_time_bars


def precompute_week_data(symbol: str, week_num: int):
    pip_size = PIP_SIZES[symbol]
    raw_dir = f"data/{symbol}"
    try:
        ticks, _ = get_week_dataset(week_num, raw_dir, warmup_prev_week_ticks=WARMUP_TICKS)
    except Exception:
        return None
    if len(ticks) <= WARMUP_TICKS + 500:
        return None

    m15 = build_time_bars(ticks, timeframe_min=15, pip_size=pip_size)
    if len(m15) < 30:
        return None

    ts = pd.to_datetime(ticks['timestamp'], unit='ms')
    mid = 0.5 * (ticks['bid'] + ticks['ask'])
    base = pd.DataFrame({'close': mid, 'high': mid, 'low': mid}, index=ts)
    h4 = base.resample('4h').agg({'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
    if len(h4) < 25:
        return None

    h4['ema20'] = h4['close'].ewm(span=20, adjust=False).mean()
    h4['bias'] = np.where(h4['close'] > h4['ema20'], 1, -1)

    bias_m15 = h4['bias'].reindex(m15.index, method='ffill').fillna(0).values.astype(np.int8)

    levels = {}
    for lb in [6, 8, 10, 12, 14, 16]:
        roll_h = h4['high'].rolling(lb).max().shift(1).bfill()
        roll_l = h4['low'].rolling(lb).min().shift(1).bfill()
        levels[lb] = {
            'supp': roll_l.reindex(m15.index, method='ffill').values.astype(np.float64),
            'res': roll_h.reindex(m15.index, method='ffill').values.astype(np.float64)
        }

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

    # Precalculate subsequent execution tick index for each bar
    exec_indices = np.searchsorted(tick_ts, time_ms + dur_ms).astype(np.int32)

    # Preallocate reusable matching buffers (zero-allocation during parameter sweep)
    signals_buf = np.zeros(n_ticks, dtype=np.int8)
    sl_buf = np.zeros(n_ticks, dtype=np.float64)
    tp_buf = np.zeros(n_ticks, dtype=np.float64)

    return {
        'ticks': ticks,
        'n_ticks': n_ticks,
        'pip_size': pip_size,
        'min_entry_ts': ticks['timestamp'][WARMUP_TICKS],
        'bias_m15': bias_m15,
        'levels': levels,
        'c': c, 'h': h, 'l': l, 'o': o,
        'spread': spread,
        'hours': hours,
        'dates': dates,
        'exec_indices': exec_indices,
        'bar_rng': ((h - l) / pip_size).astype(np.float64),
        'n_bars': len(m15),
        'signals_buf': signals_buf,
        'sl_buf': sl_buf,
        'tp_buf': tp_buf
    }


def eval_config_on_week(week_data, sl, tp, lookback_h4, dist_pips, min_wick, session_mode, max_spread, comm, slip, min_bar_rng=2.5):
    n_ticks = week_data['n_ticks']
    pip_size = week_data['pip_size']
    min_entry_ts = week_data['min_entry_ts']
    bias_m15 = week_data['bias_m15']
    supp_m15 = week_data['levels'][lookback_h4]['supp']
    res_m15 = week_data['levels'][lookback_h4]['res']
    
    c = week_data['c']
    h = week_data['h']
    l = week_data['l']
    o = week_data['o']
    spread = week_data['spread']
    hours = week_data['hours']
    dates = week_data['dates']
    exec_indices = week_data['exec_indices']
    bar_rng = week_data['bar_rng']
    n_bars = week_data['n_bars']

    signals_buf = week_data['signals_buf']
    sl_buf = week_data['sl_buf']
    tp_buf = week_data['tp_buf']

    last_trade_date = None
    fired_indices = []

    for i in range(15, n_bars):
        hr = hours[i]
        # Session filters:
        # 1: Asian + London (21-16 UTC)
        # 2: London + NY (06-18 UTC)
        # 3: All Day (0-23 UTC)
        if session_mode == 1:
            if not (hr >= 21 or hr <= 16):
                continue
        elif session_mode == 2:
            if not (6 <= hr <= 18):
                continue

        dt = dates[i]
        if dt == last_trade_date:
            continue
        if spread[i] > max_spread:
            continue

        br = bar_rng[i]
        if br < min_bar_rng:
            continue

        b = bias_m15[i]
        sig = 0

        # Bullish Retest of H4 Support
        if b > 0 and supp_m15[i] > 0:
            dist_supp = abs(l[i] - supp_m15[i]) / pip_size
            lower_wick = (min(o[i], c[i]) - l[i]) / pip_size
            if dist_supp <= dist_pips and (lower_wick / br) >= min_wick and c[i] > o[i]:
                sig = 1

        # Bearish Retest of H4 Resistance
        elif b < 0 and res_m15[i] > 0:
            dist_res = abs(res_m15[i] - h[i]) / pip_size
            upper_wick = (h[i] - max(o[i], c[i])) / pip_size
            if dist_res <= dist_pips and (upper_wick / br) >= min_wick and c[i] < o[i]:
                sig = -1

        if sig != 0:
            idx = exec_indices[i]
            if idx < n_ticks and signals_buf[idx] == 0:
                signals_buf[idx] = sig
                sl_buf[idx] = sl
                tp_buf[idx] = tp
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
        min_entry_timestamp=min_entry_ts,
        enforce_subsequent_tick=True
    )

    # Clean up only the fired indices in the buffer for next evaluation (zero-reallocation)
    for idx in fired_indices:
        signals_buf[idx] = 0
        sl_buf[idx] = 0.0
        tp_buf[idx] = 0.0

    return tr
