#!/usr/bin/env python3
"""
================================================================================
  MACRO STRUCTURE RETEST SHARED HELPER (STRICTLY CAUSAL ENGINE)
================================================================================
Modular bar precomputations and matching evaluation for Macro Retest models.
Enforces strict bar-close timestamp indexing (label='right'), zero .bfill()
leakage, 2-prior-week rolling warmup, and strict subsequent-tick execution
(T_fill > T_signal).
================================================================================
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

from tick_engine.core.engine import run_matching_loop
from tick_engine.core.dataloader import get_week_dataset
from tick_engine.core.config import PIP_SIZES
from tick_engine.core.causal_bars import (
    build_causal_time_bars,
    resample_causal_bars,
    align_htf_to_ltf,
    get_session_dates,
    get_session_hours,
)


def precompute_week_data(symbol: str, week_num: int):
    pip_size = PIP_SIZES[symbol]
    raw_dir = Path("data") / symbol
    try:
        ticks, meta = get_week_dataset(week_num, raw_dir, warmup_prev_weeks=2)
    except Exception:
        return None

    w_cnt = int(meta.get("warmup_tick_count", 0))
    if len(ticks) <= w_cnt + 500:
        return None

    m15 = build_causal_time_bars(ticks, timeframe_min=15, pip_size=pip_size)
    if len(m15) < 30:
        return None

    # H4 bars resampled from right-labeled M15 bars with closed='right', label='right'
    h4 = resample_causal_bars(m15, 240, pip_size=pip_size)
    if len(h4) < 25:
        return None

    h4["ema20"] = h4["close"].ewm(span=20, min_periods=15, adjust=False).mean()
    h4["bias"] = np.where(h4["ema20"].notna() & (h4["close"] > h4["ema20"]), 1,
                 np.where(h4["ema20"].notna() & (h4["close"] < h4["ema20"]), -1, 0))

    aligned_bias = align_htf_to_ltf(m15, h4[["bias"]])
    bias_m15 = aligned_bias["bias"].fillna(0).values.astype(np.int8)

    # For S/R Donchian boundaries, shift(1) on label='right' H4 ensures M15 bars inside the
    # active H4 candle test prior completed H4 ranges (no same-bar circularity, zero .bfill()).
    levels = {}
    ltf_idx_sorted = m15.sort_index()
    for lb in [6, 8, 10, 12, 14, 16]:
        roll_h = h4["high"].rolling(lb).max()
        roll_l = h4["low"].rolling(lb).min()
        sr_df = pd.DataFrame({"supp": roll_l, "res": roll_h}, index=h4.index).sort_index()
        sr_df.index = sr_df.index.as_unit("ms")
        # allow_exact_matches=False: for M15 bar closing at T, only use H4 S/R whose H4 bar
        # closed strictly BEFORE T (never testing the extreme of the H4 bar that M15 just finished).
        merged_sr = pd.merge_asof(
            ltf_idx_sorted,
            sr_df,
            left_index=True,
            right_index=True,
            direction="backward",
            allow_exact_matches=False,
        )
        levels[lb] = {
            "supp": merged_sr["supp"].values.astype(np.float64),
            "res": merged_sr["res"].values.astype(np.float64),
        }

    c = m15["close"].values.astype(np.float64)
    h = m15["high"].values.astype(np.float64)
    l = m15["low"].values.astype(np.float64)
    o = m15["open"].values.astype(np.float64)
    spread = m15["spread_pips"].values.astype(np.float64)
    hours = get_session_hours(m15, timeframe_min=15)
    dates = get_session_dates(m15, timeframe_min=15)
    close_ms = m15["close_time_ms"].values.astype(np.int64)
    tick_ts = ticks["timestamp"]
    n_ticks = len(ticks)

    exec_indices = np.searchsorted(tick_ts, close_ms).astype(np.int32)
    min_entry_ts = int(ticks["timestamp"][w_cnt]) if 0 < w_cnt < n_ticks else 0

    return {
        "ticks": ticks,
        "n_ticks": n_ticks,
        "pip_size": pip_size,
        "min_entry_ts": min_entry_ts,
        "bias_m15": bias_m15,
        "levels": levels,
        "c": c,
        "h": h,
        "l": l,
        "o": o,
        "spread": spread,
        "hours": hours,
        "dates": dates,
        "exec_indices": exec_indices,
        "bar_rng": ((h - l) / pip_size).astype(np.float64),
        "n_bars": len(m15),
        "signals_buf": np.zeros(n_ticks, dtype=np.int8),
        "sl_buf": np.zeros(n_ticks, dtype=np.float64),
        "tp_buf": np.zeros(n_ticks, dtype=np.float64),
    }


def eval_config_on_week(
    week_data,
    sl,
    tp,
    lookback_h4,
    dist_pips,
    min_wick,
    session_mode,
    max_spread,
    comm,
    slip,
    min_bar_rng=2.5,
):
    n_ticks = week_data["n_ticks"]
    pip_size = week_data["pip_size"]
    min_entry_ts = week_data["min_entry_ts"]
    bias_m15 = week_data["bias_m15"]
    supp_m15 = week_data["levels"][lookback_h4]["supp"]
    res_m15 = week_data["levels"][lookback_h4]["res"]

    c = week_data["c"]
    h = week_data["h"]
    l = week_data["l"]
    o = week_data["o"]
    spread = week_data["spread"]
    hours = week_data["hours"]
    dates = week_data["dates"]
    exec_indices = week_data["exec_indices"]
    bar_rng = week_data["bar_rng"]
    n_bars = week_data["n_bars"]

    signals_buf = week_data["signals_buf"]
    sl_buf = week_data["sl_buf"]
    tp_buf = week_data["tp_buf"]

    last_trade_date = None
    fired_indices = []

    for i in range(15, n_bars):
        hr = hours[i]
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
        if b == 0:
            continue
        sig = 0

        if b > 0 and not np.isnan(supp_m15[i]) and supp_m15[i] > 0:
            dist_supp = abs(l[i] - supp_m15[i]) / pip_size
            lower_wick = (min(o[i], c[i]) - l[i]) / pip_size
            if dist_supp <= dist_pips and (lower_wick / br) >= min_wick and c[i] > o[i]:
                sig = 1
        elif b < 0 and not np.isnan(res_m15[i]) and res_m15[i] > 0:
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
        ticks=week_data["ticks"],
        signals=signals_buf,
        pip_size=pip_size,
        sl_pips=sl_buf,
        tp_pips=tp_buf,
        use_slippage=True,
        close_at_end=True,
        commission_pips=comm,
        entry_slippage_pips=slip,
        min_entry_timestamp=min_entry_ts,
        enforce_subsequent_tick=True,
    )

    for idx in fired_indices:
        signals_buf[idx] = 0
        sl_buf[idx] = 0.0
        tp_buf[idx] = 0.0

    return tr
