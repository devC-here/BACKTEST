#!/usr/bin/env python3
"""
================================================================================
  CANONICAL STRICTLY-CAUSAL BAR & PIVOT ENGINE
================================================================================
Guarantees zero lookahead bias across multi-timeframe resampling and pivots:
1. All OHLCV bars use `closed='left', label='right'` so every bar's timestamp
   is its exact CLOSE time (T_close), strictly preserving datetime64[ms].
2. Higher-timeframe alignment (`align_htf_to_ltf`) merges on bar-close timestamps
   with `direction='backward'`, ensuring T_htf_close <= T_ltf_close (0 lookahead)
   and zero `.bfill()` leakage.
3. Daily & Weekly Floor/Camarilla pivots merge Sunday evening open bars
   (21:00-23:59 UTC) into Monday's session date, eliminating partial-Sunday
   anchor distortion and Monday trade skips.
================================================================================
"""

from __future__ import annotations
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd


def build_causal_time_bars(
    ticks: np.ndarray,
    timeframe_min: int = 15,
    pip_size: float = 0.0001,
) -> pd.DataFrame:
    """
    Aggregates tick arrays into strictly causal time bars indexed by BAR CLOSE
    timestamp (`closed='left', label='right'`, `datetime64[ms]`).
    """
    if len(ticks) == 0:
        return pd.DataFrame()

    p_size = float(pip_size)
    ts = pd.to_datetime(ticks["timestamp"], unit="ms").as_unit("ms")
    bid = ticks["bid"]
    ask = ticks["ask"]
    spread = (ask - bid) / p_size
    mid = (bid + ask) * 0.5

    df = pd.DataFrame(
        {"mid": mid, "spread": spread, "vol": np.ones(len(ticks), dtype=np.int32)},
        index=ts,
    )
    rule = f"{timeframe_min}min" if timeframe_min < 60 else f"{timeframe_min // 60}h"

    bars = df.resample(rule, closed="left", label="right").agg(
        {"mid": ["first", "max", "min", "last"], "spread": "mean", "vol": "count"}
    ).dropna()

    if bars.empty:
        return pd.DataFrame()

    bars.index = bars.index.as_unit("ms")
    bars.columns = ["open", "high", "low", "close", "spread_pips", "vol"]
    bars["volume"] = bars["vol"]
    bars["tick_count"] = bars["vol"]
    bars["range_pips"] = (bars["high"] - bars["low"]) / p_size
    bars["body_pips"] = (bars["close"] - bars["open"]).abs() / p_size
    bars["close_time_ms"] = bars.index.view("int64")
    return bars


def resample_causal_bars(
    ltf_bars: pd.DataFrame,
    timeframe_min: int,
    pip_size: float = 0.0001,
) -> pd.DataFrame:
    """
    Resamples right-labeled LTF causal bars (indexed by close time T_close)
    into right-labeled HTF bars using `closed='right', label='right'`.

    Because LTF bar close timestamps lie in (T_htf_open, T_htf_close], using
    `closed='right', label='right'` preserves exact equivalence with calling
    `build_causal_time_bars(ticks, timeframe_min, pip_size)` directly on raw ticks.
    """
    if ltf_bars is None or ltf_bars.empty:
        return pd.DataFrame()

    p_size = float(pip_size)
    df = ltf_bars.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, unit="ms")
    df.index = df.index.as_unit("ms")

    if timeframe_min < 60:
        rule = f"{timeframe_min}min"
    elif timeframe_min % 1440 == 0:
        rule = f"{timeframe_min // 1440}D"
    elif timeframe_min % 60 == 0:
        rule = f"{timeframe_min // 60}h"
    else:
        rule = f"{timeframe_min}min"

    agg_dict: Dict[str, Any] = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
    }
    has_vol = "vol" in df.columns
    has_spread = "spread_pips" in df.columns
    if has_vol:
        agg_dict["vol"] = "sum"
    if has_spread:
        if has_vol:
            df["_weighted_spread"] = df["spread_pips"] * df["vol"]
            agg_dict["_weighted_spread"] = "sum"
        else:
            agg_dict["spread_pips"] = "mean"

    bars = (
        df.resample(rule, closed="right", label="right")
        .agg(agg_dict)
        .dropna(subset=["open", "high", "low", "close"])
    )
    if bars.empty:
        return pd.DataFrame()

    bars.index = bars.index.as_unit("ms")
    if has_spread and has_vol:
        vol_safe = np.maximum(bars["vol"].to_numpy(dtype=np.float64), 1.0)
        bars["spread_pips"] = bars["_weighted_spread"].to_numpy(dtype=np.float64) / vol_safe
        bars = bars.drop(columns=["_weighted_spread"])
    elif not has_spread:
        bars["spread_pips"] = 0.0

    if not has_vol:
        bars["vol"] = 0

    bars["volume"] = bars["vol"]
    bars["tick_count"] = bars["vol"]
    bars["range_pips"] = (bars["high"] - bars["low"]) / p_size
    bars["body_pips"] = (bars["close"] - bars["open"]).abs() / p_size
    bars["close_time_ms"] = bars.index.view("int64")
    return bars


def align_htf_to_ltf(
    ltf_bars: pd.DataFrame,
    htf_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Causal alignment of higher-timeframe indicators/bars onto lower-timeframe bars.
    Both DataFrames MUST be indexed by bar-close timestamps (`label='right'`).
    Returns left-joined `pd.merge_asof` with `direction='backward'`, guaranteeing
    only HTF bars with T_htf_close <= T_ltf_close are visible.
    """
    ltf_sorted = ltf_bars.sort_index()
    htf_sorted = htf_df.sort_index()
    ltf_sorted.index = ltf_sorted.index.as_unit("ms")
    htf_sorted.index = htf_sorted.index.as_unit("ms")
    return pd.merge_asof(
        ltf_sorted,
        htf_sorted,
        left_index=True,
        right_index=True,
        direction="backward",
    )


def get_session_dates(m15_bars: pd.DataFrame, timeframe_min: int = 15) -> np.ndarray:
    """
    Computes trading session dates from close-indexed bars by shifting back to bar
    open time and merging Sunday evening bars (dayofweek == 6) into Monday.
    """
    open_dt = (m15_bars.index - pd.Timedelta(minutes=timeframe_min)).as_unit("ms")
    is_sunday = open_dt.dayofweek == 6
    shifted = pd.DatetimeIndex(
        np.where(is_sunday, (open_dt + pd.Timedelta(days=1)).to_numpy(), open_dt.to_numpy())
    )
    return shifted.date


def get_session_hours(bars: pd.DataFrame, timeframe_min: int = 15) -> np.ndarray:
    """Returns integer bar-open hours (0..23 UTC) from close-indexed bars."""
    open_dt = (bars.index - pd.Timedelta(minutes=timeframe_min)).as_unit("ms")
    return open_dt.hour.values.astype(np.int16)


def compute_prior_day_pivots(
    m15_bars: pd.DataFrame,
    min_bars_per_day: int = 16,
    timeframe_min: int = 15,
) -> Dict[Any, Dict[str, Any]]:
    """
    Computes strictly causal prior-session-day Floor Pivots, Camarilla levels,
    and Inside-Day status keyed by current `session_date` (Sunday merged into Monday).
    """
    if m15_bars.empty:
        return {}

    session_dates = get_session_dates(m15_bars, timeframe_min=timeframe_min)
    h = m15_bars["high"].values.astype(np.float64)
    l = m15_bars["low"].values.astype(np.float64)
    c = m15_bars["close"].values.astype(np.float64)

    uniq_dates = sorted(np.unique(session_dates))
    completed_dates = []
    daily_stats: Dict[Any, tuple[float, float, float]] = {}

    for d in uniq_dates:
        mask = session_dates == d
        if int(np.sum(mask)) >= min_bars_per_day:
            daily_stats[d] = (
                float(np.max(h[mask])),
                float(np.min(l[mask])),
                float(c[mask][-1]),
            )
            completed_dates.append(d)

    pivots: Dict[Any, Dict[str, Any]] = {}
    for d in uniq_dates:
        # Strictly prior completed days only
        prior_days = [pd_d for pd_d in completed_dates if pd_d < d]
        if not prior_days:
            continue
        d_prev1 = prior_days[-1]
        dh, dl, dc = daily_stats[d_prev1]
        p = (dh + dl + dc) / 3.0
        s1 = 2.0 * p - dh
        r1 = 2.0 * p - dl
        rng = dh - dl
        s2 = p - rng
        r2 = p + rng
        cam_h3 = dc + rng * 1.1 / 4.0
        cam_l3 = dc - rng * 1.1 / 4.0
        cam_h4 = dc + rng * 1.1 / 2.0
        cam_l4 = dc - rng * 1.1 / 2.0

        is_inside = False
        if len(prior_days) >= 2:
            d_prev2 = prior_days[-2]
            dh2, dl2, _ = daily_stats[d_prev2]
            if dh < dh2 and dl > dl2:
                is_inside = True

        pivots[d] = {
            "P": p,
            "S1": s1,
            "R1": r1,
            "S2": s2,
            "R2": r2,
            "prev_H": dh,
            "prev_L": dl,
            "prev_C": dc,
            "cam_H3": cam_h3,
            "cam_L3": cam_l3,
            "cam_H4": cam_h4,
            "cam_L4": cam_l4,
            "is_inside_day": is_inside,
        }

    return pivots


def compute_prior_week_pivots(
    m15_bars: pd.DataFrame,
    min_bars_per_week: int = 40,
    timeframe_min: int = 15,
) -> Dict[int, Dict[str, float]]:
    """
    Computes strictly causal prior-ISO-week pivots keyed by current `iso_year * 100 + iso_week`
    using Sunday-merged session dates.
    """
    if m15_bars.empty:
        return {}

    session_dates = get_session_dates(m15_bars, timeframe_min=timeframe_min)
    iso = pd.DatetimeIndex(session_dates).isocalendar()
    yw_arr = (iso.year.to_numpy(dtype=np.int64) * 100) + iso.week.to_numpy(dtype=np.int64)

    h = m15_bars["high"].values.astype(np.float64)
    l = m15_bars["low"].values.astype(np.float64)
    c = m15_bars["close"].values.astype(np.float64)

    uniq_yw = sorted(np.unique(yw_arr))
    completed_yw = []
    week_stats: Dict[int, tuple[float, float, float]] = {}

    for yw in uniq_yw:
        mask = yw_arr == yw
        if int(np.sum(mask)) >= min_bars_per_week:
            week_stats[int(yw)] = (
                float(np.max(h[mask])),
                float(np.min(l[mask])),
                float(c[mask][-1]),
            )
            completed_yw.append(int(yw))

    pivots: Dict[int, Dict[str, float]] = {}
    for yw in uniq_yw:
        prior_w = [pw for pw in completed_yw if pw < int(yw)]
        if not prior_w:
            continue
        wh, wl, wc = week_stats[prior_w[-1]]
        wp = (wh + wl + wc) / 3.0
        pivots[int(yw)] = {
            "WP": wp,
            "WS1": 2.0 * wp - wh,
            "WR1": 2.0 * wp - wl,
            "prev_wH": wh,
            "prev_wL": wl,
            "prev_wC": wc,
        }

    return pivots
