#!/usr/bin/env python3
"""EURUSD v5.0 Candidate: Causal Prior-Day Pivot P/S1/R1 Pullback (NY 12-19 UTC, H1 EMA10/30 Stack)."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

_cur = Path(__file__).resolve()
ROOT_DIR = next((p for p in _cur.parents if (p / "tick_engine").exists()), _cur.parent.parent)
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tick_engine.core.causal_bars import (
    build_causal_time_bars,
    resample_causal_bars,
    align_htf_to_ltf,
    get_session_dates,
    get_session_hours,
    compute_prior_day_pivots,
)


class EURUSDDailyPivotNYV5:
    """Strictly causal prior-day P/S1/R1 pullback during NY session aligned with H1 EMA10/30 stack."""
    symbol: str = "EURUSD"
    archetype: str = "DAILY_PIVOT_PB"
    sl_pips: float = 14.0
    tp_pips: float = 28.0
    pip_size: float = 0.0001
    spread_limit: float = 0.7

    def __init__(self, symbol="EURUSD", sl_pips=14.0, tp_pips=28.0, pip_size=0.0001, spread_limit=0.7, **kwargs):
        self.symbol = symbol
        self.asset = symbol
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.pip_size = float(pip_size)
        self.spread_limit = float(spread_limit)
        self.max_spread = float(spread_limit)
        self.name = "v5_eurusd_daily_pivot_ny"

    def generate_signals(self, ticks: np.ndarray):
        n = len(ticks)
        signals = np.zeros(n, dtype=np.int8)
        sl_arr = np.zeros(n, dtype=np.float64)
        tp_arr = np.zeros(n, dtype=np.float64)
        if n < 500:
            return signals, sl_arr, tp_arr

        m15 = build_causal_time_bars(ticks, timeframe_min=15, pip_size=self.pip_size)
        if len(m15) < 40:
            return signals, sl_arr, tp_arr
        h1 = resample_causal_bars(m15, timeframe_min=60, pip_size=self.pip_size)
        if len(h1) < 35:
            return signals, sl_arr, tp_arr

        h1_c = h1["close"]
        e10 = h1_c.ewm(span=10, adjust=False).mean()
        e30 = h1_c.ewm(span=30, adjust=False).mean()
        h1["h1_stack"] = np.where(
            (h1_c > e10) & (e10 > e30), 1,
            np.where((h1_c < e10) & (e10 < e30), -1, 0),
        )
        df = align_htf_to_ltf(m15, h1[["h1_stack"]])

        pivots = compute_prior_day_pivots(df, min_bars_per_day=15, timeframe_min=15)
        dates = get_session_dates(df, timeframe_min=15)
        hours = get_session_hours(df, timeframe_min=15)

        o = df["open"].to_numpy(dtype=np.float64)
        h = df["high"].to_numpy(dtype=np.float64)
        l = df["low"].to_numpy(dtype=np.float64)
        c = df["close"].to_numpy(dtype=np.float64)
        sp = df["spread_pips"].to_numpy(dtype=np.float64)
        bias = df["h1_stack"].fillna(0).to_numpy(dtype=np.int8)
        close_ms = df["close_time_ms"].to_numpy(dtype=np.int64)
        tick_ts = ticks["timestamp"]

        tup = 2.5 * self.pip_size
        tdn = 4.5 * self.pip_size
        last_trade_date = None

        for i in range(25, len(df)):
            hr = int(hours[i])
            if not (12 <= hr <= 19):
                continue
            dt = dates[i]
            if dt == last_trade_date:
                continue
            piv = pivots.get(dt)
            if piv is None or sp[i] > self.spread_limit:
                continue
            b = int(bias[i])
            if b == 0:
                continue
            rng = h[i] - l[i]
            if rng < 1.5 * self.pip_size:
                continue

            pv, s1, r1 = piv["P"], piv["S1"], piv["R1"]
            sig = 0
            if b == 1:
                touched = (l[i] <= pv + tup and l[i] >= pv - tdn) or (l[i] <= s1 + tup and l[i] >= s1 - tdn)
                if touched and c[i] > o[i] and (o[i] - l[i]) / rng >= 0.40:
                    sig = 1
            elif b == -1:
                touched = (h[i] >= pv - tup and h[i] <= pv + tdn) or (h[i] >= r1 - tup and h[i] <= r1 + tdn)
                if touched and c[i] < o[i] and (h[i] - o[i]) / rng >= 0.40:
                    sig = -1

            if sig != 0:
                idx = int(np.searchsorted(tick_ts, close_ms[i]))
                if idx < n and signals[idx] == 0:
                    signals[idx] = sig
                    sl_arr[idx] = self.sl_pips
                    tp_arr[idx] = self.tp_pips
                    last_trade_date = dt

        return signals, sl_arr, tp_arr


Strategy = EURUSDDailyPivotNYV5
