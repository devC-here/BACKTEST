#!/usr/bin/env python3
"""
v5.0 Strictly Causal Strategy: V5GbpjpyH4SqzBrk (GBPJPY)
================================================================================
Archetype: H4_COMP_SQZ_V5
Parameters: {'filt': 'ema20', 'sl_mode': 'fixed', 'sl': 40.0, 'tp': 60.0, 'ms': 4.0}
Discovery (Weeks 1..210): 88 trades, PF 1.38, +658.4p (t=+1.45, Ex3 +475.3p)
Execution: Strict causal bar-close subsequent-tick execution (T_fill > T_signal)
================================================================================
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import sys
from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd

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
    compute_prior_week_pivots,
)

SIGNAL_NONE = 0
SIGNAL_LONG = 1
SIGNAL_SHORT = -1
SIGNAL_FORCE_EXIT = 2


def _atr_wilder(h: np.ndarray, l: np.ndarray, c: np.ndarray, period: int = 14) -> np.ndarray:
    pc = np.roll(c, 1)
    pc[0] = c[0]
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    return pd.Series(tr).ewm(alpha=1.0 / float(period), adjust=False).mean().to_numpy()


def _adx(h: np.ndarray, l: np.ndarray, c: np.ndarray, period: int = 14) -> np.ndarray:
    up = np.diff(h, prepend=h[0])
    dn = -np.diff(l, prepend=l[0])
    pdm = np.where((up > dn) & (up > 0.0), up, 0.0)
    mdm = np.where((dn > up) & (dn > 0.0), dn, 0.0)
    atr = _atr_wilder(h, l, c, period)
    alpha = 1.0 / float(period)
    pdi = 100.0 * pd.Series(pdm).ewm(alpha=alpha, adjust=False).mean().to_numpy() / np.maximum(atr, 1e-12)
    mdi = 100.0 * pd.Series(mdm).ewm(alpha=alpha, adjust=False).mean().to_numpy() / np.maximum(atr, 1e-12)
    dx = 100.0 * np.abs(pdi - mdi) / np.maximum(pdi + mdi, 1e-12)
    return pd.Series(dx).ewm(alpha=alpha, adjust=False).mean().to_numpy()


def _er(c: np.ndarray, period: int = 20) -> np.ndarray:
    s = pd.Series(c)
    net = (s - s.shift(period)).abs()
    path = s.diff().abs().rolling(period).sum()
    return (net / path.replace(0.0, np.nan)).fillna(0.0).to_numpy()


def _rsi(c: np.ndarray, period: int = 14) -> np.ndarray:
    d = np.diff(c, prepend=c[0])
    g = np.where(d > 0.0, d, 0.0)
    ls = np.where(d < 0.0, -d, 0.0)
    a = 1.0 / float(period)
    ag = pd.Series(g).ewm(alpha=a, adjust=False).mean().to_numpy()
    al = pd.Series(ls).ewm(alpha=a, adjust=False).mean().to_numpy()
    rs = ag / np.maximum(al, 1e-12)
    return 100.0 - (100.0 / (1.0 + rs))


class V5GbpjpyH4SqzBrk:
    archetype: str = "H4_COMP_SQZ_V5"
    def __init__(
        self,
        symbol: str = "GBPJPY",
        asset: str = "GBPJPY",
        pip_size: float = 0.01,
        sl_pips: float = 40.0,
        tp_pips: float = 60.0,
        spread_limit: float = 4.0,
        **kwargs,
    ):
        self.symbol = "GBPJPY"
        self.asset = "GBPJPY"
        self.pip_size = float(pip_size)
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.spread_limit = float(spread_limit)
        self.max_spread = float(spread_limit)
        self.name = "V5GbpjpyH4SqzBrk"

    def generate_signals(self, ticks: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = len(ticks)
        sigs = np.zeros(n, dtype=np.int8)
        sl_arr = np.full(n, self.sl_pips, dtype=np.float64)
        tp_arr = np.full(n, self.tp_pips, dtype=np.float64)
        if n < 500:
            return sigs, sl_arr, tp_arr
        ps = self.pip_size
        h1 = build_causal_time_bars(ticks, timeframe_min=60, pip_size=ps)
        h4 = resample_causal_bars(h1, 240, ps)
        if len(h4) < 30 or len(h1) < 55:
            return sigs, sl_arr, tp_arr

        # Use H1 ATR(50) aligned for converged ATR_rel even on 2-week warmup
        h1c = h1["close"].to_numpy(dtype=np.float64)
        h1_a14 = _atr_wilder(h1["high"].to_numpy(), h1["low"].to_numpy(), h1c, 14) / ps
        h1_a50 = _atr_wilder(h1["high"].to_numpy(), h1["low"].to_numpy(), h1c, 50) / ps
        h1_df = pd.DataFrame({"h1_atr_rel": h1_a14 / np.maximum(h1_a50, 1e-6)}, index=h1.index)
        h4a = align_htf_to_ltf(h4, h1_df)

        o = h4a["open"].to_numpy(dtype=np.float64)
        h = h4a["high"].to_numpy(dtype=np.float64)
        l = h4a["low"].to_numpy(dtype=np.float64)
        c = h4a["close"].to_numpy(dtype=np.float64)
        hrs = get_session_hours(h4a, 240)
        close_ms = h4a["close_time_ms"].to_numpy(dtype=np.int64)
        atr14 = _atr_wilder(h, l, c, 14) / ps
        atr_s = np.clip(np.nan_to_num(atr14, nan=25.0), 10.0, 120.0)
        ema20 = pd.Series(c).ewm(span=20, adjust=False).mean().to_numpy()
        ema40 = pd.Series(c).ewm(span=40, adjust=False).mean().to_numpy()
        std20 = pd.Series(c).rolling(20).std().to_numpy()
        sqz_on = (2.0 * std20) < (1.5 * atr14 * ps)
        sqz_fire = np.roll(sqz_on, 1) & (~sqz_on)
        sqz_fire[0] = False
        tr = np.where((ema20 > ema40) & (ema20 > np.roll(ema20, 2)), 1,
                      np.where((ema20 < ema40) & (ema20 < np.roll(ema20, 2)), -1, 0))
        atr_rel = h4a["h1_atr_rel"].fillna(1.0).to_numpy(dtype=np.float64)
        body = np.abs(c - o) / np.maximum(h - l, 1e-12)

        tick_ts = ticks["timestamp"]
        tick_spr = (ticks["ask"] - ticks["bid"]) / ps
        for i in range(25, len(h4a)):
            if int(hrs[i]) == 20 or not sqz_fire[i] or body[i] < 0.40:
                continue
            if "ema20" == "nonchop" and atr_rel[i] > 1.25:
                continue
            sig = 0
            if c[i] > ema20[i] and c[i] > o[i]:
                if "ema20" != "ema20" or tr[i] >= 0:
                    sig = SIGNAL_LONG
            elif c[i] < ema20[i] and c[i] < o[i]:
                if "ema20" != "ema20" or tr[i] <= 0:
                    sig = SIGNAL_SHORT
            if sig != 0:
                idx = int(np.searchsorted(tick_ts, close_ms[i], side="left"))
                if idx < n and tick_spr[idx] <= self.spread_limit and sigs[idx] == 0:
                    sigs[idx] = sig
                    if "fixed" == "h4atr":
                        sl_arr[idx] = float(np.clip(1.8 * atr_s[i], 28.0, 95.0))
                        tp_arr[idx] = float(np.clip(3.6 * atr_s[i], 55.0, 190.0))
                    else:
                        sl_arr[idx] = self.sl_pips
                        tp_arr[idx] = self.tp_pips
        return sigs, sl_arr, tp_arr


Strategy = V5GbpjpyH4SqzBrk
