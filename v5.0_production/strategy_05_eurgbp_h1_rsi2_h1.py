#!/usr/bin/env python3
"""
v5.0_production Strictly Causal Strategy: EurgbpH1Rsi2ConnorsV5p05 (EURGBP)
================================================================================
Archetype: H1_RSI2_CONNORS (H1) | Spread Limit: 2.0p
Parameters: {'r2': 15.0, 'trend_f': True, 'sess': 'all', 'sl_m': 1.8, 'tp_m': 3.2, 'mh': 12, 'sl_floor': 13.0, 'tp_floor': 19.5, 'atr_h1_ref': 8.4}
Discovery (Weeks 1..210): 55 trades, PF 2.43, +236.6p (SevPF 2.09)
Execution: Strict causal bar-close subsequent-tick execution (T_fill > T_signal)
================================================================================
"""
from __future__ import annotations

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
    return pd.Series(tr).ewm(alpha=1.0 / float(period), adjust=False).mean().to_numpy(dtype=np.float64)


def _er_arr(c: np.ndarray, period: int = 20) -> np.ndarray:
    s = pd.Series(c)
    net = (s - s.shift(period)).abs()
    path = s.diff().abs().rolling(period).sum()
    return (net / path.replace(0.0, np.nan)).fillna(0.0).to_numpy(dtype=np.float64)


class _BarPosLock:
    """100% causal closed-bar SL/TP + Friday/MaxHold position tracker."""
    __slots__ = ("ps", "mh", "pos", "pend_sig", "pend_sl", "pend_tp", "entry_px", "sl_px", "tp_px", "enter_bar")

    def __init__(self, ps: float, mh: int = 0) -> None:
        self.ps = float(ps)
        self.mh = int(mh)
        self.pos = 0
        self.pend_sig = 0
        self.pend_sl = 0.0
        self.pend_tp = 0.0
        self.entry_px = 0.0
        self.sl_px = 0.0
        self.tp_px = 0.0
        self.enter_bar = -1

    def step(self, i: int, o: float, h: float, l: float, sp: float, fri_c: bool, ex_long: bool = False, ex_short: bool = False) -> Tuple[bool, bool]:
        if self.pend_sig != 0:
            sp_h = 0.5 * sp * self.ps
            if self.pend_sig == 1:
                self.entry_px = o + sp_h + 0.20 * self.ps
                self.sl_px = self.entry_px - self.pend_sl * self.ps
                self.tp_px = self.entry_px + self.pend_tp * self.ps
            else:
                self.entry_px = o - sp_h - 0.20 * self.ps
                self.sl_px = self.entry_px + self.pend_sl * self.ps
                self.tp_px = self.entry_px - self.pend_tp * self.ps
            self.pos = self.pend_sig
            self.enter_bar = i
            self.pend_sig = 0

        was_open = self.pos != 0
        if was_open:
            sp_h = 0.5 * sp * self.ps
            if self.pos == 1:
                if (l - sp_h) <= self.sl_px or (h - sp_h) >= self.tp_px:
                    self.pos = 0
            else:
                if (h + sp_h) >= self.sl_px or (l + sp_h) <= self.tp_px:
                    self.pos = 0

        if self.pos != 0:
            ex_hit = (self.pos == 1 and ex_long) or (self.pos == -1 and ex_short)
            if fri_c or ex_hit or (self.mh > 0 and (i - self.enter_bar) >= self.mh):
                self.pos = 0
                return False, True
            return True, False

        return was_open, False

    def enter(self, sig: int, sl_p: float, tp_p: float) -> None:
        self.pend_sig = int(sig)
        self.pend_sl = float(sl_p)
        self.pend_tp = float(tp_p)


def _rsi_wilder(c: np.ndarray, period: int = 14) -> np.ndarray:
    d = np.diff(c, prepend=c[0])
    g = np.where(d > 0.0, d, 0.0)
    ls = np.where(d < 0.0, -d, 0.0)
    a = 1.0 / float(period)
    ag = pd.Series(g).ewm(alpha=a, adjust=False).mean().to_numpy()
    al = pd.Series(ls).ewm(alpha=a, adjust=False).mean().to_numpy()
    rs = ag / np.maximum(al, 1e-12)
    out = 100.0 - (100.0 / (1.0 + rs))
    return np.where((ag < 1e-12) & (al < 1e-12), 50.0, out).astype(np.float64)


def _session_vwap(bars: pd.DataFrame, dates: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    c = bars["close"].to_numpy(dtype=np.float64)
    v = bars["vol"].to_numpy(dtype=np.float64) if "vol" in bars.columns else np.ones(len(bars), dtype=np.float64)
    v = np.maximum(v, 1.0)
    n = len(bars)
    vwap = np.full(n, np.nan, dtype=np.float64)
    vsd = np.full(n, np.nan, dtype=np.float64)
    cum_pv = cum_p2v = cum_v = 0.0
    prev_d = None
    for i in range(n):
        d = dates[i]
        if d != prev_d:
            cum_pv = cum_p2v = cum_v = 0.0
            prev_d = d
        px, wi = c[i], v[i]
        cum_pv += px * wi
        cum_p2v += (px * px) * wi
        cum_v += wi
        mu = cum_pv / cum_v
        vwap[i] = mu
        vsd[i] = np.sqrt(max(0.0, (cum_p2v / cum_v) - (mu * mu)))
    return vwap, vsd


def _causal_session_box(h: np.ndarray, l: np.ndarray, dates: np.ndarray, hrs: np.ndarray, start_hr: int, end_hr: int, min_bars: int):
    n = len(h)
    box_hi = np.full(n, np.nan, dtype=np.float64)
    box_lo = np.full(n, np.nan, dtype=np.float64)
    box_rdy = np.zeros(n, dtype=np.bool_)
    cur_hi, cur_lo, cnt, prev_d = -1e18, 1e18, 0, None
    for i in range(n):
        d = dates[i]
        if d != prev_d:
            cur_hi, cur_lo, cnt, prev_d = -1e18, 1e18, 0, d
        hr = int(hrs[i])
        if start_hr <= hr <= end_hr:
            if h[i] > cur_hi:
                cur_hi = h[i]
            if l[i] < cur_lo:
                cur_lo = l[i]
            cnt += 1
        elif hr > end_hr and cnt >= min_bars:
            box_hi[i], box_lo[i], box_rdy[i] = cur_hi, cur_lo, True
    return box_hi, box_lo, box_rdy


def _fri_flags(df: pd.DataFrame, tf: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    hrs = get_session_hours(df, tf)
    close_dt = df.index
    c_dow = close_dt.dayofweek.to_numpy(np.int8)
    c_hr = close_dt.hour.to_numpy(np.int16)
    o_dt = (close_dt - pd.Timedelta(minutes=tf)).as_unit("ms")
    o_dow = o_dt.dayofweek.to_numpy(np.int8)
    fri_close = ((c_dow == 4) & (c_hr >= (20 if tf == 240 else 21))).astype(np.uint8)
    fri_cut = ((o_dow == 4) & (hrs >= (16 if tf == 240 else 17))) | ((o_dow == 4) & (c_hr >= 18))
    return hrs, fri_close, fri_cut


class EurgbpH1Rsi2ConnorsV5p05:
    symbol: str = 'EURGBP'
    asset: str = 'EURGBP'
    archetype: str = 'H1_RSI2_CONNORS'

    def __init__(
        self,
        symbol: str = 'EURGBP',
        asset: str = 'EURGBP',
        pip_size: float = 0.0001,
        sl_pips: float = 15.1,
        tp_pips: float = 26.9,
        spread_limit: float = 2.0,
        **kwargs,
    ):
        self.symbol = 'EURGBP'
        self.asset = 'EURGBP'
        self.pip_size = float(pip_size)
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.spread_limit = float(spread_limit)
        self.max_spread = float(spread_limit)
        self.name = "EurgbpH1Rsi2ConnorsV5p05"

    def generate_signals(self, ticks: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = len(ticks)
        sigs = np.zeros(n, dtype=np.int8)
        sl_arr = np.full(n, self.sl_pips, dtype=np.float64)
        tp_arr = np.full(n, self.tp_pips, dtype=np.float64)
        if n < 500:
            return sigs, sl_arr, tp_arr
        ps = self.pip_size

        m15 = build_causal_time_bars(ticks, timeframe_min=15, pip_size=ps)
        if len(m15) < 60:
            return sigs, sl_arr, tp_arr
        bd = resample_causal_bars(m15, 60, ps)
        if len(bd) < 60:
            return sigs, sl_arr, tp_arr

        o, h, l, c = bd["open"].to_numpy(np.float64), bd["high"].to_numpy(np.float64), bd["low"].to_numpy(np.float64), bd["close"].to_numpy(np.float64)
        sp = bd["spread_pips"].to_numpy(np.float64)
        hrs, fri_close, fri_cut = _fri_flags(bd, 60)
        close_ms = bd["close_time_ms"].to_numpy(np.int64)
        atr_s = np.clip(np.nan_to_num(_atr_wilder(h, l, c, 14) / ps, nan=8.4), 10.4, 90.0)
        rsi2 = _rsi_wilder(c, 2)
        ema50 = pd.Series(c).ewm(span=50, adjust=False).mean().to_numpy(np.float64)

        tick_ts = ticks["timestamp"]
        r2 = 15.0
        trend_f = True
        sess = 'all'
        sim_p, held = 0, 0
        for i in range(55, len(bd)):
            if sim_p != 0:
                held += 1
                if fri_close[i] or (sim_p == 1 and rsi2[i] >= 75.0) or (sim_p == -1 and rsi2[i] <= 25.0) or held >= 12:
                    idx_x = int(np.searchsorted(tick_ts, close_ms[i], side="left"))
                    if idx_x < n and sigs[idx_x] == 0:
                        sigs[idx_x] = SIGNAL_FORCE_EXIT
                    sim_p = 0
                    continue
            hr = int(hrs[i])
            if fri_cut[i] or hr in (21, 22) or sp[i] > self.spread_limit:
                continue
            if sess == "00_08" and not (0 <= hr <= 8):
                continue
            if sess == "07_18" and not (7 <= hr <= 18):
                continue
            sig = 0
            if rsi2[i] <= r2 and c[i] > o[i] and (not trend_f or c[i] > ema50[i]):
                sig = SIGNAL_LONG
            elif rsi2[i] >= 100.0 - r2 and c[i] < o[i] and (not trend_f or c[i] < ema50[i]):
                sig = SIGNAL_SHORT
            if sig != 0 and sim_p == 0:
                idx = int(np.searchsorted(tick_ts, close_ms[i], side="left"))
                if idx < n and sigs[idx] == 0:
                    sigs[idx] = sig
                    sl_arr[idx] = float(np.clip(1.8 * atr_s[i], 13.0, 85.0))
                    tp_arr[idx] = float(np.clip(3.2 * atr_s[i], 19.5, 170.0))
                    sim_p, held = sig, 0
        return sigs, sl_arr, tp_arr


Strategy = EurgbpH1Rsi2ConnorsV5p05
