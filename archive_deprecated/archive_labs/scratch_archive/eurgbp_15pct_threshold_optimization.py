#!/usr/bin/env python3
"""
================================================================================
  EURGBP OPTIMIZATION LAB: PUSHING CANDIDATES OVER THE 15% ANNUAL RETURN THRESHOLD
================================================================================
Tests parameter calibrations on EURGBP to boost high-edge models (DISC51, DISC49,
DISC65, DISC46, DISC48) over the >15.0% annual return hurdle while preserving
low drawdowns.
================================================================================
"""

from __future__ import annotations
import sys
import os
import time
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Any, Tuple

import numpy as np
import pandas as pd

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["NUMBA_NUM_THREADS"] = "1"

sys.dont_write_bytecode = True

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tick_engine.core.engine import run_matching_loop
from tick_engine.core.dataloader import get_week_dataset, get_discovery_weeks
from tick_engine.strategies.base import BaseStrategy, SIGNAL_LONG, SIGNAL_SHORT
from data_engineering_suite.bars import build_time_bars

PIP_SIZE = 0.0001
MAX_SPREAD = 2.0


class EURGBPMacroStructureTuned(BaseStrategy):
    def __init__(self, name: str, sl_pips: float, tp_pips: float, dist_pips: float, min_wick: float, **kwargs):
        super().__init__(**kwargs)
        self.name = name
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.dist_pips = float(dist_pips)
        self.min_wick = float(min_wick)

    def generate_signals(self, ticks: np.ndarray):
        m15 = build_time_bars(ticks, timeframe_min=15, pip_size=PIP_SIZE)
        if len(m15) < 30:
            n = len(ticks)
            return np.zeros(n, dtype=np.int8), np.zeros(n, dtype=np.float64), np.zeros(n, dtype=np.float64)
        h1 = m15.resample('1h').agg({'high': 'max', 'low': 'min', 'close': 'last', 'open': 'first'}).dropna()
        h4 = m15.resample('4h').agg({'high': 'max', 'low': 'min', 'close': 'last', 'open': 'first'}).dropna()
        return self.generate_signals_from_bars(ticks, m15, h1, h4)

    def generate_signals_from_bars(self, ticks: np.ndarray, m15: pd.DataFrame, h1: pd.DataFrame, h4: pd.DataFrame):
        n = len(ticks)
        signals = np.zeros(n, dtype=np.int8)
        sl_arr = np.zeros(n, dtype=np.float64)
        tp_arr = np.zeros(n, dtype=np.float64)

        if len(m15) < 30 or len(h1) < 25:
            return signals, sl_arr, tp_arr

        h1_ema20 = h1['close'].ewm(span=20, adjust=False).mean()
        h1_bias = np.where(h1['close'] > h1_ema20, 1, -1)
        h1_roll_h = h1['high'].rolling(20).max().shift(1).bfill()
        h1_roll_l = h1['low'].rolling(20).min().shift(1).bfill()

        bias_m15 = pd.Series(h1_bias, index=h1.index).shift(1).reindex(m15.index, method='ffill').fillna(0).values
        supp_m15 = h1_roll_l.reindex(m15.index, method='ffill').values
        res_m15 = h1_roll_h.reindex(m15.index, method='ffill').values

        c = m15['close'].values
        h = m15['high'].values
        l = m15['low'].values
        o = m15['open'].values
        spread = m15['spread_pips'].values
        hours = m15.index.hour.values
        dates = m15.index.date
        time_ms = m15.index.astype('datetime64[ms]').view('int64')
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        last_trade_date = None

        for i in range(15, len(m15)):
            hr = hours[i]
            if not (7 <= hr <= 16):
                continue
            if spread[i] > MAX_SPREAD:
                continue
            dt = dates[i]
            if dt == last_trade_date:
                continue

            bar_rng = (h[i] - l[i]) / PIP_SIZE
            if bar_rng < 2.0:
                continue

            b = bias_m15[i]
            sig = 0
            if b > 0:
                dist_supp = abs(l[i] - supp_m15[i]) / PIP_SIZE
                lower_wick = (min(o[i], c[i]) - l[i]) / PIP_SIZE
                if dist_supp <= self.dist_pips and (lower_wick / bar_rng) >= self.min_wick and c[i] > o[i]:
                    sig = SIGNAL_LONG
            elif b < 0:
                dist_res = abs(res_m15[i] - h[i]) / PIP_SIZE
                upper_wick = (h[i] - max(o[i], c[i])) / PIP_SIZE
                if dist_res <= self.dist_pips and (upper_wick / bar_rng) >= self.min_wick and c[i] < o[i]:
                    sig = SIGNAL_SHORT

            if sig != 0:
                exec_t = time_ms[i] + dur_ms
                idx = int(np.searchsorted(tick_ts, exec_t))
                if idx < n and signals[idx] == 0:
                    signals[idx] = sig
                    sl_arr[idx] = self.sl_pips
                    tp_arr[idx] = self.tp_pips
                    last_trade_date = dt

        return signals, sl_arr, tp_arr


def build_eurgbp_lab_candidates():
    candidates = []
    # Test expanding tolerance from 2.5p to 3.5p and 4.0p for DISC51 / DISC49 variants
    for dist in [3.0, 3.5, 4.0]:
        for sl, tp in [(7.0, 14.0), (7.5, 15.0), (8.0, 16.0), (8.0, 20.0)]:
            name = f"EURGBP_MacroRetest_D{int(dist*10)}_SL{int(sl*10)}_TP{int(tp*10)}"
            candidates.append((name, EURGBPMacroStructureTuned(name=name, sl_pips=sl, tp_pips=tp, dist_pips=dist, min_wick=0.45)))
    return candidates


def evaluate_week_eurgbp(week_num: int) -> Dict[str, Any]:
    try:
        ticks, meta = get_week_dataset(week_num, raw_dir="data/EURGBP")
        if len(ticks) == 0:
            return {"week": week_num, "trades": {}}
    except Exception as e:
        return {"week": week_num, "trades": {}, "error": str(e)}

    m15 = build_time_bars(ticks, timeframe_min=15, pip_size=PIP_SIZE)
    if len(m15) < 30:
        return {"week": week_num, "trades": {}}

    h1 = m15.resample('1h').agg({'high': 'max', 'low': 'min', 'close': 'last', 'open': 'first'}).dropna()
    h4 = m15.resample('4h').agg({'high': 'max', 'low': 'min', 'close': 'last', 'open': 'first'}).dropna()

    candidates = build_eurgbp_lab_candidates()
    week_trades = {}

    for name, model in candidates:
        try:
            signals, sl_arr, tp_arr = model.generate_signals_from_bars(ticks, m15, h1, h4)
            trades = run_matching_loop(
                ticks=ticks,
                signals=signals,
                pip_size=PIP_SIZE,
                sl_pips=sl_arr,
                tp_pips=tp_arr,
                use_slippage=True
            )
            week_trades[name] = [float(t['pips']) for t in trades]
        except Exception:
            week_trades[name] = []

    return {"week": week_num, "trades": week_trades}


def main():
    t_start = time.time()
    disc_weeks = get_discovery_weeks(raw_dir="data/EURGBP")
    print(f"Screening EURGBP tuned candidates on {len(disc_weeks)} discovery weeks...")

    candidates = build_eurgbp_lab_candidates()
    all_week_trades = {name: [] for name, _ in candidates}

    max_workers = min(8, os.cpu_count() or 4)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(evaluate_week_eurgbp, w): w for w in disc_weeks}
        for fut in as_completed(futures):
            res = fut.result()
            trades_dict = res.get("trades", {})
            for m_name, pips in trades_dict.items():
                if m_name in all_week_trades:
                    all_week_trades[m_name].extend(pips)

    years = len(disc_weeks) / 52.0
    print("\n" + "=" * 115)
    print(f"{'Model Name':<42} | {'Trades':<6} | {'WR%':<6} | {'PF':<5} | {'Net Pips':<9} | {'Ann R':<7} | {'Ann Ret%':<8} | {'Max DD':<7} | {'Ret/DD'}")
    print("-" * 115)

    for name, model in candidates:
        pips = all_week_trades[name]
        if not pips:
            continue
        arr = np.array(pips)
        trades = len(arr)
        wins = np.sum(arr > 0)
        wr = wins / trades * 100.0
        gross_w = np.sum(arr[arr > 0])
        gross_l = abs(np.sum(arr[arr < 0]))
        pf = gross_w / gross_l if gross_l > 0 else 99.0
        net_p = float(np.sum(arr))
        ann_p = net_p / years
        ann_r = ann_p / model.sl_pips
        ann_ret_pct = ann_r * 1.0

        equity = np.cumsum(arr)
        peak = np.maximum.accumulate(equity)
        dd = peak - equity
        max_dd = float(np.max(dd)) if len(dd) > 0 else 0.0
        ret_to_dd = net_p / max_dd if max_dd > 0 else 0.0

        if ann_ret_pct >= 15.0 and pf >= 1.30:
            print(f"{name:<42} | {trades:<6} | {wr:<6.1f} | {pf:<5.2f} | {net_p:<+9.1f} | {ann_r:<+7.2f} | {ann_ret_pct:<+7.1f}% | {max_dd:<7.1f} | {ret_to_dd:<6.2f}x")


if __name__ == "__main__":
    main()
