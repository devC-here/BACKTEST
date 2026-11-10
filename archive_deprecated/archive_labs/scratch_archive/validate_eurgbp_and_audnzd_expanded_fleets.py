#!/usr/bin/env python3
"""
================================================================================
  UNIFIED EXPANDED FLEET AUDIT: EURGBP & AUDNZD (300 WEEKS / 5.77 YEARS)
================================================================================
Evaluates all top discovered candidates and existing champions across:
  1. 70% In-Sample Discovery (210 weeks)
  2. 30% Quarantined Out-of-Sample Holdout (90 weeks, Seed 42) - UNTOUCHED DATA
  3. 100% Full Continuous 5.77-Year Cycle (300 weeks)

Certification Hurdle:
  - Holdout Profit Factor >= 1.25
  - Zero edge degradation on holdout
  - Annual Net Return >= 15.0% / year (Annual Pips / SL_pips >= 15.0 R/year)
  - Return / Drawdown >= 4.0x
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
from tick_engine.core.dataloader import get_week_dataset, get_discovery_weeks, get_holdout_test_weeks
from tick_engine.strategies.base import SIGNAL_LONG, SIGNAL_SHORT
from data_engineering_suite.bars import build_time_bars

PIP_SIZE = 0.0001
YEARS = 5.77
RESULTS_DIR = ROOT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# EURGBP STRATEGY DEFINITIONS
# ==============================================================================

class EURGBPFibGoldenPocket:
    def __init__(self, name: str, sl_pips: float, tp_pips: float, lookback_h1: int = 48, min_wick: float = 0.35):
        self.name = name
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.lookback_h1 = int(lookback_h1)
        self.min_wick = float(min_wick)

    def generate_signals(self, ticks: np.ndarray, m15: pd.DataFrame, h1: pd.DataFrame, h4: pd.DataFrame):
        n = len(ticks)
        signals = np.zeros(n, dtype=np.int8)
        if len(m15) < 30 or len(h1) < self.lookback_h1 or len(h4) < 15:
            return signals

        h4_ema20 = h4['close'].ewm(span=20, adjust=False).mean()
        h4_bias = pd.Series(np.where(h4['close'] > h4_ema20, 1, -1), index=h4.index).shift(1).reindex(m15.index, method='ffill').fillna(0).values

        h1_high = h1['high'].rolling(self.lookback_h1).max().shift(1)
        h1_low = h1['low'].rolling(self.lookback_h1).min().shift(1)
        m15_sw_h = h1_high.reindex(m15.index, method='ffill').values
        m15_sw_l = h1_low.reindex(m15.index, method='ffill').values

        c = m15['close'].values
        h = m15['high'].values
        l = m15['low'].values
        o = m15['open'].values
        spread = m15['spread_pips'].values
        hours = m15.index.hour.values
        time_ms = m15.index.astype('datetime64[ms]').view('int64')
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        last_entry = -999
        for i in range(self.lookback_h1, len(m15)):
            if i - last_entry < 6:
                continue
            if not (6 <= hours[i] <= 17) or spread[i] > 1.8:
                continue

            bias = h4_bias[i]
            sw_h = m15_sw_h[i]
            sw_l = m15_sw_l[i]
            if np.isnan(sw_h) or np.isnan(sw_l):
                continue
            rng = sw_h - sw_l
            if rng < 20.0 * PIP_SIZE:
                continue

            bar_rng = h[i] - l[i]
            if bar_rng <= 0:
                continue

            if bias > 0:
                f618 = sw_h - 0.618 * rng
                f786 = sw_h - 0.786 * rng
                if l[i] <= f618 and c[i] >= f786:
                    if ((min(o[i], c[i]) - l[i]) / bar_rng) >= self.min_wick and c[i] > o[i]:
                        idx = np.searchsorted(tick_ts, time_ms[i] + dur_ms)
                        if idx < n:
                            signals[idx] = SIGNAL_LONG
                            last_entry = i
            elif bias < 0:
                f618 = sw_l + 0.618 * rng
                f786 = sw_l + 0.786 * rng
                if h[i] >= f618 and c[i] <= f786:
                    if ((h[i] - max(o[i], c[i])) / bar_rng) >= self.min_wick and c[i] < o[i]:
                        idx = np.searchsorted(tick_ts, time_ms[i] + dur_ms)
                        if idx < n:
                            signals[idx] = SIGNAL_SHORT
                            last_entry = i

        return signals


class EURGBPSessionVWAPFade:
    def __init__(self, name: str, sl_pips: float, tp_pips: float, band_mult: float = 2.0):
        self.name = name
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.band_mult = float(band_mult)

    def generate_signals(self, ticks: np.ndarray, m15: pd.DataFrame, h1: pd.DataFrame, h4: pd.DataFrame):
        n = len(ticks)
        signals = np.zeros(n, dtype=np.int8)
        if len(m15) < 30 or len(h4) < 15:
            return signals

        h4_ema20 = h4['close'].ewm(span=20, adjust=False).mean()
        h4_bias = pd.Series(np.where(h4['close'] > h4_ema20, 1, -1), index=h4.index).shift(1).reindex(m15.index, method='ffill').fillna(0).values

        dates = m15.index.date
        c = m15['close'].values
        h = m15['high'].values
        l = m15['low'].values
        o = m15['open'].values
        spread = m15['spread_pips'].values
        hours = m15.index.hour.values
        vol = m15['tick_volume'].values if 'tick_volume' in m15.columns else np.ones(len(m15))
        time_ms = m15.index.astype('datetime64[ms]').view('int64')
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        vwap = np.zeros(len(m15))
        sd = np.zeros(len(m15))
        cur_date = None
        cum_pv = 0.0
        cum_v = 0.0
        p_buf = []

        for i in range(len(m15)):
            d = dates[i]
            if d != cur_date:
                cur_date = d
                cum_pv = 0.0
                cum_v = 0.0
                p_buf = []
            tp = (h[i] + l[i] + c[i]) / 3.0
            v = max(1.0, vol[i])
            cum_pv += tp * v
            cum_v += v
            p_buf.append(tp)
            vwap[i] = cum_pv / cum_v
            sd[i] = np.std(p_buf) if len(p_buf) > 2 else 0.0

        last_entry = -999
        for i in range(15, len(m15)):
            if i - last_entry < 6:
                continue
            if not (7 <= hours[i] <= 16) or spread[i] > 1.8:
                continue
            if sd[i] <= 0:
                continue

            bias = h4_bias[i]
            upper = vwap[i] + self.band_mult * sd[i]
            lower = vwap[i] - self.band_mult * sd[i]
            bar_rng = h[i] - l[i]
            if bar_rng <= 0:
                continue

            if bias > 0 and l[i] <= lower and c[i] > lower:
                if ((min(o[i], c[i]) - l[i]) / bar_rng) >= 0.35:
                    idx = np.searchsorted(tick_ts, time_ms[i] + dur_ms)
                    if idx < n:
                        signals[idx] = SIGNAL_LONG
                        last_entry = i
            elif bias < 0 and h[i] >= upper and c[i] < upper:
                if ((h[i] - max(o[i], c[i])) / bar_rng) >= 0.35:
                    idx = np.searchsorted(tick_ts, time_ms[i] + dur_ms)
                    if idx < n:
                        signals[idx] = SIGNAL_SHORT
                        last_entry = i

        return signals


class EURGBPDailyPivotPullback:
    def __init__(self, name: str, sl_pips: float, tp_pips: float):
        self.name = name
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)

    def generate_signals(self, ticks: np.ndarray, m15: pd.DataFrame, h1: pd.DataFrame, h4: pd.DataFrame):
        n = len(ticks)
        signals = np.zeros(n, dtype=np.int8)
        if len(m15) < 30 or len(h4) < 15:
            return signals

        h4_ema20 = h4['close'].ewm(span=20, adjust=False).mean()
        h4_bias = pd.Series(np.where(h4['close'] > h4_ema20, 1, -1), index=h4.index).shift(1).reindex(m15.index, method='ffill').fillna(0).values

        daily = m15.resample('1D').agg({'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
        pp = (daily['high'] + daily['low'] + daily['close']) / 3.0
        s1 = 2 * pp - daily['high']
        r1 = 2 * pp - daily['low']
        pp_m15 = pp.shift(1).reindex(m15.index, method='ffill').values
        s1_m15 = s1.shift(1).reindex(m15.index, method='ffill').values
        r1_m15 = r1.shift(1).reindex(m15.index, method='ffill').values

        c = m15['close'].values
        h = m15['high'].values
        l = m15['low'].values
        o = m15['open'].values
        spread = m15['spread_pips'].values
        hours = m15.index.hour.values
        time_ms = m15.index.astype('datetime64[ms]').view('int64')
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        last_entry = -999
        for i in range(15, len(m15)):
            if i - last_entry < 6:
                continue
            if not (7 <= hours[i] <= 16) or spread[i] > 1.8:
                continue
            p_val = pp_m15[i]
            if np.isnan(p_val):
                continue

            bias = h4_bias[i]
            bar_rng = h[i] - l[i]
            if bar_rng <= 0:
                continue

            if bias > 0:
                near = abs(l[i] - p_val) <= 2.0 * PIP_SIZE or abs(l[i] - s1_m15[i]) <= 2.0 * PIP_SIZE
                if near and c[i] > o[i] and ((min(o[i], c[i]) - l[i]) / bar_rng) >= 0.35:
                    idx = np.searchsorted(tick_ts, time_ms[i] + dur_ms)
                    if idx < n:
                        signals[idx] = SIGNAL_LONG
                        last_entry = i
            elif bias < 0:
                near = abs(h[i] - p_val) <= 2.0 * PIP_SIZE or abs(h[i] - r1_m15[i]) <= 2.0 * PIP_SIZE
                if near and c[i] < o[i] and ((h[i] - max(o[i], c[i])) / bar_rng) >= 0.35:
                    idx = np.searchsorted(tick_ts, time_ms[i] + dur_ms)
                    if idx < n:
                        signals[idx] = SIGNAL_SHORT
                        last_entry = i

        return signals


# ==============================================================================
# AUDNZD STRATEGY DEFINITIONS
# ==============================================================================

class AUDNZDFibGoldenPocket:
    def __init__(self, name: str, sl_pips: float, tp_pips: float, lookback_h1: int = 72):
        self.name = name
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.lookback_h1 = int(lookback_h1)

    def generate_signals(self, ticks: np.ndarray, m15: pd.DataFrame, h1: pd.DataFrame, h4: pd.DataFrame):
        n = len(ticks)
        signals = np.zeros(n, dtype=np.int8)
        if len(m15) < 30 or len(h1) < self.lookback_h1 or len(h4) < 15:
            return signals

        h4_ema20 = h4['close'].ewm(span=20, adjust=False).mean()
        h4_bias = pd.Series(np.where(h4['close'] > h4_ema20, 1, -1), index=h4.index).shift(1).reindex(m15.index, method='ffill').fillna(0).values

        h1_high = h1['high'].rolling(self.lookback_h1).max().shift(1)
        h1_low = h1['low'].rolling(self.lookback_h1).min().shift(1)
        m15_sw_h = h1_high.reindex(m15.index, method='ffill').values
        m15_sw_l = h1_low.reindex(m15.index, method='ffill').values

        c = m15['close'].values
        h = m15['high'].values
        l = m15['low'].values
        o = m15['open'].values
        spread = m15['spread_pips'].values
        hours = m15.index.hour.values
        time_ms = m15.index.astype('datetime64[ms]').view('int64')
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        last_entry = -999
        for i in range(self.lookback_h1, len(m15)):
            if i - last_entry < 6:
                continue
            if not (hours[i] >= 21 or hours[i] <= 16) or spread[i] > 2.8:
                continue

            bias = h4_bias[i]
            sw_h = m15_sw_h[i]
            sw_l = m15_sw_l[i]
            if np.isnan(sw_h) or np.isnan(sw_l):
                continue
            rng = sw_h - sw_l
            if rng < 25.0 * PIP_SIZE:
                continue

            bar_rng = h[i] - l[i]
            if bar_rng <= 0:
                continue

            if bias > 0:
                f618 = sw_h - 0.618 * rng
                f786 = sw_h - 0.786 * rng
                if l[i] <= f618 and c[i] >= f786:
                    if ((min(o[i], c[i]) - l[i]) / bar_rng) >= 0.35 and c[i] > o[i]:
                        idx = np.searchsorted(tick_ts, time_ms[i] + dur_ms)
                        if idx < n:
                            signals[idx] = SIGNAL_LONG
                            last_entry = i
            elif bias < 0:
                f618 = sw_l + 0.618 * rng
                f786 = sw_l + 0.786 * rng
                if h[i] >= f618 and c[i] <= f786:
                    if ((h[i] - max(o[i], c[i])) / bar_rng) >= 0.35 and c[i] < o[i]:
                        idx = np.searchsorted(tick_ts, time_ms[i] + dur_ms)
                        if idx < n:
                            signals[idx] = SIGNAL_SHORT
                            last_entry = i

        return signals


# ==============================================================================
# EVALUATION HARNESS
# ==============================================================================

def build_eurgbp_candidates() -> List[Tuple[str, Any]]:
    return [
        ("EG_SessionVWAP_B2.0_SL7.0_TP14.0", EURGBPSessionVWAPFade("EG_SessionVWAP_B2.0_SL7.0_TP14.0", sl_pips=7.0, tp_pips=14.0, band_mult=2.0)),
        ("EG_FibGoldenPocket_LB36_SL7.0_TP17.5", EURGBPFibGoldenPocket("EG_FibGoldenPocket_LB36_SL7.0_TP17.5", sl_pips=7.0, tp_pips=17.5, lookback_h1=36)),
        ("EG_FibGoldenPocket_LB36_SL8.0_TP16.0", EURGBPFibGoldenPocket("EG_FibGoldenPocket_LB36_SL8.0_TP16.0", sl_pips=8.0, tp_pips=16.0, lookback_h1=36)),
        ("EG_FibGoldenPocket_LB48_SL7.0_TP17.5", EURGBPFibGoldenPocket("EG_FibGoldenPocket_LB48_SL7.0_TP17.5", sl_pips=7.0, tp_pips=17.5, lookback_h1=48)),
        ("EG_FibGoldenPocket_LB48_SL8.0_TP16.0", EURGBPFibGoldenPocket("EG_FibGoldenPocket_LB48_SL8.0_TP16.0", sl_pips=8.0, tp_pips=16.0, lookback_h1=48)),
        ("EG_FibGoldenPocket_LB48_SL7.0_TP14.0", EURGBPFibGoldenPocket("EG_FibGoldenPocket_LB48_SL7.0_TP14.0", sl_pips=7.0, tp_pips=14.0, lookback_h1=48)),
        ("EG_DailyPivot_SL7.5_TP15.0", EURGBPDailyPivotPullback("EG_DailyPivot_SL7.5_TP15.0", sl_pips=7.5, tp_pips=15.0)),
        ("EG_DailyPivot_SL7.0_TP14.0", EURGBPDailyPivotPullback("EG_DailyPivot_SL7.0_TP14.0", sl_pips=7.0, tp_pips=14.0)),
    ]


def build_audnzd_candidates() -> List[Tuple[str, Any]]:
    return [
        ("AUDNZD_Fib_LB72_SL12.0_TP30.0", AUDNZDFibGoldenPocket("AUDNZD_Fib_LB72_SL12.0_TP30.0", sl_pips=12.0, tp_pips=30.0, lookback_h1=72)),
        ("AUDNZD_Fib_LB72_SL14.0_TP28.0", AUDNZDFibGoldenPocket("AUDNZD_Fib_LB72_SL14.0_TP28.0", sl_pips=14.0, tp_pips=28.0, lookback_h1=72)),
        ("AUDNZD_Fib_LB72_SL12.0_TP24.0", AUDNZDFibGoldenPocket("AUDNZD_Fib_LB72_SL12.0_TP24.0", sl_pips=12.0, tp_pips=24.0, lookback_h1=72)),
    ]


def evaluate_asset_week(args: Tuple[str, int]) -> Dict[str, Dict[str, Any]]:
    symbol, week_num = args
    t_res = get_week_dataset(week_num, raw_dir=f"data/{symbol}")
    if t_res is None:
        return {}
    ticks, meta = t_res if isinstance(t_res, tuple) else (t_res, None)
    if ticks is None or len(ticks) < 1000:
        return {}

    m15 = build_time_bars(ticks, timeframe_min=15, pip_size=PIP_SIZE)
    if len(m15) < 30:
        return {}
    h1 = m15.resample('1h').agg({'high': 'max', 'low': 'min', 'close': 'last', 'open': 'first'}).dropna()
    h4 = m15.resample('4h').agg({'high': 'max', 'low': 'min', 'close': 'last', 'open': 'first'}).dropna()

    fleet = build_eurgbp_candidates() if symbol == "EURGBP" else build_audnzd_candidates()
    res = {}

    for name, strat in fleet:
        signals = strat.generate_signals(ticks, m15, h1, h4)
        trade_log = run_matching_loop(ticks, signals, pip_size=PIP_SIZE, sl_pips=strat.sl_pips, tp_pips=strat.tp_pips)

        trades = []
        for t in trade_log:
            trades.append({
                "pnl_pips": float(t['pips']),
                "is_win": bool(t['pips'] > 0),
                "sl_pips": strat.sl_pips,
            })
        res[name] = {"week": week_num, "trades": trades}

    return res


def compute_partition_metrics(trades: List[Dict[str, Any]], n_weeks: int, sl_val: float) -> Dict[str, Any]:
    n = len(trades)
    if n == 0:
        return {"trades": 0, "net_pips": 0.0, "pf": 0.0, "wr": 0.0, "max_dd": 0.0, "ann_pips": 0.0, "ann_ret_pct": 0.0, "ret_to_dd": 0.0}

    pnls = np.array([t["pnl_pips"] for t in trades])
    net = float(np.sum(pnls))
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    gp = float(np.sum(wins)) if len(wins) > 0 else 0.0
    gl = float(abs(np.sum(losses))) if len(losses) > 0 else 1e-9
    pf = gp / gl
    wr = len(wins) / n * 100.0

    cum = np.cumsum(pnls)
    peaks = np.maximum.accumulate(cum)
    dd_series = peaks - cum
    max_dd = float(np.max(dd_series)) if len(dd_series) > 0 else 1.0

    ann_pips = net / (n_weeks / 52.0)
    ann_ret_pct = (ann_pips / sl_val) * 1.0 * 100.0
    ret_to_dd = net / max_dd if max_dd > 0 else 0.0

    return {
        "trades": n,
        "net_pips": round(net, 1),
        "pf": round(pf, 2),
        "wr": round(wr, 1),
        "max_dd": round(max_dd, 1),
        "ann_pips": round(ann_pips, 1),
        "ann_ret_pct": round(ann_ret_pct, 1),
        "ret_to_dd": round(ret_to_dd, 2),
        "exp": round(net / n, 2)
    }


def audit_asset(symbol: str):
    start = time.time()
    print("\n" + "=" * 90)
    print(f"  {symbol} EXPANDED CANDIDATES 5.77-YEAR & HOLDOUT CERTIFICATION AUDIT")
    print("=" * 90)

    disc_weeks = set(get_discovery_weeks(raw_dir=f"data/{symbol}"))
    hold_weeks = set(get_holdout_test_weeks(raw_dir=f"data/{symbol}"))
    all_weeks = sorted(list(disc_weeks.union(hold_weeks)))

    print(f"Auditing {len(all_weeks)} weeks (Discovery: {len(disc_weeks)}w | Holdout: {len(hold_weeks)}w)...")

    fleet = build_eurgbp_candidates() if symbol == "EURGBP" else build_audnzd_candidates()
    disc_trades: Dict[str, List[Dict[str, Any]]] = {name: [] for name, _ in fleet}
    hold_trades: Dict[str, List[Dict[str, Any]]] = {name: [] for name, _ in fleet}
    full_trades: Dict[str, List[Dict[str, Any]]] = {name: [] for name, _ in fleet}

    tasks = [(symbol, w) for w in all_weeks]
    completed = 0

    with ProcessPoolExecutor(max_workers=12) as executor:
        futures = {executor.submit(evaluate_asset_week, arg): arg for arg in tasks}
        for future in as_completed(futures):
            res = future.result()
            for name, data in res.items():
                w = data["week"]
                tr = data["trades"]
                full_trades[name].extend(tr)
                if w in disc_weeks:
                    disc_trades[name].extend(tr)
                elif w in hold_weeks:
                    hold_trades[name].extend(tr)
            completed += 1
            if completed % 50 == 0 or completed == len(all_weeks):
                print(f"  Progress: {completed}/{len(all_weeks)} weeks in {time.time() - start:.1f}s")

    print("\nCalculating Holdout and Full-Cycle Certification Metrics...")
    audit_table = []

    for name, strat in fleet:
        d_m = compute_partition_metrics(disc_trades[name], len(disc_weeks), strat.sl_pips)
        h_m = compute_partition_metrics(hold_trades[name], len(hold_weeks), strat.sl_pips)
        f_m = compute_partition_metrics(full_trades[name], len(all_weeks), strat.sl_pips)

        audit_table.append({
            "name": name,
            "sl": strat.sl_pips,
            "tp": strat.tp_pips,
            "full_trades": f_m["trades"],
            "full_pips": f_m["net_pips"],
            "full_ann_ret": f_m["ann_ret_pct"],
            "full_pf": f_m["pf"],
            "full_wr": f_m["wr"],
            "full_max_dd": f_m["max_dd"],
            "full_ret_dd": f_m["ret_to_dd"],
            "full_exp": f_m["exp"],
            "disc_pips": d_m["net_pips"],
            "disc_pf": d_m["pf"],
            "hold_pips": h_m["net_pips"],
            "hold_pf": h_m["pf"],
            "hold_wr": h_m["wr"],
        })

    df = pd.DataFrame(audit_table)
    df = df.sort_values(by="full_ann_ret", ascending=False)

    out_file = RESULTS_DIR / f"{symbol.lower()}_expansion_holdout_audit.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(audit_table, f, indent=2)

    print("\n" + "=" * 135)
    print(f"  {symbol} CERTIFICATION SCORECARD (30% QUARANTINED HOLDOUT vs FULL CYCLE)")
    print("=" * 135)
    print(f"{'Rank':<4} | {'Strategy Name':<38} | {'Full Trades':<11} | {'Full Pips':<10} | {'Ann Ret%':<9} | {'Full PF':<7} | {'Hold PF':<7} | {'Full WR%':<8} | {'Max DD':<7} | {'Ret/DD':<7} | {'Verdict'}")
    print("-" * 135)

    for idx, (_, r) in enumerate(df.iterrows(), 1):
        verdict = "PASSED CERT" if r["hold_pf"] >= 1.25 and r["full_ann_ret"] >= 15.0 else "SUB-PAR"
        print(f"{idx:<4} | {r['name']:<38} | {int(r['full_trades']):<11} | {r['full_pips']:<+10.1f} | {r['full_ann_ret']:<+8.1f}% | {r['full_pf']:<7.2f} | {r['hold_pf']:<7.2f} | {r['full_wr']:<8.1f} | {r['full_max_dd']:<7.1f} | {r['full_ret_dd']:<6.2f}x | {verdict}")

    print("-" * 135)
    return df


def main():
    print("=" * 80)
    print("  UNIFIED EXPANDED FLEET CERTIFICATION: EURGBP & AUDNZD")
    print("=" * 80)
    audit_asset("EURGBP")
    audit_asset("AUDNZD")


if __name__ == "__main__":
    main()

