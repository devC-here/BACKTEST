#!/usr/bin/env python3
"""
================================================================================
  EURGBP ALPHA EXPANSION LAB: 4 NEW STRUCTURAL HYPOTHESES (LIGHTWEIGHT & FAST)
================================================================================
Building on the proven blueprint of DISC46 (Pivot Wall, 62.3% WR, PF 2.54)
and DISC42 (Macro H4/M30, 64.9% WR, PF 2.61):

Hypotheses:
1. EURGBP_Weekly_Pivot_Wall (DISC47):
   Weekly Pivot & S1/R1 levels calculated from previous week HLC.
   In H4 trend direction, test of Weekly Pivot / S1 / R1 with M30 rejection wick.

2. EURGBP_Camarilla_Floor_Elite (DISC48):
   Daily Camarilla L3 (long bounce in bull) / H3 (short bounce in bear).
   M15 rejection wick >= 45%, trend-aligned with Daily EMA(20).

3. EURGBP_Session_VWAP_Pullback (DISC49):
   London Session VWAP (07:00 UTC anchor). In H1 trend direction,
   pullback to -1.0 to -1.5 SD band (Bull) or +1.0 to +1.5 SD band (Bear)
   with M15 rejection wick.

4. EURGBP_Previous_Day_HL_Sweep_Reversal (DISC50):
   Previous Day High/Low liquidity test. In trend direction, price retests
   Prior Day High (now support in bull) or Prior Day Low (now resistance in bear),
   printing an M15 rejection wick.
================================================================================
"""

from __future__ import annotations
import sys
import os
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Any

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tick_engine.core.engine import BacktestEngine
from tick_engine.core.dataloader import get_week_dataset, load_manifest
from tick_engine.strategies.base import BaseStrategy, SIGNAL_LONG, SIGNAL_SHORT
from data_engineering_suite.bars import build_time_bars

PIP_SIZE = 0.0001
RAW_DIR = ROOT_DIR / "weekly_data" / "EURGBP"


# ==============================================================================
# 1. Weekly Pivot Wall (DISC47 Candidate)
# ==============================================================================
class EURGBPWeeklyPivotWall(BaseStrategy):
    def __init__(
        self,
        sl_pips: float = 12.0,
        tp_pips: float = 20.0,
        min_wick: float = 0.45,
        max_dist_pips: float = 5.0,
        max_spread: float = 2.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.min_wick = float(min_wick)
        self.max_dist_pips = float(max_dist_pips)
        self.max_spread = float(max_spread)
        self.name = f"Weekly_Pivot_Wall_{self.sl_pips}SL_{self.tp_pips}TP"

    def generate_signals(self, ticks: np.ndarray):
        n = len(ticks)
        signals = np.zeros(n, dtype=np.int8)
        sl_arr = np.zeros(n, dtype=np.float64)
        tp_arr = np.zeros(n, dtype=np.float64)

        if n < 500:
            return signals, sl_arr, tp_arr

        m30 = build_time_bars(ticks, timeframe_min=30)
        if len(m30) < 30:
            return signals, sl_arr, tp_arr

        # Resample from M30 bars directly (only ~200 rows!)
        daily = m30.resample('1D').agg({'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
        if len(daily) < 2:
            return signals, sl_arr, tp_arr

        weekly = daily.resample('W-SUN').agg({'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
        if len(weekly) < 2:
            wp = (daily['high'].iloc[0] + daily['low'].iloc[0] + daily['close'].iloc[0]) / 3.0
            ws1 = 2 * wp - daily['high'].iloc[0]
            wr1 = 2 * wp - daily['low'].iloc[0]
        else:
            prev_wh = weekly['high'].shift(1).iloc[-1]
            prev_wl = weekly['low'].shift(1).iloc[-1]
            prev_wc = weekly['close'].shift(1).iloc[-1]
            wp = (prev_wh + prev_wl + prev_wc) / 3.0
            ws1 = 2 * wp - prev_wh
            wr1 = 2 * wp - prev_wl

        h4 = m30.resample('4h').agg({'close': 'last'}).dropna()
        h4['ema30'] = h4['close'].ewm(span=30, adjust=False).mean()
        h4['macro'] = np.where(h4['close'] > h4['ema30'], 1, -1)
        m30_macro = h4['macro'].reindex(m30.index, method='ffill').fillna(0).values

        c = m30['close'].values
        h = m30['high'].values
        l = m30['low'].values
        o = m30['open'].values
        spread = m30['spread_pips'].values
        hours = m30.index.hour.values
        dates = m30.index.date

        time_ms = m30.index.astype('datetime64[ms]').view('int64')
        dur_ms = 30 * 60 * 1000
        tick_ts = ticks['timestamp']

        last_trade_date = None

        for i in range(10, len(m30)):
            hr = hours[i]
            if not (7 <= hr <= 16):
                continue
            if spread[i] > self.max_spread:
                continue

            dt = dates[i]
            if dt == last_trade_date:
                continue

            m = m30_macro[i]
            bar_rng = (h[i] - l[i]) / PIP_SIZE
            if bar_rng < 3.0:
                continue

            sig = 0
            if m > 0:
                dist_p = abs(l[i] - wp) / PIP_SIZE
                dist_s1 = abs(l[i] - ws1) / PIP_SIZE
                lower_wick = (min(o[i], c[i]) - l[i]) / PIP_SIZE
                if (dist_p <= self.max_dist_pips or dist_s1 <= self.max_dist_pips) and (lower_wick / bar_rng) >= self.min_wick and c[i] > o[i]:
                    sig = SIGNAL_LONG
            elif m < 0:
                dist_p = abs(h[i] - wp) / PIP_SIZE
                dist_r1 = abs(h[i] - wr1) / PIP_SIZE
                upper_wick = (h[i] - max(o[i], c[i])) / PIP_SIZE
                if (dist_p <= self.max_dist_pips or dist_r1 <= self.max_dist_pips) and (upper_wick / bar_rng) >= self.min_wick and c[i] < o[i]:
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


# ==============================================================================
# 2. Daily Camarilla Floor Rejection (DISC48 Candidate)
# ==============================================================================
class EURGBPCamarillaFloor(BaseStrategy):
    def __init__(
        self,
        sl_pips: float = 8.5,
        tp_pips: float = 14.0,
        min_wick: float = 0.45,
        max_dist_pips: float = 3.5,
        max_spread: float = 2.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.min_wick = float(min_wick)
        self.max_dist_pips = float(max_dist_pips)
        self.max_spread = float(max_spread)
        self.name = f"Camarilla_Floor_{self.sl_pips}SL_{self.tp_pips}TP"

    def generate_signals(self, ticks: np.ndarray):
        n = len(ticks)
        signals = np.zeros(n, dtype=np.int8)
        sl_arr = np.zeros(n, dtype=np.float64)
        tp_arr = np.zeros(n, dtype=np.float64)

        if n < 500:
            return signals, sl_arr, tp_arr

        m15 = build_time_bars(ticks, timeframe_min=15)
        if len(m15) < 30:
            return signals, sl_arr, tp_arr

        daily = m15.resample('1D').agg({'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
        if len(daily) < 2:
            return signals, sl_arr, tp_arr

        prev_h = daily['high'].shift(1)
        prev_l = daily['low'].shift(1)
        prev_c = daily['close'].shift(1)
        rng = prev_h - prev_l

        daily['h3'] = prev_c + rng * (1.1 / 4.0)
        daily['l3'] = prev_c - rng * (1.1 / 4.0)

        h4 = m15.resample('4h').agg({'close': 'last'}).dropna()
        h4['ema20'] = h4['close'].ewm(span=20, adjust=False).mean()
        h4['macro'] = np.where(h4['close'] > h4['ema20'], 1, -1)

        m15_l3 = daily['l3'].reindex(m15.index, method='ffill').values
        m15_h3 = daily['h3'].reindex(m15.index, method='ffill').values
        m15_macro = h4['macro'].reindex(m15.index, method='ffill').fillna(0).values

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
            if spread[i] > self.max_spread:
                continue

            dt = dates[i]
            if dt == last_trade_date:
                continue

            m = m15_macro[i]
            bar_rng = (h[i] - l[i]) / PIP_SIZE
            if bar_rng < 2.5:
                continue

            sig = 0
            if m > 0 and not np.isnan(m15_l3[i]):
                dist_l3 = abs(l[i] - m15_l3[i]) / PIP_SIZE
                lower_wick = (min(o[i], c[i]) - l[i]) / PIP_SIZE
                if dist_l3 <= self.max_dist_pips and (lower_wick / bar_rng) >= self.min_wick and c[i] > o[i]:
                    sig = SIGNAL_LONG
            elif m < 0 and not np.isnan(m15_h3[i]):
                dist_h3 = abs(h[i] - m15_h3[i]) / PIP_SIZE
                upper_wick = (h[i] - max(o[i], c[i])) / PIP_SIZE
                if dist_h3 <= self.max_dist_pips and (upper_wick / bar_rng) >= self.min_wick and c[i] < o[i]:
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


# ==============================================================================
# 3. Session VWAP Pullback Rejection (DISC49 Candidate)
# ==============================================================================
class EURGBPSessionVWAPPullback(BaseStrategy):
    def __init__(
        self,
        sl_pips: float = 9.0,
        tp_pips: float = 15.0,
        min_wick: float = 0.45,
        max_spread: float = 2.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.min_wick = float(min_wick)
        self.max_spread = float(max_spread)
        self.name = f"Session_VWAP_Pullback_{self.sl_pips}SL_{self.tp_pips}TP"

    def generate_signals(self, ticks: np.ndarray):
        n = len(ticks)
        signals = np.zeros(n, dtype=np.int8)
        sl_arr = np.zeros(n, dtype=np.float64)
        tp_arr = np.zeros(n, dtype=np.float64)

        if n < 500:
            return signals, sl_arr, tp_arr

        m15 = build_time_bars(ticks, timeframe_min=15)
        if len(m15) < 30:
            return signals, sl_arr, tp_arr

        h1 = m15.resample('1h').agg({'close': 'last'}).dropna()
        h1['ema50'] = h1['close'].ewm(span=50, adjust=False).mean()
        h1['macro'] = np.where(h1['close'] > h1['ema50'], 1, -1)
        m15_macro = h1['macro'].reindex(m15.index, method='ffill').fillna(0).values

        tp_price = (m15['high'] + m15['low'] + m15['close']) / 3.0
        vol = m15['spread_pips'].clip(lower=0.5)

        dates = m15.index.date
        hours = m15.index.hour
        session_vwap = np.zeros(len(m15))

        cum_pv = 0.0
        cum_v = 0.0
        cur_day = None

        for idx_b in range(len(m15)):
            d = dates[idx_b]
            hr = hours[idx_b]
            if d != cur_day or hr == 7:
                cur_day = d
                cum_pv = 0.0
                cum_v = 0.0
            
            p = tp_price.iloc[idx_b]
            v = vol.iloc[idx_b]
            cum_pv += p * v
            cum_v += v
            session_vwap[idx_b] = cum_pv / cum_v if cum_v > 0 else p

        c = m15['close'].values
        h = m15['high'].values
        l = m15['low'].values
        o = m15['open'].values
        spread = m15['spread_pips'].values
        time_ms = m15.index.astype('datetime64[ms]').view('int64')
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        last_trade_date = None

        for i in range(15, len(m15)):
            hr = hours[i]
            if not (8 <= hr <= 15):
                continue
            if spread[i] > self.max_spread:
                continue

            dt = dates[i]
            if dt == last_trade_date:
                continue

            m = m15_macro[i]
            bar_rng = (h[i] - l[i]) / PIP_SIZE
            if bar_rng < 2.5:
                continue

            vwap_val = session_vwap[i]
            dist_vwap = abs(l[i] - vwap_val) / PIP_SIZE if m > 0 else abs(h[i] - vwap_val) / PIP_SIZE

            sig = 0
            if m > 0:
                lower_wick = (min(o[i], c[i]) - l[i]) / PIP_SIZE
                if dist_vwap <= 3.5 and (lower_wick / bar_rng) >= self.min_wick and c[i] > o[i] and c[i] > vwap_val:
                    sig = SIGNAL_LONG
            elif m < 0:
                upper_wick = (h[i] - max(o[i], c[i])) / PIP_SIZE
                if dist_vwap <= 3.5 and (upper_wick / bar_rng) >= self.min_wick and c[i] < o[i] and c[i] < vwap_val:
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


# ==============================================================================
# 4. Previous Day High/Low Retest Wall (DISC50 Candidate)
# ==============================================================================
class EURGBPPreviousDayHLWall(BaseStrategy):
    def __init__(
        self,
        sl_pips: float = 9.5,
        tp_pips: float = 16.0,
        min_wick: float = 0.45,
        max_dist_pips: float = 4.0,
        max_spread: float = 2.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.min_wick = float(min_wick)
        self.max_dist_pips = float(max_dist_pips)
        self.max_spread = float(max_spread)
        self.name = f"Prior_Day_HL_Wall_{self.sl_pips}SL_{self.tp_pips}TP"

    def generate_signals(self, ticks: np.ndarray):
        n = len(ticks)
        signals = np.zeros(n, dtype=np.int8)
        sl_arr = np.zeros(n, dtype=np.float64)
        tp_arr = np.zeros(n, dtype=np.float64)

        if n < 500:
            return signals, sl_arr, tp_arr

        m15 = build_time_bars(ticks, timeframe_min=15)
        if len(m15) < 30:
            return signals, sl_arr, tp_arr

        daily = m15.resample('1D').agg({'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
        if len(daily) < 2:
            return signals, sl_arr, tp_arr

        daily['prev_h'] = daily['high'].shift(1)
        daily['prev_l'] = daily['low'].shift(1)

        h4 = m15.resample('4h').agg({'close': 'last'}).dropna()
        h4['ema25'] = h4['close'].ewm(span=25, adjust=False).mean()
        h4['macro'] = np.where(h4['close'] > h4['ema25'], 1, -1)

        m15_prev_h = daily['prev_h'].reindex(m15.index, method='ffill').values
        m15_prev_l = daily['prev_l'].reindex(m15.index, method='ffill').values
        m15_macro = h4['macro'].reindex(m15.index, method='ffill').fillna(0).values

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
            if spread[i] > self.max_spread:
                continue

            dt = dates[i]
            if dt == last_trade_date:
                continue

            m = m15_macro[i]
            bar_rng = (h[i] - l[i]) / PIP_SIZE
            if bar_rng < 2.5:
                continue

            sig = 0
            if m > 0 and not np.isnan(m15_prev_h[i]):
                dist_h = abs(l[i] - m15_prev_h[i]) / PIP_SIZE
                dist_l = abs(l[i] - m15_prev_l[i]) / PIP_SIZE
                lower_wick = (min(o[i], c[i]) - l[i]) / PIP_SIZE
                if (dist_h <= self.max_dist_pips or dist_l <= self.max_dist_pips) and (lower_wick / bar_rng) >= self.min_wick and c[i] > o[i]:
                    sig = SIGNAL_LONG
            elif m < 0 and not np.isnan(m15_prev_l[i]):
                dist_l = abs(h[i] - m15_prev_l[i]) / PIP_SIZE
                dist_h = abs(h[i] - m15_prev_h[i]) / PIP_SIZE
                upper_wick = (h[i] - max(o[i], c[i])) / PIP_SIZE
                if (dist_l <= self.max_dist_pips or dist_h <= self.max_dist_pips) and (upper_wick / bar_rng) >= self.min_wick and c[i] < o[i]:
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


def evaluate_week(week_num: int) -> Dict[str, Any]:
    engine = BacktestEngine(pip_size=PIP_SIZE, use_slippage=True)
    ticks, meta = get_week_dataset(week_num, raw_dir=RAW_DIR)

    models = [
        ("Weekly_Pivot_Wall", EURGBPWeeklyPivotWall()),
        ("Camarilla_Floor", EURGBPCamarillaFloor()),
        ("Session_VWAP_Pullback", EURGBPSessionVWAPPullback()),
        ("Prior_Day_HL_Wall", EURGBPPreviousDayHLWall()),
    ]

    week_res = {}
    for name, m in models:
        raw_trades = engine.run(ticks, m)
        t_list = []
        for i in range(len(raw_trades)):
            row = raw_trades[i]
            t_list.append({
                "strategy": name,
                "week": week_num,
                "entry_time": int(row["entry_time"]),
                "exit_time": int(row["exit_time"]),
                "direction": "BUY" if row["direction"] == 1 else "SELL",
                "pips": float(row["pips"]),
            })
        week_res[name] = t_list

    return {"week": week_num, "trades": week_res}


def compute_stats(trades: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(trades)
    if n == 0:
        return {"trades": 0, "wr": 0.0, "pips": 0.0, "pf": 0.0, "max_dd": 0.0, "ret_dd": 0.0}
    pips = np.array([t["pips"] for t in trades])
    wins = np.sum(pips > 0)
    wr = wins / n * 100
    gp = float(np.sum(pips[pips > 0])) if wins > 0 else 0.0
    gl = float(abs(np.sum(pips[pips < 0]))) if (n - wins) > 0 else 0.0
    pf = gp / gl if gl > 0 else (99.0 if gp > 0 else 0.0)

    cum = np.cumsum(pips)
    dd = np.maximum.accumulate(cum) - cum
    max_dd = float(np.max(dd)) if len(dd) > 0 else 0.0
    net_pips = float(np.sum(pips))
    ret_dd = net_pips / max_dd if max_dd > 0 else 0.0

    return {
        "trades": n,
        "wr": round(wr, 1),
        "pips": round(net_pips, 1),
        "pf": round(pf, 2),
        "max_dd": round(max_dd, 1),
        "ret_dd": round(ret_dd, 2)
    }


def main():
    t0 = time.time()
    manifest = load_manifest(RAW_DIR)
    weeks = sorted(list(manifest.keys()))
    total = len(weeks)

    print("=" * 90)
    print(f"  EURGBP ALPHA EXPANSION LAB: 4 NEW STRUCTURAL HYPOTHESES ({total} WEEKS)  ".center(90))
    print("=" * 90)

    all_trades = {
        "Weekly_Pivot_Wall": [],
        "Camarilla_Floor": [],
        "Session_VWAP_Pullback": [],
        "Prior_Day_HL_Wall": []
    }

    workers = 4
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(evaluate_week, w): w for w in weeks}
        done = 0
        for f in futures:
            res = f.result()
            done += 1
            for name, t_list in res["trades"].items():
                all_trades[name].extend(t_list)
            if done % 25 == 0 or done == total:
                print(f"  -> Evaluated {done}/{total} weeks ({(done/total)*100:5.1f}%) in {time.time()-t0:.1f}s...")

    print("\n" + "=" * 90)
    print("   EURGBP EXPANSION SCORECARD (FULL 157 WEEKS / 3 YEARS)   ".center(90))
    print("=" * 90)
    print(f"{'Strategy Architecture':<30} | {'Trades':>6} | {'Win Rate':>8} | {'Profit Factor':>13} | {'Net Pips':>9} | {'Max DD':>8} | {'Ret/DD':>7}")
    print("-" * 90)

    for name in all_trades:
        t_list = all_trades[name]
        t_list.sort(key=lambda x: x["entry_time"])
        s = compute_stats(t_list)
        print(f"{name:<30} | {s['trades']:>6} | {s['wr']:>7.1f}% | {s['pf']:>13.2f} | {s['pips']:>+8.1f}p | {s['max_dd']:>7.1f}p | {s['ret_dd']:>6.2f}x")

    print("=" * 90)
    print(f"Completed research in {time.time()-t0:.1f}s.")


if __name__ == "__main__":
    main()
