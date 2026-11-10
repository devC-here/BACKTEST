#!/usr/bin/env python3
"""
================================================================================
  ALPHA DISCOVERY LAB V16: 3-DAY SWEEP, ASIA MIDPOINT, SUPERTREND & MIDDAY COIL
================================================================================
Evaluates 4 Quantitative Microstructure Architectures across 70% Discovery:
  1. DISC78_3DAY_SWEEP_REVERSAL:
     Sweeps 3-day (72-hour) high/low by 2-6 pips, fails to expand, closes back
     inside with M15 rejection wick >= 50% during 07:00-16:00 UTC. 2.0R target.
  2. DISC79_ASIA_MIDPOINT_REBOUND:
     H4 20-EMA trend agreement; London morning (07:00-11:00 UTC) pullback
     testing within 3.5 pips of Asian midpoint with M15 bounce candle. 2.0R target.
  3. DISC80_SUPERTREND_M15_RUNNER:
     M15 Supertrend (10, 3.0) flips in direction of H4 20-EMA trend during
     07:00-16:00 UTC with expansion candle (body >= 0.8x ATR). 2.0R target.
  4. DISC81_MIDDAY_COIL_NY_EXPANSION:
     Deep London midday compression (11:30-13:00 UTC range < 10 pips);
     explosive breakout between 13:00 and 15:00 UTC in H4 trend direction. 2.0R target.

Tested across EURUSD, EURGBP, and USDJPY on 70% Discovery Partition ONLY.
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

sys.dont_write_bytecode = True

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tick_engine.core.engine import BacktestEngine
from tick_engine.core.dataloader import get_week_dataset, get_discovery_weeks
from tick_engine.strategies.base import BaseStrategy, SIGNAL_LONG, SIGNAL_SHORT
from data_engineering_suite.bars import build_time_bars

RESULTS_DIR = ROOT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# Model 1: 3-Day High/Low Liquidity Sweep Reversal (DISC78)
# ==============================================================================
class ThreeDaySweepReversal(BaseStrategy):
    """
    Operates during 07:00 - 16:00 UTC.
    Computes rolling 3-day (72h / 288 M15 bars) high and low.
    When price sweeps beyond the 3-day extreme by 2-6 pips, but fails and closes
    back inside with an M15 rejection wick >= 50% pointing away from the range.
    Target: 2.0R.
    """
    def __init__(self, asset: str, sl_pips: float, tp_pips: float, pip_size: float, max_spread: float, **kwargs):
        super().__init__(**kwargs)
        self.asset = asset
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.pip_size = float(pip_size)
        self.max_spread = float(max_spread)
        self.name = f"{asset}_3Day_Sweep_Reversal"

    def generate_signals(self, ticks: np.ndarray):
        n = len(ticks)
        signals = np.zeros(n, dtype=np.int8)
        sl_arr = np.zeros(n, dtype=np.float64)
        tp_arr = np.zeros(n, dtype=np.float64)
        if n < 500:
            return signals, sl_arr, tp_arr

        m15 = build_time_bars(ticks, timeframe_min=15, pip_size=self.pip_size)
        if len(m15) < 300:
            return signals, sl_arr, tp_arr

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

        # Rolling 72h (288 bars) extremes shifted by 1 to prevent lookahead
        roll_hi = pd.Series(h).rolling(288).max().shift(1).values
        roll_lo = pd.Series(l).rolling(288).min().shift(1).values

        last_trade_date = None
        for i in range(289, len(m15)):
            if not (7 <= hours[i] <= 16):
                continue
            dt = dates[i]
            if dt == last_trade_date:
                continue
            if spread[i] > self.max_spread:
                continue
            if np.isnan(roll_hi[i]) or np.isnan(roll_lo[i]):
                continue

            bar_range = h[i] - l[i]
            if bar_range <= 0:
                continue

            upper_wick = (h[i] - max(c[i], o[i])) / bar_range
            lower_wick = (min(c[i], o[i]) - l[i]) / bar_range

            sig = 0
            # Sweep High Reversal: high swept above 3-day high by >= 2 pips, but closed below with upper wick >= 50%
            if (h[i] >= roll_hi[i] + 2.0 * self.pip_size) and (c[i] < roll_hi[i]) and (upper_wick >= 0.50) and (c[i] < o[i]):
                sig = SIGNAL_SHORT
            # Sweep Low Reversal: low swept below 3-day low by >= 2 pips, but closed above with lower wick >= 50%
            elif (l[i] <= roll_lo[i] - 2.0 * self.pip_size) and (c[i] > roll_lo[i]) and (lower_wick >= 0.50) and (c[i] > o[i]):
                sig = SIGNAL_LONG

            if sig != 0:
                bar_close_ms = time_ms[i] + dur_ms
                idx = np.searchsorted(tick_ts, bar_close_ms)
                if idx < n:
                    signals[idx] = sig
                    sl_arr[idx] = self.sl_pips
                    tp_arr[idx] = self.tp_pips
                    last_trade_date = dt

        return signals, sl_arr, tp_arr


# ==============================================================================
# Model 2: Asian Midpoint Rebound (DISC79)
# ==============================================================================
class AsianMidpointRebound(BaseStrategy):
    """
    Operates during London morning (07:00 - 11:00 UTC).
    Computes Asian session range (00:00 - 06:30 UTC) and its midpoint.
    H4 20-EMA macro trend agreement.
    When price pulls back to within 3.5 pips of Asian midpoint and prints a bounce candle in trend direction.
    Target: 2.0R.
    """
    def __init__(self, asset: str, sl_pips: float, tp_pips: float, pip_size: float, max_spread: float, **kwargs):
        super().__init__(**kwargs)
        self.asset = asset
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.pip_size = float(pip_size)
        self.max_spread = float(max_spread)
        self.name = f"{asset}_Asian_Midpoint_Rebound"

    def generate_signals(self, ticks: np.ndarray):
        n = len(ticks)
        signals = np.zeros(n, dtype=np.int8)
        sl_arr = np.zeros(n, dtype=np.float64)
        tp_arr = np.zeros(n, dtype=np.float64)
        if n < 500:
            return signals, sl_arr, tp_arr

        m15 = build_time_bars(ticks, timeframe_min=15, pip_size=self.pip_size)
        h4 = build_time_bars(ticks, timeframe_min=240, pip_size=self.pip_size)
        if len(m15) < 30 or len(h4) < 10:
            return signals, sl_arr, tp_arr

        h4_c = pd.Series(h4['close'].values)
        h4_ema20 = h4_c.ewm(span=20, adjust=False).mean().values
        h4_bias = np.where(h4_c.values > h4_ema20, 1, -1)
        h4_df = pd.DataFrame({'h4_bias': h4_bias}, index=h4.index)

        m15_df = pd.merge_asof(
            m15.sort_index(),
            h4_df.sort_index(),
            left_index=True,
            right_index=True,
            direction='backward'
        )

        c = m15_df['close'].values
        h = m15_df['high'].values
        l = m15_df['low'].values
        o = m15_df['open'].values
        spread = m15_df['spread_pips'].values
        h4_trend = m15_df['h4_bias'].fillna(0).values.astype(int)
        hours = m15_df.index.hour.values
        minutes = m15_df.index.minute.values
        dates = m15_df.index.date
        time_ms = m15_df.index.astype('datetime64[ms]').view('int64')
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        # Precompute Asian midpoint per date
        unique_dates = np.unique(dates)
        asia_mid = {}
        for d in unique_dates:
            mask = (dates == d) & (hours >= 0) & ((hours < 6) | ((hours == 6) & (minutes <= 30)))
            if np.any(mask):
                hi = np.max(h[mask])
                lo = np.min(l[mask])
                if (hi - lo) / self.pip_size >= 10.0:
                    asia_mid[d] = (hi + lo) / 2.0

        tolerance = 3.5 * self.pip_size
        last_trade_date = None

        for i in range(1, len(m15_df)):
            dt = dates[i]
            hr = hours[i]
            if not (7 <= hr <= 11):
                continue
            if dt == last_trade_date:
                continue
            if dt not in asia_mid:
                continue
            if spread[i] > self.max_spread:
                continue

            macro = h4_trend[i]
            if macro == 0:
                continue

            mid_val = asia_mid[dt]
            sig = 0
            # Bullish bounce at midpoint: low touches within tolerance of midpoint, closes above midpoint & bullish
            if macro == 1 and abs(l[i] - mid_val) <= tolerance and c[i] > mid_val and c[i] > o[i]:
                sig = SIGNAL_LONG
            # Bearish bounce at midpoint: high touches within tolerance of midpoint, closes below midpoint & bearish
            elif macro == -1 and abs(h[i] - mid_val) <= tolerance and c[i] < mid_val and c[i] < o[i]:
                sig = SIGNAL_SHORT

            if sig != 0:
                bar_close_ms = time_ms[i] + dur_ms
                idx = np.searchsorted(tick_ts, bar_close_ms)
                if idx < n:
                    signals[idx] = sig
                    sl_arr[idx] = self.sl_pips
                    tp_arr[idx] = self.tp_pips
                    last_trade_date = dt

        return signals, sl_arr, tp_arr


# ==============================================================================
# Model 3: Supertrend M15 Momentum Runner (DISC80)
# ==============================================================================
class SupertrendM15Runner(BaseStrategy):
    """
    Operates during 07:00 - 16:00 UTC.
    Calculates Supertrend(10, 3.0) on M15.
    H4 20-EMA macro trend confirmation.
    Triggers on the exact bar where Supertrend flips to agree with H4 trend.
    Target: 2.0R.
    """
    def __init__(self, asset: str, sl_pips: float, tp_pips: float, pip_size: float, max_spread: float, **kwargs):
        super().__init__(**kwargs)
        self.asset = asset
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.pip_size = float(pip_size)
        self.max_spread = float(max_spread)
        self.name = f"{asset}_Supertrend_M15_Runner"

    def generate_signals(self, ticks: np.ndarray):
        n = len(ticks)
        signals = np.zeros(n, dtype=np.int8)
        sl_arr = np.zeros(n, dtype=np.float64)
        tp_arr = np.zeros(n, dtype=np.float64)
        if n < 500:
            return signals, sl_arr, tp_arr

        m15 = build_time_bars(ticks, timeframe_min=15, pip_size=self.pip_size)
        h4 = build_time_bars(ticks, timeframe_min=240, pip_size=self.pip_size)
        if len(m15) < 30 or len(h4) < 10:
            return signals, sl_arr, tp_arr

        h4_c = pd.Series(h4['close'].values)
        h4_ema20 = h4_c.ewm(span=20, adjust=False).mean().values
        h4_bias = np.where(h4_c.values > h4_ema20, 1, -1)
        h4_df = pd.DataFrame({'h4_bias': h4_bias}, index=h4.index)

        m15_df = pd.merge_asof(
            m15.sort_index(),
            h4_df.sort_index(),
            left_index=True,
            right_index=True,
            direction='backward'
        )

        c = m15_df['close'].values
        h = m15_df['high'].values
        l = m15_df['low'].values
        spread = m15_df['spread_pips'].values
        h4_trend = m15_df['h4_bias'].fillna(0).values.astype(int)
        hours = m15_df.index.hour.values
        dates = m15_df.index.date
        time_ms = m15_df.index.astype('datetime64[ms]').view('int64')
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        # ATR(10)
        tr = np.maximum(h[1:] - l[1:], np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))
        tr = np.insert(tr, 0, h[0] - l[0])
        atr = pd.Series(tr).rolling(10).mean().values

        # Supertrend calculation
        multiplier = 3.0
        n_bars = len(m15_df)
        st_trend = np.zeros(n_bars, dtype=np.int8)
        upper_band = (h + l) / 2.0 + multiplier * atr
        lower_band = (h + l) / 2.0 - multiplier * atr

        for i in range(1, n_bars):
            if c[i] > upper_band[i-1]:
                st_trend[i] = 1
            elif c[i] < lower_band[i-1]:
                st_trend[i] = -1
            else:
                st_trend[i] = st_trend[i-1]
                if st_trend[i] == 1 and lower_band[i] < lower_band[i-1]:
                    lower_band[i] = lower_band[i-1]
                if st_trend[i] == -1 and upper_band[i] > upper_band[i-1]:
                    upper_band[i] = upper_band[i-1]

        last_trade_date = None
        for i in range(15, n_bars):
            if not (7 <= hours[i] <= 16):
                continue
            dt = dates[i]
            if dt == last_trade_date:
                continue
            if spread[i] > self.max_spread:
                continue

            macro = h4_trend[i]
            if macro == 0:
                continue

            sig = 0
            # Bullish flip
            if macro == 1 and st_trend[i-1] == -1 and st_trend[i] == 1:
                sig = SIGNAL_LONG
            # Bearish flip
            elif macro == -1 and st_trend[i-1] == 1 and st_trend[i] == -1:
                sig = SIGNAL_SHORT

            if sig != 0:
                bar_close_ms = time_ms[i] + dur_ms
                idx = np.searchsorted(tick_ts, bar_close_ms)
                if idx < n:
                    signals[idx] = sig
                    sl_arr[idx] = self.sl_pips
                    tp_arr[idx] = self.tp_pips
                    last_trade_date = dt

        return signals, sl_arr, tp_arr


# ==============================================================================
# Model 4: Midday London Coil & NY Open Explosion (DISC81)
# ==============================================================================
class MiddayCoilNYExplosion(BaseStrategy):
    """
    Measures London midday compression between 11:30 and 13:00 UTC.
    When midday range compresses below asset threshold (< 10.0 pips).
    Enters upon first M15 candle breaking out between 13:00 and 15:00 UTC in H4 trend direction.
    Target: 2.0R.
    """
    def __init__(self, asset: str, sl_pips: float, tp_pips: float, pip_size: float, max_spread: float, **kwargs):
        super().__init__(**kwargs)
        self.asset = asset
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.pip_size = float(pip_size)
        self.max_spread = float(max_spread)
        self.name = f"{asset}_Midday_Coil_NY_BO"

    def generate_signals(self, ticks: np.ndarray):
        n = len(ticks)
        signals = np.zeros(n, dtype=np.int8)
        sl_arr = np.zeros(n, dtype=np.float64)
        tp_arr = np.zeros(n, dtype=np.float64)
        if n < 500:
            return signals, sl_arr, tp_arr

        m15 = build_time_bars(ticks, timeframe_min=15, pip_size=self.pip_size)
        h4 = build_time_bars(ticks, timeframe_min=240, pip_size=self.pip_size)
        if len(m15) < 30 or len(h4) < 10:
            return signals, sl_arr, tp_arr

        h4_c = pd.Series(h4['close'].values)
        h4_ema20 = h4_c.ewm(span=20, adjust=False).mean().values
        h4_bias = np.where(h4_c.values > h4_ema20, 1, -1)
        h4_df = pd.DataFrame({'h4_bias': h4_bias}, index=h4.index)

        m15_df = pd.merge_asof(
            m15.sort_index(),
            h4_df.sort_index(),
            left_index=True,
            right_index=True,
            direction='backward'
        )

        c = m15_df['close'].values
        h = m15_df['high'].values
        l = m15_df['low'].values
        spread = m15_df['spread_pips'].values
        h4_trend = m15_df['h4_bias'].fillna(0).values.astype(int)
        hours = m15_df.index.hour.values
        minutes = m15_df.index.minute.values
        dates = m15_df.index.date
        time_ms = m15_df.index.astype('datetime64[ms]').view('int64')
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        # Precompute midday ranges (11:30 to 13:00 UTC)
        unique_dates = np.unique(dates)
        mid_hi = {}
        mid_lo = {}
        max_compress = 10.0 if self.asset != "USDJPY" else 14.0

        for d in unique_dates:
            mask = (dates == d) & (((hours == 11) & (minutes >= 30)) | (hours == 12) | ((hours == 13) & (minutes == 0)))
            if np.any(mask):
                hi_val = np.max(h[mask])
                lo_val = np.min(l[mask])
                if (hi_val - lo_val) / self.pip_size <= max_compress:
                    mid_hi[d] = hi_val
                    mid_lo[d] = lo_val

        last_trade_date = None
        for i in range(1, len(m15_df)):
            dt = dates[i]
            hr = hours[i]
            if not (13 <= hr <= 15):
                continue
            if dt == last_trade_date:
                continue
            if dt not in mid_hi or dt not in mid_lo:
                continue
            if spread[i] > self.max_spread:
                continue

            macro = h4_trend[i]
            if macro == 0:
                continue

            hi_lvl = mid_hi[dt]
            lo_lvl = mid_lo[dt]

            sig = 0
            if macro == 1 and c[i] > hi_lvl and c[i-1] <= hi_lvl:
                sig = SIGNAL_LONG
            elif macro == -1 and c[i] < lo_lvl and c[i-1] >= lo_lvl:
                sig = SIGNAL_SHORT

            if sig != 0:
                bar_close_ms = time_ms[i] + dur_ms
                idx = np.searchsorted(tick_ts, bar_close_ms)
                if idx < n:
                    signals[idx] = sig
                    sl_arr[idx] = self.sl_pips
                    tp_arr[idx] = self.tp_pips
                    last_trade_date = dt

        return signals, sl_arr, tp_arr


# ==============================================================================
# Discovery Worker for Alpha Discovery Lab V16
# ==============================================================================
def evaluate_v16_discovery_week(task: Tuple[str, int]) -> Dict[str, Any]:
    asset, week_num = task
    pip_size = 0.01 if asset == "USDJPY" else 0.0001
    engine = BacktestEngine(pip_size=pip_size, use_slippage=True)

    ticks, meta = get_week_dataset(week_num, raw_dir=f"data/{asset}")
    if ticks is None or len(ticks) < 500:
        return {"asset": asset, "week": week_num, "trades": {}}

    models = []
    if asset == "EURUSD":
        models.append(("3DAY_SWEEP_REV", ThreeDaySweepReversal(asset=asset, sl_pips=9.0, tp_pips=18.0, pip_size=0.0001, max_spread=0.6)))
        models.append(("ASIA_MID_REBOUND", AsianMidpointRebound(asset=asset, sl_pips=9.0, tp_pips=18.0, pip_size=0.0001, max_spread=0.6)))
        models.append(("SUPERTREND_M15", SupertrendM15Runner(asset=asset, sl_pips=9.0, tp_pips=18.0, pip_size=0.0001, max_spread=0.6)))
        models.append(("MIDDAY_COIL_BO", MiddayCoilNYExplosion(asset=asset, sl_pips=9.0, tp_pips=18.0, pip_size=0.0001, max_spread=0.6)))
    elif asset == "USDJPY":
        models.append(("3DAY_SWEEP_REV", ThreeDaySweepReversal(asset=asset, sl_pips=12.0, tp_pips=24.0, pip_size=0.01, max_spread=0.8)))
        models.append(("ASIA_MID_REBOUND", AsianMidpointRebound(asset=asset, sl_pips=12.0, tp_pips=24.0, pip_size=0.01, max_spread=0.8)))
        models.append(("SUPERTREND_M15", SupertrendM15Runner(asset=asset, sl_pips=12.0, tp_pips=24.0, pip_size=0.01, max_spread=0.8)))
        models.append(("MIDDAY_COIL_BO", MiddayCoilNYExplosion(asset=asset, sl_pips=12.0, tp_pips=24.0, pip_size=0.01, max_spread=0.8)))
    elif asset == "EURGBP":
        models.append(("3DAY_SWEEP_REV", ThreeDaySweepReversal(asset=asset, sl_pips=8.0, tp_pips=16.0, pip_size=0.0001, max_spread=0.8)))
        models.append(("ASIA_MID_REBOUND", AsianMidpointRebound(asset=asset, sl_pips=8.0, tp_pips=16.0, pip_size=0.0001, max_spread=0.8)))
        models.append(("SUPERTREND_M15", SupertrendM15Runner(asset=asset, sl_pips=8.0, tp_pips=16.0, pip_size=0.0001, max_spread=0.8)))
        models.append(("MIDDAY_COIL_BO", MiddayCoilNYExplosion(asset=asset, sl_pips=8.0, tp_pips=16.0, pip_size=0.0001, max_spread=0.8)))

    week_trades = {}
    for m_name, model in models:
        try:
            raw_trades = engine.run(ticks, model)
            t_list = []
            for i in range(len(raw_trades)):
                row = raw_trades[i]
                t_list.append({
                    "strategy": m_name,
                    "asset": asset,
                    "week": week_num,
                    "pips": float(row["pips"]),
                    "direction": int(row["direction"])
                })
            week_trades[m_name] = t_list
        except Exception:
            week_trades[m_name] = []

    return {"asset": asset, "week": week_num, "trades": week_trades}


def run_discovery_v16():
    t_start = time.time()
    print("=" * 80, flush=True)
    print("  LAUNCHING ALPHA DISCOVERY V16 EXPEDITION (70% DISCOVERY PARTITIONS)", flush=True)
    print("=" * 80, flush=True)

    assets = [os.environ["DISCOVERY_TARGET_SYMBOL"]] if "DISCOVERY_TARGET_SYMBOL" in os.environ else ["USDJPY", "EURGBP", "EURUSD"]
    all_tasks = []
    for asset in assets:
        disc_weeks = get_discovery_weeks(f"data/{asset}")
        print(f"[*] Queued {len(disc_weeks)} discovery weeks for {asset}...", flush=True)
        for w in disc_weeks:
            all_tasks.append((asset, w))

    print(f"\n[*] Total discovery tasks across all assets: {len(all_tasks)} weeks", flush=True)

    results_by_key: Dict[Tuple[str, str], List[float]] = {}
    completed = 0

    with ProcessPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(evaluate_v16_discovery_week, t): t for t in all_tasks}
        for future in as_completed(futures):
            res = future.result()
            completed += 1
            asset = res["asset"]
            for m_name, t_list in res["trades"].items():
                key = (asset, m_name)
                if key not in results_by_key:
                    results_by_key[key] = []
                results_by_key[key].extend([t["pips"] for t in t_list])

            if completed % 100 == 0 or completed == len(all_tasks):
                print(f"  [+] Processed {completed:3d}/{len(all_tasks)} discovery weeks in {time.time() - t_start:.1f}s...", flush=True)

    # Scorecard
    print("\n" + "=" * 90, flush=True)
    print(f"{'ASSET':<8} {'MODEL NAME':<26} {'TRADES':<8} {'WIN RATE':<10} {'PROFIT FACTOR':<15} {'NET PIPS':<12} {'STATUS'}", flush=True)
    print("=" * 90, flush=True)

    summary_rows = []
    for (asset, m_name), pips in sorted(results_by_key.items()):
        n_t = len(pips)
        if n_t == 0:
            print(f"{asset:<8} {m_name:<26} {'0':<8} {'0.0%':<10} {'0.00':<15} {'+0.0p':<12} {'NO TRADES'}", flush=True)
            continue

        wins = [p for p in pips if p > 0]
        losses = [p for p in pips if p < 0]
        wr = len(wins) / n_t * 100.0
        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))
        pf = gross_profit / gross_loss if gross_loss > 0 else (99.0 if gross_profit > 0 else 0.0)
        net_pips = sum(pips)

        status = "REJECTED"
        if net_pips > 250.0 and pf >= 1.25 and n_t >= 40:
            status = "PASS_HOLDOUT_CANDIDATE"

        p_str = f"{'+' if net_pips >= 0 else ''}{net_pips:.1f}p"
        print(f"{asset:<8} {m_name:<26} {n_t:<8} {wr:5.1f}%     {pf:5.2f}           {p_str:<12} {status}", flush=True)
        summary_rows.append({
            "asset": asset,
            "model": m_name,
            "trades": n_t,
            "win_rate": round(wr, 1),
            "profit_factor": round(pf, 2),
            "net_pips": round(net_pips, 1),
            "status": status
        })

    print("=" * 90, flush=True)
    print(f"[*] Discovery Lab V16 complete in {time.time() - t_start:.2f} seconds.", flush=True)

    out_file = RESULTS_DIR / "alpha_discovery_lab_v16_summary.json"
    with open(out_file, "w") as f:
        json.dump(summary_rows, f, indent=2)
    print(f"[*] Results saved to {out_file}", flush=True)


if __name__ == "__main__":
    run_discovery_v16()
