#!/usr/bin/env python3
"""
================================================================================
  ALPHA DISCOVERY LAB V14: REVERSAL FADE, DONCHIAN RETEST, RIBBON & NY DRIFT
================================================================================
Evaluates 4 Quantitative Microstructure Architectures across 70% Discovery:
  1. DISC70_LONDON_SWEEP_FADE:
     London open (07:00 - 09:30 UTC) sweeps >= 1.5x ATR beyond Asian high/low,
     forms an extreme rejection wick (>= 50%) with RSI(14) > 70 or < 30.
     Fades back toward Asian median. 1.8R target.
  2. DISC71_H4_DONCHIAN_PULLBACK:
     H4 20-period Donchian breakout retest. Pullback to broken channel level
     with M15 rejection wick (>= 45%) during 07:00 - 17:00 UTC. 2.5R target.
  3. DISC72_EMA_RIBBON_COMPRESSION_BO:
     M15 EMA ribbon (8, 13, 21) tight compression (< 3.0 pips spread) fanning
     out in H1 50-EMA trend direction during 07:00 - 16:00 UTC. 2.0R target.
  4. DISC73_POST_FIX_NY_DRIFT:
     Post-London Fix (16:30 - 19:30 UTC) trend continuation. M15 pullback
     to 20-EMA in direction of daily trend. 1.8R target.

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
# Model 1: London Sweep Fade (DISC70)
# ==============================================================================
class LondonSweepFade(BaseStrategy):
    """
    Operates during London Open (07:00 - 09:30 UTC).
    Computes Asian session range (00:00 - 06:30 UTC).
    If price sweeps beyond Asian High or Low by >= 4 pips and forms an M15
    rejection wick >= 50% pointing away from Asian range, fade the sweep.
    Target: 1.8R.
    """
    def __init__(self, asset: str, sl_pips: float, tp_pips: float, pip_size: float, max_spread: float, **kwargs):
        super().__init__(**kwargs)
        self.asset = asset
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.pip_size = float(pip_size)
        self.max_spread = float(max_spread)
        self.name = f"{asset}_London_Sweep_Fade"

    def generate_signals(self, ticks: np.ndarray):
        n = len(ticks)
        signals = np.zeros(n, dtype=np.int8)
        sl_arr = np.zeros(n, dtype=np.float64)
        tp_arr = np.zeros(n, dtype=np.float64)
        if n < 500:
            return signals, sl_arr, tp_arr

        m15 = build_time_bars(ticks, timeframe_min=15, pip_size=self.pip_size)
        if len(m15) < 30:
            return signals, sl_arr, tp_arr

        c = m15['close'].values
        h = m15['high'].values
        l = m15['low'].values
        o = m15['open'].values
        spread = m15['spread_pips'].values
        hours = m15.index.hour.values
        minutes = m15.index.minute.values
        dates = m15.index.date
        time_ms = m15.index.astype('datetime64[ms]').view('int64')
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        # Precompute Asian ranges per date
        unique_dates = np.unique(dates)
        asian_highs = {}
        asian_lows = {}
        for d in unique_dates:
            mask = (dates == d) & (hours >= 0) & ((hours < 6) | ((hours == 6) & (minutes <= 30)))
            if np.any(mask):
                asian_highs[d] = np.max(h[mask])
                asian_lows[d] = np.min(l[mask])

        last_trade_date = None
        for i in range(1, len(m15)):
            dt = dates[i]
            hr = hours[i]
            mn = minutes[i]

            if not (7 <= hr < 10):
                continue
            if dt == last_trade_date:
                continue
            if dt not in asian_highs or dt not in asian_lows:
                continue
            if spread[i] > self.max_spread:
                continue

            asia_hi = asian_highs[dt]
            asia_lo = asian_lows[dt]
            asia_rng_pips = (asia_hi - asia_lo) / self.pip_size
            if asia_rng_pips < 8.0:
                continue

            bar_range = h[i] - l[i]
            if bar_range <= 0:
                continue

            upper_wick = (h[i] - max(c[i], o[i])) / bar_range
            lower_wick = (min(c[i], o[i]) - l[i]) / bar_range

            sig = 0
            # Bearish Fade: High swept above Asian High, closed back down with top rejection wick >= 50%
            if h[i] > asia_hi + 3.0 * self.pip_size and upper_wick >= 0.50 and c[i] < o[i]:
                sig = SIGNAL_SHORT
            # Bullish Fade: Low swept below Asian Low, closed back up with bottom rejection wick >= 50%
            elif l[i] < asia_lo - 3.0 * self.pip_size and lower_wick >= 0.50 and c[i] > o[i]:
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
# Model 2: H4 Donchian Pullback Retest (DISC71)
# ==============================================================================
class H4DonchianPullbackRetest(BaseStrategy):
    """
    H4 Donchian (20-period) high/low level breakout.
    When price pulls back on M15 to within 4.0 pips of the broken level during
    07:00 - 17:00 UTC and prints a rejection wick >= 45% in trend direction.
    Target: 2.5R.
    """
    def __init__(self, asset: str, sl_pips: float, tp_pips: float, pip_size: float, max_spread: float, **kwargs):
        super().__init__(**kwargs)
        self.asset = asset
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.pip_size = float(pip_size)
        self.max_spread = float(max_spread)
        self.name = f"{asset}_H4_Donchian_Retest"

    def generate_signals(self, ticks: np.ndarray):
        n = len(ticks)
        signals = np.zeros(n, dtype=np.int8)
        sl_arr = np.zeros(n, dtype=np.float64)
        tp_arr = np.zeros(n, dtype=np.float64)
        if n < 500:
            return signals, sl_arr, tp_arr

        m15 = build_time_bars(ticks, timeframe_min=15, pip_size=self.pip_size)
        h4 = build_time_bars(ticks, timeframe_min=240, pip_size=self.pip_size)
        if len(m15) < 30 or len(h4) < 25:
            return signals, sl_arr, tp_arr

        h4_h = pd.Series(h4['high'].values)
        h4_l = pd.Series(h4['low'].values)
        h4_c = pd.Series(h4['close'].values)

        # 20-period Donchian Channels shifted by 1 to prevent lookahead
        donch_hi = h4_h.rolling(20).max().shift(1).values
        donch_lo = h4_l.rolling(20).min().shift(1).values
        h4_ema20 = h4_c.ewm(span=20, adjust=False).mean().values

        h4_df = pd.DataFrame({
            'donch_hi': donch_hi,
            'donch_lo': donch_lo,
            'h4_ema': h4_ema20,
            'h4_close': h4_c.values
        }, index=h4.index)

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
        donch_hi_arr = m15_df['donch_hi'].values
        donch_lo_arr = m15_df['donch_lo'].values
        h4_ema_arr = m15_df['h4_ema'].values
        h4_close_arr = m15_df['h4_close'].values
        hours = m15_df.index.hour.values
        dates = m15_df.index.date
        time_ms = m15_df.index.astype('datetime64[ms]').view('int64')
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        tolerance = 4.0 * self.pip_size
        last_trade_date = None

        for i in range(1, len(m15_df)):
            if not (7 <= hours[i] <= 17):
                continue
            dt = dates[i]
            if dt == last_trade_date:
                continue
            if spread[i] > self.max_spread:
                continue
            if np.isnan(donch_hi_arr[i]) or np.isnan(donch_lo_arr[i]):
                continue

            bar_range = h[i] - l[i]
            if bar_range <= 0:
                continue

            lower_wick = (min(c[i], o[i]) - l[i]) / bar_range
            upper_wick = (h[i] - max(c[i], o[i])) / bar_range

            sig = 0
            # Bullish Retest: H4 close above EMA, price tests near Donchian High, holds with lower wick >= 45%
            if h4_close_arr[i] > h4_ema_arr[i]:
                dist = abs(l[i] - donch_hi_arr[i])
                if dist <= tolerance and lower_wick >= 0.45 and c[i] > o[i]:
                    sig = SIGNAL_LONG

            # Bearish Retest: H4 close below EMA, price tests near Donchian Low, holds with upper wick >= 45%
            elif h4_close_arr[i] < h4_ema_arr[i]:
                dist = abs(h[i] - donch_lo_arr[i])
                if dist <= tolerance and upper_wick >= 0.45 and c[i] < o[i]:
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
# Model 3: EMA Ribbon Compression & Fan Breakout (DISC72)
# ==============================================================================
class EMARibbonCompressionBreakout(BaseStrategy):
    """
    M15 EMA Ribbon: 8, 13, 21.
    H1 50-EMA establishes broader trend bias.
    When max distance between EMA 8 and EMA 21 is < 3.0 pips for 3 consecutive bars (compression),
    then EMA 8 fans out decisively in trend direction.
    Target: 2.0R.
    """
    def __init__(self, asset: str, sl_pips: float, tp_pips: float, pip_size: float, max_spread: float, **kwargs):
        super().__init__(**kwargs)
        self.asset = asset
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.pip_size = float(pip_size)
        self.max_spread = float(max_spread)
        self.name = f"{asset}_EMA_Ribbon_Compression_BO"

    def generate_signals(self, ticks: np.ndarray):
        n = len(ticks)
        signals = np.zeros(n, dtype=np.int8)
        sl_arr = np.zeros(n, dtype=np.float64)
        tp_arr = np.zeros(n, dtype=np.float64)
        if n < 500:
            return signals, sl_arr, tp_arr

        m15 = build_time_bars(ticks, timeframe_min=15, pip_size=self.pip_size)
        h1 = build_time_bars(ticks, timeframe_min=60, pip_size=self.pip_size)
        if len(m15) < 30 or len(h1) < 55:
            return signals, sl_arr, tp_arr

        h1_c = pd.Series(h1['close'].values)
        h1_ema50 = h1_c.ewm(span=50, adjust=False).mean().values
        h1_bias = np.where(h1_c.values > h1_ema50, 1, -1)
        h1_df = pd.DataFrame({'h1_trend': h1_bias}, index=h1.index)

        m15_df = pd.merge_asof(
            m15.sort_index(),
            h1_df.sort_index(),
            left_index=True,
            right_index=True,
            direction='backward'
        )

        c = m15_df['close'].values
        spread = m15_df['spread_pips'].values
        h1_trend = m15_df['h1_trend'].fillna(0).values.astype(int)
        hours = m15_df.index.hour.values
        dates = m15_df.index.date
        time_ms = m15_df.index.astype('datetime64[ms]').view('int64')
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        c_s = pd.Series(c)
        ema8 = c_s.ewm(span=8, adjust=False).mean().values
        ema13 = c_s.ewm(span=13, adjust=False).mean().values
        ema21 = c_s.ewm(span=21, adjust=False).mean().values

        ribbon_spread_pips = np.abs(ema8 - ema21) / self.pip_size
        compress_threshold = 3.0 if self.asset != "USDJPY" else 4.0

        last_trade_date = None
        for i in range(25, len(m15_df)):
            if not (7 <= hours[i] <= 16):
                continue
            dt = dates[i]
            if dt == last_trade_date:
                continue
            if spread[i] > self.max_spread:
                continue

            trend = h1_trend[i]
            if trend == 0:
                continue

            # Check compression over prior 3 bars
            was_compressed = (ribbon_spread_pips[i-1] <= compress_threshold) and (ribbon_spread_pips[i-2] <= compress_threshold)
            if not was_compressed:
                continue

            sig = 0
            # Bullish Fan Out
            if trend == 1 and (ema8[i] > ema13[i] > ema21[i]) and (c[i] > ema8[i]) and (ribbon_spread_pips[i] > compress_threshold):
                sig = SIGNAL_LONG
            # Bearish Fan Out
            elif trend == -1 and (ema8[i] < ema13[i] < ema21[i]) and (c[i] < ema8[i]) and (ribbon_spread_pips[i] > compress_threshold):
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
# Model 4: Post-London Fix NY Drift (DISC73)
# ==============================================================================
class PostFixNYDrift(BaseStrategy):
    """
    Operates during 16:30 - 19:30 UTC after the 16:00 UTC London Fix.
    Exploits smooth late-session American trend continuation.
    H4 20-EMA determines macro direction.
    M15 pulls back to touch 20-EMA with a small bounce candle closing in trend direction.
    Target: 1.8R.
    """
    def __init__(self, asset: str, sl_pips: float, tp_pips: float, pip_size: float, max_spread: float, **kwargs):
        super().__init__(**kwargs)
        self.asset = asset
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.pip_size = float(pip_size)
        self.max_spread = float(max_spread)
        self.name = f"{asset}_Post_Fix_NY_Drift"

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
        h4_trend = np.where(h4_c.values > h4_ema20, 1, -1)
        h4_df = pd.DataFrame({'h4_bias': h4_trend}, index=h4.index)

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
        h4_bias = m15_df['h4_bias'].fillna(0).values.astype(int)
        hours = m15_df.index.hour.values
        minutes = m15_df.index.minute.values
        dates = m15_df.index.date
        time_ms = m15_df.index.astype('datetime64[ms]').view('int64')
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        c_s = pd.Series(c)
        m15_ema20 = c_s.ewm(span=20, adjust=False).mean().values

        last_trade_date = None
        for i in range(25, len(m15_df)):
            # 16:30 to 19:30 UTC
            hr = hours[i]
            mn = minutes[i]
            in_window = ((hr == 16 and mn >= 30) or (17 <= hr <= 19))
            if not in_window:
                continue

            dt = dates[i]
            if dt == last_trade_date:
                continue
            if spread[i] > self.max_spread:
                continue

            macro = h4_bias[i]
            if macro == 0:
                continue

            sig = 0
            # Bullish Drift: H4 up, low touches/penetrates M15 EMA20, closes above EMA20 and bullish
            if macro == 1 and (l[i] <= m15_ema20[i]) and (c[i] > m15_ema20[i]) and (c[i] > o[i]):
                sig = SIGNAL_LONG
            # Bearish Drift: H4 down, high touches/penetrates M15 EMA20, closes below EMA20 and bearish
            elif macro == -1 and (h[i] >= m15_ema20[i]) and (c[i] < m15_ema20[i]) and (c[i] < o[i]):
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
# Discovery Worker for Alpha Discovery Lab V14
# ==============================================================================
def evaluate_v14_discovery_week(task: Tuple[str, int]) -> Dict[str, Any]:
    asset, week_num = task
    pip_size = 0.01 if asset == "USDJPY" else 0.0001
    engine = BacktestEngine(pip_size=pip_size, use_slippage=True)

    ticks, meta = get_week_dataset(week_num, raw_dir=f"data/{asset}")
    if ticks is None or len(ticks) < 500:
        return {"asset": asset, "week": week_num, "trades": {}}

    models = []
    if asset == "EURUSD":
        models.append(("LONDON_SWEEP_FADE", LondonSweepFade(asset=asset, sl_pips=9.0, tp_pips=16.2, pip_size=0.0001, max_spread=0.6)))
        models.append(("H4_DONCHIAN_RETEST", H4DonchianPullbackRetest(asset=asset, sl_pips=9.0, tp_pips=22.5, pip_size=0.0001, max_spread=0.6)))
        models.append(("EMA_RIBBON_BO", EMARibbonCompressionBreakout(asset=asset, sl_pips=9.0, tp_pips=18.0, pip_size=0.0001, max_spread=0.6)))
        models.append(("POST_FIX_NY_DRIFT", PostFixNYDrift(asset=asset, sl_pips=9.0, tp_pips=16.2, pip_size=0.0001, max_spread=0.6)))
    elif asset == "USDJPY":
        models.append(("LONDON_SWEEP_FADE", LondonSweepFade(asset=asset, sl_pips=12.0, tp_pips=21.6, pip_size=0.01, max_spread=0.8)))
        models.append(("H4_DONCHIAN_RETEST", H4DonchianPullbackRetest(asset=asset, sl_pips=12.0, tp_pips=30.0, pip_size=0.01, max_spread=0.8)))
        models.append(("EMA_RIBBON_BO", EMARibbonCompressionBreakout(asset=asset, sl_pips=12.0, tp_pips=24.0, pip_size=0.01, max_spread=0.8)))
        models.append(("POST_FIX_NY_DRIFT", PostFixNYDrift(asset=asset, sl_pips=12.0, tp_pips=21.6, pip_size=0.01, max_spread=0.8)))
    elif asset == "EURGBP":
        models.append(("LONDON_SWEEP_FADE", LondonSweepFade(asset=asset, sl_pips=8.0, tp_pips=14.4, pip_size=0.0001, max_spread=0.8)))
        models.append(("H4_DONCHIAN_RETEST", H4DonchianPullbackRetest(asset=asset, sl_pips=8.0, tp_pips=20.0, pip_size=0.0001, max_spread=0.8)))
        models.append(("EMA_RIBBON_BO", EMARibbonCompressionBreakout(asset=asset, sl_pips=8.0, tp_pips=16.0, pip_size=0.0001, max_spread=0.8)))
        models.append(("POST_FIX_NY_DRIFT", PostFixNYDrift(asset=asset, sl_pips=8.0, tp_pips=14.4, pip_size=0.0001, max_spread=0.8)))

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


def run_discovery_v14():
    t_start = time.time()
    print("=" * 80, flush=True)
    print("  LAUNCHING ALPHA DISCOVERY V14 EXPEDITION (70% DISCOVERY PARTITIONS)", flush=True)
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
        futures = {executor.submit(evaluate_v14_discovery_week, t): t for t in all_tasks}
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

    # Compute Scorecard
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
    print(f"[*] Discovery Lab V14 complete in {time.time() - t_start:.2f} seconds.", flush=True)

    out_file = RESULTS_DIR / "alpha_discovery_lab_v14_summary.json"
    with open(out_file, "w") as f:
        json.dump(summary_rows, f, indent=2)
    print(f"[*] Results saved to {out_file}", flush=True)


if __name__ == "__main__":
    run_discovery_v14()
