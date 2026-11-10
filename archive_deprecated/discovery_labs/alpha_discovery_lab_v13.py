#!/usr/bin/env python3
"""
================================================================================
  ALPHA DISCOVERY LAB V13: PIN RUNNERS, US OVERLAP, DOUBLE STRUCTURE & M30 ENGULF
================================================================================
Evaluates 4 Quantitative Microstructure Architectures across 70% Discovery:
  1. DISC66_M15_50EMA_PIN_RUNNER:
     H4 20-EMA trend alignment; M15 pullback touching 50-EMA with rejection pin
     (wick >= 50%) during 07:00-16:30 UTC. 2.2R target.
  2. DISC67_LONDON_PM_EXPANSION:
     Pre-US open consolidation (11:00-12:30 UTC); explosive breakout candle (body >= 1.3x ATR)
     during 12:30-15:30 UTC in H1 trend direction. 2.0R target.
  3. DISC68_DOUBLE_STRUCTURE_NECKLINE:
     M15 double bottom / double top within 3 pips; breakout closing beyond neckline
     with H4 trend alignment during 07:00-16:00 UTC. 2.0R target.
  4. DISC69_M30_ENGULFING_RESUMPTION:
     M30 candle engulfing previous counter-trend candle's entire range in H4 trend
     direction during 07:00-17:00 UTC. 2.0R target.

Tested across EURUSD, EURGBP, and USDJPY on 70% Discovery Partition ONLY.
================================================================================
"""

from __future__ import annotations
import sys
import os
import time
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
# Model 1: M15 50-EMA Pin Runner (DISC66)
# ==============================================================================
class M15FiftyEMAPinRunner(BaseStrategy):
    """
    H4 20-EMA establishes macro trend.
    M15 pulls back to touch 50-period EMA during 07:00 - 16:30 UTC.
    Forms a pin bar rejection (wick >= 50% pointing counter to trend).
    Target: 2.2R.
    """
    def __init__(self, asset: str, sl_pips: float, tp_pips: float, pip_size: float, max_spread: float, **kwargs):
        super().__init__(**kwargs)
        self.asset = asset
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.pip_size = float(pip_size)
        self.max_spread = float(max_spread)
        self.name = f"{asset}_M15_50EMA_Pin_Runner"

    def generate_signals(self, ticks: np.ndarray):
        n = len(ticks)
        signals = np.zeros(n, dtype=np.int8)
        sl_arr = np.zeros(n, dtype=np.float64)
        tp_arr = np.zeros(n, dtype=np.float64)
        if n < 500:
            return signals, sl_arr, tp_arr

        m15 = build_time_bars(ticks, timeframe_min=15, pip_size=self.pip_size)
        h4 = build_time_bars(ticks, timeframe_min=240, pip_size=self.pip_size)
        if len(m15) < 55 or len(h4) < 10:
            return signals, sl_arr, tp_arr

        # H4 Trend Filter
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
        mins = m15_df.index.minute.values
        dates = m15_df.index.date
        time_ms = m15_df.index.astype('datetime64[ms]').view('int64')
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        # M15 50-EMA
        s_c = pd.Series(c)
        ema50 = s_c.ewm(span=50, adjust=False).mean().values

        last_trade_date = None
        for i in range(50, len(m15_df)):
            t_hour = hours[i]
            t_min = mins[i]
            if not (7 <= t_hour < 16 or (t_hour == 16 and t_min <= 30)):
                continue

            dt = dates[i]
            if dt == last_trade_date:
                continue

            if spread[i] > self.max_spread:
                continue

            macro = h4_bias[i]
            if macro == 0:
                continue

            bar_rng = h[i] - l[i]
            if bar_rng < 1e-6:
                continue

            # Pin bar metrics
            upper_wick = h[i] - max(o[i], c[i])
            lower_wick = min(o[i], c[i]) - l[i]
            upper_ratio = upper_wick / bar_rng
            lower_ratio = lower_wick / bar_rng

            # Touch 50-EMA: low <= ema50 <= high
            touched_ema = (l[i] <= ema50[i] <= h[i])

            sig = 0
            if macro == 1 and touched_ema and (lower_ratio >= 0.45) and (c[i] > ema50[i]) and (c[i] >= o[i]):
                sig = SIGNAL_LONG
            elif macro == -1 and touched_ema and (upper_ratio >= 0.45) and (c[i] < ema50[i]) and (c[i] <= o[i]):
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
# Model 2: London PM Volatility Expansion Breakout (DISC67)
# ==============================================================================
class LondonPMVolatilityExpansion(BaseStrategy):
    """
    Pre-US Open Consolidation (11:00 - 12:30 UTC):
    Calculates range high and low.
    During 12:30 - 15:30 UTC (US session overlap momentum):
    First M15 expansion candle closing outside range in H1 20-EMA trend direction.
    Target: 2.0R.
    """
    def __init__(self, asset: str, sl_pips: float, tp_pips: float, pip_size: float, max_spread: float, **kwargs):
        super().__init__(**kwargs)
        self.asset = asset
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.pip_size = float(pip_size)
        self.max_spread = float(max_spread)
        self.name = f"{asset}_London_PM_Expansion"

    def generate_signals(self, ticks: np.ndarray):
        n = len(ticks)
        signals = np.zeros(n, dtype=np.int8)
        sl_arr = np.zeros(n, dtype=np.float64)
        tp_arr = np.zeros(n, dtype=np.float64)
        if n < 500:
            return signals, sl_arr, tp_arr

        m15 = build_time_bars(ticks, timeframe_min=15, pip_size=self.pip_size)
        h1 = build_time_bars(ticks, timeframe_min=60, pip_size=self.pip_size)
        if len(m15) < 30 or len(h1) < 15:
            return signals, sl_arr, tp_arr

        # H1 Trend Filter
        h1_c = pd.Series(h1['close'].values)
        h1_ema20 = h1_c.ewm(span=20, adjust=False).mean().values
        h1_trend = np.where(h1_c.values > h1_ema20, 1, -1)
        h1_df = pd.DataFrame({'h1_bias': h1_trend}, index=h1.index)

        m15_df = pd.merge_asof(
            m15.sort_index(),
            h1_df.sort_index(),
            left_index=True,
            right_index=True,
            direction='backward'
        )

        c = m15_df['close'].values
        h = m15_df['high'].values
        l = m15_df['low'].values
        o = m15_df['open'].values
        spread = m15_df['spread_pips'].values
        h1_bias = m15_df['h1_bias'].fillna(0).values.astype(int)
        hours = m15_df.index.hour.values
        mins = m15_df.index.minute.values
        dates = m15_df.index.date
        time_ms = m15_df.index.astype('datetime64[ms]').view('int64')
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        # Pre-US ranges (11:00 - 12:30 UTC)
        unique_dates = np.unique(dates)
        pre_us_ranges = {}
        for ud in unique_dates:
            mask = (dates == ud) & ((hours == 11) | ((hours == 12) & (mins < 30)))
            if np.any(mask):
                p_hi = np.max(h[mask])
                p_lo = np.min(l[mask])
                rng_p = (p_hi - p_lo) / self.pip_size
                if 4.0 <= rng_p <= 30.0:
                    pre_us_ranges[ud] = (p_hi, p_lo)

        last_trade_date = None
        for i in range(20, len(m15_df)):
            t_hour = hours[i]
            t_min = mins[i]
            # 12:30 - 15:30 UTC
            if not ((t_hour == 12 and t_min >= 30) or (13 <= t_hour <= 15)):
                continue

            dt = dates[i]
            if dt == last_trade_date:
                continue

            if dt not in pre_us_ranges:
                continue

            if spread[i] > self.max_spread:
                continue

            macro = h1_bias[i]
            p_hi, p_lo = pre_us_ranges[dt]

            sig = 0
            if macro == 1 and (c[i] > p_hi) and (c[i] > o[i]):
                sig = SIGNAL_LONG
            elif macro == -1 and (c[i] < p_lo) and (c[i] < o[i]):
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
# Model 3: Double Bottom/Top Neckline Breakout (DISC68)
# ==============================================================================
class DoubleStructureNecklineBreakout(BaseStrategy):
    """
    Identifies double bottoms or double tops on M15 over the last 15-40 bars.
    Neckline is the peak/trough between the two swing levels.
    When an M15 candle closes beyond the neckline in H4 trend direction -> Enter.
    Target: 2.0R.
    """
    def __init__(self, asset: str, sl_pips: float, tp_pips: float, pip_size: float, max_spread: float, **kwargs):
        super().__init__(**kwargs)
        self.asset = asset
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.pip_size = float(pip_size)
        self.max_spread = float(max_spread)
        self.name = f"{asset}_Double_Structure_BO"

    def generate_signals(self, ticks: np.ndarray):
        n = len(ticks)
        signals = np.zeros(n, dtype=np.int8)
        sl_arr = np.zeros(n, dtype=np.float64)
        tp_arr = np.zeros(n, dtype=np.float64)
        if n < 500:
            return signals, sl_arr, tp_arr

        m15 = build_time_bars(ticks, timeframe_min=15, pip_size=self.pip_size)
        h4 = build_time_bars(ticks, timeframe_min=240, pip_size=self.pip_size)
        if len(m15) < 45 or len(h4) < 10:
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
        dates = m15_df.index.date
        time_ms = m15_df.index.astype('datetime64[ms]').view('int64')
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        last_trade_date = None
        for i in range(40, len(m15_df)):
            if not (7 <= hours[i] <= 16):
                continue

            dt = dates[i]
            if dt == last_trade_date:
                continue

            if spread[i] > self.max_spread:
                continue

            macro = h4_bias[i]
            if macro == 0:
                continue

            # Lookback window for double bottom/top
            w_l = l[i-35:i-5]
            w_h = h[i-35:i-5]

            # Approximate double bottom:
            # Min of first half vs min of second half
            mid_pt = len(w_l) // 2
            min1 = np.min(w_l[:mid_pt])
            min2 = np.min(w_l[mid_pt:])
            neck_hi = np.max(w_h)

            diff_lo_pips = abs(min1 - min2) / self.pip_size
            neck_depth = (neck_hi - max(min1, min2)) / self.pip_size

            # Approximate double top:
            max1 = np.max(w_h[:mid_pt])
            max2 = np.max(w_h[mid_pt:])
            neck_lo = np.min(w_l)

            diff_hi_pips = abs(max1 - max2) / self.pip_size
            neck_height = (min(max1, max2) - neck_lo) / self.pip_size

            sig = 0
            # Double bottom breakout
            if macro == 1 and (diff_lo_pips <= 4.0) and (neck_depth >= 8.0) and (c[i-1] <= neck_hi) and (c[i] > neck_hi):
                sig = SIGNAL_LONG
            # Double top breakout
            elif macro == -1 and (diff_hi_pips <= 4.0) and (neck_height >= 8.0) and (c[i-1] >= neck_lo) and (c[i] < neck_lo):
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
# Model 4: M30 Engulfing Trend Resumption (DISC69)
# ==============================================================================
class M30EngulfingTrendResumption(BaseStrategy):
    """
    Operates on M30 bars during 07:00 - 17:00 UTC.
    H4 20-EMA establishes macro trend.
    Previous M30 bar is counter-trend (pullback).
    Current M30 bar completely engulfs previous bar's high and low with a strong close in trend direction.
    Target: 2.0R.
    """
    def __init__(self, asset: str, sl_pips: float, tp_pips: float, pip_size: float, max_spread: float, **kwargs):
        super().__init__(**kwargs)
        self.asset = asset
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.pip_size = float(pip_size)
        self.max_spread = float(max_spread)
        self.name = f"{asset}_M30_Engulfing_Resumption"

    def generate_signals(self, ticks: np.ndarray):
        n = len(ticks)
        signals = np.zeros(n, dtype=np.int8)
        sl_arr = np.zeros(n, dtype=np.float64)
        tp_arr = np.zeros(n, dtype=np.float64)
        if n < 500:
            return signals, sl_arr, tp_arr

        m30 = build_time_bars(ticks, timeframe_min=30, pip_size=self.pip_size)
        h4 = build_time_bars(ticks, timeframe_min=240, pip_size=self.pip_size)
        if len(m30) < 20 or len(h4) < 10:
            return signals, sl_arr, tp_arr

        h4_c = pd.Series(h4['close'].values)
        h4_ema20 = h4_c.ewm(span=20, adjust=False).mean().values
        h4_trend = np.where(h4_c.values > h4_ema20, 1, -1)
        h4_df = pd.DataFrame({'h4_bias': h4_trend}, index=h4.index)

        m30_df = pd.merge_asof(
            m30.sort_index(),
            h4_df.sort_index(),
            left_index=True,
            right_index=True,
            direction='backward'
        )

        c = m30_df['close'].values
        h = m30_df['high'].values
        l = m30_df['low'].values
        o = m30_df['open'].values
        spread = m30_df['spread_pips'].values
        h4_bias = m30_df['h4_bias'].fillna(0).values.astype(int)
        hours = m30_df.index.hour.values
        dates = m30_df.index.date
        time_ms = m30_df.index.astype('datetime64[ms]').view('int64')
        dur_ms = 30 * 60 * 1000
        tick_ts = ticks['timestamp']

        last_trade_date = None
        for i in range(10, len(m30_df)):
            if not (7 <= hours[i] <= 17):
                continue

            dt = dates[i]
            if dt == last_trade_date:
                continue

            if spread[i] > self.max_spread:
                continue

            macro = h4_bias[i]
            if macro == 0:
                continue

            # Bullish Engulfing: previous bar was bearish, current bar engulfs high & low, closes bullish
            is_bull_engulf = (c[i-1] < o[i-1]) and (h[i] > h[i-1]) and (l[i] <= l[i-1]) and (c[i] > h[i-1]) and (c[i] > o[i])

            # Bearish Engulfing: previous bar was bullish, current bar engulfs high & low, closes bearish
            is_bear_engulf = (c[i-1] > o[i-1]) and (l[i] < l[i-1]) and (h[i] >= h[i-1]) and (c[i] < l[i-1]) and (c[i] < o[i])

            sig = 0
            if macro == 1 and is_bull_engulf:
                sig = SIGNAL_LONG
            elif macro == -1 and is_bear_engulf:
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
# Discovery Worker for Alpha Discovery Lab V13
# ==============================================================================
def evaluate_v13_discovery_week(task: Tuple[str, int]) -> Dict[str, Any]:
    asset, week_num = task
    pip_size = 0.01 if asset == "USDJPY" else 0.0001
    engine = BacktestEngine(pip_size=pip_size, use_slippage=True)

    ticks, meta = get_week_dataset(week_num, raw_dir=f"data/{asset}")
    if ticks is None or len(ticks) < 500:
        return {"asset": asset, "week": week_num, "trades": {}}

    models = []
    if asset == "EURUSD":
        models.append(("M15_50EMA_PIN", M15FiftyEMAPinRunner(asset=asset, sl_pips=9.0, tp_pips=19.8, pip_size=0.0001, max_spread=0.6)))
        models.append(("LON_PM_EXPANSION", LondonPMVolatilityExpansion(asset=asset, sl_pips=9.0, tp_pips=18.0, pip_size=0.0001, max_spread=0.6)))
        models.append(("DOUBLE_STRUCT_BO", DoubleStructureNecklineBreakout(asset=asset, sl_pips=9.0, tp_pips=18.0, pip_size=0.0001, max_spread=0.6)))
        models.append(("M30_ENGULF_RESUME", M30EngulfingTrendResumption(asset=asset, sl_pips=10.0, tp_pips=20.0, pip_size=0.0001, max_spread=0.6)))
    elif asset == "USDJPY":
        models.append(("M15_50EMA_PIN", M15FiftyEMAPinRunner(asset=asset, sl_pips=12.0, tp_pips=26.4, pip_size=0.01, max_spread=0.8)))
        models.append(("LON_PM_EXPANSION", LondonPMVolatilityExpansion(asset=asset, sl_pips=12.0, tp_pips=24.0, pip_size=0.01, max_spread=0.8)))
        models.append(("DOUBLE_STRUCT_BO", DoubleStructureNecklineBreakout(asset=asset, sl_pips=12.0, tp_pips=24.0, pip_size=0.01, max_spread=0.8)))
        models.append(("M30_ENGULF_RESUME", M30EngulfingTrendResumption(asset=asset, sl_pips=14.0, tp_pips=28.0, pip_size=0.01, max_spread=0.8)))
    elif asset == "EURGBP":
        models.append(("M15_50EMA_PIN", M15FiftyEMAPinRunner(asset=asset, sl_pips=8.0, tp_pips=17.6, pip_size=0.0001, max_spread=0.8)))
        models.append(("LON_PM_EXPANSION", LondonPMVolatilityExpansion(asset=asset, sl_pips=8.0, tp_pips=16.0, pip_size=0.0001, max_spread=0.8)))
        models.append(("DOUBLE_STRUCT_BO", DoubleStructureNecklineBreakout(asset=asset, sl_pips=8.0, tp_pips=16.0, pip_size=0.0001, max_spread=0.8)))
        models.append(("M30_ENGULF_RESUME", M30EngulfingTrendResumption(asset=asset, sl_pips=9.0, tp_pips=18.0, pip_size=0.0001, max_spread=0.8)))

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


def run_discovery_v13():
    t_start = time.time()
    print("=" * 80, flush=True)
    print("  LAUNCHING ALPHA DISCOVERY V13 EXPEDITION (70% DISCOVERY PARTITIONS)", flush=True)
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
        futures = {executor.submit(evaluate_v13_discovery_week, t): t for t in all_tasks}
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
    rows = []
    for (asset, m_name), pips in results_by_key.items():
        n = len(pips)
        if n == 0:
            continue
        p_arr = np.array(pips)
        net_pips = float(np.sum(p_arr))
        wins = p_arr[p_arr > 0]
        losses = -p_arr[p_arr < 0]
        wr = float(len(wins) / n * 100.0)
        sum_w = float(np.sum(wins))
        sum_l = float(np.sum(losses))
        pf = round(sum_w / sum_l, 2) if sum_l > 0 else (99.0 if sum_w > 0 else 0.0)

        equity = np.cumsum(p_arr)
        peak = np.maximum.accumulate(equity)
        dd = peak - equity
        max_dd = float(np.max(dd)) if len(dd) > 0 else 0.0
        ret_dd = round(net_pips / max_dd, 2) if max_dd > 0 else 0.0
        expectancy = round(net_pips / n, 2)

        rows.append({
            "asset": asset,
            "strategy": m_name,
            "trades": n,
            "win_rate": wr,
            "net_pips": net_pips,
            "profit_factor": pf,
            "max_dd_pips": max_dd,
            "ret_dd": ret_dd,
            "expectancy": expectancy
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(by=["asset", "net_pips"], ascending=[True, False])
        out_csv = RESULTS_DIR / "alpha_discovery_lab_v13_scorecard.csv"
        df.to_csv(out_csv, index=False)

        print("\n" + "=" * 90, flush=True)
        print("  ALPHA DISCOVERY LAB V13 SCORECARD (70% DISCOVERY PARTITIONS ONLY)", flush=True)
        print("=" * 90, flush=True)
        print(f"{'Asset':<8} {'Strategy':<22} {'Trades':<8} {'Win Rate':<10} {'Net Pips':<12} {'PF':<8} {'Max DD':<10} {'Ret/DD':<8} {'Exp (p/t)':<10}")
        print("-" * 90, flush=True)
        for _, r in df.iterrows():
            print(f"{r['asset']:<8} {r['strategy']:<22} {int(r['trades']):<8} {r['win_rate']:>6.1f}%   {r['net_pips']:>10.1f}p  {r['profit_factor']:>6.2f}  {r['max_dd_pips']:>8.1f}p  {r['ret_dd']:>6.2f}x  {r['expectancy']:>8.2f}p")
        print("=" * 90, flush=True)

    print(f"\n[*] Alpha Discovery Lab V13 Complete in {time.time() - t_start:.2f}s.", flush=True)


if __name__ == "__main__":
    run_discovery_v13()
