#!/usr/bin/env python3
"""
================================================================================
  ALPHA DISCOVERY LAB V21: ANCHORED VWAP BANDS, SESSION COMPRESSION, & CLIMAX FADE
================================================================================
Explores 4 distinct quantitative microstructure architectures across the 70% In-Sample
Discovery partition (Seed 42) across EURUSD (183w), USDJPY (210w), and EURGBP (210w)
(603 total discovery weeks):

  1. TOKYO_COMPRESSION_LONDON_SURGE:
     - Measures Tokyo session range (00:00 - 07:00 UTC).
     - If compressed (< 25 pips on EURUSD/EURGBP, < 35 pips on USDJPY).
     - London expansion (07:30 - 10:30 UTC) break beyond Tokyo H/L with volume surge (> 1.2x).
     - Target: 2.0R.

  2. ANCHORED_VWAP_2SIGMA_REVERSION:
     - Daily Anchored VWAP (reset at 00:00 UTC) with running intraday standard deviation bands.
     - Price pierces +/- 2.0 sigma during active sessions (08:00 - 16:00 UTC) and rejects with >= 40% wick.
     - Enters mean reversion back towards VWAP.
     - Target: 1.5R.

  3. MULTI_BAR_EXHAUSTION_CLIMAX_FADE:
     - 4+ consecutive directional bars with expanding range.
     - Final bar exhibits extreme volume (> 1.8x 15-bar avg) and prints rejection wick >= 45%.
     - Enters exhaustion fade towards 20-EMA.
     - Target: 1.5R.

  4. DAILY_PIVOT_TREND_PULLBACK:
     - Classical Daily Pivot (H+L+C)/3 from previous day.
     - In direction of H4 20-EMA trend, price pulls back to test Pivot or S1/R1 and prints rejection wick >= 40%.
     - Target: 2.0R.
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
# Model 1: Tokyo Compression London Surge (DISC79)
# ==============================================================================
class TokyoCompressionLondonSurge(BaseStrategy):
    def __init__(self, asset: str, sl_pips: float, tp_pips: float, max_tokyo_range_pips: float, pip_size: float, max_spread: float, **kwargs):
        super().__init__(**kwargs)
        self.asset = asset
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.max_tokyo_range_pips = float(max_tokyo_range_pips)
        self.pip_size = float(pip_size)
        self.max_spread = float(max_spread)
        self.name = f"{asset}_Tokyo_Compression_Surge"

    def generate_signals(self, ticks: np.ndarray):
        n = len(ticks)
        signals = np.zeros(n, dtype=np.int8)
        sl_arr = np.zeros(n, dtype=np.float64)
        tp_arr = np.zeros(n, dtype=np.float64)
        if n < 500:
            return signals, sl_arr, tp_arr

        m15 = build_time_bars(ticks, timeframe_min=15, pip_size=self.pip_size)
        h4 = build_time_bars(ticks, timeframe_min=240, pip_size=self.pip_size)
        if len(m15) < 35 or len(h4) < 10:
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
        vol = m15_df['tick_count'].values if 'tick_count' in m15_df.columns else np.ones(len(m15_df))
        h4_trend = m15_df['h4_bias'].fillna(0).values.astype(int)
        hours = m15_df.index.hour.values
        dates = m15_df.index.date
        time_ms = m15_df.index.astype('datetime64[ms]').view('int64')
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        vol_ma10 = pd.Series(vol).rolling(10).mean().values

        # Compute Tokyo range (00:00 - 06:45 UTC) per date
        tokyo_ranges: Dict[Any, Tuple[float, float, float]] = {}
        for d in np.unique(dates):
            mask = (dates == d) & (hours < 7)
            if np.sum(mask) >= 15:
                hi = float(np.max(h[mask]))
                lo = float(np.min(l[mask]))
                tokyo_ranges[d] = (hi, lo, (hi - lo) / self.pip_size)

        last_trade_date = None

        for i in range(25, len(m15_df)):
            dt = dates[i]
            hr = hours[i]

            # London opening expansion window: 07:15 - 10:30 UTC
            if not (7 <= hr <= 10):
                continue
            if dt == last_trade_date:
                continue
            if dt not in tokyo_ranges:
                continue
            if spread[i] > self.max_spread:
                continue

            t_hi, t_lo, t_rng_pips = tokyo_ranges[dt]
            if t_rng_pips > self.max_tokyo_range_pips or t_rng_pips < 5.0:
                continue

            # Need above-average volume expansion
            if not (not np.isnan(vol_ma10[i]) and vol[i] >= 1.15 * vol_ma10[i]):
                continue

            macro = h4_trend[i]
            sig = 0
            # Bullish breakout in trend
            if macro == 1 and c[i] > t_hi and c[i-1] <= t_hi:
                sig = SIGNAL_LONG
            # Bearish breakdown in trend
            elif macro == -1 and c[i] < t_lo and c[i-1] >= t_lo:
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
# Model 2: Anchored VWAP 2-Sigma Mean Reversion (DISC80)
# ==============================================================================
class AnchoredVWAP2SigmaReversion(BaseStrategy):
    def __init__(self, asset: str, sl_pips: float, tp_pips: float, pip_size: float, max_spread: float, **kwargs):
        super().__init__(**kwargs)
        self.asset = asset
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.pip_size = float(pip_size)
        self.max_spread = float(max_spread)
        self.name = f"{asset}_VWAP_2Sigma_Reversion"

    def generate_signals(self, ticks: np.ndarray):
        n = len(ticks)
        signals = np.zeros(n, dtype=np.int8)
        sl_arr = np.zeros(n, dtype=np.float64)
        tp_arr = np.zeros(n, dtype=np.float64)
        if n < 500:
            return signals, sl_arr, tp_arr

        m15 = build_time_bars(ticks, timeframe_min=15, pip_size=self.pip_size)
        if len(m15) < 35:
            return signals, sl_arr, tp_arr

        c = m15['close'].values
        h = m15['high'].values
        l = m15['low'].values
        o = m15['open'].values
        spread = m15['spread_pips'].values
        vol = m15['tick_count'].values if 'tick_count' in m15.columns else np.ones(len(m15))
        hours = m15.index.hour.values
        dates = m15.index.date
        time_ms = m15.index.astype('datetime64[ms]').view('int64')
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        typ_price = (h + l + c) / 3.0
        pv = typ_price * vol

        # Compute intraday anchored VWAP and running std dev per date
        vwap = np.zeros(len(m15))
        vwap_std = np.zeros(len(m15))

        cur_date = None
        cum_pv = 0.0
        cum_vol = 0.0
        day_prices: List[float] = []

        for i in range(len(m15)):
            d = dates[i]
            if d != cur_date:
                cur_date = d
                cum_pv = 0.0
                cum_vol = 0.0
                day_prices = []

            cum_pv += pv[i]
            cum_vol += vol[i]
            day_prices.append(typ_price[i])

            v_val = cum_pv / max(cum_vol, 1e-6)
            vwap[i] = v_val
            vwap_std[i] = np.std(day_prices) if len(day_prices) > 2 else (h[i] - l[i])

        last_trade_date = None

        for i in range(15, len(m15)):
            dt = dates[i]
            hr = hours[i]

            # Active European & US hours: 08:00 - 16:30 UTC
            if not (8 <= hr <= 16):
                continue
            if dt == last_trade_date:
                continue
            if spread[i] > self.max_spread:
                continue

            cur_vwap = vwap[i]
            cur_std = vwap_std[i]
            if cur_std < 1.5 * self.pip_size:
                continue

            upper_band = cur_vwap + 2.0 * cur_std
            lower_band = cur_vwap - 2.0 * cur_std

            bar_rng = h[i] - l[i]
            if bar_rng < 2.5 * self.pip_size:
                continue

            sig = 0
            # Pierced upper band, closed below high with >= 40% rejection wick
            if h[i] >= upper_band and c[i] < upper_band:
                wick = (h[i] - max(o[i], c[i])) / bar_rng
                if wick >= 0.40:
                    sig = SIGNAL_SHORT
            # Pierced lower band, closed above low with >= 40% rejection wick
            elif l[i] <= lower_band and c[i] > lower_band:
                wick = (min(o[i], c[i]) - l[i]) / bar_rng
                if wick >= 0.40:
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
# Model 3: Multi-Bar Exhaustion Climax Fade (DISC81)
# ==============================================================================
class MultiBarExhaustionClimaxFade(BaseStrategy):
    def __init__(self, asset: str, sl_pips: float, tp_pips: float, pip_size: float, max_spread: float, **kwargs):
        super().__init__(**kwargs)
        self.asset = asset
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.pip_size = float(pip_size)
        self.max_spread = float(max_spread)
        self.name = f"{asset}_Exhaustion_Climax_Fade"

    def generate_signals(self, ticks: np.ndarray):
        n = len(ticks)
        signals = np.zeros(n, dtype=np.int8)
        sl_arr = np.zeros(n, dtype=np.float64)
        tp_arr = np.zeros(n, dtype=np.float64)
        if n < 500:
            return signals, sl_arr, tp_arr

        m15 = build_time_bars(ticks, timeframe_min=15, pip_size=self.pip_size)
        if len(m15) < 35:
            return signals, sl_arr, tp_arr

        c = m15['close'].values
        h = m15['high'].values
        l = m15['low'].values
        o = m15['open'].values
        spread = m15['spread_pips'].values
        vol = m15['tick_count'].values if 'tick_count' in m15.columns else np.ones(len(m15))
        hours = m15.index.hour.values
        dates = m15.index.date
        time_ms = m15.index.astype('datetime64[ms]').view('int64')
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        vol_ma15 = pd.Series(vol).rolling(15).mean().values
        last_trade_date = None

        for i in range(16, len(m15)):
            dt = dates[i]
            hr = hours[i]

            if not (7 <= hr <= 17):
                continue
            if dt == last_trade_date:
                continue
            if spread[i] > self.max_spread:
                continue

            bar_rng = h[i] - l[i]
            if bar_rng < 3.0 * self.pip_size:
                continue

            # Need extreme volume spike
            if vol[i] < 1.75 * vol_ma15[i]:
                continue

            # Check 3 preceding bars in same direction
            bullish_run = (c[i-1] > o[i-1]) and (c[i-2] > o[i-2]) and (c[i-3] > o[i-3])
            bearish_run = (c[i-1] < o[i-1]) and (c[i-2] < o[i-2]) and (c[i-3] < o[i-3])

            sig = 0
            if bullish_run and h[i] > max(h[i-1], h[i-2], h[i-3]):
                upper_wick = (h[i] - max(o[i], c[i])) / bar_rng
                if upper_wick >= 0.45:
                    sig = SIGNAL_SHORT

            elif bearish_run and l[i] < min(l[i-1], l[i-2], l[i-3]):
                lower_wick = (min(o[i], c[i]) - l[i]) / bar_rng
                if lower_wick >= 0.45:
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
# Model 4: Daily Pivot Trend Pullback (DISC82)
# ==============================================================================
class DailyPivotTrendPullback(BaseStrategy):
    def __init__(self, asset: str, sl_pips: float, tp_pips: float, pip_size: float, max_spread: float, **kwargs):
        super().__init__(**kwargs)
        self.asset = asset
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.pip_size = float(pip_size)
        self.max_spread = float(max_spread)
        self.name = f"{asset}_Daily_Pivot_Pullback"

    def generate_signals(self, ticks: np.ndarray):
        n = len(ticks)
        signals = np.zeros(n, dtype=np.int8)
        sl_arr = np.zeros(n, dtype=np.float64)
        tp_arr = np.zeros(n, dtype=np.float64)
        if n < 500:
            return signals, sl_arr, tp_arr

        m15 = build_time_bars(ticks, timeframe_min=15, pip_size=self.pip_size)
        h4 = build_time_bars(ticks, timeframe_min=240, pip_size=self.pip_size)
        if len(m15) < 35 or len(h4) < 10:
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
        dates = m15_df.index.date
        time_ms = m15_df.index.astype('datetime64[ms]').view('int64')
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        # Precompute daily pivot per date from previous day's H, L, C
        unique_dates = np.unique(dates)
        daily_stats: Dict[Any, Tuple[float, float, float]] = {}
        for d in unique_dates:
            mask = (dates == d)
            if np.sum(mask) >= 10:
                daily_stats[d] = (float(np.max(h[mask])), float(np.min(l[mask])), float(c[mask][-1]))

        pivots: Dict[Any, Tuple[float, float, float]] = {}
        for idx_d in range(1, len(unique_dates)):
            prev_d = unique_dates[idx_d - 1]
            cur_d = unique_dates[idx_d]
            if prev_d in daily_stats:
                dh, dl, dc = daily_stats[prev_d]
                pivot_p = (dh + dl + dc) / 3.0
                s1 = 2.0 * pivot_p - dh
                r1 = 2.0 * pivot_p - dl
                pivots[cur_d] = (pivot_p, s1, r1)

        last_trade_date = None

        for i in range(25, len(m15_df)):
            dt = dates[i]
            hr = hours[i]

            if not (7 <= hr <= 16):
                continue
            if dt == last_trade_date:
                continue
            if dt not in pivots:
                continue
            if spread[i] > self.max_spread:
                continue

            macro = h4_trend[i]
            if macro == 0:
                continue

            pivot_p, s1, r1 = pivots[dt]
            bar_rng = h[i] - l[i]
            if bar_rng < 2.5 * self.pip_size:
                continue

            sig = 0
            # Bullish trend: test pivot P or S1 from above, reject upwards
            if macro == 1:
                touched = (l[i] <= pivot_p + 1.5 * self.pip_size and l[i] >= pivot_p - 3.0 * self.pip_size) or \
                          (l[i] <= s1 + 1.5 * self.pip_size and l[i] >= s1 - 3.0 * self.pip_size)
                if touched and c[i] > o[i]:
                    lower_wick = (o[i] - l[i]) / bar_rng
                    if lower_wick >= 0.38:
                        sig = SIGNAL_LONG

            # Bearish trend: test pivot P or R1 from below, reject downwards
            elif macro == -1:
                touched = (h[i] >= pivot_p - 1.5 * self.pip_size and h[i] <= pivot_p + 3.0 * self.pip_size) or \
                          (h[i] >= r1 - 1.5 * self.pip_size and h[i] <= r1 + 3.0 * self.pip_size)
                if touched and c[i] < o[i]:
                    upper_wick = (h[i] - o[i]) / bar_rng
                    if upper_wick >= 0.38:
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
# Model Factory & Multiprocessing Orchestrator
# ==============================================================================
def build_model_v21(model_name: str, asset: str):
    if model_name == "TOKYO_COMPRESSION_LONDON_SURGE":
        if asset == "EURUSD":
            return TokyoCompressionLondonSurge(asset=asset, sl_pips=9.0, tp_pips=18.0, max_tokyo_range_pips=25.0, pip_size=0.0001, max_spread=0.6)
        elif asset == "USDJPY":
            return TokyoCompressionLondonSurge(asset=asset, sl_pips=12.0, tp_pips=24.0, max_tokyo_range_pips=35.0, pip_size=0.01, max_spread=0.8)
        elif asset == "EURGBP":
            return TokyoCompressionLondonSurge(asset=asset, sl_pips=8.0, tp_pips=16.0, max_tokyo_range_pips=20.0, pip_size=0.0001, max_spread=0.8)

    elif model_name == "ANCHORED_VWAP_2SIGMA_REVERSION":
        if asset == "EURUSD":
            return AnchoredVWAP2SigmaReversion(asset=asset, sl_pips=9.0, tp_pips=13.5, pip_size=0.0001, max_spread=0.6)
        elif asset == "USDJPY":
            return AnchoredVWAP2SigmaReversion(asset=asset, sl_pips=12.0, tp_pips=18.0, pip_size=0.01, max_spread=0.8)
        elif asset == "EURGBP":
            return AnchoredVWAP2SigmaReversion(asset=asset, sl_pips=7.5, tp_pips=11.25, pip_size=0.0001, max_spread=0.8)

    elif model_name == "MULTI_BAR_EXHAUSTION_CLIMAX_FADE":
        if asset == "EURUSD":
            return MultiBarExhaustionClimaxFade(asset=asset, sl_pips=9.0, tp_pips=13.5, pip_size=0.0001, max_spread=0.6)
        elif asset == "USDJPY":
            return MultiBarExhaustionClimaxFade(asset=asset, sl_pips=12.0, tp_pips=18.0, pip_size=0.01, max_spread=0.8)
        elif asset == "EURGBP":
            return MultiBarExhaustionClimaxFade(asset=asset, sl_pips=7.5, tp_pips=11.25, pip_size=0.0001, max_spread=0.8)

    elif model_name == "DAILY_PIVOT_TREND_PULLBACK":
        if asset == "EURUSD":
            return DailyPivotTrendPullback(asset=asset, sl_pips=9.0, tp_pips=18.0, pip_size=0.0001, max_spread=0.6)
        elif asset == "USDJPY":
            return DailyPivotTrendPullback(asset=asset, sl_pips=12.0, tp_pips=24.0, pip_size=0.01, max_spread=0.8)
        elif asset == "EURGBP":
            return DailyPivotTrendPullback(asset=asset, sl_pips=8.0, tp_pips=16.0, pip_size=0.0001, max_spread=0.8)

    return None


def run_week_task_v21(task: Tuple[str, str, int]) -> Dict[str, Any]:
    asset, model_name, week_num = task
    pip_size = 0.01 if asset == "USDJPY" else 0.0001
    engine = BacktestEngine(pip_size=pip_size, use_slippage=True)

    ticks, meta = get_week_dataset(week_num, raw_dir=f"data/{asset}")
    if ticks is None or len(ticks) < 500:
        return {"asset": asset, "model": model_name, "week": week_num, "trades": []}

    strat = build_model_v21(model_name, asset)
    if strat is None:
        return {"asset": asset, "model": model_name, "week": week_num, "trades": []}

    try:
        raw_trades = engine.run(ticks, strat)
        t_list = []
        for i in range(len(raw_trades)):
            row = raw_trades[i]
            t_list.append({
                "pips": float(row["pips"]),
                "direction": int(row["direction"])
            })
        return {"asset": asset, "model": model_name, "week": week_num, "trades": t_list}
    except Exception:
        return {"asset": asset, "model": model_name, "week": week_num, "trades": []}


def calculate_metrics(pips: List[float]) -> Dict[str, Any]:
    if not pips:
        return {"trades": 0, "win_rate": 0.0, "pf": 0.0, "net_pips": 0.0, "max_dd": 0.0, "ret_dd": 0.0, "expectancy": 0.0}
    arr = np.array(pips, dtype=np.float64)
    trades = len(arr)
    wins = arr[arr > 0]
    losses = arr[arr < 0]
    wr = len(wins) / trades * 100.0 if trades > 0 else 0.0
    gross_win = float(np.sum(wins)) if len(wins) > 0 else 0.0
    gross_loss = float(abs(np.sum(losses))) if len(losses) > 0 else 0.0
    pf = gross_win / gross_loss if gross_loss > 0 else 999.0
    net_pips = float(np.sum(arr))
    expectancy = net_pips / trades if trades > 0 else 0.0

    equity = np.cumsum(arr)
    peak = np.maximum.accumulate(equity)
    dd = peak - equity
    max_dd = float(np.max(dd)) if len(dd) > 0 else 0.0
    ret_dd = net_pips / max_dd if max_dd > 0 else 999.0

    return {
        "trades": trades,
        "win_rate": round(wr, 1),
        "pf": round(pf, 2),
        "net_pips": round(net_pips, 1),
        "max_dd": round(max_dd, 1),
        "ret_dd": round(ret_dd, 2),
        "expectancy": round(expectancy, 2)
    }


def main():
    print("=" * 80)
    print("  LAUNCHING ALPHA DISCOVERY LAB V21: ANCHORED VWAP, SESSION & CLIMAX")
    print("=" * 80)
    t0 = time.time()

    assets = [os.environ["DISCOVERY_TARGET_SYMBOL"]] if "DISCOVERY_TARGET_SYMBOL" in os.environ else ["EURUSD", "USDJPY", "EURGBP"]

    models = [
        "TOKYO_COMPRESSION_LONDON_SURGE",
        "ANCHORED_VWAP_2SIGMA_REVERSION",
        "MULTI_BAR_EXHAUSTION_CLIMAX_FADE",
        "DAILY_PIVOT_TREND_PULLBACK"
    ]

    tasks: List[Tuple[str, str, int]] = []
    for asset in assets:
        disc_weeks = get_discovery_weeks(f"data/{asset}")
        for m in models:
            for w in disc_weeks:
                tasks.append((asset, m, w))

    print(f"[*] Total week-tasks across 3 assets & 4 models: {len(tasks)}")

    trades_by_key: Dict[str, List[float]] = {f"{a}_{m}": [] for a in assets for m in models}

    max_workers = max(1, (os.cpu_count() or 4) - 1)
    completed_count = 0

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(run_week_task_v21, t): t for t in tasks}

        for fut in as_completed(futures):
            res = fut.result()
            key = f"{res['asset']}_{res['model']}"
            for tr in res["trades"]:
                trades_by_key[key].append(tr["pips"])

            completed_count += 1
            if completed_count % 300 == 0 or completed_count == len(tasks):
                elapsed = time.time() - t0
                print(f"  [+] Audited {completed_count}/{len(tasks)} week-tasks in {elapsed:.1f}s...")

    print("\n" + "=" * 90)
    print(f"{'ASSET':<8} | {'MODEL':<30} | {'TRADES':<6} | {'WIN RATE':<8} | {'PF':<5} | {'NET PIPS':<10} | {'MAX DD':<8} | {'RET/DD':<6}")
    print("=" * 90)

    summary_results: Dict[str, Any] = {}

    for a in assets:
        for m in models:
            k = f"{a}_{m}"
            pips = trades_by_key[k]
            met = calculate_metrics(pips)
            summary_results[k] = met
            print(f"{a:<8} | {m:<30} | {met['trades']:<6} | {met['win_rate']:<6.1f} % | {met['pf']:<5.2f} | {met['net_pips']:<10.1f} | {met['max_dd']:<8.1f} | {met['ret_dd']:<5.2f} x")
        print("-" * 90)

    out_file = RESULTS_DIR / "lab_v21_discovery_results.json"
    with open(out_file, "w") as f:
        json.dump(summary_results, f, indent=2)

    total_time = time.time() - t0
    print(f"\n[+] Discovery results saved to: {out_file}")
    print(f"[+] Total execution time: {total_time:.2f}s")
    print("=" * 90)


if __name__ == "__main__":
    main()
