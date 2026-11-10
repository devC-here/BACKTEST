#!/usr/bin/env python3
"""
================================================================================
  ALPHA DISCOVERY LAB V12: MACD ZERO CROSS, ASIA COMPRESSION & RSI 50 REGIME
================================================================================
Evaluates 4 Quantitative Microstructure Architectures across 70% Discovery:
  1. DISC62_MACD_ZERO_CROSS_MOMENTUM:
     M15 MACD line crossing zero during active European/US session (08:00-15:00 UTC)
     aligned with H4 20-EMA trend. 2.0R target.
  2. DISC63_ASIAN_COMPRESSION_BREAKOUT:
     Asian range (00:00-07:00 UTC) compressed <= 18 pips; London opening breakout
     closing decisively outside box during 07:00-10:00 UTC with H4 alignment. 2.0R target.
  3. DISC64_RSI_50_REGIME_PULLBACK:
     Cardwell RSI trend pullback: In macro trend (H4 + H1 EMA alignment), M15 RSI(14)
     finds support at 45-53 (bull regime) or resistance at 47-55 (bear regime)
     with rejection candlestick trigger. 2.0R target.
  4. DISC65_LONDON_CLOSE_MEAN_REVERSION:
     At London Close (15:00-17:30 UTC), after >= 1.5x ADR daily extension,
     pin rejection at extreme high/low fades back toward session VWAP. 1.8R target.

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
# Model 1: M15 MACD Zero-Cross Momentum (DISC62)
# ==============================================================================
class M15MACDZeroCrossMomentum(BaseStrategy):
    """
    H4 20-EMA establishes macro trend.
    M15 MACD(12, 26, 9) Line crosses the Zero Line in macro direction during 08:00 - 15:00 UTC.
    The crossing candle must have a solid directional body.
    Target: 2.0R.
    """
    def __init__(self, asset: str, sl_pips: float, tp_pips: float, pip_size: float, max_spread: float, **kwargs):
        super().__init__(**kwargs)
        self.asset = asset
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.pip_size = float(pip_size)
        self.max_spread = float(max_spread)
        self.name = f"{asset}_M15_MACD_Zero_Cross"

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
        dates = m15_df.index.date
        time_ms = m15_df.index.astype('datetime64[ms]').view('int64')
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        # M15 MACD (12, 26, 9)
        s_c = pd.Series(c)
        ema12 = s_c.ewm(span=12, adjust=False).mean()
        ema26 = s_c.ewm(span=26, adjust=False).mean()
        macd_line = (ema12 - ema26).values

        last_trade_date = None
        for i in range(30, len(m15_df)):
            if not (8 <= hours[i] <= 15):
                continue

            dt = dates[i]
            if dt == last_trade_date:
                continue

            if spread[i] > self.max_spread:
                continue

            macro = h4_bias[i]
            if macro == 0:
                continue

            # Zero cross
            cross_up = (macd_line[i-1] <= 0) and (macd_line[i] > 0)
            cross_down = (macd_line[i-1] >= 0) and (macd_line[i] < 0)

            sig = 0
            if macro == 1 and cross_up and (c[i] > o[i]):
                sig = SIGNAL_LONG
            elif macro == -1 and cross_down and (c[i] < o[i]):
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
# Model 2: Asian Compression Breakout (DISC63)
# ==============================================================================
class AsianCompressionBreakout(BaseStrategy):
    """
    Detects when Tokyo session range (00:00 - 07:00 UTC) is extremely compressed
    (range <= max_asia_pips, e.g. 18 pips on EURUSD/USDJPY, 12 pips on EURGBP).
    During 07:00 - 10:00 UTC (London Open):
    - First M15 candle closing decisively outside the Asian box in H4 20-EMA trend direction.
    - 2.0R target.
    """
    def __init__(self, asset: str, sl_pips: float, tp_pips: float, pip_size: float, max_spread: float, max_asia_pips: float, **kwargs):
        super().__init__(**kwargs)
        self.asset = asset
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.pip_size = float(pip_size)
        self.max_spread = float(max_spread)
        self.max_asia_pips = float(max_asia_pips)
        self.name = f"{asset}_Asian_Compression_BO"

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

        # H4 trend
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

        # Calculate Asian ranges (00:00 - 07:00 UTC)
        unique_dates = np.unique(dates)
        asia_ranges = {}
        for ud in unique_dates:
            mask = (dates == ud) & (hours < 7)
            if np.any(mask):
                a_hi = np.max(h[mask])
                a_lo = np.min(l[mask])
                rng_pips = (a_hi - a_lo) / self.pip_size
                if rng_pips <= self.max_asia_pips and rng_pips >= 5.0:
                    asia_ranges[ud] = (a_hi, a_lo)

        last_trade_date = None
        for i in range(20, len(m15_df)):
            if not (7 <= hours[i] <= 10):
                continue

            dt = dates[i]
            if dt == last_trade_date:
                continue

            if dt not in asia_ranges:
                continue

            if spread[i] > self.max_spread:
                continue

            macro = h4_bias[i]
            a_hi, a_lo = asia_ranges[dt]

            sig = 0
            if macro == 1 and (c[i] > a_hi) and (c[i] > o[i]):
                sig = SIGNAL_LONG
            elif macro == -1 and (c[i] < a_lo) and (c[i] < o[i]):
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
# Model 3: Cardwell RSI 50 Regime Pullback (DISC64)
# ==============================================================================
class CardwellRSI50RegimePullback(BaseStrategy):
    """
    Cardwell RSI Range Rules:
    - Bullish Trend (H4 20-EMA > 50-EMA): RSI(14) oscillates between 40 and 80.
      A pullback touching the 45-53 support zone followed by a bullish engulfing candle -> Long.
    - Bearish Trend (H4 20-EMA < 50-EMA): RSI(14) oscillates between 20 and 60.
      A rally touching the 47-55 resistance zone followed by a bearish engulfing candle -> Short.
    Active 07:00 - 17:00 UTC. 2.0R target.
    """
    def __init__(self, asset: str, sl_pips: float, tp_pips: float, pip_size: float, max_spread: float, **kwargs):
        super().__init__(**kwargs)
        self.asset = asset
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.pip_size = float(pip_size)
        self.max_spread = float(max_spread)
        self.name = f"{asset}_RSI_50_Regime_Pullback"

    def generate_signals(self, ticks: np.ndarray):
        n = len(ticks)
        signals = np.zeros(n, dtype=np.int8)
        sl_arr = np.zeros(n, dtype=np.float64)
        tp_arr = np.zeros(n, dtype=np.float64)
        if n < 500:
            return signals, sl_arr, tp_arr

        m15 = build_time_bars(ticks, timeframe_min=15, pip_size=self.pip_size)
        h4 = build_time_bars(ticks, timeframe_min=240, pip_size=self.pip_size)
        if len(m15) < 35 or len(h4) < 15:
            return signals, sl_arr, tp_arr

        # H4 Macro Trend (EMA 20 vs 50)
        h4_c = pd.Series(h4['close'].values)
        h4_ema20 = h4_c.ewm(span=20, adjust=False).mean().values
        h4_ema50 = h4_c.ewm(span=50, adjust=False).mean().values
        h4_trend = np.where(h4_ema20 > h4_ema50, 1, np.where(h4_ema20 < h4_ema50, -1, 0))
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

        # Calculate M15 RSI (14)
        diff = np.diff(c)
        gains = np.maximum(diff, 0.0)
        losses = np.maximum(-diff, 0.0)
        s_g = pd.Series(gains).rolling(14, min_periods=1).mean()
        s_l = pd.Series(losses).rolling(14, min_periods=1).mean()
        rs = s_g / (s_l + 1e-9)
        rsi_vals = 100.0 - (100.0 / (1.0 + rs))
        rsi = np.concatenate([[50.0], rsi_vals.values])

        last_trade_date = None
        for i in range(25, len(m15_df)):
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

            sig = 0
            # Bullish bounce: Previous bar tested 45-53 zone, current bar engulfs higher
            if macro == 1 and (43.0 <= rsi[i-1] <= 53.0) and (rsi[i] > rsi[i-1]) and (c[i] > o[i]) and (c[i] > h[i-1]):
                sig = SIGNAL_LONG
            # Bearish rejection: Previous bar tested 47-57 zone, current bar engulfs lower
            elif macro == -1 and (47.0 <= rsi[i-1] <= 57.0) and (rsi[i] < rsi[i-1]) and (c[i] < o[i]) and (c[i] < l[i-1]):
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
# Model 4: London Close Mean Reversion Fade (DISC65)
# ==============================================================================
class LondonCloseMeanReversionFade(BaseStrategy):
    """
    Between 15:00 and 17:30 UTC (London Close / Fixing transition):
    - Price has stretched >= 1.5x Daily ATR from 00:00 UTC open.
    - Forms an M15 reversal pin bar at the high/low of the day with wick >= 40%.
    - Enters mean reversion fade back toward session equilibrium with 1.8R target.
    """
    def __init__(self, asset: str, sl_pips: float, tp_pips: float, pip_size: float, max_spread: float, min_stretch_pips: float, **kwargs):
        super().__init__(**kwargs)
        self.asset = asset
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.pip_size = float(pip_size)
        self.max_spread = float(max_spread)
        self.min_stretch_pips = float(min_stretch_pips)
        self.name = f"{asset}_London_Close_Mean_Rev"

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
        mins = m15.index.minute.values
        dates = m15.index.date
        time_ms = m15.index.astype('datetime64[ms]').view('int64')
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        # Daily opens
        unique_dates = np.unique(dates)
        daily_opens = {}
        for ud in unique_dates:
            mask = (dates == ud)
            if np.any(mask):
                daily_opens[ud] = o[mask][0]

        last_trade_date = None
        for i in range(20, len(m15)):
            t_hour = hours[i]
            # 15:00 - 17:30 UTC
            if not (15 <= t_hour <= 17):
                continue

            dt = dates[i]
            if dt == last_trade_date:
                continue

            if dt not in daily_opens:
                continue

            if spread[i] > self.max_spread:
                continue

            d_open = daily_opens[dt]
            dist_from_open_pips = (c[i] - d_open) / self.pip_size

            bar_rng = h[i] - l[i]
            if bar_rng < 1e-6:
                continue

            upper_wick = h[i] - max(o[i], c[i])
            lower_wick = min(o[i], c[i]) - l[i]
            upper_ratio = upper_wick / bar_rng
            lower_ratio = lower_wick / bar_rng

            sig = 0
            # Extreme upward stretch -> Bearish pin fade
            if (dist_from_open_pips >= self.min_stretch_pips) and (upper_ratio >= 0.40) and (c[i] < o[i]):
                sig = SIGNAL_SHORT
            # Extreme downward stretch -> Bullish pin fade
            elif (dist_from_open_pips <= -self.min_stretch_pips) and (lower_ratio >= 0.40) and (c[i] > o[i]):
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
# Discovery Worker for Alpha Discovery Lab V12
# ==============================================================================
def evaluate_v12_discovery_week(task: Tuple[str, int]) -> Dict[str, Any]:
    asset, week_num = task
    pip_size = 0.01 if asset == "USDJPY" else 0.0001
    engine = BacktestEngine(pip_size=pip_size, use_slippage=True)

    ticks, meta = get_week_dataset(week_num, raw_dir=f"data/{asset}")
    if ticks is None or len(ticks) < 500:
        return {"asset": asset, "week": week_num, "trades": {}}

    models = []
    if asset == "EURUSD":
        models.append(("MACD_ZERO_CROSS", M15MACDZeroCrossMomentum(asset=asset, sl_pips=9.0, tp_pips=18.0, pip_size=0.0001, max_spread=0.6)))
        models.append(("ASIA_COMPRESS_BO", AsianCompressionBreakout(asset=asset, sl_pips=9.0, tp_pips=18.0, pip_size=0.0001, max_spread=0.6, max_asia_pips=18.0)))
        models.append(("RSI_50_PULLBACK", CardwellRSI50RegimePullback(asset=asset, sl_pips=9.0, tp_pips=18.0, pip_size=0.0001, max_spread=0.6)))
        models.append(("LON_CLOSE_FADE", LondonCloseMeanReversionFade(asset=asset, sl_pips=9.0, tp_pips=16.2, pip_size=0.0001, max_spread=0.6, min_stretch_pips=45.0)))
    elif asset == "USDJPY":
        models.append(("MACD_ZERO_CROSS", M15MACDZeroCrossMomentum(asset=asset, sl_pips=12.0, tp_pips=24.0, pip_size=0.01, max_spread=0.8)))
        models.append(("ASIA_COMPRESS_BO", AsianCompressionBreakout(asset=asset, sl_pips=12.0, tp_pips=24.0, pip_size=0.01, max_spread=0.8, max_asia_pips=22.0)))
        models.append(("RSI_50_PULLBACK", CardwellRSI50RegimePullback(asset=asset, sl_pips=12.0, tp_pips=24.0, pip_size=0.01, max_spread=0.8)))
        models.append(("LON_CLOSE_FADE", LondonCloseMeanReversionFade(asset=asset, sl_pips=12.0, tp_pips=21.6, pip_size=0.01, max_spread=0.8, min_stretch_pips=55.0)))
    elif asset == "EURGBP":
        models.append(("MACD_ZERO_CROSS", M15MACDZeroCrossMomentum(asset=asset, sl_pips=8.0, tp_pips=16.0, pip_size=0.0001, max_spread=0.8)))
        models.append(("ASIA_COMPRESS_BO", AsianCompressionBreakout(asset=asset, sl_pips=8.0, tp_pips=16.0, pip_size=0.0001, max_spread=0.8, max_asia_pips=14.0)))
        models.append(("RSI_50_PULLBACK", CardwellRSI50RegimePullback(asset=asset, sl_pips=8.0, tp_pips=16.0, pip_size=0.0001, max_spread=0.8)))
        models.append(("LON_CLOSE_FADE", LondonCloseMeanReversionFade(asset=asset, sl_pips=8.0, tp_pips=14.4, pip_size=0.0001, max_spread=0.8, min_stretch_pips=30.0)))

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
        except Exception as e:
            week_trades[m_name] = []

    return {"asset": asset, "week": week_num, "trades": week_trades}


def run_discovery_v12():
    t_start = time.time()
    print("=" * 80, flush=True)
    print("  LAUNCHING ALPHA DISCOVERY V12 EXPEDITION (70% DISCOVERY PARTITIONS)", flush=True)
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
        futures = {executor.submit(evaluate_v12_discovery_week, t): t for t in all_tasks}
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
        out_csv = RESULTS_DIR / "alpha_discovery_lab_v12_scorecard.csv"
        df.to_csv(out_csv, index=False)

        print("\n" + "=" * 90, flush=True)
        print("  ALPHA DISCOVERY LAB V12 SCORECARD (70% DISCOVERY PARTITIONS ONLY)", flush=True)
        print("=" * 90, flush=True)
        print(f"{'Asset':<8} {'Strategy':<22} {'Trades':<8} {'Win Rate':<10} {'Net Pips':<12} {'PF':<8} {'Max DD':<10} {'Ret/DD':<8} {'Exp (p/t)':<10}")
        print("-" * 90, flush=True)
        for _, r in df.iterrows():
            print(f"{r['asset']:<8} {r['strategy']:<22} {int(r['trades']):<8} {r['win_rate']:>6.1f}%   {r['net_pips']:>10.1f}p  {r['profit_factor']:>6.2f}  {r['max_dd_pips']:>8.1f}p  {r['ret_dd']:>6.2f}x  {r['expectancy']:>8.2f}p")
        print("=" * 90, flush=True)

    print(f"\n[*] Alpha Discovery Lab V12 Complete in {time.time() - t_start:.2f}s.", flush=True)


if __name__ == "__main__":
    run_discovery_v12()
