#!/usr/bin/env python3
"""
================================================================================
  DEEP OPTIMIZATION: "WISDOM OF THE CROWD" HIGH-WIN-RATE MODELS (> 50% WR)
================================================================================
Testing 4 advanced quantitative variations targeting:
  - Win Rate > 50% (55% - 70%+)
  - Profit Factor > 1.35
  - Return / DD > 3.0x
  - Clean trade frequency (1 to 3 trades / week)

Candidates:
  1. WOTC_Quorum_Elite:
     H1 50-EMA Macro Trend + M15 5-Indicator Voting Quorum (Ribbon, RSI, MACD, Stoch, Rejection Wick).
     Requires at least 4 out of 5 micro-crowds to agree with macro trend.

  2. WOTC_Structural_Bounce_Elite:
     Enhanced H1 Donchian 12-bar Support/Resistance test with M15 45% rejection wick.
     Fine-tuned SL (9.0p) and TP (12.5p - 13.5p) to maximize WR while keeping RR >= 1.35.

  3. WOTC_HA_Ribbon_Ignition:
     H1 Macro Trend + M15 Guppy Fast Ribbon + Zero-Wick Heiken Ashi breakout candle.
     SL 8.5p, TP 11.5p (1.35R).

  4. WOTC_Multi_Confluence_Duo:
     Joint Consensus of RSI Regime Shift (DISC40 mechanism) + MACD Acceleration (DISC38 mechanism).
     Entering ONLY when both momentum engines confirm simultaneous acceleration.
================================================================================
"""

from __future__ import annotations
import sys
import os
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tick_engine.core.engine import BacktestEngine, TRADE_DTYPE
from tick_engine.core.dataloader import get_week_dataset
from tick_engine.strategies.base import BaseStrategy, SIGNAL_LONG, SIGNAL_SHORT
from data_engineering_suite.bars import build_time_bars

PIP_SIZE = 0.0001


# ==============================================================================
# 1. WOTC_Quorum_Elite (Macro Trend + 5 Micro Indicator Consensus)
# ==============================================================================
class WOTCQuorumEliteStrategy(BaseStrategy):
    """
    Council of Crowds:
    - Macro Crowd: H1 Close > H1 50-EMA (Bullish) or H1 Close < H1 50-EMA (Bearish).
    - 5 Micro Indicators on M15:
      1. EMA Ribbon: EMA 9 > EMA 21
      2. RSI: RSI(14) between 50 and 65 (bullish expansion) or 35 and 50 (bearish)
      3. MACD: MACD Histogram positive & expanding
      4. Stoch: %K > %D and %K between 25 and 75
      5. Candle Body: Closes in outer 40% of bar range in trend direction
    Requires >= 4 out of 5 micro-crowds to agree with the Macro Crowd!
    """
    def __init__(
        self,
        sl_pips: float = 9.5,
        tp_pips: float = 12.5,
        min_votes: int = 4,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.min_votes = int(min_votes)
        self.allowed_hours = {7, 8, 9, 10, 11, 12, 13, 14, 15, 16}
        self.name = f"WOTC_Quorum_Elite_{self.sl_pips}SL_{self.tp_pips}TP"

    def generate_signals(self, ticks: np.ndarray):
        n_ticks = len(ticks)
        signals = np.zeros(n_ticks, dtype=np.int8)
        sl_pips_arr = np.zeros(n_ticks, dtype=np.float64)
        tp_pips_arr = np.zeros(n_ticks, dtype=np.float64)

        if n_ticks < 500:
            return signals, sl_pips_arr, tp_pips_arr

        m15 = build_time_bars(ticks, timeframe_min=15)
        if len(m15) < 70:
            return signals, sl_pips_arr, tp_pips_arr

        # H1 Macro Trend
        ts = pd.to_datetime(ticks['timestamp'], unit='ms')
        mid = 0.5 * (ticks['bid'] + ticks['ask'])
        base = pd.DataFrame({'close': mid}, index=ts)
        h1 = base.resample('1h').last().dropna()
        if len(h1) < 55:
            return signals, sl_pips_arr, tp_pips_arr

        h1['ema50'] = h1['close'].ewm(span=50, adjust=False).mean()
        h1['macro'] = np.where(h1['close'] > h1['ema50'], 1, -1)
        h1_macro_m15 = h1['macro'].reindex(m15.index, method='ffill').fillna(0).values

        c = m15['close'].values
        h = m15['high'].values
        l = m15['low'].values
        o = m15['open'].values
        n_bars = len(m15)

        # 1. EMA Ribbon: 9 & 21
        ema9 = m15['close'].ewm(span=9, adjust=False).mean().values
        ema21 = m15['close'].ewm(span=21, adjust=False).mean().values
        v1_bull = (c > ema9) & (ema9 > ema21)
        v1_bear = (c < ema9) & (ema9 < ema21)

        # 2. RSI 14
        delta = pd.Series(c).diff()
        gain = delta.where(delta > 0, 0.0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
        rs = gain / (loss + 1e-9)
        rsi = (100 - (100 / (1 + rs))).values
        v2_bull = (rsi >= 50) & (rsi <= 68)
        v2_bear = (rsi <= 50) & (rsi >= 32)

        # 3. MACD (12, 26, 9)
        ema12 = pd.Series(c).ewm(span=12, adjust=False).mean()
        ema26 = pd.Series(c).ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        hist = (macd_line - signal_line).values
        v3_bull = (hist > 0) & (hist > np.roll(hist, 1))
        v3_bear = (hist < 0) & (hist < np.roll(hist, 1))

        # 4. Stochastic (14, 3)
        low14 = pd.Series(l).rolling(14).min()
        high14 = pd.Series(h).rolling(14).max()
        stoch_k = ((pd.Series(c) - low14) / (high14 - low14 + 1e-9) * 100).rolling(3).mean().values
        stoch_d = pd.Series(stoch_k).rolling(3).mean().values
        v4_bull = (stoch_k > stoch_d) & (stoch_k >= 30) & (stoch_k <= 75)
        v4_bear = (stoch_k < stoch_d) & (stoch_k <= 70) & (stoch_k >= 25)

        # 5. Candle Directional Body
        bar_range = np.maximum(h - l, 1e-6)
        v5_bull = (c > o) & ((c - l) / bar_range >= 0.60)
        v5_bear = (c < o) & ((h - c) / bar_range >= 0.60)

        # Total Votes
        votes_bull = v1_bull.astype(int) + v2_bull.astype(int) + v3_bull.astype(int) + v4_bull.astype(int) + v5_bull.astype(int)
        votes_bear = v1_bear.astype(int) + v2_bear.astype(int) + v3_bear.astype(int) + v4_bear.astype(int) + v5_bear.astype(int)

        tick_ts = ticks['timestamp']
        time_ms = m15.index.astype(np.int64) // 10**6
        dur_ms = 15 * 60 * 1000
        hours = m15.index.hour

        traded_today = False
        last_day = -1

        for i in range(50, n_bars - 1):
            cur_day = m15.index[i].dayofyear
            if cur_day != last_day:
                last_day = cur_day
                traded_today = False

            if traded_today:
                continue

            if hours[i] not in self.allowed_hours:
                continue

            exec_time = time_ms[i] + dur_ms
            t_idx = int(np.searchsorted(tick_ts, exec_time))
            if t_idx >= n_ticks:
                break

            spread = (ticks['ask'][t_idx] - ticks['bid'][t_idx]) / PIP_SIZE
            if spread > 0.60:
                continue

            # Bullish Quorum: H1 Macro Bull + >= 4 Micro Votes
            if h1_macro_m15[i] == 1 and votes_bull[i] >= self.min_votes:
                signals[t_idx] = SIGNAL_LONG
                sl_pips_arr[t_idx] = self.sl_pips
                tp_pips_arr[t_idx] = self.tp_pips
                traded_today = True

            # Bearish Quorum: H1 Macro Bear + >= 4 Micro Votes
            elif h1_macro_m15[i] == -1 and votes_bear[i] >= self.min_votes:
                signals[t_idx] = SIGNAL_SHORT
                sl_pips_arr[t_idx] = self.sl_pips
                tp_pips_arr[t_idx] = self.tp_pips
                traded_today = True

        return signals, sl_pips_arr, tp_pips_arr


# ==============================================================================
# 2. WOTC_Structural_Bounce_Elite (Optimized H1 Support/Resistance + M15 Wick)
# ==============================================================================
class WOTCStructuralBounceEliteStrategy(BaseStrategy):
    """
    DISC42 Mechanism calibrated on H1 / M15:
    - H1 20-EMA Macro Trend
    - H1 12-bar Donchian Support / Resistance
    - M15 tests the level within 5.5 pips and prints an institutional rejection wick >= 45%.
    - Parameter Sweep: sl_pips=9.0, tp_pips=13.0 (1.44R) or tp_pips=12.0 (1.33R).
    """
    def __init__(
        self,
        sl_pips: float = 9.0,
        tp_pips: float = 12.5,
        wick_pct: float = 0.45,
        max_dist: float = 5.5,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.wick_pct = float(wick_pct)
        self.max_dist = float(max_dist)
        self.allowed_hours = {7, 8, 9, 10, 11, 12, 13, 14, 15, 16}
        self.name = f"WOTC_Struct_Bounce_{self.sl_pips}SL_{self.tp_pips}TP"

    def generate_signals(self, ticks: np.ndarray):
        n_ticks = len(ticks)
        signals = np.zeros(n_ticks, dtype=np.int8)
        sl_pips_arr = np.zeros(n_ticks, dtype=np.float64)
        tp_pips_arr = np.zeros(n_ticks, dtype=np.float64)

        if n_ticks < 500:
            return signals, sl_pips_arr, tp_pips_arr

        m15 = build_time_bars(ticks, timeframe_min=15)
        if len(m15) < 40:
            return signals, sl_pips_arr, tp_pips_arr

        ts = pd.to_datetime(ticks['timestamp'], unit='ms')
        mid = 0.5 * (ticks['bid'] + ticks['ask'])
        base = pd.DataFrame({'close': mid, 'high': mid, 'low': mid, 'open': mid}, index=ts)
        h1 = base.resample('1h').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
        if len(h1) < 25:
            return signals, sl_pips_arr, tp_pips_arr

        h1['ema20'] = h1['close'].ewm(span=20, adjust=False).mean()
        h1['macro'] = np.where(h1['close'] > h1['ema20'], 1, -1)
        h1['roll_h'] = h1['high'].rolling(12).max().shift(1).bfill()
        h1['roll_l'] = h1['low'].rolling(12).min().shift(1).bfill()

        h1_macro = h1['macro'].reindex(m15.index, method='ffill').fillna(0).values
        h1_supp = h1['roll_l'].reindex(m15.index, method='ffill').values
        h1_res = h1['roll_h'].reindex(m15.index, method='ffill').values

        c = m15['close'].values
        h = m15['high'].values
        l = m15['low'].values
        o = m15['open'].values
        n_bars = len(m15)

        tick_ts = ticks['timestamp']
        time_ms = m15.index.astype(np.int64) // 10**6
        dur_ms = 15 * 60 * 1000
        hours = m15.index.hour

        traded_today = False
        last_day = -1

        for i in range(25, n_bars - 1):
            cur_day = m15.index[i].dayofyear
            if cur_day != last_day:
                last_day = cur_day
                traded_today = False

            if traded_today:
                continue

            if hours[i] not in self.allowed_hours:
                continue

            exec_time = time_ms[i] + dur_ms
            t_idx = int(np.searchsorted(tick_ts, exec_time))
            if t_idx >= n_ticks:
                break

            spread = (ticks['ask'][t_idx] - ticks['bid'][t_idx]) / PIP_SIZE
            if spread > 0.60:
                continue

            bar_range = max(h[i] - l[i], 1e-6)
            bar_pips = bar_range / PIP_SIZE
            if bar_pips < 3.2:
                continue

            m = h1_macro[i]

            # Bullish: Macro Up, tests H1 Support, lower wick >= wick_pct, closes green
            if m > 0:
                dist_supp = abs(l[i] - h1_supp[i]) / PIP_SIZE
                lower_wick = min(o[i], c[i]) - l[i]
                if dist_supp <= self.max_dist and (lower_wick / bar_range) >= self.wick_pct and c[i] > o[i]:
                    signals[t_idx] = SIGNAL_LONG
                    sl_pips_arr[t_idx] = self.sl_pips
                    tp_pips_arr[t_idx] = self.tp_pips
                    traded_today = True

            # Bearish: Macro Down, tests H1 Resistance, upper wick >= wick_pct, closes red
            elif m < 0:
                dist_res = abs(h1_res[i] - h[i]) / PIP_SIZE
                upper_wick = h[i] - max(o[i], c[i])
                if dist_res <= self.max_dist and (upper_wick / bar_range) >= self.wick_pct and c[i] < o[i]:
                    signals[t_idx] = SIGNAL_SHORT
                    sl_pips_arr[t_idx] = self.sl_pips
                    tp_pips_arr[t_idx] = self.tp_pips
                    traded_today = True

        return signals, sl_pips_arr, tp_pips_arr


# ==============================================================================
# 3. WOTC_HA_Ribbon_Ignition (High-Win-Rate Heiken Ashi Scalper)
# ==============================================================================
class WOTCHARibbonIgnitionStrategy(BaseStrategy):
    """
    Heiken Ashi Fast Ribbon Ignition:
    - Macro Filter: H1 50-EMA
    - M15 Fast Ribbon: EMA 5 > EMA 8 > EMA 13 > EMA 34
    - Pullback into EMA 13, then ignition candle with ZERO opposite wick on Heiken Ashi.
    - Sized with compact SL: 8.5p, TP: 11.5p (1.35R).
    """
    def __init__(
        self,
        sl_pips: float = 8.5,
        tp_pips: float = 11.5,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.allowed_hours = {7, 8, 9, 10, 11, 12, 13, 14, 15, 16}
        self.name = f"WOTC_HA_Ribbon_{self.sl_pips}SL_{self.tp_pips}TP"

    def generate_signals(self, ticks: np.ndarray):
        n_ticks = len(ticks)
        signals = np.zeros(n_ticks, dtype=np.int8)
        sl_pips_arr = np.zeros(n_ticks, dtype=np.float64)
        tp_pips_arr = np.zeros(n_ticks, dtype=np.float64)

        if n_ticks < 500:
            return signals, sl_pips_arr, tp_pips_arr

        m15 = build_time_bars(ticks, timeframe_min=15)
        if len(m15) < 70:
            return signals, sl_pips_arr, tp_pips_arr

        ts = pd.to_datetime(ticks['timestamp'], unit='ms')
        mid = 0.5 * (ticks['bid'] + ticks['ask'])
        base = pd.DataFrame({'close': mid}, index=ts)
        h1 = base.resample('1h').last().dropna()
        if len(h1) < 55:
            return signals, sl_pips_arr, tp_pips_arr

        h1['ema50'] = h1['close'].ewm(span=50, adjust=False).mean()
        h1['macro'] = np.where(h1['close'] > h1['ema50'], 1, -1)
        h1_macro_m15 = h1['macro'].reindex(m15.index, method='ffill').fillna(0).values

        c = m15['close'].values
        h = m15['high'].values
        l = m15['low'].values
        o = m15['open'].values
        n_bars = len(m15)

        ema5 = m15['close'].ewm(span=5, adjust=False).mean().values
        ema8 = m15['close'].ewm(span=8, adjust=False).mean().values
        ema13 = m15['close'].ewm(span=13, adjust=False).mean().values
        ema34 = m15['close'].ewm(span=34, adjust=False).mean().values

        # Heiken Ashi
        ha_close = (o + h + l + c) / 4.0
        ha_open = np.zeros(n_bars, dtype=np.float64)
        ha_open[0] = (o[0] + c[0]) / 2.0
        for b in range(1, n_bars):
            ha_open[b] = (ha_open[b-1] + ha_close[b-1]) / 2.0
        ha_high = np.maximum(h, np.maximum(ha_open, ha_close))
        ha_low = np.minimum(l, np.minimum(ha_open, ha_close))

        ribbon_bull = (ema5 > ema8) & (ema8 > ema13) & (ema13 > ema34)
        ribbon_bear = (ema5 < ema8) & (ema8 < ema13) & (ema13 < ema34)

        tick_ts = ticks['timestamp']
        time_ms = m15.index.astype(np.int64) // 10**6
        dur_ms = 15 * 60 * 1000
        hours = m15.index.hour

        traded_today = False
        last_day = -1

        for i in range(50, n_bars - 1):
            cur_day = m15.index[i].dayofyear
            if cur_day != last_day:
                last_day = cur_day
                traded_today = False

            if traded_today:
                continue

            if hours[i] not in self.allowed_hours:
                continue

            exec_time = time_ms[i] + dur_ms
            t_idx = int(np.searchsorted(tick_ts, exec_time))
            if t_idx >= n_ticks:
                break

            spread = (ticks['ask'][t_idx] - ticks['bid'][t_idx]) / PIP_SIZE
            if spread > 0.60:
                continue

            # Bullish Trigger
            if h1_macro_m15[i] == 1 and ribbon_bull[i] and (l[i-1] <= ema13[i-1] or l[i] <= ema13[i]):
                if ha_close[i] > ha_open[i] and (ha_open[i] - ha_low[i]) <= (0.5 * PIP_SIZE):
                    signals[t_idx] = SIGNAL_LONG
                    sl_pips_arr[t_idx] = self.sl_pips
                    tp_pips_arr[t_idx] = self.tp_pips
                    traded_today = True

            # Bearish Trigger
            elif h1_macro_m15[i] == -1 and ribbon_bear[i] and (h[i-1] >= ema13[i-1] or h[i] >= ema13[i]):
                if ha_close[i] < ha_open[i] and (ha_high[i] - ha_open[i]) <= (0.5 * PIP_SIZE):
                    signals[t_idx] = SIGNAL_SHORT
                    sl_pips_arr[t_idx] = self.sl_pips
                    tp_pips_arr[t_idx] = self.tp_pips
                    traded_today = True

        return signals, sl_pips_arr, tp_pips_arr


# ==============================================================================
# 4. WOTC_RSI_MACD_Confluence (Dual Momentum Crowd Agreement)
# ==============================================================================
class WOTCRSIMACDConfluenceStrategy(BaseStrategy):
    """
    Dual Momentum Crowd Consensus:
    - Merges the core edge of DISC40 (RSI Bull/Bear Regime Range Shift) with
      DISC38 (MACD Momentum Acceleration).
    - When RSI enters strong bull regime (RSI > 55) at the exact bar where MACD
      histogram accelerates upward across zero in London/NY hours:
      both momentum crowds agree!
    - Risk: 9.0p SL, 13.0p TP (1.44R).
    """
    def __init__(
        self,
        sl_pips: float = 9.0,
        tp_pips: float = 13.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.allowed_hours = {7, 8, 9, 10, 11, 12, 13, 14, 15}
        self.name = f"WOTC_RSI_MACD_{self.sl_pips}SL_{self.tp_pips}TP"

    def generate_signals(self, ticks: np.ndarray):
        n_ticks = len(ticks)
        signals = np.zeros(n_ticks, dtype=np.int8)
        sl_pips_arr = np.zeros(n_ticks, dtype=np.float64)
        tp_pips_arr = np.zeros(n_ticks, dtype=np.float64)

        if n_ticks < 500:
            return signals, sl_pips_arr, tp_pips_arr

        m15 = build_time_bars(ticks, timeframe_min=15)
        if len(m15) < 60:
            return signals, sl_pips_arr, tp_pips_arr

        c = m15['close'].values
        h = m15['high'].values
        l = m15['low'].values
        o = m15['open'].values
        n_bars = len(m15)

        # H1 50-EMA Trend
        ts = pd.to_datetime(ticks['timestamp'], unit='ms')
        mid = 0.5 * (ticks['bid'] + ticks['ask'])
        base = pd.DataFrame({'close': mid}, index=ts)
        h1 = base.resample('1h').last().dropna()
        if len(h1) < 55:
            return signals, sl_pips_arr, tp_pips_arr

        h1['ema50'] = h1['close'].ewm(span=50, adjust=False).mean()
        h1['macro'] = np.where(h1['close'] > h1['ema50'], 1, -1)
        h1_macro_m15 = h1['macro'].reindex(m15.index, method='ffill').fillna(0).values

        # RSI 14
        delta = pd.Series(c).diff()
        gain = delta.where(delta > 0, 0.0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
        rs = gain / (loss + 1e-9)
        rsi = (100 - (100 / (1 + rs))).values

        # MACD (12, 26, 9)
        ema12 = pd.Series(c).ewm(span=12, adjust=False).mean()
        ema26 = pd.Series(c).ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        hist = (macd - signal).values

        tick_ts = ticks['timestamp']
        time_ms = m15.index.astype(np.int64) // 10**6
        dur_ms = 15 * 60 * 1000
        hours = m15.index.hour

        traded_today = False
        last_day = -1

        for i in range(50, n_bars - 1):
            cur_day = m15.index[i].dayofyear
            if cur_day != last_day:
                last_day = cur_day
                traded_today = False

            if traded_today:
                continue

            if hours[i] not in self.allowed_hours:
                continue

            exec_time = time_ms[i] + dur_ms
            t_idx = int(np.searchsorted(tick_ts, exec_time))
            if t_idx >= n_ticks:
                break

            spread = (ticks['ask'][t_idx] - ticks['bid'][t_idx]) / PIP_SIZE
            if spread > 0.60:
                continue

            # Bullish Confluence: Macro Bull, RSI crosses above 54, MACD Hist expands upward
            if h1_macro_m15[i] == 1 and (rsi[i] >= 54 and rsi[i-1] < 54) and (hist[i] > 0 and hist[i] > hist[i-1]):
                if c[i] > o[i]:
                    signals[t_idx] = SIGNAL_LONG
                    sl_pips_arr[t_idx] = self.sl_pips
                    tp_pips_arr[t_idx] = self.tp_pips
                    traded_today = True

            # Bearish Confluence: Macro Bear, RSI crosses below 46, MACD Hist expands downward
            elif h1_macro_m15[i] == -1 and (rsi[i] <= 46 and rsi[i-1] > 46) and (hist[i] < 0 and hist[i] < hist[i-1]):
                if c[i] < o[i]:
                    signals[t_idx] = SIGNAL_SHORT
                    sl_pips_arr[t_idx] = self.sl_pips
                    tp_pips_arr[t_idx] = self.tp_pips
                    traded_today = True

        return signals, sl_pips_arr, tp_pips_arr


# ==============================================================================
# AUDIT RUNNER ACROSS 52 WEEKS
# ==============================================================================
def _worker_eval(args):
    w, StratCls, kwargs = args
    ticks, _ = get_week_dataset(w)
    engine = BacktestEngine(pip_size=0.0001, use_slippage=True)
    strat = StratCls(**kwargs)
    trades = engine.run(ticks, strat)
    return trades


def screen_candidate(StratCls, **kwargs):
    weeks = [(w, StratCls, kwargs) for w in range(1, 53)]
    num_workers = min(8, os.cpu_count() or 4)
    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        results = list(ex.map(_worker_eval, weeks))

    trades = [t for r in results for t in r]
    n = len(trades)
    strat_inst = StratCls(**kwargs)
    if n == 0:
        return {'name': strat_inst.name, 'trades': 0, 'wr': 0, 'pips': 0, 'pf': 0, 'max_dd': 0, 'rr': 0}

    pips = sum(t['pips'] for t in trades)
    wins = [t for t in trades if t['pips'] > 0]
    losses = [t for t in trades if t['pips'] < 0]
    wr = len(wins) / n * 100.0
    gp = sum(t['pips'] for t in wins)
    gl = abs(sum(t['pips'] for t in losses))
    pf = gp / gl if gl > 0 else 99.0
    realized_rr = (gp / max(len(wins), 1)) / (gl / max(len(losses), 1)) if gl > 0 else 0.0

    equity = np.cumsum([t['pips'] for t in trades])
    peak = np.maximum.accumulate(equity)
    dd = peak - equity
    max_dd = dd.max()

    return {
        'name': strat_inst.name,
        'trades': n,
        'wr': wr,
        'pips': pips,
        'pf': pf,
        'max_dd': max_dd,
        'rr': realized_rr
    }


if __name__ == "__main__":
    t0 = time.time()
    print("=" * 80)
    print("  OPTIMIZATION: 4 REFINED WISDOM OF CROWD CANDIDATES ACROSS 52 WEEKS")
    print("  TARGET CRITERIA: WIN RATE > 50%, PROFIT FACTOR > 1.35, LOW DRAWDOWN")
    print("=" * 80)

    candidates = [
        (WOTCQuorumEliteStrategy, {'sl_pips': 9.5, 'tp_pips': 12.5, 'min_votes': 4}),
        (WOTCStructuralBounceEliteStrategy, {'sl_pips': 9.0, 'tp_pips': 12.5, 'wick_pct': 0.45, 'max_dist': 5.5}),
        (WOTCStructuralBounceEliteStrategy, {'sl_pips': 9.0, 'tp_pips': 11.5, 'wick_pct': 0.45, 'max_dist': 5.5}),
        (WOTCHARibbonIgnitionStrategy, {'sl_pips': 8.5, 'tp_pips': 11.5}),
        (WOTCRSIMACDConfluenceStrategy, {'sl_pips': 9.0, 'tp_pips': 13.0}),
    ]

    for StratCls, kwargs in candidates:
        sub_t0 = time.time()
        res = screen_candidate(StratCls, **kwargs)
        sub_dt = time.time() - sub_t0
        print(f"Strategy: {res['name']}")
        print(f"  Trades:   {res['trades']} ({res['trades']/52:.1f}/wk)")
        print(f"  Win Rate: {res['wr']:.1f}%  <--- [WR TARGET: > 50%]")
        print(f"  PF:       {res['pf']:.2f}")
        print(f"  Real RR:  {res['rr']:.2f} : 1")
        print(f"  Net Pips: {res['pips']:+.1f} pips")
        print(f"  Max DD:   {res['max_dd']:.1f} pips")
        print(f"  Duration: {sub_dt:.1f}s")
        print("-" * 60)

    print(f"Screening complete in {time.time() - t0:.1f}s total.")
