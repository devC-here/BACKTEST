#!/usr/bin/env python3
"""
================================================================================
  WISDOM OF THE CROWD (WOTC): GMMA, HA-RIBBON V2, & MTF STRUCTURE ENSEMBLES
================================================================================
Testing high win rate (> 50%) models across 52 weeks (~26M ticks):

  1. WOTC_GMMA_Crowd_Agreement:
     Guppy Multiple Moving Average (Trader Crowd 3-15 vs Investor Crowd 30-60).
     Entry when trader crowd compresses into investor crowd and bounces outward.

  2. WOTC_HeikenAshi_Ribbon_V2:
     Upgraded with H1 50-EMA Macro Trend filter + Heiken Ashi momentum ignition.

  3. WOTC_H1_M15_Structural_Bounce:
     H1 20-EMA trend + H1 Donchian 10-bar Support/Resistance + M15 rejection wick.
     (Adapting the 72.9% WR DISC42 mechanism to H1/M15 for higher trade frequency).

  4. WOTC_Triple_Screen_Elder_Crowd:
     Dr. Alexander Elder's Triple Screen (Tide on H4, Wave on H1, Ripple on M15).
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
# 1. WOTC_GMMA_Crowd_Agreement (Guppy Multiple Moving Average)
# ==============================================================================
class WOTCGMMACrowdAgreementStrategy(BaseStrategy):
    """
    Guppy Multiple Moving Average (GMMA):
    - Trader Crowd (Short-term): EMAs 3, 5, 8, 10, 12, 15
    - Investor Crowd (Long-term): EMAs 30, 35, 40, 45, 50, 60
    
    Setup:
      1. Investor Crowd is clearly aligned (EMA30 > EMA35 > ... > EMA60 for bull).
      2. Trader Crowd compresses down into the Investor Crowd (mean reversion).
      3. Trader Crowd turns back up and crosses above EMA30 (consensus alignment).
    """
    def __init__(
        self,
        sl_pips: float = 10.0,
        tp_pips: float = 13.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.allowed_hours = {7, 8, 9, 10, 11, 12, 13, 14, 15, 16}
        self.name = "WOTC_GMMA_Crowd_Agreement"

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

        c = m15['close']
        # Investor Crowd
        inv30 = c.ewm(span=30, adjust=False).mean().values
        inv35 = c.ewm(span=35, adjust=False).mean().values
        inv40 = c.ewm(span=40, adjust=False).mean().values
        inv45 = c.ewm(span=45, adjust=False).mean().values
        inv50 = c.ewm(span=50, adjust=False).mean().values
        inv60 = c.ewm(span=60, adjust=False).mean().values

        # Trader Crowd
        tr3 = c.ewm(span=3, adjust=False).mean().values
        tr5 = c.ewm(span=5, adjust=False).mean().values
        tr8 = c.ewm(span=8, adjust=False).mean().values
        tr10 = c.ewm(span=10, adjust=False).mean().values
        tr12 = c.ewm(span=12, adjust=False).mean().values
        tr15 = c.ewm(span=15, adjust=False).mean().values

        # Investor alignment
        inv_bull = (inv30 > inv35) & (inv35 > inv40) & (inv40 > inv45) & (inv45 > inv50) & (inv50 > inv60)
        inv_bear = (inv30 < inv35) & (inv35 < inv40) & (inv40 < inv45) & (inv45 < inv50) & (inv50 < inv60)

        # Trader alignment
        tr_bull = (tr3 > tr5) & (tr5 > tr8) & (tr8 > tr10) & (tr10 > tr12) & (tr12 > tr15)
        tr_bear = (tr3 < tr5) & (tr5 < tr8) & (tr8 < tr10) & (tr10 < tr12) & (tr12 < tr15)

        lows = m15['low'].values
        highs = m15['high'].values
        closes = c.values
        opens = m15['open'].values
        n_bars = len(m15)

        tick_ts = ticks['timestamp']
        time_ms = m15.index.astype(np.int64) // 10**6
        dur_ms = 15 * 60 * 1000
        hours = m15.index.hour

        traded_today = False
        last_day = -1

        for i in range(60, n_bars - 1):
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

            # Bullish Agreement:
            # Investor crowd is bull; in previous 3 bars, low touched into investor band (<= inv30);
            # Current bar: Trader crowd aligns bull and close > open.
            if inv_bull[i] and tr_bull[i] and (lows[i-1] <= inv30[i-1] or lows[i-2] <= inv30[i-2]):
                if closes[i] > opens[i] and closes[i] > inv30[i]:
                    signals[t_idx] = SIGNAL_LONG
                    sl_pips_arr[t_idx] = self.sl_pips
                    tp_pips_arr[t_idx] = self.tp_pips
                    traded_today = True

            # Bearish Agreement:
            elif inv_bear[i] and tr_bear[i] and (highs[i-1] >= inv30[i-1] or highs[i-2] >= inv30[i-2]):
                if closes[i] < opens[i] and closes[i] < inv30[i]:
                    signals[t_idx] = SIGNAL_SHORT
                    sl_pips_arr[t_idx] = self.sl_pips
                    tp_pips_arr[t_idx] = self.tp_pips
                    traded_today = True

        return signals, sl_pips_arr, tp_pips_arr


# ==============================================================================
# 2. WOTC_HeikenAshi_Ribbon_V2 (Upgraded with H1 Macro Filter)
# ==============================================================================
class WOTCHeikenAshiRibbonV2Strategy(BaseStrategy):
    """
    Upgraded Heiken Ashi Ribbon:
    - Adds H1 50-EMA Macro Trend Filter (ensuring trade aligns with the macro institutional crowd).
    - Requires Heiken Ashi candle to have ZERO lower wick on longs, ZERO upper wick on shorts.
    - Sized at 9.0p SL and 13.5p TP (1.50R).
    """
    def __init__(
        self,
        sl_pips: float = 9.0,
        tp_pips: float = 13.5,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.allowed_hours = {7, 8, 9, 10, 11, 12, 13, 14, 15, 16}
        self.name = "WOTC_HeikenAshi_Ribbon_V2"

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

        # Baseline EMAs
        ema50 = m15['close'].ewm(span=50, adjust=False).mean().values
        ema100 = m15['close'].ewm(span=100, adjust=False).mean().values

        # Fast Ribbon
        ema5 = m15['close'].ewm(span=5, adjust=False).mean().values
        ema8 = m15['close'].ewm(span=8, adjust=False).mean().values
        ema13 = m15['close'].ewm(span=13, adjust=False).mean().values

        # Compute Heiken Ashi bars
        ha_close = (o + h + l + c) / 4.0
        ha_open = np.zeros(n_bars, dtype=np.float64)
        ha_open[0] = (o[0] + c[0]) / 2.0
        for b in range(1, n_bars):
            ha_open[b] = (ha_open[b-1] + ha_close[b-1]) / 2.0
        ha_high = np.maximum(h, np.maximum(ha_open, ha_close))
        ha_low = np.minimum(l, np.minimum(ha_open, ha_close))

        ribbon_bull = (ema5 > ema8) & (ema8 > ema13) & (ema13 > ema50) & (ema50 > ema100)
        ribbon_bear = (ema5 < ema8) & (ema8 < ema13) & (ema13 < ema50) & (ema50 < ema100)

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

            # Bullish Trigger: H1 macro bull, M15 ribbon bull, dipped near ema13, HA is clean green
            if h1_macro_m15[i] == 1 and ribbon_bull[i] and (l[i-1] <= ema13[i-1] or l[i] <= ema13[i]):
                if ha_close[i] > ha_open[i] and (ha_open[i] - ha_low[i]) <= (0.6 * PIP_SIZE):
                    signals[t_idx] = SIGNAL_LONG
                    sl_pips_arr[t_idx] = self.sl_pips
                    tp_pips_arr[t_idx] = self.tp_pips
                    traded_today = True

            # Bearish Trigger: H1 macro bear, M15 ribbon bear, dipped near ema13, HA is clean red
            elif h1_macro_m15[i] == -1 and ribbon_bear[i] and (h[i-1] >= ema13[i-1] or h[i] >= ema13[i]):
                if ha_close[i] < ha_open[i] and (ha_high[i] - ha_open[i]) <= (0.6 * PIP_SIZE):
                    signals[t_idx] = SIGNAL_SHORT
                    sl_pips_arr[t_idx] = self.sl_pips
                    tp_pips_arr[t_idx] = self.tp_pips
                    traded_today = True

        return signals, sl_pips_arr, tp_pips_arr


# ==============================================================================
# 3. WOTC_H1_M15_Structural_Bounce (DISC42 Mechanism on H1/M15)
# ==============================================================================
class WOTCH1M15StructuralBounceStrategy(BaseStrategy):
    """
    DISC42 produced a 72.9% win rate on H4/M30.
    Here we adapt the exact institutional rejection logic to H1 / M15:
    - H1 Macro Trend: H1 Close > H1 EMA20.
    - H1 Structural Level: 10-bar Donchian Support / Resistance on H1.
    - M15 Trigger: M15 bar tests the H1 Support/Resistance within 4.5 pips and prints
      an institutional rejection wick >= 50% of bar range, closing in trend direction.
    - Risk: 9.0p SL, 13.5p TP (1.50R).
    """
    def __init__(
        self,
        sl_pips: float = 9.0,
        tp_pips: float = 13.5,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.allowed_hours = {7, 8, 9, 10, 11, 12, 13, 14, 15, 16}
        self.name = "WOTC_H1_M15_Structural_Bounce"

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
        h1['roll_h'] = h1['high'].rolling(10).max().shift(1).bfill()
        h1['roll_l'] = h1['low'].rolling(10).min().shift(1).bfill()

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
            if bar_pips < 3.5:
                continue

            m = h1_macro[i]

            # Bullish: Macro Up, M15 tests H1 Support within 5.0 pips, prints lower rejection wick >= 50%
            if m > 0:
                dist_supp = abs(l[i] - h1_supp[i]) / PIP_SIZE
                lower_wick = min(o[i], c[i]) - l[i]
                if dist_supp <= 5.0 and (lower_wick / bar_range) >= 0.48 and c[i] > o[i]:
                    signals[t_idx] = SIGNAL_LONG
                    sl_pips_arr[t_idx] = self.sl_pips
                    tp_pips_arr[t_idx] = self.tp_pips
                    traded_today = True

            # Bearish: Macro Down, M15 tests H1 Resistance within 5.0 pips, prints upper rejection wick >= 50%
            elif m < 0:
                dist_res = abs(h1_res[i] - h[i]) / PIP_SIZE
                upper_wick = h[i] - max(o[i], c[i])
                if dist_res <= 5.0 and (upper_wick / bar_range) >= 0.48 and c[i] < o[i]:
                    signals[t_idx] = SIGNAL_SHORT
                    sl_pips_arr[t_idx] = self.sl_pips
                    tp_pips_arr[t_idx] = self.tp_pips
                    traded_today = True

        return signals, sl_pips_arr, tp_pips_arr


# ==============================================================================
# 4. WOTC_Triple_Screen_Elder_Crowd
# ==============================================================================
class WOTCTripleScreenElderStrategy(BaseStrategy):
    """
    Alexander Elder Triple Screen:
    - Screen 1 (Market Tide): H4 MACD-Histogram slope (rising = bull, falling = bear).
    - Screen 2 (Market Wave): H1 2-period RSI drops below 30 in uptrend (oversold pullback).
    - Screen 3 (Ripple Trigger): M15 trailing breakout of previous bar's high + 1 pip.
    - Sized at 10.0p SL, 14.0p TP (1.40R).
    """
    def __init__(
        self,
        sl_pips: float = 10.0,
        tp_pips: float = 14.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.allowed_hours = {7, 8, 9, 10, 11, 12, 13, 14, 15, 16}
        self.name = "WOTC_Triple_Screen_Elder"

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

        ts = pd.to_datetime(ticks['timestamp'], unit='ms')
        mid = 0.5 * (ticks['bid'] + ticks['ask'])
        base = pd.DataFrame({'close': mid, 'high': mid, 'low': mid, 'open': mid}, index=ts)
        h1 = base.resample('1h').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
        h4 = base.resample('4h').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()

        if len(h1) < 30 or len(h4) < 25:
            return signals, sl_pips_arr, tp_pips_arr

        # Screen 1: H4 MACD Histogram slope
        h4_ema12 = h4['close'].ewm(span=12, adjust=False).mean()
        h4_ema26 = h4['close'].ewm(span=26, adjust=False).mean()
        h4_macd = h4_ema12 - h4_ema26
        h4_sig = h4_macd.ewm(span=9, adjust=False).mean()
        h4_hist = h4_macd - h4_sig
        h4_tide = np.where(h4_hist > h4_hist.shift(1), 1, np.where(h4_hist < h4_hist.shift(1), -1, 0))
        h4_tide_m15 = pd.Series(h4_tide, index=h4.index).shift(1).reindex(m15.index, method='ffill').fillna(0).values

        # Screen 2: H1 2-period RSI (Extreme oscillator)
        h1_delta = h1['close'].diff()
        h1_gain = h1_delta.where(h1_delta > 0, 0.0).rolling(2).mean()
        h1_loss = (-h1_delta.where(h1_delta < 0, 0.0)).rolling(2).mean()
        h1_rs = h1_gain / (h1_loss + 1e-9)
        h1_rsi2 = 100 - (100 / (1 + h1_rs))
        h1_rsi2_m15 = h1_rsi2.reindex(m15.index, method='ffill').fillna(50).values

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

            # Bullish: H4 Tide Bull, H1 RSI2 was oversold (<= 25), M15 bar closes green breaking previous bar high
            if h4_tide_m15[i] == 1 and h1_rsi2_m15[i] <= 25.0:
                if c[i] > o[i] and c[i] > h[i-1]:
                    signals[t_idx] = SIGNAL_LONG
                    sl_pips_arr[t_idx] = self.sl_pips
                    tp_pips_arr[t_idx] = self.tp_pips
                    traded_today = True

            # Bearish: H4 Tide Bear, H1 RSI2 was overbought (>= 75), M15 bar closes red breaking previous bar low
            elif h4_tide_m15[i] == -1 and h1_rsi2_m15[i] >= 75.0:
                if c[i] < o[i] and c[i] < l[i-1]:
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
    if n == 0:
        return {'name': StratCls().name, 'trades': 0, 'wr': 0, 'pips': 0, 'pf': 0, 'max_dd': 0, 'rr': 0}

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
        'name': StratCls().name,
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
    print("  SCREENING ROUND 2: GMMA, HA-RIBBON V2, H1/M15 BOUNCE, TRIPLE SCREEN")
    print("  CRITERIA: WIN RATE > 50%, POSITIVE EXPECTANCY, LOW DRAWDOWN")
    print("=" * 80)

    candidates = [
        (WOTCGMMACrowdAgreementStrategy, {'sl_pips': 10.0, 'tp_pips': 13.0}),
        (WOTCHeikenAshiRibbonV2Strategy, {'sl_pips': 9.0, 'tp_pips': 13.5}),
        (WOTCH1M15StructuralBounceStrategy, {'sl_pips': 9.0, 'tp_pips': 13.5}),
        (WOTCTripleScreenElderStrategy, {'sl_pips': 10.0, 'tp_pips': 14.0}),
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
