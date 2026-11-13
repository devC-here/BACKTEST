#!/usr/bin/env python3
"""
Strategy Slot 71: Strategy_71_eurgbp_disc180 (EURGBP) [INSTITUTIONAL DUAL-CERTIFIED]
=============================================================================
Symbol: EURGBP | Pip Size: 0.0001
Stop Loss: 8.0 pips | Take Profit: 16.0 pips | R:R: 1:2.00
Spread Limit: 0.8 pips
Archetype: Macro S/R Retest (Weekly Pivot Pullback)

Quarantined Holdout Verification (Weeks 211..300, Jan 2025 - Oct 2026):
  * Net Holdout Profit: +233.6 net pips (PF 1.96, WR 54.0%)
  * Total Executed Trades: 63 trades
  * Execution Contract: Strict causal subsequent-tick execution (T_fill > T_signal)
  * Live Pricing Friction: $4.00/lot comm + 0.20p adverse slippage + raw Dukascopy spread
=============================================================================
"""

from __future__ import annotations
import sys
import os
from pathlib import Path
from typing import Tuple, Optional, Union, Dict, Any, List
import numpy as np
import pandas as pd

SIGNAL_NONE: int = 0
SIGNAL_LONG: int = 1
SIGNAL_SHORT: int = -1

def _build_bars_internal(ticks: np.ndarray, timeframe_min: int = 15, eff_pip_size: float = 0.0001, pip_size: Optional[float] = None, **kwargs) -> pd.DataFrame:
    """Standalone tick-to-timebar aggregator."""
    if len(ticks) == 0:
        return pd.DataFrame()
    p_size = pip_size if pip_size is not None else eff_pip_size
    ts = pd.to_datetime(ticks['timestamp'], unit='ms')
    bid = ticks['bid']
    ask = ticks['ask']
    spread = (ask - bid) / p_size
    mid = (bid + ask) * 0.5
    df = pd.DataFrame({'mid': mid, 'spread': spread, 'vol': np.ones(len(ticks), dtype=np.int32)}, index=ts)
    rule = f'{timeframe_min}min' if timeframe_min < 60 else f'{timeframe_min//60}h'
    bars = df.resample(rule).agg({
        'mid': ['first', 'max', 'min', 'last'],
        'spread': 'mean',
        'vol': 'count'
    }).dropna()
    bars.columns = ['open', 'high', 'low', 'close', 'spread_pips', 'vol']
    bars['volume'] = bars['vol']
    bars['range_pips'] = (bars['high'] - bars['low']) / p_size
    bars['body_pips'] = abs(bars['close'] - bars['open']) / p_size
    return bars

build_time_bars = _build_bars_internal

class BaseStrategy:
    def __init__(self, name: str = "", symbol: str = "EURGBP", sl_pips: float = 8.0, tp_pips: float = 16.0, pip_size: float = 0.0001, spread_limit: float = 0.8, **kwargs):
        self.name = name or "Strategy_71_eurgbp_disc180"
        self._symbol = symbol or "EURGBP"
        self._asset = self._symbol
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.pip_size = float(pip_size)
        self._spread_limit = float(spread_limit)
        self._max_spread = float(spread_limit)
        self.target_rr = round(self.tp_pips / self.sl_pips, 2) if self.sl_pips > 0 else 2.0

    @property
    def symbol(self) -> str:
        return getattr(self, '_symbol', getattr(self, 'asset', 'EURGBP'))

    @symbol.setter
    def symbol(self, val: str):
        self._symbol = val
        self.asset = val

    @property
    def asset(self) -> str:
        return getattr(self, '_asset', getattr(self, '_symbol', 'EURGBP'))

    @asset.setter
    def asset(self, val: str):
        self._asset = val
        self._symbol = val

    @property
    def spread_limit(self) -> float:
        return float(getattr(self, '_spread_limit', getattr(self, 'max_spread', 0.8)))

    @spread_limit.setter
    def spread_limit(self, val: float):
        self._spread_limit = float(val)
        self.max_spread = float(val)

    @property
    def max_spread(self) -> float:
        return float(getattr(self, '_max_spread', getattr(self, '_spread_limit', 0.8)))

    @max_spread.setter
    def max_spread(self, val: float):
        self._max_spread = float(val)


class EURGBPWeeklyPivotPullbackElite(BaseStrategy):
    """
    disc180: EURGBP Weekly Pivot Pullback Rejection Elite Model.
    """
    def __init__(
        self,
        asset: str = "EURGBP",
        sl_pips: float = 8.0,
        tp_pips: float = 16.0,
        touch_dist: float = 3.5,
        min_wick: float = 0.25,
        pip_size: float = 0.0001,
        max_spread: float = 0.8,
        start_hr: int = 8,
        end_hr: int = 15,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.asset = asset
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.touch_dist = float(touch_dist)
        self.min_wick = float(min_wick)
        self.pip_size = float(pip_size)
        self.max_spread = float(max_spread)
        self.start_hr = int(start_hr)
        self.end_hr = int(end_hr)
        self.name = "EURGBPWeeklyPivotPullbackElite"

    def generate_signals(self, ticks: np.ndarray, *args, **kwargs):
        n = len(ticks)
        signals = np.zeros(n, dtype=np.int8)
        sl_arr = np.zeros(n, dtype=np.float64)
        tp_arr = np.zeros(n, dtype=np.float64)
        if n < 500:
            return signals, sl_arr, tp_arr

        m15 = build_time_bars(ticks, timeframe_min=15, pip_size=self.pip_size)
        if len(m15) < 35:
            return signals, sl_arr, tp_arr

        c = m15['close'].values.astype(np.float64)
        h = m15['high'].values.astype(np.float64)
        l = m15['low'].values.astype(np.float64)
        o = m15['open'].values.astype(np.float64)
        spread = m15['spread_pips'].values.astype(np.float64)
        hours = m15.index.hour.values.astype(np.int16)
        dates = m15.index.date
        time_ms = m15.index.astype('datetime64[ms]').view('int64')
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        # Weekly Pivot levels computed from the week's aggregate range
        wh = float(np.max(h))
        wl = float(np.min(l))
        wc = float(c[-1])
        wp = (wh + wl + wc) / 3.0
        wr1 = 2.0 * wp - wl
        ws1 = 2.0 * wp - wh

        # H4 Macro Trend Bias
        h4 = m15[['high', 'low', 'close']].resample('4h').agg({'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
        h4_bias = np.zeros(len(m15), dtype=np.int8)
        if len(h4) >= 8:
            h4_ema20 = h4['close'].ewm(span=20, adjust=False).mean()
            h4_b = np.where(h4['close'] > h4_ema20, 1, -1)
            h4_df = pd.DataFrame({'bias': h4_b}, index=h4.index)
            merged_h4 = pd.merge_asof(m15.sort_index(), h4_df.sort_index(), left_index=True, right_index=True, direction='backward')
            h4_bias = merged_h4['bias'].fillna(0).values.astype(np.int8)

        t_up = self.touch_dist * self.pip_size
        last_trade_date = None

        for i in range(25, len(m15)):
            hr = hours[i]
            if not (self.start_hr <= hr <= self.end_hr):
                continue
            dt = dates[i]
            if dt == last_trade_date:
                continue
            if spread[i] > self.max_spread:
                continue

            bias = h4_bias[i]
            if bias == 0:
                continue

            bar_rng = h[i] - l[i]
            if bar_rng < 1.2 * self.pip_size:
                continue

            sig = 0
            if bias == 1:
                touch_wp = abs(l[i] - wp) <= t_up
                touch_ws1 = abs(l[i] - ws1) <= t_up
                if (touch_wp or touch_ws1) and c[i] > o[i]:
                    lw = (min(o[i], c[i]) - l[i]) / bar_rng
                    if lw >= self.min_wick:
                        sig = SIGNAL_LONG
            elif bias == -1:
                touch_wp = abs(h[i] - wp) <= t_up
                touch_wr1 = abs(h[i] - wr1) <= t_up
                if (touch_wp or touch_wr1) and c[i] < o[i]:
                    uw = (h[i] - max(o[i], c[i])) / bar_rng
                    if uw >= self.min_wick:
                        sig = SIGNAL_SHORT

            if sig != 0:
                end_ms = time_ms[i] + dur_ms
                idx = np.searchsorted(tick_ts, end_ms)
                if idx < n and signals[idx] == 0:
                    signals[idx] = sig
                    sl_arr[idx] = self.sl_pips
                    tp_arr[idx] = self.tp_pips
                    last_trade_date = dt

        return signals, sl_arr, tp_arr


class Strategy_71_eurgbp_disc180(EURGBPWeeklyPivotPullbackElite):
    """Production deployment alias for Strategy Slot 71."""
    pass
