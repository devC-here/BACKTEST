#!/usr/bin/env python3
"""
Strategy Slot 84: Strategy_84_usdjpy_micro_trend_breakout (USDJPY) [DUAL-CERTIFIED CHAMPION]
=============================================================================
Symbol: USDJPY | Pip Size: 0.01
Architecture: Micro-Trend Momentum Breakout (London Session Natural Expansion)
Directional Alignment: Strictly causal H1 EMA12 > EMA36 trend filter (.shift(1))
Execution Window: 07:00 to 14:00 UTC (Next-tick execution after M15 bar close)
Sweep Delta: 3.5 pips | SL: 10.0 pips | TP: 18.0 pips | RR: 1:1.80
Range Bounds: [8.0, 35.0] pips | Max Spread: 1.5 pips
=============================================================================
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Tuple

def _build_bars_internal(ticks: np.ndarray, timeframe_min: int = 15, eff_pip_size: float = 0.01) -> pd.DataFrame:
    ts = pd.to_datetime(ticks['timestamp'], unit='ms')
    bid = ticks['bid']
    ask = ticks['ask']
    spread = (ask - bid) / eff_pip_size
    mid = (bid + ask) * 0.5
    df = pd.DataFrame({'mid': mid, 'spread': spread, 'vol': np.ones(len(ticks), dtype=np.int32)}, index=ts)
    rule = f'{timeframe_min}min'
    bars = df.resample(rule).agg({
        'mid': ['first', 'max', 'min', 'last'],
        'spread': 'mean',
        'vol': 'count'
    }).dropna()
    bars.columns = ['open', 'high', 'low', 'close', 'spread_pips', 'vol']
    bars['volume'] = bars['vol']
    bars['range_pips'] = (bars['high'] - bars['low']) / eff_pip_size
    bars['body_pips'] = abs(bars['close'] - bars['open']) / eff_pip_size
    return bars

build_time_bars = _build_bars_internal

SIGNAL_NONE: int = 0
SIGNAL_LONG: int = 1
SIGNAL_SHORT: int = -1

class BaseStrategy:
    def __init__(self, name: str = "", symbol: str = "USDJPY", sl_pips: float = 10.0, tp_pips: float = 18.0, pip_size: float = 0.01, spread_limit: float = 1.5, **kwargs):
        self.name = name
        self.symbol = symbol
        self.asset = symbol
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.pip_size = float(pip_size)
        self.spread_limit = float(spread_limit)

class Strategy_84_usdjpy_micro_trend_breakout(BaseStrategy):
    def __init__(
        self,
        asset: str = "USDJPY",
        delta_pips: float = 3.5,
        sl_pips: float = 10.0,
        tp_pips: float = 18.0,
        min_range_pips: float = 8.0,
        max_range_pips: float = 35.0,
        max_spread: float = 1.5,
        pip_size: float = 0.01,
        **kwargs
    ):
        super().__init__(
            name="Strategy_84_usdjpy_micro_trend_breakout",
            symbol=asset,
            sl_pips=sl_pips,
            tp_pips=tp_pips,
            pip_size=pip_size,
            spread_limit=max_spread
        )
        self.delta_pips = float(delta_pips)
        self.min_range_pips = float(min_range_pips)
        self.max_range_pips = float(max_range_pips)
        self.max_spread = float(max_spread)

    def generate_signals(self, ticks: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n_ticks = len(ticks)
        signals = np.zeros(n_ticks, dtype=np.int8)
        sl_arr = np.zeros(n_ticks, dtype=np.float64)
        tp_arr = np.zeros(n_ticks, dtype=np.float64)

        if n_ticks < 500:
            return signals, sl_arr, tp_arr

        m15 = build_time_bars(ticks, timeframe_min=15, eff_pip_size=self.pip_size)
        if len(m15) < 30:
            return signals, sl_arr, tp_arr

        ts = pd.to_datetime(ticks['timestamp'], unit='ms')
        mid = 0.5 * (ticks['bid'] + ticks['ask'])
        mid_series = pd.Series(mid, index=ts)
        h1_base = mid_series.resample("1h").agg(["first", "max", "min", "last"]).dropna()

        # Strict Causal H1 EMA 12/36 Directional Bias
        if len(h1_base) >= 36 + 5:
            ema_f = h1_base["last"].ewm(span=12, adjust=False).mean()
            ema_s = h1_base["last"].ewm(span=36, adjust=False).mean()
            h1_bias = np.where(ema_f > ema_s, 1, -1)
            bias_m15 = pd.Series(h1_bias, index=h1_base.index).shift(1).reindex(m15.index, method="ffill").fillna(0).values
        else:
            bias_m15 = np.zeros(len(m15), dtype=np.int8)

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

        curr_d = None
        last_d = None
        sh, sl = 0.0, 999999.0
        ready = False

        for i in range(len(m15)):
            d = dates[i]
            hr = hours[i]
            if d != curr_d:
                curr_d = d
                sh, sl = 0.0, 999999.0
                ready = False

            if hr < 7:
                if h[i] > sh: sh = h[i]
                if l[i] < sl: sl = l[i]
            elif hr == 7 and not ready:
                rng = (sh - sl) / self.pip_size
                if self.min_range_pips <= rng <= self.max_range_pips:
                    ready = True

            if ready and (7 <= hr <= 14):
                if d == last_d or spread[i] > self.max_spread:
                    continue

                sig = 0
                h1_b = bias_m15[i]

                # Pro-Trend Breakout Long: Breaks above Asian High with Bullish candle + H1 Bullish
                if h[i] >= (sh + self.delta_pips * self.pip_size) and c[i] > sh and c[i] > o[i] and h1_b > 0:
                    sig = SIGNAL_LONG
                # Pro-Trend Breakout Short: Breaks below Asian Low with Bearish candle + H1 Bearish
                elif l[i] <= (sl - self.delta_pips * self.pip_size) and c[i] < sl and c[i] < o[i] and h1_b < 0:
                    sig = SIGNAL_SHORT

                if sig != 0:
                    exec_t = time_ms[i] + dur_ms
                    idx = int(np.searchsorted(tick_ts, exec_t))
                    if idx < n_ticks and signals[idx] == 0:
                        signals[idx] = sig
                        sl_arr[idx] = self.sl_pips
                        tp_arr[idx] = self.tp_pips
                        last_d = d

        return signals, sl_arr, tp_arr

Strategy = Strategy_84_usdjpy_micro_trend_breakout
