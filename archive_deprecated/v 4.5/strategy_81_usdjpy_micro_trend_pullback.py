#!/usr/bin/env python3
"""
Strategy Slot 81: Strategy_81_usdjpy_micro_trend_pullback (USDJPY) [DUAL-CERTIFIED CHAMPION]
=============================================================================
Symbol: USDJPY | Pip Size: 0.01
Architecture: Micro-Trend Pullback Continuation (M15 EMA20 Dip-Buy in H1 Trend)
Directional Alignment: Strictly causal H1 EMA12 > EMA36 trend filter (.shift(1))
Execution Window: 07:00 to 18:00 UTC (Next-tick execution after M15 bar close)
SL: 14.0 pips | TP: 28.0 pips | RR: 1:2.00
Min Body Ratio: 0.40 | Max Spread: 1.5 pips
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
    def __init__(self, name: str = "", symbol: str = "USDJPY", sl_pips: float = 14.0, tp_pips: float = 28.0, pip_size: float = 0.01, spread_limit: float = 1.5, **kwargs):
        self.name = name
        self.symbol = symbol
        self.asset = symbol
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.pip_size = float(pip_size)
        self.spread_limit = float(spread_limit)

class Strategy_81_usdjpy_micro_trend_pullback(BaseStrategy):
    def __init__(
        self,
        asset: str = "USDJPY",
        ema_fast: int = 12,
        ema_slow: int = 36,
        sl_pips: float = 14.0,
        tp_pips: float = 28.0,
        min_body: float = 0.4,
        max_spread: float = 1.5,
        pip_size: float = 0.01,
        **kwargs
    ):
        super().__init__(
            name="Strategy_81_usdjpy_micro_trend_pullback",
            symbol=asset,
            sl_pips=sl_pips,
            tp_pips=tp_pips,
            pip_size=pip_size,
            spread_limit=max_spread
        )
        self.ema_fast = int(ema_fast)
        self.ema_slow = int(ema_slow)
        self.min_body = float(min_body)
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

        # Strict Causal H1 EMA Directional Bias
        if len(h1_base) >= self.ema_slow + 5:
            ema_f = h1_base["last"].ewm(span=self.ema_fast, adjust=False).mean()
            ema_s = h1_base["last"].ewm(span=self.ema_slow, adjust=False).mean()
            h1_bias = np.where(ema_f > ema_s, 1, -1)
            bias_m15 = pd.Series(h1_bias, index=h1_base.index).shift(1).reindex(m15.index, method="ffill").fillna(0).values
        else:
            bias_m15 = np.zeros(len(m15), dtype=np.int8)

        # Precompute M15 EMA20 (causal shifted 1)
        m15_ema20 = m15['close'].ewm(span=20, adjust=False).mean().shift(1).values

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

        last_d = None

        for i in range(20, len(m15)):
            hr = hours[i]
            if not (7 <= hr <= 18):
                continue
            d = dates[i]
            if d == last_d or spread[i] > self.max_spread:
                continue

            bar_rng = (h[i] - l[i]) / self.pip_size
            if bar_rng < 2.0:
                continue
            body_ratio = abs(c[i] - o[i]) / (h[i] - l[i] + 1e-9)

            sig = 0
            ema_val = m15_ema20[i]
            h1_b = bias_m15[i]

            # Dip-buy in Uptrend: Low tags or dips below EMA20, Close rebounds above EMA20 with strong body
            if h1_b > 0 and l[i] <= ema_val and c[i] > ema_val and c[i] > o[i] and body_ratio >= self.min_body:
                sig = SIGNAL_LONG
            # Rip-sell in Downtrend: High tags or rallies above EMA20, Close rebounds below EMA20 with strong body
            elif h1_b < 0 and h[i] >= ema_val and c[i] < ema_val and c[i] < o[i] and body_ratio >= self.min_body:
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

Strategy = Strategy_81_usdjpy_micro_trend_pullback
