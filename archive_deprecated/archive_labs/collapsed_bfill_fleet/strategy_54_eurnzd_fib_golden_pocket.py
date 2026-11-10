"""
Strategy Slot 54: EURNZD Fibonacci Golden Pocket Elite
=============================================================================
Asset: EURNZD
Timeframe: M15 execution, H1 rolling swing levels, H4 EMA20 trend bias
Architecture: Deep Fibonacci Retracement (61.8% to 78.6% Golden Pocket)
Strict Zero-Lookahead:
  - H4 EMA20 shifted by 1 H4 candle before reindexing
  - Fibonacci levels explicitly shifted by 1 H1 candle before reindexing
  - Reindex strictly causal (method='ffill')
  - Execution at next tick on or after candle close
Friction: 0.5p commission + 0.2p slippage + dynamic floating spread
Hurdles: Standalone In-Sample >= 12.0% profit, Positive Holdout Persistence
"""

import numpy as np
import pandas as pd
from typing import Tuple

def _build_bars_internal(ticks: np.ndarray, timeframe_min: int = 15, eff_pip_size: float = 0.0001) -> pd.DataFrame:
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
    def __init__(self, name: str = "", symbol: str = "EURNZD", sl_pips: float = 45.0, tp_pips: float = 105.0, pip_size: float = 0.0001, spread_limit: float = 4.5, **kwargs):
        self.name = name
        self.symbol = symbol
        self.asset = symbol
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.pip_size = float(pip_size)
        self.spread_limit = float(spread_limit)

class Strategy_54_eurnzd_fib_golden_pocket(BaseStrategy):
    """
    EURNZD Fibonacci Golden Pocket Elite (Slot 54).
    """
    def __init__(
        self,
        asset: str = "EURNZD",
        sl_pips: float = 45.0,
        tp_pips: float = 105.0,
        lookback_h1: int = 72,
        min_wick: float = 0.35,
        max_spread: float = 4.5,
        pip_size: float = 0.0001
    ):
        super().__init__(
            name="Strategy_54_eurnzd_fib_golden_pocket",
            symbol=asset,
            sl_pips=sl_pips,
            tp_pips=tp_pips,
            pip_size=pip_size,
            spread_limit=max_spread
        )
        self.lookback_h1 = lookback_h1
        self.min_wick = min_wick
        self.max_spread = max_spread

    def generate_signals(self, ticks: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = len(ticks)
        signals = np.zeros(n, dtype=np.int8)
        sl_arr = np.zeros(n, dtype=np.float64)
        tp_arr = np.zeros(n, dtype=np.float64)

        if n < 500:
            return signals, sl_arr, tp_arr

        m15 = build_time_bars(ticks, timeframe_min=15, eff_pip_size=self.pip_size)
        if len(m15) < 30:
            return signals, sl_arr, tp_arr

        h1 = m15.resample('1h').agg({'high': 'max', 'low': 'min', 'close': 'last', 'open': 'first'}).dropna()
        h4 = m15.resample('4h').agg({'high': 'max', 'low': 'min', 'close': 'last', 'open': 'first'}).dropna()

        if len(h1) < self.lookback_h1 + 5 or len(h4) < 15:
            return signals, sl_arr, tp_arr

        h4_ema20 = h4['close'].ewm(span=20, adjust=False).mean()
        h4_macro = np.where(h4['close'] > h4_ema20, 1, -1)

        roll_h = h1['high'].rolling(self.lookback_h1).max()
        roll_l = h1['low'].rolling(self.lookback_h1).min()
        swing_range = roll_h - roll_l

        fib618_long = (roll_h - 0.618 * swing_range).shift(1)
        fib786_long = (roll_h - 0.786 * swing_range).shift(1)
        fib618_short = (roll_l + 0.618 * swing_range).shift(1)
        fib786_short = (roll_l + 0.786 * swing_range).shift(1)

        m15_fib618_l = fib618_long.reindex(m15.index, method='ffill').values
        m15_fib786_l = fib786_long.reindex(m15.index, method='ffill').values
        m15_fib618_s = fib618_short.reindex(m15.index, method='ffill').values
        m15_fib786_s = fib786_short.reindex(m15.index, method='ffill').values
        m15_macro = pd.Series(h4_macro, index=h4.index).shift(1).reindex(m15.index, method='ffill').fillna(0).values

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

        last_trade_date = None

        for i in range(15, len(m15)):
            hr = hours[i]
            if not (6 <= hr <= 18):
                continue
            if spread[i] > self.max_spread:
                continue

            dt = dates[i]
            if dt == last_trade_date:
                continue

            bar_rng = (h[i] - l[i]) / self.pip_size
            if bar_rng < 3.0:
                continue

            m = m15_macro[i]
            sig = 0

            if m > 0:
                f618 = m15_fib618_l[i]
                f786 = m15_fib786_l[i]
                if np.isnan(f618) or np.isnan(f786):
                    continue
                if l[i] <= f618 and l[i] >= f786 - 3.0 * self.pip_size:
                    lower_wick = (min(o[i], c[i]) - l[i]) / self.pip_size
                    if (lower_wick / bar_rng) >= self.min_wick and c[i] > o[i]:
                        sig = SIGNAL_LONG

            elif m < 0:
                f618 = m15_fib618_s[i]
                f786 = m15_fib786_s[i]
                if np.isnan(f618) or np.isnan(f786):
                    continue
                if h[i] >= f618 and h[i] <= f786 + 3.0 * self.pip_size:
                    upper_wick = (h[i] - max(o[i], c[i])) / self.pip_size
                    if (upper_wick / bar_rng) >= self.min_wick and c[i] < o[i]:
                        sig = SIGNAL_SHORT

            if sig != 0:
                exec_t = time_ms[i] + dur_ms
                idx = int(np.searchsorted(tick_ts, exec_t))
                if idx < n and signals[idx] == 0:
                    signals[idx] = sig
                    sl_arr[idx] = self.sl_pips
                    tp_arr[idx] = self.tp_pips
                    last_trade_date = dt

        return signals, sl_arr, tp_arr

Strategy = Strategy_54_eurnzd_fib_golden_pocket
