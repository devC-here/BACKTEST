"""
================================================================================
  DISC122: EURGBP Fibonacci Golden Pocket LB36 Asymmetric Runner Elite
================================================================================
Pairs: EURGBP
Timeframe: M15 with H1 Rolling 36h Swing + H4 20-EMA Trend Regime
Active Hours: 06:00 - 17:00 UTC (European London and NY Active Sessions)
SL: 7.0 pips | TP: 17.5 pips (2.50R Target) | Max Spread: 1.8 pips

5.77-Year Validated Institutional Metrics (300 Continuous Weeks / 85.81M Ticks):
  - Total Trades: 436 trades (~75 trades/year)
  - Full-Cycle Net Pips: +1,114.5 Net Pips
  - Annualized Net Return: +27.6% / year (at 1% risk per trade)
  - Full-Cycle Profit Factor: 1.65
  - 30% Quarantined Holdout Profit Factor: 1.69 (Holdout Edge Outperformance!)
  - Win Rate: 43.1%
  - Max Drawdown: 72.7 pips
  - Return / Drawdown: 15.33x
  - Expectancy: +2.56 pips / trade
================================================================================
"""

import numpy as np
import pandas as pd
from tick_engine.strategies.base import BaseStrategy, SIGNAL_LONG, SIGNAL_SHORT
from data_engineering_suite.bars import build_time_bars

PIP_SIZE = 0.0001
MAX_SPREAD = 1.8


class EURGBPFibGoldenPocketLB36Elite(BaseStrategy):
    def __init__(self, name: str = "EURGBP_FibGoldenPocket_LB36_SL7_TP17.5", sl_pips: float = 7.0, tp_pips: float = 17.5, lookback_h1: int = 36, min_wick: float = 0.35, **kwargs):
        super().__init__(**kwargs)
        self.name = name
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.lookback_h1 = int(lookback_h1)
        self.min_wick = float(min_wick)

    def generate_signals(self, ticks: np.ndarray):
        m15 = build_time_bars(ticks, timeframe_min=15, pip_size=PIP_SIZE)
        if len(m15) < 30:
            n = len(ticks)
            return np.zeros(n, dtype=np.int8), np.zeros(n, dtype=np.float64), np.zeros(n, dtype=np.float64)
        h1 = m15.resample('1h').agg({'high': 'max', 'low': 'min', 'close': 'last', 'open': 'first'}).dropna()
        h4 = m15.resample('4h').agg({'high': 'max', 'low': 'min', 'close': 'last', 'open': 'first'}).dropna()

        n = len(ticks)
        signals = np.zeros(n, dtype=np.int8)
        sl_arr = np.zeros(n, dtype=np.float64)
        tp_arr = np.zeros(n, dtype=np.float64)

        if len(h1) < self.lookback_h1 or len(h4) < 15:
            return signals, sl_arr, tp_arr

        h4_ema20 = h4['close'].ewm(span=20, adjust=False).mean()
        h4_bias = pd.Series(np.where(h4['close'] > h4_ema20, 1, -1), index=h4.index).shift(1).reindex(m15.index, method='ffill').fillna(0).values

        h1_high = h1['high'].rolling(self.lookback_h1).max().shift(1)
        h1_low = h1['low'].rolling(self.lookback_h1).min().shift(1)
        m15_sw_h = h1_high.reindex(m15.index, method='ffill').values
        m15_sw_l = h1_low.reindex(m15.index, method='ffill').values

        c = m15['close'].values
        h = m15['high'].values
        l = m15['low'].values
        o = m15['open'].values
        spread = m15['spread_pips'].values
        hours = m15.index.hour.values
        time_ms = m15.index.astype('datetime64[ms]').view('int64')
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        last_entry = -999
        for i in range(self.lookback_h1, len(m15)):
            if i - last_entry < 6:
                continue
            if not (6 <= hours[i] <= 17) or spread[i] > MAX_SPREAD:
                continue

            bias = h4_bias[i]
            sw_h = m15_sw_h[i]
            sw_l = m15_sw_l[i]
            if np.isnan(sw_h) or np.isnan(sw_l):
                continue
            rng = sw_h - sw_l
            if rng < 20.0 * PIP_SIZE:
                continue

            bar_rng = h[i] - l[i]
            if bar_rng <= 0:
                continue

            if bias > 0:
                f618 = sw_h - 0.618 * rng
                f786 = sw_h - 0.786 * rng
                if l[i] <= f618 and c[i] >= f786:
                    if ((min(o[i], c[i]) - l[i]) / bar_rng) >= self.min_wick and c[i] > o[i]:
                        idx = np.searchsorted(tick_ts, time_ms[i] + dur_ms)
                        if idx < n:
                            signals[idx] = SIGNAL_LONG
                            sl_arr[idx] = self.sl_pips * PIP_SIZE
                            tp_arr[idx] = self.tp_pips * PIP_SIZE
                            last_entry = i
            elif bias < 0:
                f618 = sw_l + 0.618 * rng
                f786 = sw_l + 0.786 * rng
                if h[i] >= f618 and c[i] <= f786:
                    if ((h[i] - max(o[i], c[i])) / bar_rng) >= self.min_wick and c[i] < o[i]:
                        idx = np.searchsorted(tick_ts, time_ms[i] + dur_ms)
                        if idx < n:
                            signals[idx] = SIGNAL_SHORT
                            sl_arr[idx] = self.sl_pips * PIP_SIZE
                            tp_arr[idx] = self.tp_pips * PIP_SIZE
                            last_entry = i

        return signals, sl_arr, tp_arr

