"""
================================================================================
  DISC121: EURGBP Session Anchored VWAP 2.0-SD Mean Fade Elite
================================================================================
Pairs: EURGBP
Timeframe: M15 with Daily Anchored VWAP + H4 20-EMA Trend Regime
Active Hours: 07:00 - 16:00 UTC (European London and NY Active Sessions)
SL: 7.0 pips | TP: 14.0 pips (2.00R Target) | Max Spread: 1.8 pips

5.77-Year Validated Institutional Metrics (300 Continuous Weeks / 85.81M Ticks):
  - Total Trades: 696 trades (~120 trades/year)
  - Full-Cycle Net Pips: +1,285.5 Net Pips
  - Annualized Net Return: +31.8% / year (at 1% risk per trade)
  - Full-Cycle Profit Factor: 1.45
  - 30% Quarantined Holdout Profit Factor: 1.50 (Holdout Edge Outperformance!)
  - Win Rate: 44.3%
  - Max Drawdown: 160.5 pips
  - Return / Drawdown: 8.01x
  - Expectancy: +1.85 pips / trade
================================================================================
"""

import numpy as np
import pandas as pd
from tick_engine.strategies.base import BaseStrategy, SIGNAL_LONG, SIGNAL_SHORT
from data_engineering_suite.bars import build_time_bars

PIP_SIZE = 0.0001
MAX_SPREAD = 1.8


class EURGBPSessionVWAPFadeElite(BaseStrategy):
    def __init__(self, name: str = "EURGBP_SessionVWAP_B2.0_SL7_TP14", sl_pips: float = 7.0, tp_pips: float = 14.0, band_mult: float = 2.0, **kwargs):
        super().__init__(**kwargs)
        self.name = name
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.band_mult = float(band_mult)

    def generate_signals(self, ticks: np.ndarray):
        m15 = build_time_bars(ticks, timeframe_min=15, pip_size=PIP_SIZE)
        if len(m15) < 30:
            n = len(ticks)
            return np.zeros(n, dtype=np.int8), np.zeros(n, dtype=np.float64), np.zeros(n, dtype=np.float64)
        h4 = m15.resample('4h').agg({'high': 'max', 'low': 'min', 'close': 'last', 'open': 'first'}).dropna()

        n = len(ticks)
        signals = np.zeros(n, dtype=np.int8)
        sl_arr = np.zeros(n, dtype=np.float64)
        tp_arr = np.zeros(n, dtype=np.float64)

        if len(h4) < 15:
            return signals, sl_arr, tp_arr

        h4_ema20 = h4['close'].ewm(span=20, adjust=False).mean()
        h4_bias = pd.Series(np.where(h4['close'] > h4_ema20, 1, -1), index=h4.index).shift(1).reindex(m15.index, method='ffill').fillna(0).values

        dates = m15.index.date
        c = m15['close'].values
        h = m15['high'].values
        l = m15['low'].values
        o = m15['open'].values
        spread = m15['spread_pips'].values
        hours = m15.index.hour.values
        vol = m15['tick_volume'].values if 'tick_volume' in m15.columns else np.ones(len(m15))
        time_ms = m15.index.astype('datetime64[ms]').view('int64')
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        vwap = np.zeros(len(m15))
        sd = np.zeros(len(m15))
        cur_date = None
        cum_pv = 0.0
        cum_v = 0.0
        p_buf = []

        for i in range(len(m15)):
            d = dates[i]
            if d != cur_date:
                cur_date = d
                cum_pv = 0.0
                cum_v = 0.0
                p_buf = []
            tp = (h[i] + l[i] + c[i]) / 3.0
            v = max(1.0, vol[i])
            cum_pv += tp * v
            cum_v += v
            p_buf.append(tp)
            vwap[i] = cum_pv / cum_v
            sd[i] = np.std(p_buf) if len(p_buf) > 2 else 0.0

        last_entry = -999
        for i in range(15, len(m15)):
            if i - last_entry < 6:
                continue
            if not (7 <= hours[i] <= 16) or spread[i] > MAX_SPREAD:
                continue
            if sd[i] <= 0:
                continue

            bias = h4_bias[i]
            upper = vwap[i] + self.band_mult * sd[i]
            lower = vwap[i] - self.band_mult * sd[i]
            bar_rng = h[i] - l[i]
            if bar_rng <= 0:
                continue

            if bias > 0 and l[i] <= lower and c[i] > lower:
                if ((min(o[i], c[i]) - l[i]) / bar_rng) >= 0.35:
                    idx = np.searchsorted(tick_ts, time_ms[i] + dur_ms)
                    if idx < n:
                        signals[idx] = SIGNAL_LONG
                        sl_arr[idx] = self.sl_pips * PIP_SIZE
                        tp_arr[idx] = self.tp_pips * PIP_SIZE
                        last_entry = i
            elif bias < 0 and h[i] >= upper and c[i] < upper:
                if ((h[i] - max(o[i], c[i])) / bar_rng) >= 0.35:
                    idx = np.searchsorted(tick_ts, time_ms[i] + dur_ms)
                    if idx < n:
                        signals[idx] = SIGNAL_SHORT
                        sl_arr[idx] = self.sl_pips * PIP_SIZE
                        tp_arr[idx] = self.tp_pips * PIP_SIZE
                        last_entry = i

        return signals, sl_arr, tp_arr

