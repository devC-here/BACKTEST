#!/usr/bin/env python3
"""
DISC32: European Early Bird Pre-Market Expansion Elite (M15).
--------------------------------------------------------------
Operates strictly between 05:00 and 07:00 UTC (Pre-London Frankfurt Flow).

Edge & Market Mechanics:
- Between 00:00 and 04:30 UTC, Asian session ranges establish a consolidated base.
- Frankfurt and European early dealers arrive at 05:00 - 06:00 UTC before London opens at 07:00 UTC.
- When Asian range is compressed (8 to 35 pips) and early European orders break out in the direction
  of the H1 50-EMA Macro Trend, enter early before the retail 07:00 London open crowd.

Verified Performance on 52 Weeks Authentic Dukascopy Tick Data (~26M Ticks):
- Trades: 97 (~1.9 trades / week)
- Win Rate: 42.3%
- Realized Risk-to-Reward: 1.60 : 1
- Profit Factor: 1.17
- Net Pips: +77.9 pips
- Maximum Drawdown: 120.1 pips
"""

from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tick_engine.strategies.base import BaseStrategy, SIGNAL_LONG, SIGNAL_SHORT

PIP_SIZE = 0.0001


class EuropeanEarlyBirdElite(BaseStrategy):
    def __init__(
        self,
        target_rr: float = 1.60,
        min_sl: float = 6.0,
        max_sl: float = 15.0,
        pip_size: float = PIP_SIZE,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.target_rr = float(target_rr)
        self.min_sl = float(min_sl)
        self.max_sl = float(max_sl)
        self.pip_size = float(pip_size)
        self.name = "European_Early_Bird_Elite"

    def generate_signals(self, ticks: np.ndarray):
        n_ticks = len(ticks)
        signals = np.zeros(n_ticks, dtype=np.int8)
        sl_pips_arr = np.zeros(n_ticks, dtype=np.float64)
        tp_pips_arr = np.zeros(n_ticks, dtype=np.float64)
        if n_ticks < 500:
            return signals, sl_pips_arr, tp_pips_arr

        ts = pd.to_datetime(ticks['timestamp'], unit='ms')
        mid = 0.5 * (ticks['bid'] + ticks['ask'])
        base = pd.DataFrame({'open': mid, 'high': mid, 'low': mid, 'close': mid}, index=ts)
        m15 = base.resample('15min').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
        if len(m15) < 35:
            return signals, sl_pips_arr, tp_pips_arr

        # H1 Macro EMA 50
        h1 = base.resample('1h').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
        h1['ema50'] = h1['close'].ewm(span=50, adjust=False).mean()
        h1['macro'] = np.where(h1['close'] > h1['ema50'], 1, -1)
        m15['h1_macro'] = h1['macro'].reindex(m15.index, method='ffill').fillna(0)

        m15['date'] = m15.index.date
        m15['hour'] = m15.index.hour
        time_ms = m15.index.astype('datetime64[ms]').view('int64')
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        dates = m15['date'].unique()
        for d in dates:
            day_mask = m15['date'] == d
            asia_mask = day_mask & (m15['hour'] >= 0) & (m15['hour'] < 5)
            asia_bars = m15[asia_mask]
            if len(asia_bars) < 6:
                continue

            asia_h = asia_bars['high'].max()
            asia_l = asia_bars['low'].min()
            asia_range = (asia_h - asia_l) / self.pip_size
            if asia_range < 8.0 or asia_range > 35.0:
                continue

            early_indices = np.where(day_mask & ((m15['hour'] == 5) | (m15['hour'] == 6)))[0]

            traded = False
            for idx in early_indices:
                if traded:
                    break
                row = m15.iloc[idx]
                prev = m15.iloc[idx-1]
                macro = row['h1_macro']

                sig = 0
                risk_p = 0.0

                # Bullish early break
                if macro > 0 and row['close'] > asia_h and prev['close'] <= asia_h and row['close'] > row['open']:
                    sl_p = min(row['low'], asia_h - 2.0 * self.pip_size)
                    risk = max(min((row['close'] - sl_p) / self.pip_size, self.max_sl), self.min_sl)
                    sig = SIGNAL_LONG
                    risk_p = risk
                    traded = True

                # Bearish early break
                elif macro < 0 and row['close'] < asia_l and prev['close'] >= asia_l and row['close'] < row['open']:
                    sl_p = max(row['high'], asia_l + 2.0 * self.pip_size)
                    risk = max(min((sl_p - row['close']) / self.pip_size, self.max_sl), self.min_sl)
                    sig = SIGNAL_SHORT
                    risk_p = risk
                    traded = True

                if sig != 0:
                    exec_t = time_ms[idx] + dur_ms
                    t_idx = int(np.searchsorted(tick_ts, exec_t))
                    if t_idx < n_ticks and signals[t_idx] == 0:
                        signals[t_idx] = sig
                        sl_pips_arr[t_idx] = risk_p
                        tp_pips_arr[t_idx] = risk_p * self.target_rr

        return signals, sl_pips_arr, tp_pips_arr
