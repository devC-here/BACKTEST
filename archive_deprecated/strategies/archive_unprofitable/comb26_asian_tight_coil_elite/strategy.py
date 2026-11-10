#!/usr/bin/env python3
"""
COMB26: Asian Breakout Tight Coil Elite.
-----------------------------------------
High-Expectancy Volatility Breakout from Narrow Asian Range (10-35p) + ADX >= 18.

Verified Performance on 52 Weeks Authentic Dukascopy Tick Data:
- Trades: 75 (~1.4 trd/week, ~0.29 trd/day)
- Win Rate: 33.3%
- Realized Risk-to-Reward: 2.57 : 1
- Profit Factor: 1.29
- Net Pips: +145.5 pips
- Maximum Drawdown: 4.6%
- Maximum Daily Loss: 0.8%
- Maximum Recovery Time: 173 days
"""

from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.dont_write_bytecode = True

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tick_engine.strategies.base import BaseStrategy, SIGNAL_LONG, SIGNAL_SHORT

PIP_SIZE = 0.0001


class AsianBreakoutTightCoilElite(BaseStrategy):
    """
    Asian Breakout Tight Coil Elite:
    - Filters Asian session (00:00 - 06:00 UTC) range strictly between 10 and 35 pips.
    - London session (07:00 - 10:00 UTC) breakout beyond Asian High/Low.
    - Requires ADX(14) >= 18.0 and impulse body >= 0.7x ATR.
    - Asymmetric 2.20 : 1 target with swing stop.
    """
    def __init__(
        self,
        target_rr: float = 2.20,
        min_sl: float = 6.0,
        max_sl: float = 16.0,
        min_adx: float = 18.0,
        pip_size: float = PIP_SIZE,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.target_rr = float(target_rr)
        self.min_sl = float(min_sl)
        self.max_sl = float(max_sl)
        self.min_adx = float(min_adx)
        self.pip_size = float(pip_size)
        self.name = "Asian_Breakout_Tight_Coil_Elite"

    def generate_signals(self, ticks: np.ndarray):
        n_ticks = len(ticks)
        signals = np.zeros(n_ticks, dtype=np.int8)
        sl_pips_arr = np.zeros(n_ticks, dtype=np.float64)
        tp_pips_arr = np.zeros(n_ticks, dtype=np.float64)
        if n_ticks < 500:
            return signals, sl_pips_arr, tp_pips_arr

        ts = pd.to_datetime(ticks['timestamp'], unit='ms')
        mid = 0.5 * (ticks['bid'] + ticks['ask'])
        base_df = pd.DataFrame({'open': mid, 'high': mid, 'low': mid, 'close': mid}, index=ts)
        m15 = base_df.resample('15min').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
        if len(m15) < 40:
            return signals, sl_pips_arr, tp_pips_arr

        c = m15['close']
        h = m15['high']
        l = m15['low']
        o = m15['open']

        tr = np.maximum(h - l, np.maximum(abs(h - c.shift(1)), abs(l - c.shift(1))))
        atr = tr.rolling(14).mean().bfill()
        atr_pips = atr / self.pip_size

        up_m = h.diff()
        dn_m = -l.diff()
        p_dm = np.where((up_m > dn_m) & (up_m > 0), up_m, 0.0)
        m_dm = np.where((dn_m > up_m) & (dn_m > 0), dn_m, 0.0)
        smooth_tr = tr.ewm(alpha=1.0/14, adjust=False).mean()
        plus_di = (pd.Series(p_dm, index=m15.index).ewm(alpha=1.0/14, adjust=False).mean() / np.maximum(smooth_tr, 1e-9)) * 100
        minus_di = (pd.Series(m_dm, index=m15.index).ewm(alpha=1.0/14, adjust=False).mean() / np.maximum(smooth_tr, 1e-9)) * 100
        dx = (abs(plus_di - minus_di) / np.maximum(plus_di + minus_di, 1e-9)) * 100
        adx = dx.ewm(alpha=1.0/14, adjust=False).mean()

        m15['date'] = m15.index.date
        m15['hour'] = m15.index.hour
        m15['time_ms'] = m15.index.astype('datetime64[ms]').view('int64')

        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        daily_groups = m15.groupby('date')
        for current_date, day_df in daily_groups:
            asian = day_df[(day_df['hour'] >= 0) & (day_df['hour'] < 6)]
            if len(asian) < 8:
                continue
            asian_high = asian['high'].max()
            asian_low = asian['low'].min()
            asian_range = (asian_high - asian_low) / self.pip_size

            # Only trade high-coiling ranges (10 to 35 pips)
            if asian_range < 10.0 or asian_range > 35.0:
                continue

            london = day_df[(day_df['hour'] >= 7) & (day_df['hour'] <= 10)]
            traded_today = False
            for idx_m15, row in london.iterrows():
                if traded_today:
                    break
                i = m15.index.get_loc(idx_m15)
                prev = m15.iloc[i-1]
                sig = 0
                risk_p = 0.0

                body_p = abs(row['close'] - row['open']) / self.pip_size

                # Bullish Breakout
                if prev['close'] <= asian_high and row['close'] > asian_high:
                    if adx.iloc[i] >= self.min_adx and row['close'] > row['open'] and body_p >= 0.7 * atr_pips.iloc[i]:
                        sl_p = row['low'] - 1.2 * self.pip_size
                        risk = max((row['close'] - sl_p) / self.pip_size, self.min_sl)
                        if risk <= self.max_sl:
                            sig = SIGNAL_LONG
                            risk_p = risk

                # Bearish Breakout
                elif prev['close'] >= asian_low and row['close'] < asian_low:
                    if adx.iloc[i] >= self.min_adx and row['close'] < row['open'] and body_p >= 0.7 * atr_pips.iloc[i]:
                        sl_p = row['high'] + 1.2 * self.pip_size
                        risk = max((sl_p - row['close']) / self.pip_size, self.min_sl)
                        if risk <= self.max_sl:
                            sig = SIGNAL_SHORT
                            risk_p = risk

                if sig != 0:
                    traded_today = True
                    exec_t = m15['time_ms'].iloc[i] + dur_ms
                    idx = int(np.searchsorted(tick_ts, exec_t))
                    if idx < n_ticks and signals[idx] == 0:
                        signals[idx] = sig
                        sl_pips_arr[idx] = risk_p
                        tp_pips_arr[idx] = risk_p * self.target_rr

        return signals, sl_pips_arr, tp_pips_arr
