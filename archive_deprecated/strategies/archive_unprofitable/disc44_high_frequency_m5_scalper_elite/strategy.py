"""
================================================================================
  DISC44: HIGH-FREQUENCY M5 SCALPER ELITE (ALL-WEATHER EDITION)
================================================================================
Asset: Pure EUR/USD
Timeframe: 5-Minute (M5)
Target R:R: 2.20 : 1
Execution: Millisecond Tick Lookup with Authentic Floating Spreads & Slippage

Core Mechanism:
Combines 3 institutional M5 microstructure setups calibrated to conquer spread friction:
  1. Calibrated FVG Mitigation (gap >= 3.5 pips)
  2. M5 Displacement 50% Equilibrium Retest (body >= 70% of 7p impulse)
  3. M5 Narrow-Range Compression Box Breakout (NR4 on M5)
Filtered strictly by H1 50-EMA macro regime during peak liquidity hours (07-10 & 12-16 UTC).
================================================================================
"""

from __future__ import annotations
import numpy as np
import pandas as pd

from tick_engine.strategies.base import BaseStrategy, SIGNAL_LONG, SIGNAL_SHORT

PIP_SIZE = 0.0001


class HighFrequencyM5ScalperElite(BaseStrategy):
    def __init__(
        self,
        target_rr: float = 2.2,
        min_sl: float = 7.0,
        max_sl: float = 16.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.target_rr = target_rr
        self.min_sl = min_sl
        self.max_sl = max_sl
        self.name = "HighFrequency_M5_Scalper_Elite"
        self.allowed_hours = {7, 8, 9, 10, 12, 13, 14, 15, 16}

    def generate_signals(self, ticks: np.ndarray):
        n_ticks = len(ticks)
        signals = np.zeros(n_ticks, dtype=np.int8)
        sl_pips_arr = np.zeros(n_ticks, dtype=np.float64)
        tp_pips_arr = np.zeros(n_ticks, dtype=np.float64)

        if n_ticks < 1000:
            return signals, sl_pips_arr, tp_pips_arr

        ts = pd.to_datetime(ticks['timestamp'], unit='ms')
        mid = 0.5 * (ticks['bid'] + ticks['ask'])
        base_df = pd.DataFrame({'open': mid, 'high': mid, 'low': mid, 'close': mid}, index=ts)

        m5 = base_df.resample('5min').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
        if len(m5) < 60:
            return signals, sl_pips_arr, tp_pips_arr

        # Macro Trend (H1 50-EMA with strict shift(1))
        h1 = base_df.resample('1h').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
        h1['ema50'] = h1['close'].ewm(span=50, adjust=False).mean()
        h1['trend'] = np.where(h1['close'] > h1['ema50'], 1, -1)
        m5['macro'] = h1['trend'].shift(1).reindex(m5.index, method='ffill').fillna(0)

        tr = np.maximum(m5['high'] - m5['low'], np.maximum(abs(m5['high'] - m5['close'].shift(1)), abs(m5['low'] - m5['close'].shift(1))))
        m5['atr'] = tr.rolling(14).mean().bfill()
        m5['ranges'] = (m5['high'] - m5['low']) / PIP_SIZE

        c = m5['close'].values
        h = m5['high'].values
        l = m5['low'].values
        o = m5['open'].values
        macro = m5['macro'].values
        ranges = m5['ranges'].values
        atr_pips = (m5['atr'] / PIP_SIZE).values
        hours = m5.index.hour.values
        weekdays = m5.index.dayofweek.values
        time_ms = m5.index.astype('datetime64[ms]').view('int64')
        dur_ms = 5 * 60 * 1000
        tick_ts = ticks['timestamp']
        n_bars = len(m5)

        for i in range(25, n_bars - 1):
            hr = hours[i]
            wd = weekdays[i]
            if hr not in self.allowed_hours:
                continue
            if wd == 4 and hr >= 16:
                continue

            sig = 0
            risk_p = 0.0
            m_dir = macro[i]

            dyn_min = max(self.min_sl, 0.9 * atr_pips[i])
            dyn_max = max(self.max_sl, 1.8 * atr_pips[i])

            # 1. Calibrated FVG Mitigation (gap >= 3.5 pips)
            gap_up = (l[i] - h[i-2]) / PIP_SIZE
            if gap_up >= 3.5 and c[i-1] > o[i-1]:
                if m_dir >= 0 and c[i] > o[i]:
                    sl_p = h[i-2] - 1.2 * PIP_SIZE
                    r = max((c[i] - sl_p) / PIP_SIZE, dyn_min)
                    if r <= dyn_max:
                        sig = SIGNAL_LONG
                        risk_p = r

            gap_dn = (l[i-2] - h[i]) / PIP_SIZE
            if gap_dn >= 3.5 and c[i-1] < o[i-1]:
                if m_dir <= 0 and c[i] < o[i]:
                    sl_p = l[i-2] + 1.2 * PIP_SIZE
                    r = max((sl_p - c[i]) / PIP_SIZE, dyn_min)
                    if r <= dyn_max:
                        sig = SIGNAL_SHORT
                        risk_p = r

            # 2. M5 Displacement 50% Equilibrium Retest
            if sig == 0:
                prev_rng = ranges[i-1]
                prev_body = abs(c[i-1] - o[i-1]) / PIP_SIZE
                if prev_rng >= 7.0 and (prev_body / max(prev_rng, 0.1)) >= 0.70:
                    midpoint = 0.5 * (o[i-1] + c[i-1])

                    if c[i-1] > o[i-1] and m_dir >= 0:
                        if l[i] <= midpoint and c[i] > midpoint and c[i] > o[i]:
                            sl_p = l[i-1] - 1.0 * PIP_SIZE
                            r = max((c[i] - sl_p) / PIP_SIZE, dyn_min)
                            if r <= dyn_max:
                                sig = SIGNAL_LONG
                                risk_p = r

                    elif c[i-1] < o[i-1] and m_dir <= 0:
                        if h[i] >= midpoint and c[i] < midpoint and c[i] < o[i]:
                            sl_p = h[i-1] + 1.0 * PIP_SIZE
                            r = max((sl_p - c[i]) / PIP_SIZE, dyn_min)
                            if r <= dyn_max:
                                sig = SIGNAL_SHORT
                                risk_p = r

            # 3. M5 NR4 Compression Box Breakout
            if sig == 0:
                is_nr4 = (ranges[i-1] <= 3.2) and (ranges[i-2] <= 3.2) and (ranges[i-3] <= 3.2) and (ranges[i-4] <= 3.2)
                if is_nr4:
                    box_h = np.max(h[i-4:i])
                    box_l = np.min(l[i-4:i])

                    if c[i] > box_h and c[i] > o[i] and m_dir >= 0:
                        sl_p = box_l - 1.0 * PIP_SIZE
                        r = max((c[i] - sl_p) / PIP_SIZE, dyn_min)
                        if r <= dyn_max:
                            sig = SIGNAL_LONG
                            risk_p = r

                    elif c[i] < box_l and c[i] < o[i] and m_dir <= 0:
                        sl_p = box_h + 1.0 * PIP_SIZE
                        r = max((sl_p - c[i]) / PIP_SIZE, dyn_min)
                        if r <= dyn_max:
                            sig = SIGNAL_SHORT
                            risk_p = r

            if sig != 0:
                exec_t = time_ms[i] + dur_ms
                idx = int(np.searchsorted(tick_ts, exec_t))
                if idx < n_ticks and signals[idx] == 0:
                    signals[idx] = sig
                    sl_pips_arr[idx] = risk_p
                    tp_pips_arr[idx] = risk_p * self.target_rr

        return signals, sl_pips_arr, tp_pips_arr
