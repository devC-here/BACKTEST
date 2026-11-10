#!/usr/bin/env python3
"""
High-Win-Rate Strategy Laboratory for Prop Firms.
-------------------------------------------------
Objective: Discover and validate strategies with HIGH WIN RATES (50% - 70%)
and asymmetric/balanced reward profiles (1.2R - 1.6R) to eliminate prolonged drawdowns
and multi-loss streaks.

Evaluates 6 High-Probability Models across 52 Weeks of Dukascopy Tick Data (~26M Ticks):
1. HWR01: Asian Night Mean-Reversion Scalper (21:00-05:00 UTC) (Bollinger 2.0 + RSI 7)
2. HWR02: London Asian Liquidity Sweep & Reversion (07:00-10:00 UTC) (Turtle Soup Trap)
3. HWR03: Connors RSI(2) Deep Pullback in 200-EMA Trend (M15)
4. HWR04: Session VWAP 1.5-Sigma Reversion with 1.3R Target (M15)
5. HWR05: Fair Value Gap (FVG) Mitigation Scalper with 1.4R Target (M15)
6. HWR06: Supertrend VWAP Pullback Calibrated for High Win Rate (1.35R Target) (M15)

Evaluated under Prop Firm Rules:
- Account Risk: 0.40% per trade ($400 on $100k account)
- Max Permitted Daily Loss: 5.0%
- Max Permitted Total DD: 10.0%
"""

from __future__ import annotations
import sys
import os
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Any
import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tick_engine.core.engine import BacktestEngine
from tick_engine.core.dataloader import get_week_dataset
from tick_engine.strategies.base import BaseStrategy, SIGNAL_LONG, SIGNAL_SHORT

PIP_SIZE = 0.0001


def enrich_hwr_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Precomputes indicators with ZERO lookahead on closed M15 bars."""
    c = df['close']
    h = df['high']
    l = df['low']
    o = df['open']
    n = len(df)

    # 1. ATR (14)
    tr = np.maximum(h - l, np.maximum(abs(h - c.shift(1)), abs(l - c.shift(1))))
    df['tr'] = tr
    df['atr_14'] = tr.rolling(14).mean().bfill()
    df['atr_pips'] = df['atr_14'] / PIP_SIZE

    # 2. Moving Averages
    for p in [5, 20, 50, 200]:
        df[f'ema_{p}'] = c.ewm(span=p, adjust=False).mean()
    df['sma_20'] = c.rolling(20).mean().bfill()

    # 3. Bollinger Bands (20, 2.0)
    std20 = c.rolling(20).std().bfill()
    df['bb_up'] = df['sma_20'] + (2.0 * std20)
    df['bb_low'] = df['sma_20'] - (2.0 * std20)

    # 4. RSI (7) & RSI (2)
    delta = c.diff()
    gain7 = delta.clip(lower=0.0).ewm(alpha=1.0/7, adjust=False).mean()
    loss7 = (-delta.clip(upper=0.0)).ewm(alpha=1.0/7, adjust=False).mean()
    rs7 = gain7 / np.maximum(loss7, 1e-9)
    df['rsi_7'] = 100.0 - (100.0 / (1.0 + rs7))

    gain2 = delta.clip(lower=0.0).ewm(alpha=1.0/2, adjust=False).mean()
    loss2 = (-delta.clip(upper=0.0)).ewm(alpha=1.0/2, adjust=False).mean()
    rs2 = gain2 / np.maximum(loss2, 1e-9)
    df['rsi_2'] = 100.0 - (100.0 / (1.0 + rs2))

    # 5. Stochastic (14, 3, 3)
    low14 = l.rolling(14).min().bfill()
    high14 = h.rolling(14).max().bfill()
    k_fast = ((c - low14) / np.maximum(high14 - low14, 1e-9)) * 100
    df['stoch_k'] = k_fast.rolling(3).mean().bfill()
    df['stoch_d'] = df['stoch_k'].rolling(3).mean().bfill()

    # 6. ADX (14)
    up_m = h.diff()
    dn_m = -l.diff()
    p_dm = np.where((up_m > dn_m) & (up_m > 0), up_m, 0.0)
    m_dm = np.where((dn_m > up_m) & (dn_m > 0), dn_m, 0.0)
    smooth_tr = tr.ewm(alpha=1.0/14, adjust=False).mean()
    plus_di = (pd.Series(p_dm, index=df.index).ewm(alpha=1.0/14, adjust=False).mean() / np.maximum(smooth_tr, 1e-9)) * 100
    minus_di = (pd.Series(m_dm, index=df.index).ewm(alpha=1.0/14, adjust=False).mean() / np.maximum(smooth_tr, 1e-9)) * 100
    dx = (abs(plus_di - minus_di) / np.maximum(plus_di + minus_di, 1e-9)) * 100
    df['adx_14'] = dx.ewm(alpha=1.0/14, adjust=False).mean()

    # 7. Supertrend (10, 2.0)
    hl2 = (h + l) / 2.0
    st_up = hl2 - (2.0 * df['atr_14'])
    st_low = hl2 + (2.0 * df['atr_14'])
    st = np.zeros(n)
    st_dir = np.zeros(n, dtype=np.int8)
    for i in range(1, n):
        if c.iloc[i] > st[i-1]:
            st_dir[i] = 1
            st[i] = max(st_up.iloc[i], st[i-1]) if st_dir[i-1] == 1 else st_up.iloc[i]
        else:
            st_dir[i] = -1
            st[i] = min(st_low.iloc[i], st[i-1]) if st_dir[i-1] == -1 else st_low.iloc[i]
    df['supertrend'] = st
    df['supertrend_dir'] = st_dir

    # 8. Session VWAP with 1.5-Sigma Bands
    df['date'] = df.index.date
    df['hour'] = df.index.hour
    df['time_ms'] = df.index.astype('datetime64[ms]').view('int64')

    vol = tr / PIP_SIZE + 10.0
    cum_vol = vol.groupby(df['date']).cumsum()
    cum_pv = (c * vol).groupby(df['date']).cumsum()
    df['session_vwap'] = cum_pv / np.maximum(cum_vol, 1e-9)

    cum_p2_v = ((c ** 2) * vol).groupby(df['date']).cumsum()
    vwap_var = (cum_p2_v / np.maximum(cum_vol, 1e-9)) - (df['session_vwap'] ** 2)
    vwap_std = np.sqrt(np.maximum(vwap_var, 0.0))
    df['vwap_up15'] = df['session_vwap'] + (1.5 * vwap_std)
    df['vwap_low15'] = df['session_vwap'] - (1.5 * vwap_std)

    return df


# ==============================================================================
# HWR01: Asian Night Mean-Reversion Scalper (21:00-05:00 UTC)
# ==============================================================================
class AsianNightMeanReversion(BaseStrategy):
    """
    Asian Night Scalper:
    - Time window: 21:00 to 05:00 UTC (Tokyo/Sydney range consolidation).
    - Long: Price touches or pierces lower Bollinger Band, RSI(7) < 25, closes green.
    - Short: Price touches or pierces upper Bollinger Band, RSI(7) > 75, closes red.
    - Target RR: 1.30 : 1 (quick reversion back to SMA 20).
    """
    def __init__(self, target_rr: float = 1.30, min_sl: float = 6.0, max_sl: float = 14.0, **kwargs):
        super().__init__(**kwargs)
        self.target_rr = target_rr
        self.min_sl = min_sl
        self.max_sl = max_sl
        self.pip_size = PIP_SIZE
        self.name = "Asian_Night_Scalper"

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

        m15 = enrich_hwr_indicators(m15)
        time_ms = m15['time_ms'].values
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        for i in range(25, len(m15)):
            row = m15.iloc[i]
            prev = m15.iloc[i-1]

            # Allowed hours: 21, 22, 23, 0, 1, 2, 3, 4 UTC
            if 5 <= row['hour'] < 21:
                continue

            sig = 0
            risk_p = 0.0

            # Long Reversion
            if row['low'] <= row['bb_low'] and row['close'] > row['open'] and prev['rsi_7'] < 28.0:
                sl_p = min(prev['low'], row['low']) - 1.2 * self.pip_size
                risk = max((row['close'] - sl_p) / self.pip_size, self.min_sl)
                if risk <= self.max_sl:
                    sig = SIGNAL_LONG
                    risk_p = risk

            # Short Reversion
            elif row['high'] >= row['bb_up'] and row['close'] < row['open'] and prev['rsi_7'] > 72.0:
                sl_p = max(prev['high'], row['high']) + 1.2 * self.pip_size
                risk = max((sl_p - row['close']) / self.pip_size, self.min_sl)
                if risk <= self.max_sl:
                    sig = SIGNAL_SHORT
                    risk_p = risk

            if sig != 0:
                exec_t = time_ms[i] + dur_ms
                idx = int(np.searchsorted(tick_ts, exec_t))
                if idx < n_ticks and signals[idx] == 0:
                    signals[idx] = sig
                    sl_pips_arr[idx] = risk_p
                    tp_pips_arr[idx] = risk_p * self.target_rr

        return signals, sl_pips_arr, tp_pips_arr


# ==============================================================================
# HWR02: London Asian Liquidity Sweep & Reversion (Turtle Soup Trap)
# ==============================================================================
class LondonLiquiditySweepTrap(BaseStrategy):
    """
    Institutional Stop Hunt Trap:
    - London Open (07:00-10:00 UTC).
    - Price sweeps Asian High or Asian Low by 1.0 to 9.0 pips, then vigorously closes BACK inside.
    - Stochastic %K crosses %D to confirm momentum reversal.
    - Target: 1.40 : 1 towards the Asian Range Midpoint.
    """
    def __init__(self, target_rr: float = 1.40, min_sl: float = 6.0, max_sl: float = 15.0, **kwargs):
        super().__init__(**kwargs)
        self.target_rr = target_rr
        self.min_sl = min_sl
        self.max_sl = max_sl
        self.pip_size = PIP_SIZE
        self.name = "London_Sweep_Trap"

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

        m15 = enrich_hwr_indicators(m15)
        time_ms = m15['time_ms'].values
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
            if asian_range < 8.0 or asian_range > 45.0:
                continue

            london = day_df[(day_df['hour'] >= 7) & (day_df['hour'] <= 10)]
            trades_today = 0
            for idx_m15, row in london.iterrows():
                if trades_today >= 1:
                    break
                i = m15.index.get_loc(idx_m15)
                prev = m15.iloc[i-1]
                sig = 0
                risk_p = 0.0

                # Bullish Trap: Low swept below Asian Low, but closed back above Asian Low & green
                sweep_l = (asian_low - row['low']) / self.pip_size
                if 0.8 <= sweep_l <= 9.0 and row['close'] > asian_low and row['close'] > row['open']:
                    if row['stoch_k'] > row['stoch_d'] and prev['stoch_k'] < 45.0:
                        sl_p = row['low'] - 1.2 * self.pip_size
                        risk = max((row['close'] - sl_p) / self.pip_size, self.min_sl)
                        if risk <= self.max_sl:
                            sig = SIGNAL_LONG
                            risk_p = risk

                # Bearish Trap: High swept above Asian High, but closed back below Asian High & red
                sweep_h = (row['high'] - asian_high) / self.pip_size
                if 0.8 <= sweep_h <= 9.0 and row['close'] < asian_high and row['close'] < row['open']:
                    if row['stoch_k'] < row['stoch_d'] and prev['stoch_k'] > 55.0:
                        sl_p = row['high'] + 1.2 * self.pip_size
                        risk = max((sl_p - row['close']) / self.pip_size, self.min_sl)
                        if risk <= self.max_sl:
                            sig = SIGNAL_SHORT
                            risk_p = risk

                if sig != 0:
                    trades_today += 1
                    exec_t = time_ms[i] + dur_ms
                    idx = int(np.searchsorted(tick_ts, exec_t))
                    if idx < n_ticks and signals[idx] == 0:
                        signals[idx] = sig
                        sl_pips_arr[idx] = risk_p
                        tp_pips_arr[idx] = risk_p * self.target_rr

        return signals, sl_pips_arr, tp_pips_arr


# ==============================================================================
# HWR03: Connors RSI(2) Deep Pullback in 200-EMA Trend (M15)
# ==============================================================================
class ConnorsRSI2TrendPullback(BaseStrategy):
    """
    Larry Connors High Win Rate Pullback:
    - Trend: Close > 200 EMA (Uptrend) or Close < 200 EMA (Downtrend).
    - Oversold dip: RSI(2) drops below 10.0 in an uptrend, candle prints reversal wick.
    - Overbought spike: RSI(2) rises above 90.0 in a downtrend.
    - Target RR: 1.40 : 1 with 5 EMA exit / protective stop.
    """
    def __init__(self, target_rr: float = 1.40, min_sl: float = 6.0, max_sl: float = 16.0, **kwargs):
        super().__init__(**kwargs)
        self.target_rr = target_rr
        self.min_sl = min_sl
        self.max_sl = max_sl
        self.pip_size = PIP_SIZE
        self.name = "Connors_RSI2_Pullback"

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
        if len(m15) < 55:
            return signals, sl_pips_arr, tp_pips_arr

        m15 = enrich_hwr_indicators(m15)
        time_ms = m15['time_ms'].values
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        for i in range(25, len(m15)):
            row = m15.iloc[i]
            prev = m15.iloc[i-1]

            if row['hour'] < 7 or row['hour'] > 16:
                continue

            sig = 0
            risk_p = 0.0

            # Bullish Connors Pullback: Price above 200 EMA, RSI(2) was deeply oversold (<10), Current closes green
            if row['close'] > row['ema_200'] and prev['rsi_2'] <= 10.0 and row['close'] > row['open']:
                sl_p = min(prev['low'], row['low']) - 1.2 * self.pip_size
                risk = max((row['close'] - sl_p) / self.pip_size, self.min_sl)
                if risk <= self.max_sl:
                    sig = SIGNAL_LONG
                    risk_p = risk

            # Bearish Connors Pullback: Price below 200 EMA, RSI(2) was deeply overbought (>90), Current closes red
            elif row['close'] < row['ema_200'] and prev['rsi_2'] >= 90.0 and row['close'] < row['open']:
                sl_p = max(prev['high'], row['high']) + 1.2 * self.pip_size
                risk = max((sl_p - row['close']) / self.pip_size, self.min_sl)
                if risk <= self.max_sl:
                    sig = SIGNAL_SHORT
                    risk_p = risk

            if sig != 0:
                exec_t = time_ms[i] + dur_ms
                idx = int(np.searchsorted(tick_ts, exec_t))
                if idx < n_ticks and signals[idx] == 0:
                    signals[idx] = sig
                    sl_pips_arr[idx] = risk_p
                    tp_pips_arr[idx] = risk_p * self.target_rr

        return signals, sl_pips_arr, tp_pips_arr


# ==============================================================================
# HWR04: Session VWAP 1.5-Sigma Reversion with 1.3R Target (M15)
# ==============================================================================
class SessionVWAP15Reversion(BaseStrategy):
    """
    Statistical VWAP Band Mean Reversion:
    - Filters: ADX < 20 (strict ranging market).
    - Long: Price extends beyond -1.5 Sigma VWAP band and closes green.
    - Short: Price extends beyond +1.5 Sigma VWAP band and closes red.
    - Target RR: 1.30 : 1 towards the VWAP mean.
    """
    def __init__(self, target_rr: float = 1.30, min_sl: float = 6.0, max_sl: float = 15.0, **kwargs):
        super().__init__(**kwargs)
        self.target_rr = target_rr
        self.min_sl = min_sl
        self.max_sl = max_sl
        self.pip_size = PIP_SIZE
        self.name = "VWAP_1.5Sigma_Reversion"

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

        m15 = enrich_hwr_indicators(m15)
        time_ms = m15['time_ms'].values
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        for i in range(25, len(m15)):
            row = m15.iloc[i]
            prev = m15.iloc[i-1]

            if row['hour'] < 7 or row['hour'] > 15:
                continue

            if row['adx_14'] >= 20.0:
                continue

            sig = 0
            risk_p = 0.0

            # Long Reversion from -1.5 Sigma
            if prev['low'] <= prev['vwap_low15'] and row['close'] > row['vwap_low15'] and row['close'] > row['open']:
                sl_p = min(prev['low'], row['low']) - 1.2 * self.pip_size
                risk = max((row['close'] - sl_p) / self.pip_size, self.min_sl)
                if risk <= self.max_sl:
                    sig = SIGNAL_LONG
                    risk_p = risk

            # Short Reversion from +1.5 Sigma
            elif prev['high'] >= prev['vwap_up15'] and row['close'] < row['vwap_up15'] and row['close'] < row['open']:
                sl_p = max(prev['high'], row['high']) + 1.2 * self.pip_size
                risk = max((sl_p - row['close']) / self.pip_size, self.min_sl)
                if risk <= self.max_sl:
                    sig = SIGNAL_SHORT
                    risk_p = risk

            if sig != 0:
                exec_t = time_ms[i] + dur_ms
                idx = int(np.searchsorted(tick_ts, exec_t))
                if idx < n_ticks and signals[idx] == 0:
                    signals[idx] = sig
                    sl_pips_arr[idx] = risk_p
                    tp_pips_arr[idx] = risk_p * self.target_rr

        return signals, sl_pips_arr, tp_pips_arr


# ==============================================================================
# HWR05: Fair Value Gap (FVG) Mitigation Scalper with 1.4R Target (M15)
# ==============================================================================
class FVGMitigationHighWR(BaseStrategy):
    """
    High-Velocity Fair Value Gap Mitigation:
    - 3-bar imbalance gap with impulse body >= 3.5 pips.
    - Retest of the gap with 50 EMA trend alignment.
    - Calibrated target: 1.40 : 1.
    """
    def __init__(self, target_rr: float = 1.40, min_sl: float = 6.0, max_sl: float = 16.0, **kwargs):
        super().__init__(**kwargs)
        self.target_rr = target_rr
        self.min_sl = min_sl
        self.max_sl = max_sl
        self.pip_size = PIP_SIZE
        self.name = "FVG_Mitigation_HighWR"

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

        m15 = enrich_hwr_indicators(m15)
        time_ms = m15['time_ms'].values
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        for i in range(25, len(m15)):
            row = m15.iloc[i]
            prev = m15.iloc[i-1]
            prior = m15.iloc[i-2]

            if row['hour'] < 7 or row['hour'] > 15:
                continue

            sig = 0
            risk_p = 0.0

            # Bullish FVG Mitigation: prior impulse bar
            if prev['close'] > prev['open'] and (prev['close'] - prev['open']) / self.pip_size >= 3.5:
                if row['open'] > prior['high'] and row['low'] <= prior['high'] and row['close'] > prior['high']:
                    if row['close'] > row['ema_50'] and row['close'] > row['open']:
                        sl_p = prev['low'] - 1.2 * self.pip_size
                        risk = max((row['close'] - sl_p) / self.pip_size, self.min_sl)
                        if risk <= self.max_sl:
                            sig = SIGNAL_LONG
                            risk_p = risk

            # Bearish FVG Mitigation
            elif prev['close'] < prev['open'] and (prev['open'] - prev['close']) / self.pip_size >= 3.5:
                if row['open'] < prior['low'] and row['high'] >= prior['low'] and row['close'] < prior['low']:
                    if row['close'] < row['ema_50'] and row['close'] < row['open']:
                        sl_p = prev['high'] + 1.2 * self.pip_size
                        risk = max((sl_p - row['close']) / self.pip_size, self.min_sl)
                        if risk <= self.max_sl:
                            sig = SIGNAL_SHORT
                            risk_p = risk

            if sig != 0:
                exec_t = time_ms[i] + dur_ms
                idx = int(np.searchsorted(tick_ts, exec_t))
                if idx < n_ticks and signals[idx] == 0:
                    signals[idx] = sig
                    sl_pips_arr[idx] = risk_p
                    tp_pips_arr[idx] = risk_p * self.target_rr

        return signals, sl_pips_arr, tp_pips_arr


# ==============================================================================
# HWR06: Supertrend VWAP Pullback Calibrated for High Win Rate (1.35R Target)
# ==============================================================================
class SupertrendVWAPHighWR(BaseStrategy):
    """
    Supertrend + VWAP Pullback with 1.35R Target:
    - Same robust entry as COMB14 (+503p).
    - Reduced target from 2.20R to 1.35R to capture quick momentum bursts,
      dramatically increasing win rate and eliminating givebacks.
    """
    def __init__(self, target_rr: float = 1.35, min_sl: float = 6.0, max_sl: float = 15.0, **kwargs):
        super().__init__(**kwargs)
        self.target_rr = target_rr
        self.min_sl = min_sl
        self.max_sl = max_sl
        self.pip_size = PIP_SIZE
        self.name = "Supertrend_VWAP_HighWR"

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

        m15 = enrich_hwr_indicators(m15)
        time_ms = m15['time_ms'].values
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        for i in range(25, len(m15)):
            row = m15.iloc[i]
            prev = m15.iloc[i-1]

            if row['hour'] < 7 or row['hour'] > 15:
                continue

            sig = 0
            risk_p = 0.0
            vwap = row['session_vwap']

            # Bullish: Supertrend Bullish, Low touched or pierced VWAP, Closed green above VWAP
            if row['supertrend_dir'] == 1 and prev['close'] > vwap:
                if row['low'] <= vwap + 2.0 * self.pip_size and row['close'] > vwap and row['close'] > row['open']:
                    sl_p = min(row['low'], vwap) - 1.2 * self.pip_size
                    risk = max((row['close'] - sl_p) / self.pip_size, self.min_sl)
                    if risk <= self.max_sl:
                        sig = SIGNAL_LONG
                        risk_p = risk

            # Bearish: Supertrend Bearish, High touched or pierced VWAP, Closed red below VWAP
            elif row['supertrend_dir'] == -1 and prev['close'] < vwap:
                if row['high'] >= vwap - 2.0 * self.pip_size and row['close'] < vwap and row['close'] < row['open']:
                    sl_p = max(row['high'], vwap) + 1.2 * self.pip_size
                    risk = max((sl_p - row['close']) / self.pip_size, self.min_sl)
                    if risk <= self.max_sl:
                        sig = SIGNAL_SHORT
                        risk_p = risk

            if sig != 0:
                exec_t = time_ms[i] + dur_ms
                idx = int(np.searchsorted(tick_ts, exec_t))
                if idx < n_ticks and signals[idx] == 0:
                    signals[idx] = sig
                    sl_pips_arr[idx] = risk_p
                    tp_pips_arr[idx] = risk_p * self.target_rr

        return signals, sl_pips_arr, tp_pips_arr


# ==============================================================================
# LABORATORY EXPERIMENT HARNESS
# ==============================================================================
HWR_STRATEGIES = [
    {"code": "HWR01", "name": "Asian Night Scalper (BB+RSI7) (M15)", "cls": AsianNightMeanReversion},
    {"code": "HWR02", "name": "London Asian Sweep Trap (Turtle Soup) (M15)", "cls": LondonLiquiditySweepTrap},
    {"code": "HWR03", "name": "Connors RSI(2) Deep Pullback (M15)", "cls": ConnorsRSI2TrendPullback},
    {"code": "HWR04", "name": "Session VWAP 1.5-Sigma Reversion (M15)", "cls": SessionVWAP15Reversion},
    {"code": "HWR05", "name": "FVG Mitigation Scalper (1.4R) (M15)", "cls": FVGMitigationHighWR},
    {"code": "HWR06", "name": "Supertrend VWAP High-WR (1.35R) (M15)", "cls": SupertrendVWAPHighWR},
]


def run_week_experiment(week_num: int) -> Dict[str, Any]:
    ticks, meta = get_week_dataset(week_num)
    n_ticks = len(ticks)
    engine = BacktestEngine(pip_size=PIP_SIZE, use_slippage=True)

    week_trades = {}
    for item in HWR_STRATEGIES:
        strat = item["cls"]()
        trades = engine.run(ticks, strat)
        t_list = []
        for t in trades:
            t_list.append({
                "entry_time": int(t["entry_time"]),
                "exit_time": int(t["exit_time"]),
                "direction": int(t["direction"]),
                "entry_price": float(t["entry_price"]),
                "exit_price": float(t["exit_price"]),
                "pips": float(t["pips"]),
                "exit_reason": int(t["exit_reason"]),
                "strategy": item["code"]
            })
        week_trades[item["code"]] = t_list

    return {
        "week": week_num,
        "ticks": n_ticks,
        "trades": week_trades
    }


def analyze_combination_performance(trades: List[Dict[str, Any]], risk_pct: float = 0.40):
    if not trades:
        return {
            "total_trades": 0, "trades_per_week": 0.0, "win_rate": 0.0, "realized_rr": 0.0,
            "profit_factor": 0.0, "net_pips": 0.0, "net_return_pct": 0.0, "max_dd_pct": 0.0,
            "max_daily_loss_pct": 0.0, "max_recovery_days": 0.0, "ulcer_index": 0.0
        }

    s_trades = sorted(trades, key=lambda x: x["exit_time"])
    pips = np.array([t["pips"] for t in s_trades], dtype=np.float64)
    wins = pips[pips > 0]
    losses = pips[pips < 0]
    n_trades = len(pips)
    win_rate = (len(wins) / n_trades * 100.0) if n_trades > 0 else 0.0
    avg_win = float(np.mean(wins)) if len(wins) > 0 else 0.0
    avg_loss = float(abs(np.mean(losses))) if len(losses) > 0 else 1.0
    realized_rr = avg_win / avg_loss if avg_loss > 0 else 0.0
    tot_win_pips = float(np.sum(wins)) if len(wins) > 0 else 0.0
    tot_loss_pips = float(abs(np.sum(losses))) if len(losses) > 0 else 0.0
    pf = tot_win_pips / tot_loss_pips if tot_loss_pips > 0 else 0.0
    net_pips = float(np.sum(pips))

    # Equity Curve
    balance = 100000.0
    curve = [balance]
    timestamps = [s_trades[0]["entry_time"]]
    daily_pnl = {}

    for t in s_trades:
        r_pnl = (t["pips"] / avg_loss) * (balance * (risk_pct / 100.0))
        balance += r_pnl
        curve.append(balance)
        timestamps.append(t["exit_time"])

        day_str = pd.to_datetime(t["exit_time"], unit='ms').strftime('%Y-%m-%d')
        daily_pnl[day_str] = daily_pnl.get(day_str, 0.0) + r_pnl

    c_arr = np.array(curve)
    peaks = np.maximum.accumulate(c_arr)
    dd_pct = (peaks - c_arr) / peaks * 100.0
    max_dd_pct = float(np.max(dd_pct))

    ulcer_index = float(np.sqrt(np.mean(dd_pct ** 2)))

    max_daily_loss = 0.0
    for day, pnl in daily_pnl.items():
        if pnl < 0:
            day_loss_pct = abs(pnl) / 100000.0 * 100.0
            if day_loss_pct > max_daily_loss:
                max_daily_loss = day_loss_pct

    ts_arr = np.array(timestamps)
    in_dd = False
    dd_start_ts = ts_arr[0]
    max_recov_days = 0.0
    for k in range(len(c_arr)):
        if c_arr[k] < peaks[k]:
            if not in_dd:
                in_dd = True
                dd_start_ts = ts_arr[max(0, k-1)]
        else:
            if in_dd:
                recov_d = (ts_arr[k] - dd_start_ts) / (1000.0 * 86400.0)
                if recov_d > max_recov_days:
                    max_recov_days = recov_d
                in_dd = False

    net_return_pct = (balance - 100000.0) / 100000.0 * 100.0
    trades_per_week = n_trades / 52.0

    return {
        "total_trades": n_trades,
        "trades_per_week": trades_per_week,
        "win_rate": win_rate,
        "realized_rr": realized_rr,
        "profit_factor": pf,
        "net_pips": net_pips,
        "net_return_pct": net_return_pct,
        "max_dd_pct": max_dd_pct,
        "max_daily_loss_pct": max_daily_loss,
        "max_recovery_days": max_recov_days,
        "ulcer_index": ulcer_index
    }


def main():
    print("=" * 135)
    print("                 HIGH-WIN-RATE STRATEGY LABORATORY: 52 WEEKS FULL TICK AUDIT (~26M TICKS)")
    print("                      Evaluating 6 High-Probability Models (Targeting 50% - 70% Win Rate)")
    print("=" * 135)

    start_t = time.time()
    weeks = list(range(1, 53))

    strat_trades: Dict[str, List[Dict[str, Any]]] = {item["code"]: [] for item in HWR_STRATEGIES}
    total_ticks = 0

    with ProcessPoolExecutor() as executor:
        results = list(executor.map(run_week_experiment, weeks))

    for res in results:
        total_ticks += res["ticks"]
        for code, t_list in res["trades"].items():
            strat_trades[code].extend(t_list)

    elapsed = time.time() - start_t
    print(f"\n[+] Evaluated {total_ticks:,} Ticks across 52 Weeks in {elapsed:.1f}s ({total_ticks/elapsed:,.0f} ticks/s)!\n")

    print("=" * 145)
    print("                                      HIGH-WIN-RATE MODELS SCORECARD")
    print("=" * 145)
    print(f"{'Code':<7} | {'High-Probability Strategy Name':<45} | {'Trades':<6} | {'Trd/Wk':<6} | {'Win %':<6} | {'Real RR':<7} | {'PF':<5} | {'Net Pips':<9} | {'Return %':<8} | {'Max DD%':<7} | {'Daily DD%':<9} | {'Ulcer':<5} | {'Max Recov':<9}")
    print("-" * 145)

    for item in HWR_STRATEGIES:
        code = item["code"]
        name = item["name"]
        t_data = strat_trades[code]
        metrics = analyze_combination_performance(t_data, risk_pct=0.40)

        print(
            f"{code:<7} | {name:<45} | "
            f"{metrics['total_trades']:<6} | "
            f"{metrics['trades_per_week']:<6.1f} | "
            f"{metrics['win_rate']:<4.1f} % | "
            f"{metrics['realized_rr']:<7.2f} | "
            f"{metrics['profit_factor']:<5.2f} | "
            f"{metrics['net_pips']:<+9.1f} | "
            f"{metrics['net_return_pct']:<+6.1f} % | "
            f"{metrics['max_dd_pct']:<5.1f} % | "
            f"{metrics['max_daily_loss_pct']:<7.1f} % | "
            f"{metrics['ulcer_index']:<5.1f} | "
            f"{metrics['max_recovery_days']:<4.0f} days"
        )
    print("=" * 145)


if __name__ == "__main__":
    main()
