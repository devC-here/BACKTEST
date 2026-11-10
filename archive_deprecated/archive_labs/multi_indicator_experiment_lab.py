#!/usr/bin/env python3
"""
Comprehensive Multi-Indicator Strategy Laboratory for Prop Firm Trading.
Evaluates 8 Diverse Indicator Combinations on 52 Weeks of Authentic Dukascopy Tick Data (~26M Ticks):

1. COMB01: Session VWAP + RSI Pullback (M15)
2. COMB02: TTM Squeeze Breakout (Bollinger inside Keltner + Momentum Flip) (M15)
3. COMB03: Donchian 20-Channel Trend Breakout + ADX > 22 (M15)
4. COMB04: New York Open Expansion (12:30-15:30 UTC) + MACD Acceleration (M15)
5. COMB05: Triple EMA Ribbon (8/21/55) Value Zone Pullback (M15)
6. COMB06: Supertrend (10, 2.5) + H1 50-EMA Macro Engulfing (M15)
7. COMB07: Keltner Envelope Reversal + Stochastic Extreme (M15)
8. COMB08: London Asian Breakout + ADX Momentum (Validated Benchmark) (M15)

Evaluated under Prop Firm Rules:
- Account Risk: 0.50% per trade ($500 on $100k account)
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

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tick_engine.core.engine import BacktestEngine
from tick_engine.core.dataloader import get_week_dataset
from tick_engine.strategies.base import BaseStrategy, SIGNAL_LONG, SIGNAL_SHORT

PIP_SIZE = 0.0001


def enrich_full_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Precomputes all indicators needed across the 8 experimental combinations."""
    c = df['close']
    h = df['high']
    l = df['low']
    o = df['open']

    # 1. EMAs & SMAs
    for p in [8, 20, 21, 50, 55, 200]:
        df[f'ema_{p}'] = c.ewm(span=p, adjust=False).mean()
    df['sma_20'] = c.rolling(20).mean().bfill()

    # 2. ATR (14)
    tr = np.maximum(h - l, np.maximum(abs(h - c.shift(1)), abs(l - c.shift(1))))
    df['atr_14'] = tr.rolling(14).mean().bfill()
    df['atr_pips'] = df['atr_14'] / PIP_SIZE

    # 3. RSI (14)
    delta = c.diff()
    gain = delta.clip(lower=0.0).ewm(alpha=1.0/14, adjust=False).mean()
    loss = (-delta.clip(upper=0.0)).ewm(alpha=1.0/14, adjust=False).mean()
    rs = gain / np.maximum(loss, 1e-9)
    df['rsi_14'] = 100.0 - (100.0 / (1.0 + rs))

    # 4. MACD (12, 26, 9)
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    df['macd_line'] = ema12 - ema26
    df['macd_sig'] = df['macd_line'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd_line'] - df['macd_sig']

    # 5. Bollinger Bands (20, 2.0)
    std20 = c.rolling(20).std().bfill()
    df['bb_up'] = df['sma_20'] + (2.0 * std20)
    df['bb_low'] = df['sma_20'] - (2.0 * std20)
    df['bb_width'] = (df['bb_up'] - df['bb_low']) / PIP_SIZE

    # 6. Keltner Channels (20, 1.5 ATR)
    df['kc_up'] = df['ema_20'] + (1.5 * df['atr_14'])
    df['kc_low'] = df['ema_20'] - (1.5 * df['atr_14'])

    # TTM Squeeze Condition: BB is completely inside KC
    df['is_squeeze'] = (df['bb_up'] < df['kc_up']) & (df['bb_low'] > df['kc_low'])

    # 7. Donchian Channels (20)
    df['donchian_high'] = h.rolling(20).max().shift(1).bfill()
    df['donchian_low'] = l.rolling(20).min().shift(1).bfill()

    # 8. ADX & DMI (14)
    up_m = h.diff()
    dn_m = -l.diff()
    p_dm = np.where((up_m > dn_m) & (up_m > 0), up_m, 0.0)
    m_dm = np.where((dn_m > up_m) & (dn_m > 0), dn_m, 0.0)
    smooth_tr = tr.ewm(alpha=1.0/14, adjust=False).mean()
    df['plus_di'] = (pd.Series(p_dm, index=df.index).ewm(alpha=1.0/14, adjust=False).mean() / np.maximum(smooth_tr, 1e-9)) * 100
    df['minus_di'] = (pd.Series(m_dm, index=df.index).ewm(alpha=1.0/14, adjust=False).mean() / np.maximum(smooth_tr, 1e-9)) * 100
    dx = (abs(df['plus_di'] - df['minus_di']) / np.maximum(df['plus_di'] + df['minus_di'], 1e-9)) * 100
    df['adx_14'] = dx.ewm(alpha=1.0/14, adjust=False).mean()

    # 9. Stochastic (14, 3, 3)
    low14 = l.rolling(14).min().bfill()
    high14 = h.rolling(14).max().bfill()
    k_fast = ((c - low14) / np.maximum(high14 - low14, 1e-9)) * 100
    df['stoch_k'] = k_fast.rolling(3).mean().bfill()
    df['stoch_d'] = df['stoch_k'].rolling(3).mean().bfill()

    # 10. Supertrend (10, 2.5)
    hl2 = (h + l) / 2.0
    st_up = hl2 - (2.5 * df['atr_14'])
    st_low = hl2 + (2.5 * df['atr_14'])
    n = len(df)
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

    # 11. Time & Session metadata
    df['hour'] = df.index.hour
    df['date'] = df.index.date
    df['weekday'] = df.index.dayofweek
    df['time_ms'] = df.index.astype('datetime64[ms]').view('int64')

    # 12. Session VWAP (resets daily 00:00 UTC)
    typical_p = (h + l + c) / 3.0
    # Approximate volume with bar range / activity
    vol = tr / PIP_SIZE + 10.0
    cum_vol = vol.groupby(df['date']).cumsum()
    cum_pv = (typical_p * vol).groupby(df['date']).cumsum()
    df['session_vwap'] = cum_pv / np.maximum(cum_vol, 1e-9)

    return df


# ==============================================================================
# MODEL 1: Session VWAP + RSI Pullback (M15)
# ==============================================================================
class SessionVWAP_RSIPullback(BaseStrategy):
    """
    Volume-Weighted Mean Reversion:
    Trend: Price > 200 EMA.
    Setup: Pullback touches Session VWAP with RSI cooling to 40-50.
    Trigger: Bullish candle closes above VWAP with RSI turning up.
    """
    def __init__(self, target_rr=2.0, **kwargs):
        super().__init__(**kwargs)
        self.target_rr = target_rr
        self.name = "COMB01_VWAP_RSI_Pullback"

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
        if len(m15) < 40:
            return signals, sl_pips_arr, tp_pips_arr

        m15 = enrich_full_indicators(m15)
        time_ms = m15['time_ms'].values
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        for i in range(25, len(m15)):
            row = m15.iloc[i]
            prev = m15.iloc[i - 1]
            if row['hour'] < 7 or row['hour'] > 17:
                continue

            sig = 0
            risk_p = 0.0

            # Bullish VWAP Bounce
            if row['close'] > row['ema_200']:
                touched_vwap = (m15['low'].iloc[i-2:i+1] <= m15['session_vwap'].iloc[i-2:i+1] + 1.5*PIP_SIZE).any()
                if touched_vwap and prev['rsi_14'] <= 50.0 and row['rsi_14'] > 50.0 and row['close'] > row['session_vwap'] and row['close'] > row['open']:
                    swing_l = float(m15['low'].iloc[i-2:i+1].min())
                    sl_dist = (row['close'] - swing_l) / PIP_SIZE + 1.2
                    risk_p = np.clip(sl_dist, 6.0, 16.0)
                    sig = SIGNAL_LONG

            # Bearish VWAP Bounce
            elif row['close'] < row['ema_200']:
                touched_vwap = (m15['high'].iloc[i-2:i+1] >= m15['session_vwap'].iloc[i-2:i+1] - 1.5*PIP_SIZE).any()
                if touched_vwap and prev['rsi_14'] >= 50.0 and row['rsi_14'] < 50.0 and row['close'] < row['session_vwap'] and row['close'] < row['open']:
                    swing_h = float(m15['high'].iloc[i-2:i+1].max())
                    sl_dist = (swing_h - row['close']) / PIP_SIZE + 1.2
                    risk_p = np.clip(sl_dist, 6.0, 16.0)
                    sig = SIGNAL_SHORT

            if sig != 0:
                exec_t = time_ms[i] + dur_ms
                idx = int(np.searchsorted(tick_ts, exec_t))
                if idx < n_ticks and signals[idx] == 0:
                    signals[idx] = sig
                    sl_pips_arr[idx] = risk_p
                    tp_pips_arr[idx] = risk_p * self.target_rr

        return signals, sl_pips_arr, tp_pips_arr


# ==============================================================================
# MODEL 2: TTM Squeeze Breakout (Bollinger inside Keltner + MACD Flip) (M15)
# ==============================================================================
class TTMSqueezeBreakout(BaseStrategy):
    """
    Volatility Squeeze Expansion:
    Setup: Bollinger Bands were squeezed inside Keltner Channels in past 6 bars.
    Trigger: Squeeze releases with MACD histogram acceleration in direction of 50 EMA.
    """
    def __init__(self, target_rr=2.2, **kwargs):
        super().__init__(**kwargs)
        self.target_rr = target_rr
        self.name = "COMB02_TTM_Squeeze_Breakout"

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
        if len(m15) < 40:
            return signals, sl_pips_arr, tp_pips_arr

        m15 = enrich_full_indicators(m15)
        time_ms = m15['time_ms'].values
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        for i in range(25, len(m15)):
            row = m15.iloc[i]
            prev = m15.iloc[i - 1]
            if row['hour'] < 7 or row['hour'] > 17:
                continue

            sig = 0
            risk_p = 0.0

            # Squeeze released: was squeezed in past 6 bars, now expanding
            was_squeezed = m15['is_squeeze'].iloc[i-6:i].any()

            if was_squeezed and not row['is_squeeze']:
                # Bullish Squeeze Fired
                if row['close'] > row['ema_50'] and prev['macd_hist'] <= 0.0 and row['macd_hist'] > 0.0 and row['close'] > row['open']:
                    sl_dist = (row['close'] - row['kc_low']) / PIP_SIZE
                    risk_p = np.clip(sl_dist, 7.0, 18.0)
                    sig = SIGNAL_LONG

                # Bearish Squeeze Fired
                elif row['close'] < row['ema_50'] and prev['macd_hist'] >= 0.0 and row['macd_hist'] < 0.0 and row['close'] < row['open']:
                    sl_dist = (row['kc_up'] - row['close']) / PIP_SIZE
                    risk_p = np.clip(sl_dist, 7.0, 18.0)
                    sig = SIGNAL_SHORT

            if sig != 0:
                exec_t = time_ms[i] + dur_ms
                idx = int(np.searchsorted(tick_ts, exec_t))
                if idx < n_ticks and signals[idx] == 0:
                    signals[idx] = sig
                    sl_pips_arr[idx] = risk_p
                    tp_pips_arr[idx] = risk_p * self.target_rr

        return signals, sl_pips_arr, tp_pips_arr


# ==============================================================================
# MODEL 3: Donchian 20-Channel Trend Breakout + ADX > 22 (M15)
# ==============================================================================
class DonchianADXBreakout(BaseStrategy):
    """
    Turtle Momentum Scalper:
    Trend: Filtered by 50 EMA and 200 EMA.
    Setup: ADX > 22 (confirmed trending environment).
    Trigger: Price breaks beyond 20-bar Donchian high/low with expansion candle.
    """
    def __init__(self, target_rr=2.0, **kwargs):
        super().__init__(**kwargs)
        self.target_rr = target_rr
        self.name = "COMB03_Donchian_ADX_Breakout"

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
        if len(m15) < 40:
            return signals, sl_pips_arr, tp_pips_arr

        m15 = enrich_full_indicators(m15)
        time_ms = m15['time_ms'].values
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        for i in range(25, len(m15)):
            row = m15.iloc[i]
            prev = m15.iloc[i - 1]
            if row['hour'] < 7 or row['hour'] > 17:
                continue

            sig = 0
            risk_p = 0.0

            if row['adx_14'] >= 22.0:
                # Bullish Breakout
                if prev['close'] <= prev['donchian_high'] and row['close'] > prev['donchian_high'] and row['close'] > row['ema_50']:
                    if row['plus_di'] > row['minus_di'] and row['close'] > row['open']:
                        sl_dist = (row['close'] - row['ema_20']) / PIP_SIZE + 1.5
                        risk_p = np.clip(sl_dist, 7.0, 18.0)
                        sig = SIGNAL_LONG

                # Bearish Breakout
                elif prev['close'] >= prev['donchian_low'] and row['close'] < prev['donchian_low'] and row['close'] < row['ema_50']:
                    if row['minus_di'] > row['plus_di'] and row['close'] < row['open']:
                        sl_dist = (row['ema_20'] - row['close']) / PIP_SIZE + 1.5
                        risk_p = np.clip(sl_dist, 7.0, 18.0)
                        sig = SIGNAL_SHORT

            if sig != 0:
                exec_t = time_ms[i] + dur_ms
                idx = int(np.searchsorted(tick_ts, exec_t))
                if idx < n_ticks and signals[idx] == 0:
                    signals[idx] = sig
                    sl_pips_arr[idx] = risk_p
                    tp_pips_arr[idx] = risk_p * self.target_rr

        return signals, sl_pips_arr, tp_pips_arr


# ==============================================================================
# MODEL 4: New York Open Expansion (12:30-15:30 UTC) + MACD Acceleration (M15)
# ==============================================================================
class NewYorkOpenExpansion(BaseStrategy):
    """
    Session Momentum Breakout:
    Tracks European Morning Range (07:00 to 12:00 UTC).
    At US Open (12:30 to 15:30 UTC), enters on displacement breakout of European range
    confirmed by MACD histogram momentum and H1 50-EMA macro trend.
    """
    def __init__(self, target_rr=2.0, **kwargs):
        super().__init__(**kwargs)
        self.target_rr = target_rr
        self.name = "COMB04_NY_Open_Expansion"

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
        if len(m15) < 40:
            return signals, sl_pips_arr, tp_pips_arr

        m15 = enrich_full_indicators(m15)
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        dates = m15['date'].unique()
        for d in dates:
            day_mask = m15['date'] == d
            eu_mask = day_mask & (m15['hour'] >= 7) & (m15['hour'] < 12)
            eu_bars = m15[eu_mask]
            if len(eu_bars) < 10:
                continue

            eu_h = eu_bars['high'].max()
            eu_l = eu_bars['low'].min()
            eu_mid = 0.5 * (eu_h + eu_l)

            # NY window: 12:30 to 15:30 UTC (hours 12 to 15)
            ny_indices = np.where(day_mask & (m15['hour'] >= 12) & (m15['hour'] <= 15))[0]
            traded_today = False

            for idx in ny_indices:
                if traded_today:
                    break
                row = m15.iloc[idx]
                prev = m15.iloc[idx - 1]

                sig = 0
                risk_p = 0.0

                # Bullish Breakout above European High
                if prev['close'] <= eu_h and row['close'] > eu_h + 1.0*PIP_SIZE and row['close'] > row['open']:
                    if row['macd_hist'] > 0 and row['close'] > row['ema_50']:
                        sl_dist = (row['close'] - eu_mid) / PIP_SIZE
                        risk_p = np.clip(sl_dist, 8.0, 20.0)
                        sig = SIGNAL_LONG
                        traded_today = True

                # Bearish Breakout below European Low
                elif prev['close'] >= eu_l and row['close'] < eu_l - 1.0*PIP_SIZE and row['close'] < row['open']:
                    if row['macd_hist'] < 0 and row['close'] < row['ema_50']:
                        sl_dist = (eu_mid - row['close']) / PIP_SIZE
                        risk_p = np.clip(sl_dist, 8.0, 20.0)
                        sig = SIGNAL_SHORT
                        traded_today = True

                if traded_today and sig != 0:
                    exec_t = m15['time_ms'].iloc[idx] + dur_ms
                    t_idx = int(np.searchsorted(tick_ts, exec_t))
                    if t_idx < n_ticks and signals[t_idx] == 0:
                        signals[t_idx] = sig
                        sl_pips_arr[t_idx] = risk_p
                        tp_pips_arr[t_idx] = risk_p * self.target_rr

        return signals, sl_pips_arr, tp_pips_arr


# ==============================================================================
# MODEL 5: Triple EMA Ribbon (8/21/55) Value Zone Pullback (M15)
# ==============================================================================
class TripleEMARibbonPullback(BaseStrategy):
    """
    Fibonacci EMA Stack Trend Rider:
    Trend: Perfect Ribbon Alignment (8 > 21 > 55 EMA for Long).
    Setup: Low touches 21 EMA pocket without violating 55 EMA.
    Trigger: Candle closes bullish above 8 EMA breaking previous candle high.
    """
    def __init__(self, target_rr=2.2, **kwargs):
        super().__init__(**kwargs)
        self.target_rr = target_rr
        self.name = "COMB05_Triple_EMA_Ribbon"

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
        if len(m15) < 60:
            return signals, sl_pips_arr, tp_pips_arr

        m15 = enrich_full_indicators(m15)
        time_ms = m15['time_ms'].values
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        for i in range(30, len(m15)):
            row = m15.iloc[i]
            prev = m15.iloc[i - 1]
            if row['hour'] < 7 or row['hour'] > 17:
                continue

            sig = 0
            risk_p = 0.0

            # Bullish Ribbon: 8 > 21 > 55
            if row['ema_8'] > row['ema_21'] and row['ema_21'] > row['ema_55']:
                # Pullback in last 3 bars touched 21 EMA but remained above 55 EMA
                touched_21 = (m15['low'].iloc[i-3:i] <= m15['ema_21'].iloc[i-3:i] + 1.5*PIP_SIZE).any()
                above_55 = (m15['low'].iloc[i-3:i+1] > m15['ema_55'].iloc[i-3:i+1] - 1.0*PIP_SIZE).all()

                if touched_21 and above_55 and row['close'] > row['open'] and row['close'] > row['ema_8'] and row['close'] > prev['high']:
                    swing_l = float(m15['low'].iloc[i-3:i+1].min())
                    sl_dist = (row['close'] - swing_l) / PIP_SIZE + 1.2
                    risk_p = np.clip(sl_dist, 6.0, 16.0)
                    sig = SIGNAL_LONG

            # Bearish Ribbon: 8 < 21 < 55
            elif row['ema_8'] < row['ema_21'] and row['ema_21'] < row['ema_55']:
                touched_21 = (m15['high'].iloc[i-3:i] >= m15['ema_21'].iloc[i-3:i] - 1.5*PIP_SIZE).any()
                below_55 = (m15['high'].iloc[i-3:i+1] < m15['ema_55'].iloc[i-3:i+1] + 1.0*PIP_SIZE).all()

                if touched_21 and below_55 and row['close'] < row['open'] and row['close'] < row['ema_8'] and row['close'] < prev['low']:
                    swing_h = float(m15['high'].iloc[i-3:i+1].max())
                    sl_dist = (swing_h - row['close']) / PIP_SIZE + 1.2
                    risk_p = np.clip(sl_dist, 6.0, 16.0)
                    sig = SIGNAL_SHORT

            if sig != 0:
                exec_t = time_ms[i] + dur_ms
                idx = int(np.searchsorted(tick_ts, exec_t))
                if idx < n_ticks and signals[idx] == 0:
                    signals[idx] = sig
                    sl_pips_arr[idx] = risk_p
                    tp_pips_arr[idx] = risk_p * self.target_rr

        return signals, sl_pips_arr, tp_pips_arr


# ==============================================================================
# MODEL 6: Supertrend (10, 2.5) + Macro Trend Engulfing (M15)
# ==============================================================================
class SupertrendMacroEngulfing(BaseStrategy):
    """
    Momentum Reversal:
    Supertrend (10, 2.5) flips direction aligned with 200 EMA
    confirmed by a high-volatility engulfing candle (body >= 1.0x ATR).
    """
    def __init__(self, target_rr=2.2, **kwargs):
        super().__init__(**kwargs)
        self.target_rr = target_rr
        self.name = "COMB06_Supertrend_Engulfing"

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
        if len(m15) < 40:
            return signals, sl_pips_arr, tp_pips_arr

        m15 = enrich_full_indicators(m15)
        time_ms = m15['time_ms'].values
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        for i in range(25, len(m15)):
            row = m15.iloc[i]
            prev = m15.iloc[i - 1]
            if row['hour'] < 7 or row['hour'] > 17:
                continue

            sig = 0
            risk_p = 0.0
            body = abs(row['close'] - row['open'])

            # Supertrend flips bullish
            if prev['supertrend_dir'] == -1 and row['supertrend_dir'] == 1 and row['close'] > row['ema_200']:
                if body >= 0.9 * row['atr_14'] and row['close'] > row['open']:
                    sl_dist = (row['close'] - row['supertrend']) / PIP_SIZE + 1.2
                    risk_p = np.clip(sl_dist, 7.0, 18.0)
                    sig = SIGNAL_LONG

            # Supertrend flips bearish
            elif prev['supertrend_dir'] == 1 and row['supertrend_dir'] == -1 and row['close'] < row['ema_200']:
                if body >= 0.9 * row['atr_14'] and row['close'] < row['open']:
                    sl_dist = (row['supertrend'] - row['close']) / PIP_SIZE + 1.2
                    risk_p = np.clip(sl_dist, 7.0, 18.0)
                    sig = SIGNAL_SHORT

            if sig != 0:
                exec_t = time_ms[i] + dur_ms
                idx = int(np.searchsorted(tick_ts, exec_t))
                if idx < n_ticks and signals[idx] == 0:
                    signals[idx] = sig
                    sl_pips_arr[idx] = risk_p
                    tp_pips_arr[idx] = risk_p * self.target_rr

        return signals, sl_pips_arr, tp_pips_arr


# ==============================================================================
# MODEL 7: Keltner Envelope Reversal + Stochastic Extreme (M15)
# ==============================================================================
class KeltnerStochasticReversal(BaseStrategy):
    """
    Overextended Mean Reversion:
    Setup: Price pierces outside 1.5 ATR Keltner Envelope with Stochastic Extreme (<20 / >80).
    Trigger: Candle closes back inside Keltner band with reversal hammer/shooting star.
    """
    def __init__(self, target_rr=1.8, **kwargs):
        super().__init__(**kwargs)
        self.target_rr = target_rr
        self.name = "COMB07_Keltner_Stoch_Reversal"

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
        if len(m15) < 40:
            return signals, sl_pips_arr, tp_pips_arr

        m15 = enrich_full_indicators(m15)
        time_ms = m15['time_ms'].values
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        for i in range(25, len(m15)):
            row = m15.iloc[i]
            prev = m15.iloc[i - 1]
            if row['hour'] < 7 or row['hour'] > 17:
                continue

            sig = 0
            risk_p = 0.0

            # Bullish Reversal: previous bar tested below Keltner Low with oversold Stoch (<25)
            if prev['low'] < prev['kc_low'] and prev['stoch_k'] < 25.0:
                if row['close'] > row['kc_low'] and row['close'] > row['open']:
                    sl_dist = (row['close'] - min(prev['low'], row['low'])) / PIP_SIZE + 1.2
                    risk_p = np.clip(sl_dist, 6.0, 16.0)
                    sig = SIGNAL_LONG

            # Bearish Reversal: previous bar tested above Keltner High with overbought Stoch (>75)
            elif prev['high'] > prev['kc_up'] and prev['stoch_k'] > 75.0:
                if row['close'] < row['kc_up'] and row['close'] < row['open']:
                    sl_dist = (max(prev['high'], row['high']) - row['close']) / PIP_SIZE + 1.2
                    risk_p = np.clip(sl_dist, 6.0, 16.0)
                    sig = SIGNAL_SHORT

            if sig != 0:
                exec_t = time_ms[i] + dur_ms
                idx = int(np.searchsorted(tick_ts, exec_t))
                if idx < n_ticks and signals[idx] == 0:
                    signals[idx] = sig
                    sl_pips_arr[idx] = risk_p
                    tp_pips_arr[idx] = risk_p * self.target_rr

        return signals, sl_pips_arr, tp_pips_arr


# ==============================================================================
# MODEL 8: London Asian Breakout + ADX Momentum (Validated Benchmark) (M15)
# ==============================================================================
class LondonAsianBreakoutADX(BaseStrategy):
    """
    Session Momentum Breakout:
    Asian Range (00:00 to 06:00 UTC).
    London Open (07:00 to 10:00 UTC) breaks Asian Range with ADX >= 16 and body >= 0.8x ATR.
    """
    def __init__(self, target_rr=2.2, min_adx=16.0, **kwargs):
        super().__init__(**kwargs)
        self.target_rr = target_rr
        self.min_adx = min_adx
        self.name = "COMB08_London_Asian_Breakout"

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
        if len(m15) < 40:
            return signals, sl_pips_arr, tp_pips_arr

        m15 = enrich_full_indicators(m15)
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        dates = m15['date'].unique()
        for d in dates:
            day_mask = m15['date'] == d
            asian_mask = day_mask & (m15['hour'] < 6)
            asian_bars = m15[asian_mask]
            if len(asian_bars) < 8:
                continue

            asian_h = asian_bars['high'].max()
            asian_l = asian_bars['low'].min()
            asian_mid = 0.5 * (asian_h + asian_l)
            asian_range_pips = (asian_h - asian_l) / PIP_SIZE
            if asian_range_pips < 8.0 or asian_range_pips > 35.0:
                continue

            london_indices = np.where(day_mask & (m15['hour'] >= 7) & (m15['hour'] <= 10))[0]
            traded_today = False

            for idx in london_indices:
                if traded_today:
                    break
                row = m15.iloc[idx]
                prev = m15.iloc[idx - 1]
                body = abs(row['close'] - row['open'])

                if row['adx_14'] < self.min_adx or body < 0.8 * row['atr_14']:
                    continue

                sig = 0
                risk_p = 0.0

                # Bullish Breakout
                if prev['close'] <= asian_h and row['close'] > asian_h + 1.0*PIP_SIZE and row['close'] > row['open']:
                    if row['close'] > row['ema_50']:
                        sl_dist = (row['close'] - asian_mid) / PIP_SIZE
                        risk_p = np.clip(sl_dist, 8.0, 18.0)
                        sig = SIGNAL_LONG
                        traded_today = True

                # Bearish Breakout
                elif prev['close'] >= asian_l and row['close'] < asian_l - 1.0*PIP_SIZE and row['close'] < row['open']:
                    if row['close'] < row['ema_50']:
                        sl_dist = (asian_mid - row['close']) / PIP_SIZE
                        risk_p = np.clip(sl_dist, 8.0, 18.0)
                        sig = SIGNAL_SHORT
                        traded_today = True

                if traded_today and sig != 0:
                    exec_t = m15['time_ms'].iloc[idx] + dur_ms
                    t_idx = int(np.searchsorted(tick_ts, exec_t))
                    if t_idx < n_ticks and signals[t_idx] == 0:
                        signals[t_idx] = sig
                        sl_pips_arr[t_idx] = risk_p
                        tp_pips_arr[t_idx] = risk_p * self.target_rr

        return signals, sl_pips_arr, tp_pips_arr


EXPERIMENTAL_STRATEGIES = [
    {"id": "COMB01", "name": "Session VWAP + RSI Pullback (M15)", "cls": SessionVWAP_RSIPullback},
    {"id": "COMB02", "name": "TTM Squeeze Momentum Breakout (M15)", "cls": TTMSqueezeBreakout},
    {"id": "COMB03", "name": "Donchian 20-Channel + ADX Breakout (M15)", "cls": DonchianADXBreakout},
    {"id": "COMB04", "name": "NY Open Expansion + MACD Momentum (M15)", "cls": NewYorkOpenExpansion},
    {"id": "COMB05", "name": "Triple EMA Ribbon (8/21/55) Pullback (M15)", "cls": TripleEMARibbonPullback},
    {"id": "COMB06", "name": "Supertrend + Macro Engulfing (M15)", "cls": SupertrendMacroEngulfing},
    {"id": "COMB07", "name": "Keltner Envelope + Stoch Reversal (M15)", "cls": KeltnerStochasticReversal},
    {"id": "COMB08", "name": "London Asian Breakout + ADX (Benchmark) (M15)", "cls": LondonAsianBreakoutADX},
]


def run_week_experiment(week_num: int) -> Dict[str, Any]:
    ticks, meta = get_week_dataset(week_num)
    n_ticks = len(ticks)
    engine = BacktestEngine(pip_size=0.0001, use_slippage=True)

    week_res = {}
    for s in EXPERIMENTAL_STRATEGIES:
        strat = s["cls"]()
        trades = engine.run(ticks, strat)
        t_list = []
        for t in trades:
            t_list.append({
                "entry_time": int(t["entry_time"]),
                "exit_time": int(t["exit_time"]),
                "direction": int(t["direction"]),
                "pips": float(t["pips"]),
                "strategy": s["id"]
            })
        week_res[s["id"]] = t_list

    return {"week": week_num, "ticks": n_ticks, "trades": week_res}


def evaluate_model_results(trades: List[Dict[str, Any]], risk_pct: float = 0.50) -> Dict[str, Any]:
    if not trades:
        return {"n_trades": 0, "wr": 0.0, "real_rr": 0.0, "pf": 0.0, "net_pips": 0.0, "return_pct": 0.0, "max_dd_pct": 0.0, "max_daily_loss": 0.0, "max_recovery_days": 0}

    trades.sort(key=lambda x: x["exit_time"])
    df = pd.DataFrame(trades)
    df["exit_dt"] = pd.to_datetime(df["exit_time"], unit="ms", utc=True)
    df["exit_date"] = df["exit_dt"].dt.date

    n = len(df)
    wins = df[df["pips"] > 0]
    losses = df[df["pips"] < 0]

    wr = (len(wins) / n * 100.0) if n > 0 else 0.0
    gp = wins["pips"].sum() if len(wins) > 0 else 0.0
    gl = abs(losses["pips"].sum()) if len(losses) > 0 else 0.0
    pf = (gp / gl) if gl > 0 else 99.0

    avg_w = wins["pips"].mean() if len(wins) > 0 else 0.0
    avg_l = abs(losses["pips"].mean()) if len(losses) > 0 else 1.0
    real_rr = (avg_w / avg_l) if avg_l > 0 else 0.0

    df["pct_return"] = (df["pips"] / avg_l) * risk_pct
    daily_pnl = df.groupby("exit_date")["pct_return"].sum()

    cum = daily_pnl.cumsum()
    peak = cum.cummax()
    dd = cum - peak
    max_dd = abs(dd.min()) if len(dd) > 0 else 0.0
    max_daily_loss = abs(min(daily_pnl.min(), 0.0)) if len(daily_pnl) > 0 else 0.0

    # Drawdown recovery in trading days
    underwater = []
    c_uw = 0
    for d_val in dd:
        if d_val < -1e-5:
            c_uw += 1
        else:
            if c_uw > 0:
                underwater.append(c_uw)
            c_uw = 0
    if c_uw > 0:
        underwater.append(c_uw)
    max_recov = max(underwater) if underwater else 0

    return {
        "n_trades": n,
        "wr": wr,
        "real_rr": real_rr,
        "pf": pf,
        "net_pips": df["pips"].sum(),
        "return_pct": cum.iloc[-1] if len(cum) > 0 else 0.0,
        "max_dd_pct": max_dd,
        "max_daily_loss": max_daily_loss,
        "max_recovery_days": max_recov
    }


def main():
    print("=" * 125)
    print("        MULTI-INDICATOR STRATEGY LABORATORY: 52 WEEKS FULL TICK AUDIT (~26M TICKS)        ".center(125))
    print("        Evaluating 8 Advanced Indicator Combinations with Discrete Event Triggers          ".center(125))
    print("=" * 125)

    weeks = list(range(1, 53))
    num_workers = min(8, os.cpu_count() or 4)

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(run_week_experiment, weeks))
    dt = time.time() - t0

    total_ticks = sum(r["ticks"] for r in results)
    print(f"\n[+] Evaluated {total_ticks:,} Ticks across {len(weeks)} Weeks in {dt:.1f}s ({total_ticks/max(dt, 0.001):,.0f} ticks/s)!\n")

    strat_trades = {s["id"]: [] for s in EXPERIMENTAL_STRATEGIES}
    for r in results:
        for sid, t_list in r["trades"].items():
            strat_trades[sid].extend(t_list)

    print("=" * 135)
    print("                            EXPERIMENTAL INDICATOR COMBINATIONS SCORECARD                            ".center(135))
    print("=" * 135)
    print(f"{'Code':<7} | {'Indicator Combination Strategy':<45} | {'Trades':<6} | {'Trd/Wk':<6} | {'Win %':<6} | {'Real RR':<7} | {'PF':<5} | {'Net Pips':<9} | {'Return %':<8} | {'Max DD%':<7} | {'Daily DD%':<9} | {'Max Recov':<10}")
    print("-" * 135)

    for s in EXPERIMENTAL_STRATEGIES:
        sid = s["id"]
        t = strat_trades[sid]
        res = evaluate_model_results(t)
        trd_wk = res["n_trades"] / 52.0
        print(f"{sid:<7} | {s['name']:<45} | {res['n_trades']:<6} | {trd_wk:<6.1f} | {res['wr']:<5.1f}% | {res['real_rr']:<7.2f} | {res['pf']:<5.2f} | {res['net_pips']:<+9.1f} | {res['return_pct']:<+7.1f}% | {res['max_dd_pct']:<6.1f}% | {res['max_daily_loss']:<8.1f}% | {res['max_recovery_days']:<2.0f} days")

    print("=" * 135)


if __name__ == "__main__":
    main()
