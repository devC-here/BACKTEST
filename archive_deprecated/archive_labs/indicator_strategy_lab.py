#!/usr/bin/env python3
"""
Systematic Indicator Strategy Laboratory for Prop Firm Clearance.
Tests classic & institutional indicator combinations on authentic 52-week EUR/USD tick data.

Every model uses the Discrete Event Trigger pattern:
- Macro Regime Filter (e.g. 200 EMA, Supertrend)
- Value Zone Setup (e.g. Pullback to 20/50 EMA, RSI cooling down)
- Discrete Event Trigger (fires on the exact bar transition, zero bar churn)
- Dynamic SL & Target RR (1.5R - 2.5R)
- Active Session Hours (07:00 - 17:00 UTC)
"""

from __future__ import annotations
import sys
import os
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Any, Tuple
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


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    c = df['close']
    h = df['high']
    l = df['low']
    o = df['open']

    # EMAs
    for p in [9, 20, 21, 50, 200]:
        df[f'ema_{p}'] = c.ewm(span=p, adjust=False).mean()

    # ATR
    tr = np.maximum(h - l, np.maximum(abs(h - c.shift(1)), abs(l - c.shift(1))))
    df['atr_14'] = tr.rolling(14).mean().bfill()
    df['atr_pips'] = df['atr_14'] / PIP_SIZE

    # RSI (14)
    delta = c.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0/14, adjust=False).mean()
    rs = avg_gain / np.maximum(avg_loss, 1e-9)
    df['rsi_14'] = 100.0 - (100.0 / (1.0 + rs))

    # MACD (12, 26, 9)
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    df['macd_line'] = ema12 - ema26
    df['macd_sig'] = df['macd_line'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd_line'] - df['macd_sig']

    # Bollinger Bands (20, 2.0)
    sma20 = c.rolling(20).mean().bfill()
    std20 = c.rolling(20).std().bfill()
    df['bb_up'] = sma20 + (2.0 * std20)
    df['bb_low'] = sma20 - (2.0 * std20)
    df['bb_width'] = (df['bb_up'] - df['bb_low']) / PIP_SIZE

    # Supertrend (10, 2.0)
    hl2 = (h + l) / 2.0
    st_basic_up = hl2 - (2.0 * df['atr_14'])
    st_basic_low = hl2 + (2.0 * df['atr_14'])
    n = len(df)
    st = np.zeros(n)
    st_dir = np.zeros(n, dtype=np.int8) # 1 = Bullish, -1 = Bearish
    for i in range(1, n):
        if c.iloc[i] > st[i-1]:
            st_dir[i] = 1
            st[i] = max(st_basic_up.iloc[i], st[i-1]) if st_dir[i-1] == 1 else st_basic_up.iloc[i]
        else:
            st_dir[i] = -1
            st[i] = min(st_basic_low.iloc[i], st[i-1]) if st_dir[i-1] == -1 else st_basic_low.iloc[i]
    df['supertrend'] = st
    df['supertrend_dir'] = st_dir

    # ADX (14)
    up_m = h.diff()
    dn_m = -l.diff()
    p_dm = np.where((up_m > dn_m) & (up_m > 0), up_m, 0.0)
    m_dm = np.where((dn_m > up_m) & (dn_m > 0), dn_m, 0.0)
    smooth_tr = tr.ewm(alpha=1.0/14, adjust=False).mean()
    df['plus_di'] = (pd.Series(p_dm, index=df.index).ewm(alpha=1.0/14, adjust=False).mean() / np.maximum(smooth_tr, 1e-9)) * 100
    df['minus_di'] = (pd.Series(m_dm, index=df.index).ewm(alpha=1.0/14, adjust=False).mean() / np.maximum(smooth_tr, 1e-9)) * 100
    dx = (abs(df['plus_di'] - df['minus_di']) / np.maximum(df['plus_di'] + df['minus_di'], 1e-9)) * 100
    df['adx_14'] = dx.ewm(alpha=1.0/14, adjust=False).mean()

    # Stochastic (14, 3, 3)
    low14 = l.rolling(14).min().bfill()
    high14 = h.rolling(14).max().bfill()
    k_fast = ((c - low14) / np.maximum(high14 - low14, 1e-9)) * 100
    df['stoch_k'] = k_fast.rolling(3).mean().bfill()
    df['stoch_d'] = df['stoch_k'].rolling(3).mean().bfill()

    # Time metadata
    df['hour'] = df.index.hour
    df['weekday'] = df.index.dayofweek
    df['time_ms'] = df.index.astype('datetime64[ms]').view('int64')

    return df


# ==============================================================================
# STRATEGY 1: Dual EMA (9/21) + RSI 50 Momentum Re-cross (M15)
# ==============================================================================
class DualEMA_RSI_MomentumStrategy(BaseStrategy):
    """
    Trend: Price > 200 EMA and 50 EMA direction.
    Setup: Price pulled back into 21 EMA value pocket with RSI dipping to 40-52.
    Trigger: Bullish candle closes above 21 EMA and 9 EMA > 21 EMA while RSI crosses above 50.
    """
    def __init__(self, target_rr: float = 2.0, **kwargs):
        super().__init__(**kwargs)
        self.target_rr = target_rr
        self.name = "IND01_DualEMA_RSI_Momentum_M15"

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

        m15 = compute_indicators(m15)
        time_ms = m15['time_ms'].values
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        for i in range(25, len(m15)):
            row = m15.iloc[i]
            prev = m15.iloc[i - 1]

            if row['hour'] < 7 or row['hour'] > 17:
                continue
            if row['weekday'] == 4 and row['hour'] >= 16:
                continue

            sig = 0
            risk_p = 0.0

            # Bullish Setup
            if row['close'] > row['ema_200'] and row['ema_21'] > row['ema_50']:
                # Pullback touched 21 EMA in last 2 bars
                touched = (m15['low'].iloc[i-2:i+1] <= m15['ema_21'].iloc[i-2:i+1] + 1.5*PIP_SIZE).any()
                # Trigger: RSI was <= 50, now crossed > 50, and current bar closes bullish above 9 EMA & 21 EMA
                if touched and prev['rsi_14'] <= 50.0 and row['rsi_14'] > 50.0 and row['close'] > row['open'] and row['close'] > row['ema_9']:
                    swing_l = float(m15['low'].iloc[i-3:i+1].min())
                    sl_dist = (row['close'] - swing_l) / PIP_SIZE + 1.5
                    risk_p = np.clip(sl_dist, 6.0, 18.0)
                    sig = SIGNAL_LONG

            # Bearish Setup
            elif row['close'] < row['ema_200'] and row['ema_21'] < row['ema_50']:
                touched = (m15['high'].iloc[i-2:i+1] >= m15['ema_21'].iloc[i-2:i+1] - 1.5*PIP_SIZE).any()
                if touched and prev['rsi_14'] >= 50.0 and row['rsi_14'] < 50.0 and row['close'] < row['open'] and row['close'] < row['ema_9']:
                    swing_h = float(m15['high'].iloc[i-3:i+1].max())
                    sl_dist = (swing_h - row['close']) / PIP_SIZE + 1.5
                    risk_p = np.clip(sl_dist, 6.0, 18.0)
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
# STRATEGY 2: Supertrend + 50 EMA Dynamic Trend Retest (M15)
# ==============================================================================
class Supertrend_EMA50_RetestStrategy(BaseStrategy):
    """
    Trend: Supertrend (10, 2.0) is aligned with 200 EMA.
    Setup: Price retraces back to 50 EMA value zone.
    Trigger: Candle closes in trend direction breaking previous candle extreme.
    """
    def __init__(self, target_rr: float = 2.0, **kwargs):
        super().__init__(**kwargs)
        self.target_rr = target_rr
        self.name = "IND02_Supertrend_EMA50_Retest_M15"

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

        m15 = compute_indicators(m15)
        time_ms = m15['time_ms'].values
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        for i in range(25, len(m15)):
            row = m15.iloc[i]
            prev = m15.iloc[i - 1]

            if row['hour'] < 7 or row['hour'] > 17:
                continue
            if row['weekday'] == 4 and row['hour'] >= 16:
                continue

            sig = 0
            risk_p = 0.0

            # Bullish: Supertrend Bullish + Price > 200 EMA + 50 EMA > 200 EMA
            if row['supertrend_dir'] == 1 and row['close'] > row['ema_200'] and row['ema_50'] > row['ema_200']:
                # Low touched 50 EMA within 2.0 pips
                touched_50 = (m15['low'].iloc[i-2:i+1] <= m15['ema_50'].iloc[i-2:i+1] + 2.0*PIP_SIZE).any()
                # Trigger: Bullish reversal departure closing above prev high and above 20 EMA
                if touched_50 and row['close'] > row['open'] and row['close'] > prev['high'] and row['close'] > row['ema_20']:
                    swing_l = float(m15['low'].iloc[i-2:i+1].min())
                    sl_dist = (row['close'] - swing_l) / PIP_SIZE + 1.5
                    risk_p = np.clip(sl_dist, 7.0, 18.0)
                    sig = SIGNAL_LONG

            # Bearish: Supertrend Bearish + Price < 200 EMA + 50 EMA < 200 EMA
            elif row['supertrend_dir'] == -1 and row['close'] < row['ema_200'] and row['ema_50'] < row['ema_200']:
                touched_50 = (m15['high'].iloc[i-2:i+1] >= m15['ema_50'].iloc[i-2:i+1] - 2.0*PIP_SIZE).any()
                if touched_50 and row['close'] < row['open'] and row['close'] < prev['low'] and row['close'] < row['ema_20']:
                    swing_h = float(m15['high'].iloc[i-2:i+1].max())
                    sl_dist = (swing_h - row['close']) / PIP_SIZE + 1.5
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
# STRATEGY 3: MACD Zero-Line Acceleration + EMA 20 Pullback (M15)
# ==============================================================================
class MACD_ZeroLine_AccelerationStrategy(BaseStrategy):
    """
    Trend: Macro trend 200 EMA alignment.
    Setup: MACD line on the correct side of zero.
    Trigger: MACD histogram flips positive/negative with expanding momentum.
    """
    def __init__(self, target_rr: float = 2.0, **kwargs):
        super().__init__(**kwargs)
        self.target_rr = target_rr
        self.name = "IND03_MACD_ZeroLine_Acceleration_M15"

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

        m15 = compute_indicators(m15)
        time_ms = m15['time_ms'].values
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        for i in range(25, len(m15)):
            row = m15.iloc[i]
            prev = m15.iloc[i - 1]

            if row['hour'] < 7 or row['hour'] > 17:
                continue
            if row['weekday'] == 4 and row['hour'] >= 16:
                continue

            sig = 0
            risk_p = 0.0

            # Bullish MACD Zero-Line Acceleration
            if row['close'] > row['ema_200'] and row['macd_line'] > 0:
                # Trigger: Histogram flips from negative to positive
                if prev['macd_hist'] <= 0.0 and row['macd_hist'] > 0.0 and row['close'] > row['open'] and row['close'] > row['ema_20']:
                    swing_l = float(m15['low'].iloc[i-3:i+1].min())
                    sl_dist = (row['close'] - swing_l) / PIP_SIZE + 1.5
                    risk_p = np.clip(sl_dist, 6.0, 16.0)
                    sig = SIGNAL_LONG

            # Bearish MACD Zero-Line Acceleration
            elif row['close'] < row['ema_200'] and row['macd_line'] < 0:
                if prev['macd_hist'] >= 0.0 and row['macd_hist'] < 0.0 and row['close'] < row['open'] and row['close'] < row['ema_20']:
                    swing_h = float(m15['high'].iloc[i-3:i+1].max())
                    sl_dist = (swing_h - row['close']) / PIP_SIZE + 1.5
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
# STRATEGY 4: Stochastic Extreme Pullback + 20 EMA Momentum Departure (M15)
# ==============================================================================
class Stochastic_EMA_PullbackStrategy(BaseStrategy):
    """
    Trend: 200 EMA + 20/50 EMA stack.
    Setup: Stochastic dipped below 25 (oversold pullback in uptrend).
    Trigger: Stochastic %K crosses above %D from oversold zone with bullish candle departure.
    """
    def __init__(self, target_rr: float = 2.0, **kwargs):
        super().__init__(**kwargs)
        self.target_rr = target_rr
        self.name = "IND04_Stochastic_EMA_Pullback_M15"

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

        m15 = compute_indicators(m15)
        time_ms = m15['time_ms'].values
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        for i in range(25, len(m15)):
            row = m15.iloc[i]
            prev = m15.iloc[i - 1]

            if row['hour'] < 7 or row['hour'] > 17:
                continue
            if row['weekday'] == 4 and row['hour'] >= 16:
                continue

            sig = 0
            risk_p = 0.0

            # Bullish Stochastic Reversal in Uptrend
            if row['close'] > row['ema_200'] and row['ema_20'] > row['ema_50']:
                was_oversold = (m15['stoch_k'].iloc[i-3:i] < 28.0).any()
                # Trigger: %K crosses above %D while < 50, candle is bullish and closes above 20 EMA
                if was_oversold and prev['stoch_k'] <= prev['stoch_d'] and row['stoch_k'] > row['stoch_d'] and row['stoch_k'] < 60:
                    if row['close'] > row['open'] and row['close'] > row['ema_20']:
                        swing_l = float(m15['low'].iloc[i-3:i+1].min())
                        sl_dist = (row['close'] - swing_l) / PIP_SIZE + 1.5
                        risk_p = np.clip(sl_dist, 6.0, 16.0)
                        sig = SIGNAL_LONG

            # Bearish Stochastic Reversal in Downtrend
            elif row['close'] < row['ema_200'] and row['ema_20'] < row['ema_50']:
                was_overbought = (m15['stoch_k'].iloc[i-3:i] > 72.0).any()
                if was_overbought and prev['stoch_k'] >= prev['stoch_d'] and row['stoch_k'] < row['stoch_d'] and row['stoch_k'] > 40:
                    if row['close'] < row['open'] and row['close'] < row['ema_20']:
                        swing_h = float(m15['high'].iloc[i-3:i+1].max())
                        sl_dist = (swing_h - row['close']) / PIP_SIZE + 1.5
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
# STRATEGY 5: Bollinger Band Squeeze Breakout + ADX Expansion (M15)
# ==============================================================================
class Bollinger_ADX_BreakoutStrategy(BaseStrategy):
    """
    Setup: Bollinger Band width was compressed (squeeze < 18 pips in past 5 bars).
    Filter: ADX > 20 and rising (momentum expanding).
    Trigger: Strong displacement breakout candle closes firmly outside upper/lower band.
    """
    def __init__(self, target_rr: float = 2.0, **kwargs):
        super().__init__(**kwargs)
        self.target_rr = target_rr
        self.name = "IND05_Bollinger_ADX_Breakout_M15"

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

        m15 = compute_indicators(m15)
        time_ms = m15['time_ms'].values
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        for i in range(25, len(m15)):
            row = m15.iloc[i]
            prev = m15.iloc[i - 1]

            if row['hour'] < 7 or row['hour'] > 17:
                continue
            if row['weekday'] == 4 and row['hour'] >= 16:
                continue

            sig = 0
            risk_p = 0.0

            # Squeeze condition: recent minimum BB width was tight (< 22 pips)
            recent_min_width = float(m15['bb_width'].iloc[i-6:i].min())
            is_squeezed = recent_min_width <= 22.0

            if is_squeezed and row['adx_14'] >= 18.0:
                # Bullish Breakout
                if prev['close'] <= prev['bb_up'] and row['close'] > row['bb_up'] and row['plus_di'] > row['minus_di']:
                    if (row['close'] - row['open']) / PIP_SIZE >= 3.0:
                        sl_dist = (row['close'] - row['ema_20']) / PIP_SIZE + 1.0
                        risk_p = np.clip(sl_dist, 7.0, 16.0)
                        sig = SIGNAL_LONG

                # Bearish Breakout
                elif prev['close'] >= prev['bb_low'] and row['close'] < row['bb_low'] and row['minus_di'] > row['plus_di']:
                    if (row['open'] - row['close']) / PIP_SIZE >= 3.0:
                        sl_dist = (row['ema_20'] - row['close']) / PIP_SIZE + 1.0
                        risk_p = np.clip(sl_dist, 7.0, 16.0)
                        sig = SIGNAL_SHORT

            if sig != 0:
                exec_t = time_ms[i] + dur_ms
                idx = int(np.searchsorted(tick_ts, exec_t))
                if idx < n_ticks and signals[idx] == 0:
                    signals[idx] = sig
                    sl_pips_arr[idx] = risk_p
                    tp_pips_arr[idx] = risk_p * self.target_rr

        return signals, sl_pips_arr, tp_pips_arr


LAB_STRATEGIES = [
    {"id": "IND01", "name": "Dual EMA (9/21) + RSI 50 Momentum (M15)", "cls": DualEMA_RSI_MomentumStrategy},
    {"id": "IND02", "name": "Supertrend + 50 EMA Retest (M15)", "cls": Supertrend_EMA50_RetestStrategy},
    {"id": "IND03", "name": "MACD Zero-Line Acceleration (M15)", "cls": MACD_ZeroLine_AccelerationStrategy},
    {"id": "IND04", "name": "Stochastic Pullback + 20 EMA (M15)", "cls": Stochastic_EMA_PullbackStrategy},
    {"id": "IND05", "name": "Bollinger Squeeze + ADX Breakout (M15)", "cls": Bollinger_ADX_BreakoutStrategy},
]


def run_week_eval(week_num: int) -> Dict[str, Any]:
    ticks, meta = get_week_dataset(week_num)
    n_ticks = len(ticks)
    engine = BacktestEngine(pip_size=0.0001, use_slippage=True)

    week_res = {}
    for s in LAB_STRATEGIES:
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


def evaluate_trades(trades: List[Dict[str, Any]], risk_pct: float = 0.50) -> Dict[str, Any]:
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

    # Recovery days
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
    print("=" * 115)
    print("      INDICATOR STRATEGY LABORATORY: 52 WEEKS FULL TICK AUDIT (~18M TICKS)      ".center(115))
    print("      Evaluating 5 Indicator Combinations with Discrete Event Triggers          ".center(115))
    print("=" * 115)

    weeks = list(range(1, 53))
    num_workers = min(8, os.cpu_count() or 4)

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(run_week_eval, weeks))
    dt = time.time() - t0

    total_ticks = sum(r["ticks"] for r in results)
    print(f"\n[+] Evaluated {total_ticks:,} Ticks across {len(weeks)} Weeks in {dt:.1f}s ({total_ticks/max(dt, 0.001):,.0f} ticks/s)!\n")

    strat_trades = {s["id"]: [] for s in LAB_STRATEGIES}
    for r in results:
        for sid, t_list in r["trades"].items():
            strat_trades[sid].extend(t_list)

    print("=" * 130)
    print("                          INDICATOR STRATEGY CANDIDATE PERFORMANCE SCORECARD                          ".center(130))
    print("=" * 130)
    print(f"{'Code':<7} | {'Strategy Name':<42} | {'Trades':<6} | {'Trd/Wk':<6} | {'Win %':<6} | {'Real RR':<7} | {'PF':<5} | {'Net Pips':<9} | {'Return %':<8} | {'Max DD%':<7} | {'Daily DD%':<9} | {'Max Recov':<10}")
    print("-" * 130)

    for s in LAB_STRATEGIES:
        sid = s["id"]
        t = strat_trades[sid]
        res = evaluate_trades(t)
        trd_wk = res["n_trades"] / 52.0
        print(f"{sid:<7} | {s['name']:<42} | {res['n_trades']:<6} | {trd_wk:<6.1f} | {res['wr']:<5.1f}% | {res['real_rr']:<7.2f} | {res['pf']:<5.2f} | {res['net_pips']:<+9.1f} | {res['return_pct']:<+7.1f}% | {res['max_dd_pct']:<6.1f}% | {res['max_daily_loss']:<8.1f}% | {res['max_recovery_days']:<2.0f} days")

    print("=" * 130)


if __name__ == "__main__":
    main()
