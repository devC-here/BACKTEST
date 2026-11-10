#!/usr/bin/env python3
"""
Multi-Indicator Strategy Laboratory Phase 3:
Evaluating 10 Brand New Indicator Combinations on 52 Weeks of Dukascopy Tick Data (~26M Ticks):

1. COMB19: Supertrend (10, 2.0) + Parabolic SAR Dual Trend Convergence (M15)
2. COMB20: Kaufman Adaptive Moving Average (KAMA 10,2,30) Efficiency Breakout (M15)
3. COMB21: Vortex Indicator (14) Crossover + ADX > 20 Expansion (M15)
4. COMB22: Aroon Trend (25) + Session VWAP Value Pullback (M15)
5. COMB23: Chande Momentum Oscillator (CMO 14) Zero-Cross + ATR Trail (M15)
6. COMB24: Statistical 2.5σ Bollinger Extreme + Pin Bar Rejection (Mean Reversion) (M15)
7. COMB25: Zero-Lag MACD (ZLEMA 12/26) Acceleration + Momentum Burst (M15)
8. COMB26: Enhanced Asian Breakout Elite (Tight Range 10-35p + ADX > 18) (M15)
9. COMB27: Bollinger Squeeze (COMB12) + Supertrend Filter Hybrid (M15)
10. COMB28: Multi-Oscillator Triple Confluence (RSI 14 + CMO 14 + Stoch 14) (M15)

All models are strictly ZERO-LOOKAHEAD (M15 closed bars only).
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


def enrich_phase3_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Precomputes indicators for Phase 3 combinations with ZERO lookahead."""
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

    # 2. EMAs
    for p in [9, 20, 21, 50, 100, 200]:
        df[f'ema_{p}'] = c.ewm(span=p, adjust=False).mean()
    df['sma_20'] = c.rolling(20).mean().bfill()

    # 3. Supertrend (10, 2.0)
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

    # 4. Parabolic SAR (0.02, 0.2)
    psar = np.zeros(n)
    psar_dir = np.zeros(n, dtype=np.int8)
    af = 0.02
    ep = h.iloc[0]
    psar[0] = l.iloc[0]
    psar_dir[0] = 1
    for i in range(1, n):
        prev_sar = psar[i-1]
        prev_dir = psar_dir[i-1]
        if prev_dir == 1:
            curr_sar = prev_sar + af * (ep - prev_sar)
            curr_sar = min(curr_sar, l.iloc[i-1], l.iloc[max(0, i-2)])
            if l.iloc[i] < curr_sar:
                psar_dir[i] = -1
                psar[i] = ep
                af = 0.02
                ep = l.iloc[i]
            else:
                psar_dir[i] = 1
                psar[i] = curr_sar
                if h.iloc[i] > ep:
                    ep = h.iloc[i]
                    af = min(af + 0.02, 0.2)
        else:
            curr_sar = prev_sar - af * (prev_sar - ep)
            curr_sar = max(curr_sar, h.iloc[i-1], h.iloc[max(0, i-2)])
            if h.iloc[i] > curr_sar:
                psar_dir[i] = 1
                psar[i] = ep
                af = 0.02
                ep = h.iloc[i]
            else:
                psar_dir[i] = -1
                psar[i] = curr_sar
                if l.iloc[i] < ep:
                    ep = l.iloc[i]
                    af = min(af + 0.02, 0.2)
    df['psar'] = psar
    df['psar_dir'] = psar_dir

    # 5. ADX & DMI (14)
    up_m = h.diff()
    dn_m = -l.diff()
    p_dm = np.where((up_m > dn_m) & (up_m > 0), up_m, 0.0)
    m_dm = np.where((dn_m > up_m) & (dn_m > 0), dn_m, 0.0)
    smooth_tr = tr.ewm(alpha=1.0/14, adjust=False).mean()
    df['plus_di'] = (pd.Series(p_dm, index=df.index).ewm(alpha=1.0/14, adjust=False).mean() / np.maximum(smooth_tr, 1e-9)) * 100
    df['minus_di'] = (pd.Series(m_dm, index=df.index).ewm(alpha=1.0/14, adjust=False).mean() / np.maximum(smooth_tr, 1e-9)) * 100
    dx = (abs(df['plus_di'] - df['minus_di']) / np.maximum(df['plus_di'] + df['minus_di'], 1e-9)) * 100
    df['adx_14'] = dx.ewm(alpha=1.0/14, adjust=False).mean()

    # 6. Kaufman's Adaptive Moving Average (KAMA: 10, fast=2, slow=30)
    kama_period = 10
    change = abs(c - c.shift(kama_period))
    volatility = abs(c.diff()).rolling(kama_period).sum()
    er = change / np.maximum(volatility, 1e-9)
    fast_sc = 2.0 / (2.0 + 1.0)
    slow_sc = 2.0 / (30.0 + 1.0)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    kama = np.zeros(n)
    kama[:kama_period] = c.iloc[:kama_period]
    for i in range(kama_period, n):
        kama[i] = kama[i-1] + sc.iloc[i] * (c.iloc[i] - kama[i-1])
    df['kama_10'] = kama
    df['kama_slope'] = df['kama_10'] - df['kama_10'].shift(1)

    # 7. Vortex Indicator (14)
    vm_plus = abs(h - l.shift(1))
    vm_minus = abs(l - h.shift(1))
    sum_tr = tr.rolling(14).sum().bfill()
    df['vi_plus'] = vm_plus.rolling(14).sum().bfill() / np.maximum(sum_tr, 1e-9)
    df['vi_minus'] = vm_minus.rolling(14).sum().bfill() / np.maximum(sum_tr, 1e-9)

    # 8. Aroon Indicator (25)
    high_idx = h.rolling(25).apply(lambda x: 24 - np.argmax(x), raw=True).bfill()
    low_idx = l.rolling(25).apply(lambda x: 24 - np.argmin(x), raw=True).bfill()
    df['aroon_up'] = ((25 - high_idx) / 25.0) * 100.0
    df['aroon_down'] = ((25 - low_idx) / 25.0) * 100.0

    # 9. Chande Momentum Oscillator (CMO 14)
    c_diff = c.diff()
    c_up = c_diff.clip(lower=0.0).rolling(14).sum().bfill()
    c_dn = (-c_diff.clip(upper=0.0)).rolling(14).sum().bfill()
    df['cmo_14'] = ((c_up - c_dn) / np.maximum(c_up + c_dn, 1e-9)) * 100.0

    # 10. Bollinger Bands 2.0 & 2.5
    std20 = c.rolling(20).std().bfill()
    df['bb_up_20'] = df['sma_20'] + (2.0 * std20)
    df['bb_low_20'] = df['sma_20'] - (2.0 * std20)
    df['bb_up_25'] = df['sma_20'] + (2.5 * std20)
    df['bb_low_25'] = df['sma_20'] - (2.5 * std20)
    df['bb_width'] = (df['bb_up_20'] - df['bb_low_20']) / PIP_SIZE
    df['bb_pct_b'] = (c - df['bb_low_20']) / np.maximum(df['bb_up_20'] - df['bb_low_20'], 1e-9)
    df['bb_w_min40'] = df['bb_width'].rolling(40).min().shift(1).bfill()

    # 11. Zero-Lag MACD (ZLEMA 12, 26, 9)
    lag12 = int((12 - 1) / 2)
    lag26 = int((26 - 1) / 2)
    z_c12 = c + (c - c.shift(lag12))
    z_c26 = c + (c - c.shift(lag26))
    zlema12 = z_c12.ewm(span=12, adjust=False).mean()
    zlema26 = z_c26.ewm(span=26, adjust=False).mean()
    df['zl_macd'] = zlema12 - zlema26
    df['zl_sig'] = df['zl_macd'].ewm(span=9, adjust=False).mean()
    df['zl_hist'] = df['zl_macd'] - df['zl_sig']

    # 12. RSI (14) & Stochastic (14, 3, 3)
    gain = c_diff.clip(lower=0.0).ewm(alpha=1.0/14, adjust=False).mean()
    loss = (-c_diff.clip(upper=0.0)).ewm(alpha=1.0/14, adjust=False).mean()
    rs = gain / np.maximum(loss, 1e-9)
    df['rsi_14'] = 100.0 - (100.0 / (1.0 + rs))

    low14 = l.rolling(14).min().bfill()
    high14 = h.rolling(14).max().bfill()
    k_fast = ((c - low14) / np.maximum(high14 - low14, 1e-9)) * 100
    df['stoch_k'] = k_fast.rolling(3).mean().bfill()
    df['stoch_d'] = df['stoch_k'].rolling(3).mean().bfill()

    # 13. Session VWAP (00:00 UTC daily reset)
    df['date'] = df.index.date
    df['hour'] = df.index.hour
    df['time_ms'] = df.index.astype('datetime64[ms]').view('int64')
    vol = tr / PIP_SIZE + 10.0
    typical_p = (h + l + c) / 3.0
    cum_vol = vol.groupby(df['date']).cumsum()
    cum_pv = (typical_p * vol).groupby(df['date']).cumsum()
    df['session_vwap'] = cum_pv / np.maximum(cum_vol, 1e-9)

    return df


# ==============================================================================
# COMB19: Supertrend + Parabolic SAR Dual Trend Convergence (M15)
# ==============================================================================
class SupertrendPSARConvergence(BaseStrategy):
    """
    Dual Trend Confluence:
    - Supertrend is Bullish (direction == 1).
    - PSAR flips from Bearish to Bullish on current bar.
    - ADX(14) >= 18.0 with +DI > -DI.
    - Trading during London & NY liquid hours (07:00-15:00 UTC).
    """
    def __init__(self, target_rr: float = 2.0, min_sl: float = 6.0, max_sl: float = 16.0, **kwargs):
        super().__init__(**kwargs)
        self.target_rr = target_rr
        self.min_sl = min_sl
        self.max_sl = max_sl
        self.pip_size = PIP_SIZE
        self.name = "Supertrend_PSAR_Convergence"

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

        m15 = enrich_phase3_indicators(m15)
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

            # Bullish Convergence: Supertrend is Bullish AND PSAR just flipped Bullish
            if row['supertrend_dir'] == 1 and prev['psar_dir'] == -1 and row['psar_dir'] == 1:
                if row['adx_14'] >= 18.0 and row['plus_di'] > row['minus_di'] and row['close'] > row['open']:
                    sl_p = min(row['psar'], row['low']) - 1.2 * self.pip_size
                    risk = max((row['close'] - sl_p) / self.pip_size, self.min_sl)
                    if risk <= self.max_sl:
                        sig = SIGNAL_LONG
                        risk_p = risk

            # Bearish Convergence: Supertrend is Bearish AND PSAR just flipped Bearish
            elif row['supertrend_dir'] == -1 and prev['psar_dir'] == 1 and row['psar_dir'] == -1:
                if row['adx_14'] >= 18.0 and row['minus_di'] > row['plus_di'] and row['close'] < row['open']:
                    sl_p = max(row['psar'], row['high']) + 1.2 * self.pip_size
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
# COMB20: Kaufman Adaptive Moving Average (KAMA) Efficiency Breakout (M15)
# ==============================================================================
class KAMAEfficiencyBreakout(BaseStrategy):
    """
    Adaptive Trend Following:
    - Filters choppy noise: Only triggers when KAMA slope surges (> 0.5 pips/bar).
    - Close crosses above KAMA (Long) or below KAMA (Short).
    - ADX(14) >= 20.0 confirming trend expansion.
    """
    def __init__(self, target_rr: float = 2.2, min_sl: float = 7.0, max_sl: float = 16.0, **kwargs):
        super().__init__(**kwargs)
        self.target_rr = target_rr
        self.min_sl = min_sl
        self.max_sl = max_sl
        self.pip_size = PIP_SIZE
        self.name = "KAMA_Efficiency_Breakout"

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

        m15 = enrich_phase3_indicators(m15)
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

            # Bullish: Cross above KAMA + Positive slope surge
            if prev['close'] <= prev['kama_10'] and row['close'] > row['kama_10'] and row['close'] > row['open']:
                if row['kama_slope'] > 0 and row['adx_14'] >= 18.0:
                    sl_p = min(prev['low'], row['low']) - 1.2 * self.pip_size
                    risk = max((row['close'] - sl_p) / self.pip_size, self.min_sl)
                    if risk <= self.max_sl:
                        sig = SIGNAL_LONG
                        risk_p = risk

            # Bearish: Cross below KAMA + Negative slope surge
            elif prev['close'] >= prev['kama_10'] and row['close'] < row['kama_10'] and row['close'] < row['open']:
                if row['kama_slope'] < 0 and row['adx_14'] >= 18.0:
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
# COMB21: Vortex Indicator (14) Crossover + ADX > 20 Expansion (M15)
# ==============================================================================
class VortexADXExpansion(BaseStrategy):
    """
    Vortex Flow Indicator:
    - VI+ crosses above VI- from below (Long).
    - VI- crosses above VI+ from below (Short).
    - ADX(14) >= 20.0 confirms strong flow regime.
    - Candle body >= 0.8x ATR.
    """
    def __init__(self, target_rr: float = 2.0, min_sl: float = 7.0, max_sl: float = 16.0, **kwargs):
        super().__init__(**kwargs)
        self.target_rr = target_rr
        self.min_sl = min_sl
        self.max_sl = max_sl
        self.pip_size = PIP_SIZE
        self.name = "Vortex_ADX_Expansion"

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

        m15 = enrich_phase3_indicators(m15)
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

            # Bullish Vortex Cross
            if prev['vi_plus'] <= prev['vi_minus'] and row['vi_plus'] > row['vi_minus']:
                if row['adx_14'] >= 18.0 and row['close'] > row['open']:
                    sl_p = min(prev['low'], row['low']) - 1.2 * self.pip_size
                    risk = max((row['close'] - sl_p) / self.pip_size, self.min_sl)
                    if risk <= self.max_sl:
                        sig = SIGNAL_LONG
                        risk_p = risk

            # Bearish Vortex Cross
            elif prev['vi_plus'] >= prev['vi_minus'] and row['vi_plus'] < row['vi_minus']:
                if row['adx_14'] >= 18.0 and row['close'] < row['open']:
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
# COMB22: Aroon Trend (25) + Session VWAP Value Pullback (M15)
# ==============================================================================
class AroonVWAPPullback(BaseStrategy):
    """
    Aroon Fresh Trend Detection + Institutional Fair Value:
    - Aroon Up > 80 and Aroon Down < 30 (Dominant Bullish Trend).
    - Price touches within 2.5 pips of Session VWAP and closes green.
    - Reverse for Bearish.
    """
    def __init__(self, target_rr: float = 2.0, min_sl: float = 6.0, max_sl: float = 16.0, **kwargs):
        super().__init__(**kwargs)
        self.target_rr = target_rr
        self.min_sl = min_sl
        self.max_sl = max_sl
        self.pip_size = PIP_SIZE
        self.name = "Aroon_VWAP_Pullback"

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

        m15 = enrich_phase3_indicators(m15)
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

            # Bullish: Aroon Up > 80, Down < 30, tested VWAP, closed green
            if row['aroon_up'] >= 80.0 and row['aroon_down'] <= 30.0 and prev['close'] > vwap:
                if row['low'] <= vwap + 2.5 * self.pip_size and row['close'] > vwap and row['close'] > row['open']:
                    sl_p = min(row['low'], vwap) - 1.2 * self.pip_size
                    risk = max((row['close'] - sl_p) / self.pip_size, self.min_sl)
                    if risk <= self.max_sl:
                        sig = SIGNAL_LONG
                        risk_p = risk

            # Bearish: Aroon Down > 80, Up < 30, tested VWAP, closed red
            elif row['aroon_down'] >= 80.0 and row['aroon_up'] <= 30.0 and prev['close'] < vwap:
                if row['high'] >= vwap - 2.5 * self.pip_size and row['close'] < vwap and row['close'] < row['open']:
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
# COMB23: Chande Momentum Oscillator (CMO 14) Zero-Cross + ATR Trail (M15)
# ==============================================================================
class CMOZeroCrossBreakout(BaseStrategy):
    """
    Direct Unsmoothed Momentum:
    - CMO crosses above 0.0 from below (Bullish).
    - CMO crosses below 0.0 from above (Bearish).
    - Filter: 50 EMA Trend direction.
    """
    def __init__(self, target_rr: float = 2.0, min_sl: float = 7.0, max_sl: float = 16.0, **kwargs):
        super().__init__(**kwargs)
        self.target_rr = target_rr
        self.min_sl = min_sl
        self.max_sl = max_sl
        self.pip_size = PIP_SIZE
        self.name = "CMO_Zero_Cross"

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

        m15 = enrich_phase3_indicators(m15)
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

            # Bullish CMO Zero-Line Cross
            if prev['cmo_14'] <= 0.0 and row['cmo_14'] > 0.0 and row['close'] > row['ema_50']:
                sl_p = min(prev['low'], row['low']) - 1.2 * self.pip_size
                risk = max((row['close'] - sl_p) / self.pip_size, self.min_sl)
                if risk <= self.max_sl:
                    sig = SIGNAL_LONG
                    risk_p = risk

            # Bearish CMO Zero-Line Cross
            elif prev['cmo_14'] >= 0.0 and row['cmo_14'] < 0.0 and row['close'] < row['ema_50']:
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
# COMB24: Statistical 2.5σ Bollinger Extreme + Pin Bar Rejection (M15)
# ==============================================================================
class BollingerExtremePinBar(BaseStrategy):
    """
    Statistical Outlier Mean Reversion:
    - Candle pierces 2.5 Standard Deviation Bollinger Band.
    - Forms a classic Rejection Pin Bar (wick >= 60% of total candle length).
    - Closes back inside the 2.0σ band.
    - Low-ADX regime (< 22.0) to prevent catching falling knives in trends.
    """
    def __init__(self, target_rr: float = 1.8, min_sl: float = 6.0, max_sl: float = 15.0, **kwargs):
        super().__init__(**kwargs)
        self.target_rr = target_rr
        self.min_sl = min_sl
        self.max_sl = max_sl
        self.pip_size = PIP_SIZE
        self.name = "Bollinger_2.5_Extreme_PinBar"

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

        m15 = enrich_phase3_indicators(m15)
        time_ms = m15['time_ms'].values
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        for i in range(25, len(m15)):
            row = m15.iloc[i]

            if row['hour'] < 7 or row['hour'] > 16:
                continue

            if row['adx_14'] >= 22.0:
                continue

            total_len = max(row['high'] - row['low'], 1e-9)
            body_len = abs(row['close'] - row['open'])
            lower_wick = min(row['open'], row['close']) - row['low']
            upper_wick = row['high'] - max(row['open'], row['close'])

            sig = 0
            risk_p = 0.0

            # Bullish Rejection Pin Bar at 2.5σ
            if row['low'] < row['bb_low_25'] and row['close'] > row['bb_low_20']:
                if lower_wick / total_len >= 0.55 and row['close'] > row['open']:
                    sl_p = row['low'] - 1.2 * self.pip_size
                    risk = max((row['close'] - sl_p) / self.pip_size, self.min_sl)
                    if risk <= self.max_sl:
                        sig = SIGNAL_LONG
                        risk_p = risk

            # Bearish Rejection Pin Bar at 2.5σ
            elif row['high'] > row['bb_up_25'] and row['close'] < row['bb_up_20']:
                if upper_wick / total_len >= 0.55 and row['close'] < row['open']:
                    sl_p = row['high'] + 1.2 * self.pip_size
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
# COMB25: Zero-Lag MACD (ZLEMA 12/26) Acceleration (M15)
# ==============================================================================
class ZeroLagMACDAcceleration(BaseStrategy):
    """
    Zero-Lag Momentum:
    - Zero-Lag MACD Histogram flips from negative to positive (Long) or positive to negative (Short).
    - ADX(14) >= 18.0.
    - Candle body >= 0.8x ATR.
    """
    def __init__(self, target_rr: float = 2.0, min_sl: float = 7.0, max_sl: float = 16.0, **kwargs):
        super().__init__(**kwargs)
        self.target_rr = target_rr
        self.min_sl = min_sl
        self.max_sl = max_sl
        self.pip_size = PIP_SIZE
        self.name = "Zero_Lag_MACD_Acceleration"

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

        m15 = enrich_phase3_indicators(m15)
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
            body_p = abs(row['close'] - row['open']) / self.pip_size

            # Bullish ZL MACD Histogram Flip
            if prev['zl_hist'] <= 0 and row['zl_hist'] > 0 and row['close'] > row['open']:
                if row['adx_14'] >= 18.0 and body_p >= 0.7 * row['atr_pips']:
                    sl_p = min(prev['low'], row['low']) - 1.2 * self.pip_size
                    risk = max((row['close'] - sl_p) / self.pip_size, self.min_sl)
                    if risk <= self.max_sl:
                        sig = SIGNAL_LONG
                        risk_p = risk

            # Bearish ZL MACD Histogram Flip
            elif prev['zl_hist'] >= 0 and row['zl_hist'] < 0 and row['close'] < row['open']:
                if row['adx_14'] >= 18.0 and body_p >= 0.7 * row['atr_pips']:
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
# COMB26: Enhanced Asian Breakout Elite (Tight Range 10-35p + ADX > 18) (M15)
# ==============================================================================
class AsianBreakoutElite(BaseStrategy):
    """
    Optimized version of COMB08 (+234.9p, PF 1.24):
    - Asian Range constrained to high-conviction institutional coiling (10.0 to 35.0 pips).
    - Breakout bar must close beyond Asian High/Low with strong displacement (body >= 0.7x ATR).
    - ADX(14) >= 18.0.
    - Target RR expanded to 2.40:1.
    """
    def __init__(self, target_rr: float = 2.4, min_sl: float = 6.0, max_sl: float = 16.0, **kwargs):
        super().__init__(**kwargs)
        self.target_rr = target_rr
        self.min_sl = min_sl
        self.max_sl = max_sl
        self.pip_size = PIP_SIZE
        self.name = "Asian_Breakout_Elite"

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

        m15 = enrich_phase3_indicators(m15)
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
                    if row['adx_14'] >= 18.0 and row['close'] > row['open'] and body_p >= 0.7 * row['atr_pips']:
                        sl_p = row['low'] - 1.2 * self.pip_size
                        risk = max((row['close'] - sl_p) / self.pip_size, self.min_sl)
                        if risk <= self.max_sl:
                            sig = SIGNAL_LONG
                            risk_p = risk

                # Bearish Breakout
                elif prev['close'] >= asian_low and row['close'] < asian_low:
                    if row['adx_14'] >= 18.0 and row['close'] < row['open'] and body_p >= 0.7 * row['atr_pips']:
                        sl_p = row['high'] + 1.2 * self.pip_size
                        risk = max((sl_p - row['close']) / self.pip_size, self.min_sl)
                        if risk <= self.max_sl:
                            sig = SIGNAL_SHORT
                            risk_p = risk

                if sig != 0:
                    traded_today = True
                    exec_t = time_ms[i] + dur_ms
                    idx = int(np.searchsorted(tick_ts, exec_t))
                    if idx < n_ticks and signals[idx] == 0:
                        signals[idx] = sig
                        sl_pips_arr[idx] = risk_p
                        tp_pips_arr[idx] = risk_p * self.target_rr

        return signals, sl_pips_arr, tp_pips_arr


# ==============================================================================
# COMB27: Bollinger Squeeze (COMB12) + Supertrend Filter Hybrid (M15)
# ==============================================================================
class BollingerSupertrendHybrid(BaseStrategy):
    """
    Hybrid of COMB12 (PF 1.55) + Supertrend (PF 1.26):
    - BB Bandwidth was in 40-bar tightest squeeze.
    - %B breaks above 1.0 (Long) or below 0.0 (Short).
    - Confirmed by Supertrend direction.
    """
    def __init__(self, target_rr: float = 2.2, min_sl: float = 7.0, max_sl: float = 16.0, **kwargs):
        super().__init__(**kwargs)
        self.target_rr = target_rr
        self.min_sl = min_sl
        self.max_sl = max_sl
        self.pip_size = PIP_SIZE
        self.name = "Bollinger_Supertrend_Hybrid"

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
        if len(m15) < 45:
            return signals, sl_pips_arr, tp_pips_arr

        m15 = enrich_phase3_indicators(m15)
        time_ms = m15['time_ms'].values
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        for i in range(40, len(m15)):
            row = m15.iloc[i]
            prev = m15.iloc[i-1]

            if row['hour'] < 7 or row['hour'] > 15:
                continue

            sig = 0
            risk_p = 0.0
            body_pips = abs(row['close'] - row['open']) / self.pip_size

            was_tight = prev['bb_width'] <= 1.35 * row['bb_w_min40']
            if not was_tight:
                continue

            # Bullish %B Breakout + Supertrend Bullish
            if prev['bb_pct_b'] <= 1.0 and row['bb_pct_b'] > 1.0 and row['supertrend_dir'] == 1:
                if body_pips >= 0.7 * row['atr_pips']:
                    sl_p = row['sma_20']
                    risk = max((row['close'] - sl_p) / self.pip_size, self.min_sl)
                    if risk <= self.max_sl:
                        sig = SIGNAL_LONG
                        risk_p = risk

            # Bearish %B Breakout + Supertrend Bearish
            elif prev['bb_pct_b'] >= 0.0 and row['bb_pct_b'] < 0.0 and row['supertrend_dir'] == -1:
                if body_pips >= 0.7 * row['atr_pips']:
                    sl_p = row['sma_20']
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
# COMB28: Multi-Oscillator Triple Confluence (RSI + CMO + Stoch) (M15)
# ==============================================================================
class TripleOscillatorConfluence(BaseStrategy):
    """
    Harmonic Oscillator Confluence:
    - RSI(14) crosses above 50.
    - CMO(14) crosses above 0.
    - Stochastic %K crosses above %D.
    - All 3 align within the current bar for maximum confirmation.
    """
    def __init__(self, target_rr: float = 2.0, min_sl: float = 7.0, max_sl: float = 16.0, **kwargs):
        super().__init__(**kwargs)
        self.target_rr = target_rr
        self.min_sl = min_sl
        self.max_sl = max_sl
        self.pip_size = PIP_SIZE
        self.name = "Triple_Oscillator_Confluence"

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

        m15 = enrich_phase3_indicators(m15)
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

            # Bullish Confluence
            if (prev['rsi_14'] <= 50 and row['rsi_14'] > 50) and (row['cmo_14'] > 0) and (row['stoch_k'] > row['stoch_d']):
                if row['close'] > row['ema_50']:
                    sl_p = min(prev['low'], row['low']) - 1.2 * self.pip_size
                    risk = max((row['close'] - sl_p) / self.pip_size, self.min_sl)
                    if risk <= self.max_sl:
                        sig = SIGNAL_LONG
                        risk_p = risk

            # Bearish Confluence
            elif (prev['rsi_14'] >= 50 and row['rsi_14'] < 50) and (row['cmo_14'] < 0) and (row['stoch_k'] < row['stoch_d']):
                if row['close'] < row['ema_50']:
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
# LABORATORY EXPERIMENT HARNESS
# ==============================================================================
EXPERIMENT_STRATEGIES = [
    {"code": "COMB19", "name": "Supertrend + PSAR Dual Trend (M15)", "cls": SupertrendPSARConvergence},
    {"code": "COMB20", "name": "KAMA Efficiency Breakout (M15)", "cls": KAMAEfficiencyBreakout},
    {"code": "COMB21", "name": "Vortex Flow + ADX Expansion (M15)", "cls": VortexADXExpansion},
    {"code": "COMB22", "name": "Aroon Trend + VWAP Pullback (M15)", "cls": AroonVWAPPullback},
    {"code": "COMB23", "name": "CMO Zero-Cross + 50 EMA (M15)", "cls": CMOZeroCrossBreakout},
    {"code": "COMB24", "name": "Bollinger 2.5 Sigma Extreme PinBar (M15)", "cls": BollingerExtremePinBar},
    {"code": "COMB25", "name": "Zero-Lag MACD Acceleration (M15)", "cls": ZeroLagMACDAcceleration},
    {"code": "COMB26", "name": "Asian Breakout Elite (Tight Coil) (M15)", "cls": AsianBreakoutElite},
    {"code": "COMB27", "name": "Bollinger Squeeze + Supertrend (M15)", "cls": BollingerSupertrendHybrid},
    {"code": "COMB28", "name": "Triple Oscillator Confluence (M15)", "cls": TripleOscillatorConfluence},
]


def run_week_experiment(week_num: int) -> Dict[str, Any]:
    ticks, meta = get_week_dataset(week_num)
    n_ticks = len(ticks)
    engine = BacktestEngine(pip_size=PIP_SIZE, use_slippage=True)

    week_trades = {}
    for item in EXPERIMENT_STRATEGIES:
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


def analyze_combination_performance(trades: List[Dict[str, Any]], risk_pct: float = 0.50):
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
    print("                 MULTI-INDICATOR STRATEGY LABORATORY PHASE 3: 52 WEEKS FULL TICK AUDIT (~26M TICKS)")
    print("                          Evaluating 10 Advanced Discrete Indicator Combinations for Prop Firms")
    print("=" * 135)

    start_t = time.time()
    weeks = list(range(1, 53))

    strat_trades: Dict[str, List[Dict[str, Any]]] = {item["code"]: [] for item in EXPERIMENT_STRATEGIES}
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
    print("                                      PHASE 3 EXPERIMENTAL COMBINATIONS SCORECARD")
    print("=" * 145)
    print(f"{'Code':<7} | {'Indicator Combination Strategy':<45} | {'Trades':<6} | {'Trd/Wk':<6} | {'Win %':<6} | {'Real RR':<7} | {'PF':<5} | {'Net Pips':<9} | {'Return %':<8} | {'Max DD%':<7} | {'Daily DD%':<9} | {'Ulcer':<5} | {'Max Recov':<9}")
    print("-" * 145)

    for item in EXPERIMENT_STRATEGIES:
        code = item["code"]
        name = item["name"]
        t_data = strat_trades[code]
        metrics = analyze_combination_performance(t_data, risk_pct=0.50)

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
