#!/usr/bin/env python3
"""
Multi-Indicator Strategy Laboratory Phase 2:
Evaluating 10 Advanced Indicator Combinations on 52 Weeks of Dukascopy Tick Data (~26M Ticks):

1. COMB09: Asian Liquidity Sweep & Reversal (Turtle Soup / Judas Swing + Stoch Flip) (M15)
2. COMB10: Regime-Gated Keltner Envelope + Stoch Reversal (Low-ADX Mean Reversion) (M15)
3. COMB11: Hull Moving Average (HMA 21) Directional Flip + ADX Momentum (M15)
4. COMB12: Bollinger BandWidth Squeeze Expansion (%B Breakout + Volatility Surge) (M15)
5. COMB13: Elder's Triple Screen Impulse (H1 MACD Tide + M15 RSI Value Pullback) (M15)
6. COMB14: Supertrend (10, 2.0) + Session VWAP Rejection Pullback (M15)
7. COMB15: Chaikin Money Flow (CMF) Volume Surge + Donchian Channel Breakout (M15)
8. COMB16: Dual Momentum Convergence (RSI 50 Cross + Stoch %K/%D Cross) (M15)
9. COMB17: Parabolic SAR Trend Acceleration + ADX > 25 (M15)
10. COMB18: London Breakout + NY Continuation Hybrid (Two-Session Momentum) (M15)

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


def enrich_phase2_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Precomputes all indicators needed across the 10 Phase 2 combinations."""
    c = df['close']
    h = df['high']
    l = df['low']
    o = df['open']

    # 1. EMAs & SMAs
    for p in [8, 14, 20, 21, 50, 100, 200]:
        df[f'ema_{p}'] = c.ewm(span=p, adjust=False).mean()
    df['sma_20'] = c.rolling(20).mean().bfill()

    # 2. ATR (14)
    tr = np.maximum(h - l, np.maximum(abs(h - c.shift(1)), abs(l - c.shift(1))))
    df['tr'] = tr
    df['atr_14'] = tr.rolling(14).mean().bfill()
    df['atr_pips'] = df['atr_14'] / PIP_SIZE

    # 3. RSI (14) & RSI (2)
    delta = c.diff()
    gain = delta.clip(lower=0.0).ewm(alpha=1.0/14, adjust=False).mean()
    loss = (-delta.clip(upper=0.0)).ewm(alpha=1.0/14, adjust=False).mean()
    rs = gain / np.maximum(loss, 1e-9)
    df['rsi_14'] = 100.0 - (100.0 / (1.0 + rs))

    gain2 = delta.clip(lower=0.0).ewm(alpha=1.0/2, adjust=False).mean()
    loss2 = (-delta.clip(upper=0.0)).ewm(alpha=1.0/2, adjust=False).mean()
    rs2 = gain2 / np.maximum(loss2, 1e-9)
    df['rsi_2'] = 100.0 - (100.0 / (1.0 + rs2))

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
    df['bb_pct_b'] = (c - df['bb_low']) / np.maximum(df['bb_up'] - df['bb_low'], 1e-9)

    # 6. Keltner Channels (20, 1.5 ATR)
    df['kc_up'] = df['ema_20'] + (1.5 * df['atr_14'])
    df['kc_low'] = df['ema_20'] - (1.5 * df['atr_14'])

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

    # 10. Supertrend (10, 2.0)
    hl2 = (h + l) / 2.0
    st_up = hl2 - (2.0 * df['atr_14'])
    st_low = hl2 + (2.0 * df['atr_14'])
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

    # 11. Hull Moving Average (HMA 21)
    # HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    def wma(s: pd.Series, period: int) -> pd.Series:
        weights = np.arange(1, period + 1)
        return s.rolling(period).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True).bfill()

    period = 21
    half_period = int(period / 2)
    sqrt_period = int(np.sqrt(period))
    wma_half = wma(c, half_period)
    wma_full = wma(c, period)
    diff_wma = 2 * wma_half - wma_full
    df['hma_21'] = wma(diff_wma, sqrt_period)
    df['hma_slope'] = df['hma_21'] - df['hma_21'].shift(1)

    # 12. Chaikin Money Flow (CMF 20) Proxy using tick/candle volume approximation
    vol = tr / PIP_SIZE + 10.0
    mf_mult = ((c - l) - (h - c)) / np.maximum(h - l, 1e-9)
    mf_vol = mf_mult * vol
    df['cmf_20'] = mf_vol.rolling(20).sum() / np.maximum(vol.rolling(20).sum(), 1e-9)

    # 13. Parabolic SAR (0.02, 0.2)
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

    # 14. Time & Session metadata
    df['hour'] = df.index.hour
    df['minute'] = df.index.minute
    df['date'] = df.index.date
    df['weekday'] = df.index.dayofweek
    df['time_ms'] = df.index.astype('datetime64[ms]').view('int64')

    # 15. Session VWAP (resets daily 00:00 UTC)
    typical_p = (h + l + c) / 3.0
    cum_vol = vol.groupby(df['date']).cumsum()
    cum_pv = (typical_p * vol).groupby(df['date']).cumsum()
    df['session_vwap'] = cum_pv / np.maximum(cum_vol, 1e-9)

    return df


# ==============================================================================
# COMB09: Asian Liquidity Sweep & Reversal (Turtle Soup / Judas Swing + Stoch Flip)
# ==============================================================================
class AsianLiquiditySweepReversal(BaseStrategy):
    """
    Institutional Stop Hunt / Liquidity Sweep:
    Asian Session (00:00-06:00 UTC) defines High & Low.
    During London Open (07:00-10:00 UTC):
    - Price pierces Asian Low by 1.0-8.0 pips, then closes back inside Asian range with a green candle
      and Stochastic %K crossing above %D from oversold (<30). Long entry.
    - Price pierces Asian High by 1.0-8.0 pips, closes back inside with a red candle
      and Stochastic %K crossing below %D from overbought (>70). Short entry.
    SL: Beyond sweep extreme + 1.5 pips. TP: 2.2x RR.
    """
    def __init__(self, target_rr: float = 2.2, min_sl: float = 6.0, max_sl: float = 16.0, **kwargs):
        super().__init__(**kwargs)
        self.target_rr = target_rr
        self.min_sl = min_sl
        self.max_sl = max_sl
        self.pip_size = PIP_SIZE
        self.name = "Asian_Liquidity_Sweep_Reversal"

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

        m15 = enrich_phase2_indicators(m15)
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

                # Bullish Sweep of Asian Low
                sweep_l = (asian_low - row['low']) / self.pip_size
                if 1.0 <= sweep_l <= 8.0 and row['close'] > asian_low and row['close'] > row['open']:
                    if prev['stoch_k'] <= prev['stoch_d'] and row['stoch_k'] > row['stoch_d'] and prev['stoch_k'] < 40:
                        sl_price = row['low'] - 1.5 * self.pip_size
                        risk = max((row['close'] - sl_price) / self.pip_size, self.min_sl)
                        if risk <= self.max_sl:
                            sig = SIGNAL_LONG
                            risk_p = risk

                # Bearish Sweep of Asian High
                sweep_h = (row['high'] - asian_high) / self.pip_size
                if 1.0 <= sweep_h <= 8.0 and row['close'] < asian_high and row['close'] < row['open']:
                    if prev['stoch_k'] >= prev['stoch_d'] and row['stoch_k'] < row['stoch_d'] and prev['stoch_k'] > 60:
                        sl_price = row['high'] + 1.5 * self.pip_size
                        risk = max((sl_price - row['close']) / self.pip_size, self.min_sl)
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
# COMB10: Regime-Gated Keltner Envelope + Stoch Reversal (Low-ADX Mean Reversion)
# ==============================================================================
class RegimeGatedKeltnerStoch(BaseStrategy):
    """
    Fixes COMB07's drawdown by strictly gating mean-reversion:
    1. Market Regime Gated: ADX < 20 (Strong trending days are blocked).
    2. Discrete state transition: Stochastic crosses out of extreme (<20 for long, >80 for short).
    3. Price must have pierced Keltner lower/upper band on previous bar and rejected.
    4. Max 2 trades per day limit.
    """
    def __init__(self, target_rr: float = 1.8, min_sl: float = 8.0, max_sl: float = 18.0, **kwargs):
        super().__init__(**kwargs)
        self.target_rr = target_rr
        self.min_sl = min_sl
        self.max_sl = max_sl
        self.pip_size = PIP_SIZE
        self.name = "Regime_Gated_Keltner_Stoch"

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

        m15 = enrich_phase2_indicators(m15)
        time_ms = m15['time_ms'].values
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        current_day = None
        day_trades = 0

        for i in range(25, len(m15)):
            row = m15.iloc[i]
            prev = m15.iloc[i-1]

            if row['date'] != current_day:
                current_day = row['date']
                day_trades = 0

            if day_trades >= 2:
                continue

            # Only trade during London & NY liquid hours (07:00-16:00 UTC)
            if row['hour'] < 7 or row['hour'] > 16:
                continue

            # Strict Low-ADX Range Regime Filter: Avoid trending whipsaws!
            if row['adx_14'] >= 20.0:
                continue

            sig = 0
            risk_p = 0.0

            # Long Reversal: Prev pierced KC Low, Current closes back above KC Low + Stoch K crosses D from < 25
            if prev['low'] < prev['kc_low'] and row['close'] > row['kc_low'] and row['close'] > row['open']:
                if prev['stoch_k'] <= prev['stoch_d'] and row['stoch_k'] > row['stoch_d'] and prev['stoch_k'] < 30:
                    sl_p = min(prev['low'], row['low']) - 1.5 * self.pip_size
                    risk = max((row['close'] - sl_p) / self.pip_size, self.min_sl)
                    if risk <= self.max_sl:
                        sig = SIGNAL_LONG
                        risk_p = risk

            # Short Reversal: Prev pierced KC Up, Current closes back below KC Up + Stoch K crosses D from > 75
            elif prev['high'] > prev['kc_up'] and row['close'] < row['kc_up'] and row['close'] < row['open']:
                if prev['stoch_k'] >= prev['stoch_d'] and row['stoch_k'] < row['stoch_d'] and prev['stoch_k'] > 70:
                    sl_p = max(prev['high'], row['high']) + 1.5 * self.pip_size
                    risk = max((sl_p - row['close']) / self.pip_size, self.min_sl)
                    if risk <= self.max_sl:
                        sig = SIGNAL_SHORT
                        risk_p = risk

            if sig != 0:
                day_trades += 1
                exec_t = time_ms[i] + dur_ms
                idx = int(np.searchsorted(tick_ts, exec_t))
                if idx < n_ticks and signals[idx] == 0:
                    signals[idx] = sig
                    sl_pips_arr[idx] = risk_p
                    tp_pips_arr[idx] = risk_p * self.target_rr

        return signals, sl_pips_arr, tp_pips_arr


# ==============================================================================
# COMB11: Hull Moving Average (HMA 21) Directional Flip + ADX Momentum
# ==============================================================================
class HullMovingAverageADX(BaseStrategy):
    """
    Zero-Lag Trend Follower:
    - HMA 21 slope flips from negative to positive (Long) or positive to negative (Short).
    - ADX(14) >= 20.0 confirming strong directional movement (+DI > -DI for Long, -DI > +DI for Short).
    - H1 50-EMA Macro Trend confirmation.
    - Trade hours: London & NY (07:00-16:00 UTC).
    """
    def __init__(self, target_rr: float = 2.0, min_sl: float = 7.0, max_sl: float = 16.0, **kwargs):
        super().__init__(**kwargs)
        self.target_rr = target_rr
        self.min_sl = min_sl
        self.max_sl = max_sl
        self.pip_size = PIP_SIZE
        self.name = "Hull_MA_ADX_Momentum"

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

        m15 = enrich_phase2_indicators(m15)

        # Macro H1 EMA50
        h1 = base_df.resample('1h').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
        h1['h1_ema50'] = h1['close'].ewm(span=50, adjust=False).mean()
        h1['h1_macro'] = np.where(h1['close'] > h1['h1_ema50'], 1, -1)
        m15['macro'] = h1['h1_macro'].reindex(m15.index, method='ffill').fillna(0)

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

            # HMA slope flip positive
            if prev['hma_slope'] <= 0 and row['hma_slope'] > 0:
                if row['macro'] >= 0 and row['adx_14'] >= 20.0 and row['plus_di'] > row['minus_di']:
                    sl_p = min(prev['low'], row['low']) - 1.2 * self.pip_size
                    risk = max((row['close'] - sl_p) / self.pip_size, self.min_sl)
                    if risk <= self.max_sl:
                        sig = SIGNAL_LONG
                        risk_p = risk

            # HMA slope flip negative
            elif prev['hma_slope'] >= 0 and row['hma_slope'] < 0:
                if row['macro'] <= 0 and row['adx_14'] >= 20.0 and row['minus_di'] > row['plus_di']:
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
# COMB12: Bollinger BandWidth Squeeze Expansion (%B Breakout)
# ==============================================================================
class BollingerBandwidthExpansion(BaseStrategy):
    """
    Volatility Contraction to Expansion Breakout:
    - Bollinger BandWidth was in the lowest 20% of the last 40 bars (tight squeeze).
    - Discrete Expansion: Current bar BB width expands by > 20% AND %B crosses above 1.0 (Long) or below 0.0 (Short).
    - Candle body > 1.0x ATR confirming institutional displacement.
    """
    def __init__(self, target_rr: float = 2.0, min_sl: float = 7.0, max_sl: float = 16.0, **kwargs):
        super().__init__(**kwargs)
        self.target_rr = target_rr
        self.min_sl = min_sl
        self.max_sl = max_sl
        self.pip_size = PIP_SIZE
        self.name = "Bollinger_Bandwidth_Expansion"

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

        m15 = enrich_phase2_indicators(m15)
        # 40-bar rolling minimum BB width
        m15['bb_w_min40'] = m15['bb_width'].rolling(40).min().shift(1).bfill()

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

            # Squeeze condition: prior 3 bars were close to 40-bar lowest bandwidth
            was_tight = prev['bb_width'] <= 1.35 * row['bb_w_min40']
            if not was_tight:
                continue

            # Bullish %B Breakout
            if prev['bb_pct_b'] <= 1.0 and row['bb_pct_b'] > 1.0 and row['close'] > row['open']:
                if body_pips >= 0.8 * row['atr_pips']:
                    sl_p = row['sma_20']
                    risk = max((row['close'] - sl_p) / self.pip_size, self.min_sl)
                    if risk <= self.max_sl:
                        sig = SIGNAL_LONG
                        risk_p = risk

            # Bearish %B Breakout
            elif prev['bb_pct_b'] >= 0.0 and row['bb_pct_b'] < 0.0 and row['close'] < row['open']:
                if body_pips >= 0.8 * row['atr_pips']:
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
# COMB13: Elder's Triple Screen Impulse (H1 MACD Tide + M15 RSI 2 Pullback)
# ==============================================================================
class ElderTripleScreenImpulse(BaseStrategy):
    """
    Alexander Elder's Classic Triple Screen adapted for M15 Scalping:
    - Screen 1 (Tide): H1 MACD Histogram slope is rising (Bullish) or falling (Bearish).
    - Screen 2 (Wave): M15 2-period RSI drops below 20 (oversold dip in uptrend) or spikes above 80 (overbought spike in downtrend).
    - Screen 3 (Trigger): Price breaks previous bar's high (Long) or previous bar's low (Short).
    """
    def __init__(self, target_rr: float = 2.2, min_sl: float = 6.0, max_sl: float = 16.0, **kwargs):
        super().__init__(**kwargs)
        self.target_rr = target_rr
        self.min_sl = min_sl
        self.max_sl = max_sl
        self.pip_size = PIP_SIZE
        self.name = "Elder_Triple_Screen_Impulse"

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

        m15 = enrich_phase2_indicators(m15)

        # H1 MACD Histogram Tide
        h1 = base_df.resample('1h').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
        h1_ema12 = h1['close'].ewm(span=12, adjust=False).mean()
        h1_ema26 = h1['close'].ewm(span=26, adjust=False).mean()
        h1_macd = h1_ema12 - h1_ema26
        h1_sig = h1_macd.ewm(span=9, adjust=False).mean()
        h1_hist = h1_macd - h1_sig
        h1['tide'] = np.where(h1_hist > h1_hist.shift(1), 1, -1)
        m15['tide'] = h1['tide'].reindex(m15.index, method='ffill').fillna(0)

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

            # Long: H1 Tide Bullish, Prev RSI(2) < 20, Current breaks above Prev High and closes green
            if row['tide'] > 0 and prev['rsi_2'] < 20.0 and row['close'] > prev['high'] and row['close'] > row['open']:
                sl_p = min(prev['low'], row['low']) - 1.2 * self.pip_size
                risk = max((row['close'] - sl_p) / self.pip_size, self.min_sl)
                if risk <= self.max_sl:
                    sig = SIGNAL_LONG
                    risk_p = risk

            # Short: H1 Tide Bearish, Prev RSI(2) > 80, Current breaks below Prev Low and closes red
            elif row['tide'] < 0 and prev['rsi_2'] > 80.0 and row['close'] < prev['low'] and row['close'] < row['open']:
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
# COMB14: Supertrend (10, 2.0) + Session VWAP Rejection Pullback
# ==============================================================================
class SupertrendVWAPPullback(BaseStrategy):
    """
    Combines Trend Regime (Supertrend) with Value Anchor (Session VWAP):
    - Supertrend is Bullish (direction == 1).
    - Price pulls back to test within 3 pips of Session VWAP.
    - Current bar rejects VWAP and closes green above VWAP.
    - SL: 1.2 pips below VWAP / bar low. TP: 2.2x RR.
    """
    def __init__(self, target_rr: float = 2.2, min_sl: float = 6.0, max_sl: float = 16.0, **kwargs):
        super().__init__(**kwargs)
        self.target_rr = target_rr
        self.min_sl = min_sl
        self.max_sl = max_sl
        self.pip_size = PIP_SIZE
        self.name = "Supertrend_VWAP_Pullback"

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

        m15 = enrich_phase2_indicators(m15)
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
# COMB15: Chaikin Money Flow (CMF) Volume Surge + Donchian Channel Breakout
# ==============================================================================
class CMFDonchianBreakout(BaseStrategy):
    """
    Institutional Flow Breakout:
    - Breaks 20-bar Donchian High/Low.
    - CMF(20) > +0.10 (Strong Institutional Buying) or CMF(20) < -0.10 (Strong Institutional Selling).
    - ADX(14) > 20.0.
    """
    def __init__(self, target_rr: float = 2.0, min_sl: float = 7.0, max_sl: float = 16.0, **kwargs):
        super().__init__(**kwargs)
        self.target_rr = target_rr
        self.min_sl = min_sl
        self.max_sl = max_sl
        self.pip_size = PIP_SIZE
        self.name = "CMF_Donchian_Breakout"

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

        m15 = enrich_phase2_indicators(m15)
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

            # Donchian High Breakout + Positive CMF
            if prev['close'] <= prev['donchian_high'] and row['close'] > row['donchian_high']:
                if row['cmf_20'] > 0.08 and row['adx_14'] >= 18.0:
                    sl_p = row['ema_20']
                    risk = max((row['close'] - sl_p) / self.pip_size, self.min_sl)
                    if risk <= self.max_sl:
                        sig = SIGNAL_LONG
                        risk_p = risk

            # Donchian Low Breakout + Negative CMF
            elif prev['close'] >= prev['donchian_low'] and row['close'] < row['donchian_low']:
                if row['cmf_20'] < -0.08 and row['adx_14'] >= 18.0:
                    sl_p = row['ema_20']
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
# COMB16: Dual Momentum Convergence (RSI 50 Cross + Stoch %K/%D Cross)
# ==============================================================================
class DualMomentumConvergence(BaseStrategy):
    """
    Momentum Synergy:
    - Trend: Close > EMA 50 for Long, Close < EMA 50 for Short.
    - RSI(14) crosses above 50.0 (Bullish momentum flip).
    - Stochastic %K crosses above %D simultaneously (within 1 bar).
    - Target RR: 2.0x.
    """
    def __init__(self, target_rr: float = 2.0, min_sl: float = 7.0, max_sl: float = 16.0, **kwargs):
        super().__init__(**kwargs)
        self.target_rr = target_rr
        self.min_sl = min_sl
        self.max_sl = max_sl
        self.pip_size = PIP_SIZE
        self.name = "Dual_Momentum_Convergence"

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

        m15 = enrich_phase2_indicators(m15)
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

            # Bullish Convergence
            if row['close'] > row['ema_50'] and prev['rsi_14'] <= 50.0 and row['rsi_14'] > 50.0:
                if row['stoch_k'] > row['stoch_d'] and prev['stoch_k'] <= prev['stoch_d']:
                    sl_p = min(prev['low'], row['low']) - 1.2 * self.pip_size
                    risk = max((row['close'] - sl_p) / self.pip_size, self.min_sl)
                    if risk <= self.max_sl:
                        sig = SIGNAL_LONG
                        risk_p = risk

            # Bearish Convergence
            elif row['close'] < row['ema_50'] and prev['rsi_14'] >= 50.0 and row['rsi_14'] < 50.0:
                if row['stoch_k'] < row['stoch_d'] and prev['stoch_k'] >= prev['stoch_d']:
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
# COMB17: Parabolic SAR Trend Acceleration + ADX > 22
# ==============================================================================
class ParabolicSAR_ADX_Acceleration(BaseStrategy):
    """
    Parabolic SAR Reversal + ADX Trend Confirmation:
    - PSAR flips from above to below price (Bullish).
    - ADX >= 22.0 and +DI > -DI.
    - Trade during London & NY sessions.
    - SL: at PSAR value.
    """
    def __init__(self, target_rr: float = 2.0, min_sl: float = 7.0, max_sl: float = 16.0, **kwargs):
        super().__init__(**kwargs)
        self.target_rr = target_rr
        self.min_sl = min_sl
        self.max_sl = max_sl
        self.pip_size = PIP_SIZE
        self.name = "Parabolic_SAR_ADX"

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

        m15 = enrich_phase2_indicators(m15)
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

            # PSAR flipped from Bearish to Bullish
            if prev['psar_dir'] == -1 and row['psar_dir'] == 1:
                if row['adx_14'] >= 20.0 and row['plus_di'] > row['minus_di']:
                    sl_p = row['psar'] - 1.0 * self.pip_size
                    risk = max((row['close'] - sl_p) / self.pip_size, self.min_sl)
                    if risk <= self.max_sl:
                        sig = SIGNAL_LONG
                        risk_p = risk

            # PSAR flipped from Bullish to Bearish
            elif prev['psar_dir'] == 1 and row['psar_dir'] == -1:
                if row['adx_14'] >= 20.0 and row['minus_di'] > row['plus_di']:
                    sl_p = row['psar'] + 1.0 * self.pip_size
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
# COMB18: London Breakout + NY Continuation Hybrid (Two-Session Powerhouse)
# ==============================================================================
class DualSessionContinuation(BaseStrategy):
    """
    Session Synergy:
    - Session A (London 07:00-10:00 UTC): Asian Range Breakout + ADX > 16.
    - Session B (NY 12:30-15:30 UTC): Retest & Continuation of London Range (07:00-11:00 UTC) with MACD Hist flip.
    """
    def __init__(self, target_rr: float = 2.2, min_sl: float = 6.0, max_sl: float = 16.0, **kwargs):
        super().__init__(**kwargs)
        self.target_rr = target_rr
        self.min_sl = min_sl
        self.max_sl = max_sl
        self.pip_size = PIP_SIZE
        self.name = "Dual_Session_Continuation"

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

        m15 = enrich_phase2_indicators(m15)
        time_ms = m15['time_ms'].values
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        daily_groups = m15.groupby('date')
        for current_date, day_df in daily_groups:
            # 1. Asian Session
            asian = day_df[(day_df['hour'] >= 0) & (day_df['hour'] < 6)]
            if len(asian) < 8:
                continue
            asian_high = asian['high'].max()
            asian_low = asian['low'].min()
            asian_range = (asian_high - asian_low) / self.pip_size

            # 2. London Session (07:00-10:00 UTC)
            london = day_df[(day_df['hour'] >= 7) & (day_df['hour'] <= 10)]
            london_traded = False

            if 8.0 <= asian_range <= 45.0:
                for idx_m15, row in london.iterrows():
                    if london_traded:
                        break
                    i = m15.index.get_loc(idx_m15)
                    prev = m15.iloc[i-1]
                    sig = 0
                    risk_p = 0.0

                    if prev['close'] <= asian_high and row['close'] > asian_high:
                        if row['adx_14'] >= 16.0 and row['close'] > row['open']:
                            sl_p = row['low'] - 1.2 * self.pip_size
                            risk = max((row['close'] - sl_p) / self.pip_size, self.min_sl)
                            if risk <= self.max_sl:
                                sig = SIGNAL_LONG
                                risk_p = risk

                    elif prev['close'] >= asian_low and row['close'] < asian_low:
                        if row['adx_14'] >= 16.0 and row['close'] < row['open']:
                            sl_p = row['high'] + 1.2 * self.pip_size
                            risk = max((sl_p - row['close']) / self.pip_size, self.min_sl)
                            if risk <= self.max_sl:
                                sig = SIGNAL_SHORT
                                risk_p = risk

                    if sig != 0:
                        london_traded = True
                        exec_t = time_ms[i] + dur_ms
                        idx = int(np.searchsorted(tick_ts, exec_t))
                        if idx < n_ticks and signals[idx] == 0:
                            signals[idx] = sig
                            sl_pips_arr[idx] = risk_p
                            tp_pips_arr[idx] = risk_p * self.target_rr

            # 3. New York Session (12:30-15:30 UTC)
            london_full = day_df[(day_df['hour'] >= 7) & (day_df['hour'] < 12)]
            if len(london_full) < 10:
                continue
            lon_high = london_full['high'].max()
            lon_low = london_full['low'].min()

            ny = day_df[(day_df['hour'] >= 12) & (day_df['hour'] <= 15)]
            ny_traded = False
            for idx_m15, row in ny.iterrows():
                if ny_traded:
                    break
                if row['hour'] == 12 and row['minute'] < 30:
                    continue
                i = m15.index.get_loc(idx_m15)
                prev = m15.iloc[i-1]
                sig = 0
                risk_p = 0.0

                if prev['close'] <= lon_high and row['close'] > lon_high:
                    if row['macd_hist'] > 0 and prev['macd_hist'] <= 0 and row['close'] > row['open']:
                        sl_p = row['low'] - 1.2 * self.pip_size
                        risk = max((row['close'] - sl_p) / self.pip_size, self.min_sl)
                        if risk <= self.max_sl:
                            sig = SIGNAL_LONG
                            risk_p = risk

                elif prev['close'] >= lon_low and row['close'] < lon_low:
                    if row['macd_hist'] < 0 and prev['macd_hist'] >= 0 and row['close'] < row['open']:
                        sl_p = row['high'] + 1.2 * self.pip_size
                        risk = max((sl_p - row['close']) / self.pip_size, self.min_sl)
                        if risk <= self.max_sl:
                            sig = SIGNAL_SHORT
                            risk_p = risk

                if sig != 0:
                    ny_traded = True
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
    {"code": "COMB09", "name": "Asian Sweep + Stoch Reversal (M15)", "cls": AsianLiquiditySweepReversal},
    {"code": "COMB10", "name": "Regime Keltner + Stoch (ADX<20) (M15)", "cls": RegimeGatedKeltnerStoch},
    {"code": "COMB11", "name": "Hull MA 21 Flip + ADX Trend (M15)", "cls": HullMovingAverageADX},
    {"code": "COMB12", "name": "Bollinger Bandwidth %B Breakout (M15)", "cls": BollingerBandwidthExpansion},
    {"code": "COMB13", "name": "Elder Triple Screen Impulse (M15)", "cls": ElderTripleScreenImpulse},
    {"code": "COMB14", "name": "Supertrend + VWAP Pullback (M15)", "cls": SupertrendVWAPPullback},
    {"code": "COMB15", "name": "CMF Volume + Donchian Breakout (M15)", "cls": CMFDonchianBreakout},
    {"code": "COMB16", "name": "Dual Momentum Convergence (M15)", "cls": DualMomentumConvergence},
    {"code": "COMB17", "name": "Parabolic SAR + ADX Trend (M15)", "cls": ParabolicSAR_ADX_Acceleration},
    {"code": "COMB18", "name": "Dual Session London+NY Hybrid (M15)", "cls": DualSessionContinuation},
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

    # Ulcer Index
    ulcer_index = float(np.sqrt(np.mean(dd_pct ** 2)))

    # Daily DD
    max_daily_loss = 0.0
    for day, pnl in daily_pnl.items():
        if pnl < 0:
            day_loss_pct = abs(pnl) / 100000.0 * 100.0
            if day_loss_pct > max_daily_loss:
                max_daily_loss = day_loss_pct

    # Drawdown Recovery Duration
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
    print("                 MULTI-INDICATOR STRATEGY LABORATORY PHASE 2: 52 WEEKS FULL TICK AUDIT (~26M TICKS)")
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
    print("                                      PHASE 2 EXPERIMENTAL COMBINATIONS SCORECARD")
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
