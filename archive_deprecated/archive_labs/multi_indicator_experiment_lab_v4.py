#!/usr/bin/env python3
"""
Multi-Indicator Strategy Laboratory Phase 4:
Evaluating 10 Brand New Cutting-Edge Indicator Combinations on 52 Weeks of Dukascopy Tick Data (~26M Ticks):

1. COMB29: Linear Regression Channel (LRC 20) 2.0-StdErr Mean Reversion (M15)
2. COMB30: Commodity Channel Index (CCI 20) Zero-Line Bounce + Supertrend (M15)
3. COMB31: Williams %R (14) Failure Swing + EMA 21 Trend (M15)
4. COMB32: Relative Vigor Index (RVI 10) Conviction Crossover (M15)
5. COMB33: Volume-Weighted MACD (VW-MACD 12/26/9) Acceleration (M15)
6. COMB34: Squeeze Pro Multi-Compression (BB vs KC 1.5 ATR) Momentum Flip (M15)
7. COMB35: Directional Movement (DMI) Parabolic Exhaustion Fader (M15)
8. COMB36: Session VWAP 2.0-Sigma Statistical Band Reversion (M15)
9. COMB37: Hull MA 9 + EMA 50 Value Pocket Scalper (M15)
10. COMB38: True Strength Index (TSI 25, 13) Double-Smoothed Cross (M15)

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


def enrich_phase4_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Precomputes indicators for Phase 4 combinations with ZERO lookahead."""
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

    # 2. EMAs & SMAs
    for p in [9, 20, 21, 50, 100, 200]:
        df[f'ema_{p}'] = c.ewm(span=p, adjust=False).mean()
    df['sma_20'] = c.rolling(20).mean().bfill()

    # 3. Linear Regression Channel (20)
    # y = a + b*x, slope b = Cov(x, y) / Var(x)
    x = np.arange(20)
    x_mean = 9.5
    x_var = np.var(x) * 20.0  # sum of (x - x_mean)^2

    def calc_linreg(sub_s):
        y = sub_s
        y_mean = np.mean(y)
        slope = np.sum((x - x_mean) * (y - y_mean)) / x_var
        intercept = y_mean - slope * x_mean
        pred_last = intercept + slope * 19.0
        # Standard error
        residuals = y - (intercept + slope * x)
        stderr = np.std(residuals)
        return pred_last, stderr

    lr_center = np.zeros(n)
    lr_err = np.zeros(n)
    c_vals = c.values
    for i in range(20, n):
        pred, err = calc_linreg(c_vals[i-19:i+1])
        lr_center[i] = pred
        lr_err[i] = err

    df['lr_center'] = lr_center
    df['lr_up2'] = lr_center + (2.0 * lr_err)
    df['lr_low2'] = lr_center - (2.0 * lr_err)

    # 4. Supertrend (10, 2.0)
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

    # 5. Commodity Channel Index (CCI 20)
    tp = (h + l + c) / 3.0
    sma_tp = tp.rolling(20).mean().bfill()
    mad = tp.rolling(20).apply(lambda s: np.mean(np.abs(s - np.mean(s))), raw=True).bfill()
    df['cci_20'] = (tp - sma_tp) / np.maximum(0.015 * mad, 1e-9)

    # 6. Williams %R (14)
    low14 = l.rolling(14).min().bfill()
    high14 = h.rolling(14).max().bfill()
    df['will_r_14'] = ((high14 - c) / np.maximum(high14 - low14, 1e-9)) * -100.0

    # 7. Relative Vigor Index (RVI 10)
    num = (c - o) + 2.0 * (c.shift(1) - o.shift(1)) + 2.0 * (c.shift(2) - o.shift(2)) + (c.shift(3) - o.shift(3))
    den = (h - l) + 2.0 * (h.shift(1) - l.shift(1)) + 2.0 * (h.shift(2) - l.shift(2)) + (h.shift(3) - l.shift(3))
    rvi_raw = num.rolling(10).mean().bfill() / np.maximum(den.rolling(10).mean().bfill(), 1e-9)
    df['rvi_main'] = rvi_raw
    df['rvi_sig'] = (rvi_raw + 2.0 * rvi_raw.shift(1) + 2.0 * rvi_raw.shift(2) + rvi_raw.shift(3)) / 6.0

    # 8. Volume-Weighted MACD (VW-MACD 12/26/9)
    vol = tr / PIP_SIZE + 10.0
    pv = c * vol
    vw_ema12 = pv.ewm(span=12, adjust=False).mean() / np.maximum(vol.ewm(span=12, adjust=False).mean(), 1e-9)
    vw_ema26 = pv.ewm(span=26, adjust=False).mean() / np.maximum(vol.ewm(span=26, adjust=False).mean(), 1e-9)
    df['vw_macd'] = vw_ema12 - vw_ema26
    df['vw_sig'] = df['vw_macd'].ewm(span=9, adjust=False).mean()
    df['vw_hist'] = df['vw_macd'] - df['vw_sig']

    # 9. Bollinger Bands & Keltner Channels (Squeeze Pro)
    std20 = c.rolling(20).std().bfill()
    df['bb_up'] = df['sma_20'] + (2.0 * std20)
    df['bb_low'] = df['sma_20'] - (2.0 * std20)
    df['kc_up15'] = df['ema_20'] + (1.5 * df['atr_14'])
    df['kc_low15'] = df['ema_20'] - (1.5 * df['atr_14'])
    df['is_squeeze_pro'] = (df['bb_up'] < df['kc_up15']) & (df['bb_low'] > df['kc_low15'])

    # 10. ADX & DMI (14)
    up_m = h.diff()
    dn_m = -l.diff()
    p_dm = np.where((up_m > dn_m) & (up_m > 0), up_m, 0.0)
    m_dm = np.where((dn_m > up_m) & (dn_m > 0), dn_m, 0.0)
    smooth_tr = tr.ewm(alpha=1.0/14, adjust=False).mean()
    df['plus_di'] = (pd.Series(p_dm, index=df.index).ewm(alpha=1.0/14, adjust=False).mean() / np.maximum(smooth_tr, 1e-9)) * 100
    df['minus_di'] = (pd.Series(m_dm, index=df.index).ewm(alpha=1.0/14, adjust=False).mean() / np.maximum(smooth_tr, 1e-9)) * 100
    dx = (abs(df['plus_di'] - df['minus_di']) / np.maximum(df['plus_di'] + df['minus_di'], 1e-9)) * 100
    df['adx_14'] = dx.ewm(alpha=1.0/14, adjust=False).mean()

    # 11. Hull Moving Average (HMA 9)
    def wma(s: pd.Series, period: int) -> pd.Series:
        weights = np.arange(1, period + 1)
        return s.rolling(period).apply(lambda z: np.dot(z, weights) / weights.sum(), raw=True).bfill()

    period = 9
    half_period = int(period / 2)
    sqrt_period = int(np.sqrt(period))
    wma_half = wma(c, half_period)
    wma_full = wma(c, period)
    diff_wma = 2 * wma_half - wma_full
    df['hma_9'] = wma(diff_wma, sqrt_period)
    df['hma9_slope'] = df['hma_9'] - df['hma_9'].shift(1)

    # 12. True Strength Index (TSI 25, 13)
    m_diff = c.diff()
    smooth_m = m_diff.ewm(span=25, adjust=False).mean().ewm(span=13, adjust=False).mean()
    abs_m = abs(m_diff).ewm(span=25, adjust=False).mean().ewm(span=13, adjust=False).mean()
    df['tsi'] = (smooth_m / np.maximum(abs_m, 1e-9)) * 100.0
    df['tsi_sig'] = df['tsi'].ewm(span=7, adjust=False).mean()

    # 13. Session VWAP with 2.0-Sigma Standard Deviation Bands
    df['date'] = df.index.date
    df['hour'] = df.index.hour
    df['time_ms'] = df.index.astype('datetime64[ms]').view('int64')

    cum_vol = vol.groupby(df['date']).cumsum()
    cum_pv = (c * vol).groupby(df['date']).cumsum()
    df['session_vwap'] = cum_pv / np.maximum(cum_vol, 1e-9)

    cum_p2_v = ((c ** 2) * vol).groupby(df['date']).cumsum()
    vwap_var = (cum_p2_v / np.maximum(cum_vol, 1e-9)) - (df['session_vwap'] ** 2)
    vwap_std = np.sqrt(np.maximum(vwap_var, 0.0))
    df['vwap_up2'] = df['session_vwap'] + (2.0 * vwap_std)
    df['vwap_low2'] = df['session_vwap'] - (2.0 * vwap_std)

    # 14. RSI 14
    gain = m_diff.clip(lower=0.0).ewm(alpha=1.0/14, adjust=False).mean()
    loss = (-m_diff.clip(upper=0.0)).ewm(alpha=1.0/14, adjust=False).mean()
    rs = gain / np.maximum(loss, 1e-9)
    df['rsi_14'] = 100.0 - (100.0 / (1.0 + rs))

    return df


# ==============================================================================
# COMB29: Linear Regression Channel (LRC 20) 2.0-StdErr Mean Reversion (M15)
# ==============================================================================
class LRCMeanReversion(BaseStrategy):
    """
    Statistical Channel Mean Reversion:
    - Prior bar pierces 2.0 Standard Error lower/upper band.
    - Current bar closes back inside with rejection candle and RSI confirmation (<35 for long, >65 for short).
    - Regime filter: ADX < 22 to prevent trading against trending runaways.
    """
    def __init__(self, target_rr: float = 1.8, min_sl: float = 6.0, max_sl: float = 16.0, **kwargs):
        super().__init__(**kwargs)
        self.target_rr = target_rr
        self.min_sl = min_sl
        self.max_sl = max_sl
        self.pip_size = PIP_SIZE
        self.name = "LRC_Mean_Reversion"

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

        m15 = enrich_phase4_indicators(m15)
        time_ms = m15['time_ms'].values
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        for i in range(25, len(m15)):
            row = m15.iloc[i]
            prev = m15.iloc[i-1]

            if row['hour'] < 7 or row['hour'] > 15:
                continue

            if row['adx_14'] >= 22.0:
                continue

            sig = 0
            risk_p = 0.0

            # Long: Pierced lower 2.0-stderr, closed back inside & green
            if prev['low'] < prev['lr_low2'] and row['close'] > row['lr_low2'] and row['close'] > row['open']:
                if row['rsi_14'] < 40.0:
                    sl_p = min(prev['low'], row['low']) - 1.2 * self.pip_size
                    risk = max((row['close'] - sl_p) / self.pip_size, self.min_sl)
                    if risk <= self.max_sl:
                        sig = SIGNAL_LONG
                        risk_p = risk

            # Short: Pierced upper 2.0-stderr, closed back inside & red
            elif prev['high'] > prev['lr_up2'] and row['close'] < row['lr_up2'] and row['close'] < row['open']:
                if row['rsi_14'] > 60.0:
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
# COMB30: Commodity Channel Index (CCI 20) Zero-Line Bounce + Supertrend (M15)
# ==============================================================================
class CCIZeroBounceSupertrend(BaseStrategy):
    """
    Momentum Pullback Rebound:
    - Supertrend confirms trend direction.
    - CCI(20) dips near 0 line (-40 to +20 for Long), then rebounds sharply above +50.
    - Reverse for Short.
    """
    def __init__(self, target_rr: float = 2.2, min_sl: float = 7.0, max_sl: float = 16.0, **kwargs):
        super().__init__(**kwargs)
        self.target_rr = target_rr
        self.min_sl = min_sl
        self.max_sl = max_sl
        self.pip_size = PIP_SIZE
        self.name = "CCI_Zero_Bounce_Supertrend"

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

        m15 = enrich_phase4_indicators(m15)
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

            # Bullish CCI Bounce
            if row['supertrend_dir'] == 1 and prev['cci_20'] <= 40.0 and row['cci_20'] > 50.0 and row['close'] > row['open']:
                sl_p = min(prev['low'], row['low']) - 1.2 * self.pip_size
                risk = max((row['close'] - sl_p) / self.pip_size, self.min_sl)
                if risk <= self.max_sl:
                    sig = SIGNAL_LONG
                    risk_p = risk

            # Bearish CCI Bounce
            elif row['supertrend_dir'] == -1 and prev['cci_20'] >= -40.0 and row['cci_20'] < -50.0 and row['close'] < row['open']:
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
# COMB31: Williams %R (14) Failure Swing + EMA 21 Trend (M15)
# ==============================================================================
class WilliamsRFailureSwing(BaseStrategy):
    """
    Momentum Failure Swing:
    - Uptrend: Price > 21 EMA.
    - Williams %R dropped below -80 (oversold dip), then on current bar crosses back above -50.
    - Reverse for Downtrend.
    """
    def __init__(self, target_rr: float = 2.0, min_sl: float = 6.0, max_sl: float = 16.0, **kwargs):
        super().__init__(**kwargs)
        self.target_rr = target_rr
        self.min_sl = min_sl
        self.max_sl = max_sl
        self.pip_size = PIP_SIZE
        self.name = "Williams_R_Failure_Swing"

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

        m15 = enrich_phase4_indicators(m15)
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

            # Bullish Failure Swing: Prior was oversold (<-80), Current crosses above -50, Trend is up
            if row['close'] > row['ema_21']:
                if prior['will_r_14'] < -80.0 and prev['will_r_14'] <= -50.0 and row['will_r_14'] > -50.0 and row['close'] > row['open']:
                    sl_p = min(prior['low'], prev['low'], row['low']) - 1.2 * self.pip_size
                    risk = max((row['close'] - sl_p) / self.pip_size, self.min_sl)
                    if risk <= self.max_sl:
                        sig = SIGNAL_LONG
                        risk_p = risk

            # Bearish Failure Swing: Prior was overbought (>-20), Current crosses below -50, Trend is down
            elif row['close'] < row['ema_21']:
                if prior['will_r_14'] > -20.0 and prev['will_r_14'] >= -50.0 and row['will_r_14'] < -50.0 and row['close'] < row['open']:
                    sl_p = max(prior['high'], prev['high'], row['high']) + 1.2 * self.pip_size
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
# COMB32: Relative Vigor Index (RVI 10) Conviction Crossover (M15)
# ==============================================================================
class RVIConvictionCrossover(BaseStrategy):
    """
    RVI Conviction Momentum:
    - RVI main line crosses above RVI signal line from below 0 (Long).
    - RVI main line crosses below RVI signal line from above 0 (Short).
    - ADX(14) >= 18.0.
    """
    def __init__(self, target_rr: float = 2.0, min_sl: float = 7.0, max_sl: float = 16.0, **kwargs):
        super().__init__(**kwargs)
        self.target_rr = target_rr
        self.min_sl = min_sl
        self.max_sl = max_sl
        self.pip_size = PIP_SIZE
        self.name = "RVI_Conviction_Cross"

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

        m15 = enrich_phase4_indicators(m15)
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

            # Bullish RVI Cross
            if prev['rvi_main'] <= prev['rvi_sig'] and row['rvi_main'] > row['rvi_sig']:
                if row['adx_14'] >= 18.0 and row['close'] > row['open']:
                    sl_p = min(prev['low'], row['low']) - 1.2 * self.pip_size
                    risk = max((row['close'] - sl_p) / self.pip_size, self.min_sl)
                    if risk <= self.max_sl:
                        sig = SIGNAL_LONG
                        risk_p = risk

            # Bearish RVI Cross
            elif prev['rvi_main'] >= prev['rvi_sig'] and row['rvi_main'] < row['rvi_sig']:
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
# COMB33: Volume-Weighted MACD (VW-MACD 12/26/9) Acceleration (M15)
# ==============================================================================
class VWMACDAcceleration(BaseStrategy):
    """
    Volume-Weighted MACD Acceleration:
    - VW-MACD histogram flips from negative to positive with high volume conviction (Long).
    - Reverse for Short.
    - Filter: 50 EMA Trend direction.
    """
    def __init__(self, target_rr: float = 2.0, min_sl: float = 7.0, max_sl: float = 16.0, **kwargs):
        super().__init__(**kwargs)
        self.target_rr = target_rr
        self.min_sl = min_sl
        self.max_sl = max_sl
        self.pip_size = PIP_SIZE
        self.name = "VW_MACD_Acceleration"

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

        m15 = enrich_phase4_indicators(m15)
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

            # Bullish VW-MACD Flip
            if prev['vw_hist'] <= 0 and row['vw_hist'] > 0 and row['close'] > row['ema_50'] and row['close'] > row['open']:
                sl_p = min(prev['low'], row['low']) - 1.2 * self.pip_size
                risk = max((row['close'] - sl_p) / self.pip_size, self.min_sl)
                if risk <= self.max_sl:
                    sig = SIGNAL_LONG
                    risk_p = risk

            # Bearish VW-MACD Flip
            elif prev['vw_hist'] >= 0 and row['vw_hist'] < 0 and row['close'] < row['ema_50'] and row['close'] < row['open']:
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
# COMB34: Squeeze Pro Multi-Compression Momentum Flip (M15)
# ==============================================================================
class SqueezeProMomentumFlip(BaseStrategy):
    """
    Squeeze Pro:
    - BB inside KC 1.5 ATR was active for at least 2 consecutive bars.
    - Current bar fires out of squeeze (BB expands beyond KC) with VW-MACD momentum alignment.
    """
    def __init__(self, target_rr: float = 2.2, min_sl: float = 7.0, max_sl: float = 16.0, **kwargs):
        super().__init__(**kwargs)
        self.target_rr = target_rr
        self.min_sl = min_sl
        self.max_sl = max_sl
        self.pip_size = PIP_SIZE
        self.name = "Squeeze_Pro_Momentum_Flip"

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

        m15 = enrich_phase4_indicators(m15)
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

            # Was in squeeze prior 2 bars, and now releases
            was_in_squeeze = prior['is_squeeze_pro'] and prev['is_squeeze_pro']
            is_firing = not row['is_squeeze_pro']

            if was_in_squeeze and is_firing:
                if row['vw_hist'] > 0 and row['close'] > row['open']:
                    sl_p = row['sma_20']
                    risk = max((row['close'] - sl_p) / self.pip_size, self.min_sl)
                    if risk <= self.max_sl:
                        sig = SIGNAL_LONG
                        risk_p = risk

                elif row['vw_hist'] < 0 and row['close'] < row['open']:
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
# COMB35: Directional Movement (DMI) Parabolic Exhaustion Fader (M15)
# ==============================================================================
class DMIParabolicExhaustionFader(BaseStrategy):
    """
    Exhaustion Top/Bottom Fade:
    - Parabolic Climax: +DI > 40 and ADX > 32 (extreme buyers exhaustion).
    - Current candle prints an engulfing reversal closing below prior candle low.
    - Reverse for -DI > 40 (extreme sellers exhaustion).
    """
    def __init__(self, target_rr: float = 2.0, min_sl: float = 6.0, max_sl: float = 16.0, **kwargs):
        super().__init__(**kwargs)
        self.target_rr = target_rr
        self.min_sl = min_sl
        self.max_sl = max_sl
        self.pip_size = PIP_SIZE
        self.name = "DMI_Parabolic_Exhaustion_Fader"

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

        m15 = enrich_phase4_indicators(m15)
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

            # Fading Buyers Exhaustion: prev had +DI > 40 & ADX > 30, current closes below prev low & red
            if prev['plus_di'] >= 40.0 and prev['adx_14'] >= 30.0:
                if row['close'] < prev['low'] and row['close'] < row['open']:
                    sl_p = max(prev['high'], row['high']) + 1.2 * self.pip_size
                    risk = max((sl_p - row['close']) / self.pip_size, self.min_sl)
                    if risk <= self.max_sl:
                        sig = SIGNAL_SHORT
                        risk_p = risk

            # Fading Sellers Exhaustion: prev had -DI > 40 & ADX > 30, current closes above prev high & green
            elif prev['minus_di'] >= 40.0 and prev['adx_14'] >= 30.0:
                if row['close'] > prev['high'] and row['close'] > row['open']:
                    sl_p = min(prev['low'], row['low']) - 1.2 * self.pip_size
                    risk = max((row['close'] - sl_p) / self.pip_size, self.min_sl)
                    if risk <= self.max_sl:
                        sig = SIGNAL_LONG
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
# COMB36: Session VWAP 2.0-Sigma Statistical Band Reversion (M15)
# ==============================================================================
class SessionVWAP2SigmaReversion(BaseStrategy):
    """
    VWAP 2.0-Sigma Mean Reversion:
    - Low-ADX regime (< 20.0).
    - Prior candle pierced VWAP 2.0-Sigma band.
    - Current candle closes back inside with rejection body.
    """
    def __init__(self, target_rr: float = 1.8, min_sl: float = 6.0, max_sl: float = 16.0, **kwargs):
        super().__init__(**kwargs)
        self.target_rr = target_rr
        self.min_sl = min_sl
        self.max_sl = max_sl
        self.pip_size = PIP_SIZE
        self.name = "VWAP_2Sigma_Reversion"

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

        m15 = enrich_phase4_indicators(m15)
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

            # Bullish: Pierced VWAP -2.0 sigma, closed back above & green
            if prev['low'] < prev['vwap_low2'] and row['close'] > row['vwap_low2'] and row['close'] > row['open']:
                sl_p = min(prev['low'], row['low']) - 1.2 * self.pip_size
                risk = max((row['close'] - sl_p) / self.pip_size, self.min_sl)
                if risk <= self.max_sl:
                    sig = SIGNAL_LONG
                    risk_p = risk

            # Bearish: Pierced VWAP +2.0 sigma, closed back below & red
            elif prev['high'] > prev['vwap_up2'] and row['close'] < row['vwap_up2'] and row['close'] < row['open']:
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
# COMB37: Hull MA 9 + EMA 50 Value Pocket Scalper (M15)
# ==============================================================================
class HullMA9EMA50Pocket(BaseStrategy):
    """
    Dynamic EMA Value Pocket:
    - Macro Trend: Price > 100 EMA.
    - Pullback: Low enters the dynamic pocket between 20 EMA and 50 EMA.
    - Trigger: Hull MA 9 slope flips positive while bar closes green.
    """
    def __init__(self, target_rr: float = 2.0, min_sl: float = 6.0, max_sl: float = 16.0, **kwargs):
        super().__init__(**kwargs)
        self.target_rr = target_rr
        self.min_sl = min_sl
        self.max_sl = max_sl
        self.pip_size = PIP_SIZE
        self.name = "Hull_EMA50_Pocket"

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

        m15 = enrich_phase4_indicators(m15)
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

            # Bullish Pocket
            if row['close'] > row['ema_100']:
                if row['low'] <= row['ema_20'] and row['low'] >= row['ema_50'] - 2.0 * self.pip_size:
                    if prev['hma9_slope'] <= 0 and row['hma9_slope'] > 0 and row['close'] > row['open']:
                        sl_p = min(row['low'], row['ema_50']) - 1.2 * self.pip_size
                        risk = max((row['close'] - sl_p) / self.pip_size, self.min_sl)
                        if risk <= self.max_sl:
                            sig = SIGNAL_LONG
                            risk_p = risk

            # Bearish Pocket
            elif row['close'] < row['ema_100']:
                if row['high'] >= row['ema_20'] and row['high'] <= row['ema_50'] + 2.0 * self.pip_size:
                    if prev['hma9_slope'] >= 0 and row['hma9_slope'] < 0 and row['close'] < row['open']:
                        sl_p = max(row['high'], row['ema_50']) + 1.2 * self.pip_size
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
# COMB38: True Strength Index (TSI 25, 13) Double-Smoothed Cross (M15)
# ==============================================================================
class TSIDoubleSmoothedCross(BaseStrategy):
    """
    True Strength Index Cross:
    - Double-smoothed momentum filters whipsaws.
    - TSI crosses above signal line from below (Long).
    - TSI crosses below signal line from above (Short).
    - Filter: 50 EMA trend alignment.
    """
    def __init__(self, target_rr: float = 2.0, min_sl: float = 7.0, max_sl: float = 16.0, **kwargs):
        super().__init__(**kwargs)
        self.target_rr = target_rr
        self.min_sl = min_sl
        self.max_sl = max_sl
        self.pip_size = PIP_SIZE
        self.name = "TSI_Double_Smoothed_Cross"

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

        m15 = enrich_phase4_indicators(m15)
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

            # Bullish TSI Cross
            if row['close'] > row['ema_50']:
                if prev['tsi'] <= prev['tsi_sig'] and row['tsi'] > row['tsi_sig'] and row['close'] > row['open']:
                    sl_p = min(prev['low'], row['low']) - 1.2 * self.pip_size
                    risk = max((row['close'] - sl_p) / self.pip_size, self.min_sl)
                    if risk <= self.max_sl:
                        sig = SIGNAL_LONG
                        risk_p = risk

            # Bearish TSI Cross
            elif row['close'] < row['ema_50']:
                if prev['tsi'] >= prev['tsi_sig'] and row['tsi'] < row['tsi_sig'] and row['close'] < row['open']:
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
    {"code": "COMB29", "name": "LRC 2.0-StdErr Mean Reversion (M15)", "cls": LRCMeanReversion},
    {"code": "COMB30", "name": "CCI Zero Bounce + Supertrend (M15)", "cls": CCIZeroBounceSupertrend},
    {"code": "COMB31", "name": "Williams %R Failure Swing (M15)", "cls": WilliamsRFailureSwing},
    {"code": "COMB32", "name": "RVI Conviction Crossover (M15)", "cls": RVIConvictionCrossover},
    {"code": "COMB33", "name": "Volume-Weighted MACD Accel (M15)", "cls": VWMACDAcceleration},
    {"code": "COMB34", "name": "Squeeze Pro Momentum Flip (M15)", "cls": SqueezeProMomentumFlip},
    {"code": "COMB35", "name": "DMI Parabolic Exhaustion Fade (M15)", "cls": DMIParabolicExhaustionFader},
    {"code": "COMB36", "name": "VWAP 2.0-Sigma Band Reversion (M15)", "cls": SessionVWAP2SigmaReversion},
    {"code": "COMB37", "name": "Hull MA 9 + EMA 50 Pocket (M15)", "cls": HullMA9EMA50Pocket},
    {"code": "COMB38", "name": "TSI Double-Smoothed Cross (M15)", "cls": TSIDoubleSmoothedCross},
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
    print("                 MULTI-INDICATOR STRATEGY LABORATORY PHASE 4: 52 WEEKS FULL TICK AUDIT (~26M TICKS)")
    print("                          Evaluating 10 Brand New Cutting-Edge Combinations for Prop Firms")
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
    print("                                      PHASE 4 EXPERIMENTAL COMBINATIONS SCORECARD")
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
