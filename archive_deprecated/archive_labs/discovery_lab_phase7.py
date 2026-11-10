#!/usr/bin/env python3
"""
Prop Firm Strategy Discovery Laboratory Phase 7:
Evaluating 5 Institutional Anchored & Microstructure Models on 52 Weeks (~26M Ticks):

1. DISC11: London Anchored VWAP (AVWAP) Pullback Scalper (M15)
   - VWAP resets precisely at 07:00 UTC (London Open).
   - Initial breakout followed by pullback retesting London AVWAP during 08:15-11:30 UTC.
   - Rejection bounce trigger with H1 50-EMA macro alignment. Target: 1.6R.

2. DISC12: New York Anchored VWAP (AVWAP) Pullback Scalper (M15)
   - VWAP resets precisely at 13:00 UTC (US Open).
   - Initial US impulse followed by pullback retesting US AVWAP during 13:45-16:30 UTC.
   - Rejection bounce trigger with H1 50-EMA macro alignment. Target: 1.6R.

3. DISC13: Kaufman Efficiency Ratio (KER) Dynamic Breakout (M15) [Perry Kaufman]
   - KER(10) = |Close_t - Close_{t-10}| / Sum(|Close_i - Close_{i-1}|)
   - Filters out market noise: KER bursts from <= 0.30 to >= 0.55 during 07:00-15:00 UTC.
   - Trigger: Directional impulse candle with H1 50-EMA macro alignment. Target: 1.6R.

4. DISC14: Connors 2-Period RSI Trend-Pullback Scalper (M15) [Larry Connors]
   - M15 Trend: Close > EMA50, H1 Macro Bullish.
   - Extreme oversold pullback: RSI(2) <= 10.0 (or RSI(2) >= 90.0 for shorts).
   - Reversal trigger: Next bar closes green (bullish engulfing/rejection). Target: 1.5R.

5. DISC15: London-to-NY Momentum Handover (M15)
   - London Morning (07:00-12:45 UTC) range establishment.
   - If London closes near its high (top 20% of range) and Macro Bullish,
     enters long at US Open (13:00-13:30 UTC) on initial M15 expansion.
   - Target: 1.6R with Midpoint stop.

All models evaluated strictly with zero-lookahead on authentic tick data (floating spread + slippage).
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


def enrich_phase7_indicators(df: pd.DataFrame, base_df: pd.DataFrame) -> pd.DataFrame:
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

    # 2. EMAs (8, 20, 50)
    df['ema_8'] = c.ewm(span=8, adjust=False).mean()
    df['ema_20'] = c.ewm(span=20, adjust=False).mean()
    df['ema_50'] = c.ewm(span=50, adjust=False).mean()

    # 3. H1 50-EMA Macro Trend (zero-lookahead shift(1))
    h1 = base_df.resample('1h').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
    h1['ema50'] = h1['close'].ewm(span=50, adjust=False).mean()
    h1['macro_bias'] = np.where(h1['close'] > h1['ema50'], 1, -1)
    df['macro_bias'] = h1['macro_bias'].shift(1).reindex(df.index, method='ffill').fillna(0)

    # 4. Session Timings
    df['date'] = df.index.date
    df['hour'] = df.index.hour
    df['minute'] = df.index.minute
    df['time_ms'] = df.index.astype('datetime64[ms]').view('int64')

    vol = tr / PIP_SIZE + 10.0
    df['vol'] = vol

    # 5. London Anchored VWAP (resets at 07:00 UTC)
    # Mask bars from 07:00 UTC onwards until midnight
    is_london = (df['hour'] >= 7)
    df['london_vol'] = np.where(is_london, vol, 0.0)
    df['london_pv'] = np.where(is_london, c * vol, 0.0)
    cum_l_vol = df.groupby(df['date'])['london_vol'].cumsum()
    cum_l_pv = df.groupby(df['date'])['london_pv'].cumsum()
    df['avwap_london'] = np.where(is_london, cum_l_pv / np.maximum(cum_l_vol, 1e-9), np.nan)

    # 6. New York Anchored VWAP (resets at 13:00 UTC)
    is_ny = (df['hour'] >= 13)
    df['ny_vol'] = np.where(is_ny, vol, 0.0)
    df['ny_pv'] = np.where(is_ny, c * vol, 0.0)
    cum_ny_vol = df.groupby(df['date'])['ny_vol'].cumsum()
    cum_ny_pv = df.groupby(df['date'])['ny_pv'].cumsum()
    df['avwap_ny'] = np.where(is_ny, cum_ny_pv / np.maximum(cum_ny_vol, 1e-9), np.nan)

    # 7. Kaufman Efficiency Ratio (KER 10)
    c_diff_10 = (c - c.shift(10)).abs()
    sum_abs_diff = (c.diff().abs()).rolling(10).sum()
    df['ker_10'] = c_diff_10 / np.maximum(sum_abs_diff, 1e-9)

    # 8. Connors RSI (2-period RSI)
    c_diff = c.diff()
    gain2 = c_diff.clip(lower=0.0).ewm(alpha=0.5, adjust=False).mean()
    loss2 = (-c_diff.clip(upper=0.0)).ewm(alpha=0.5, adjust=False).mean()
    rs2 = gain2 / np.maximum(loss2, 1e-9)
    df['rsi_2'] = 100.0 - (100.0 / (1.0 + rs2))

    return df


# ==============================================================================
# DISC11: London Anchored VWAP (AVWAP) Pullback Scalper (M15)
# ==============================================================================
class LondonAVWAPPullbackScalper(BaseStrategy):
    """
    Anchored VWAP resets at 07:00 UTC (London Open).
    Between 08:15 and 11:30 UTC, if price pulls back to retest AVWAP and rejects,
    enters in direction of H1 Macro Bias.
    """
    def __init__(self, target_rr: float = 1.60, min_sl: float = 6.0, max_sl: float = 16.0, **kwargs):
        super().__init__(**kwargs)
        self.target_rr = target_rr
        self.min_sl = min_sl
        self.max_sl = max_sl
        self.pip_size = PIP_SIZE
        self.name = "London_AVWAP_Pullback"

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

        m15 = enrich_phase7_indicators(m15, base_df)
        time_ms = m15['time_ms'].values
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        for d, day_df in m15.groupby('date'):
            # Trade window: 08:15 to 11:30 UTC
            trade_bars = day_df[(day_df['hour'] >= 8) & (day_df['hour'] <= 11)]
            traded = False

            for idx_val, row in trade_bars.iterrows():
                if traded:
                    break

                i = m15.index.get_loc(idx_val)
                prev = m15.iloc[i-1]
                avwap = row['avwap_london']
                prev_avwap = prev['avwap_london']

                if pd.isna(avwap) or pd.isna(prev_avwap):
                    continue

                sig = 0
                risk_p = 0.0

                # Bullish Pullback to AVWAP
                # Previous or current bar dipped to or below AVWAP, but current close is green and above AVWAP
                if row['macro_bias'] > 0 and (prev['low'] <= prev_avwap or row['low'] <= avwap):
                    if row['close'] > avwap and row['close'] > row['open']:
                        sl_p = min(row['low'], prev['low']) - 1.2 * self.pip_size
                        risk = max((row['close'] - sl_p) / self.pip_size, self.min_sl)
                        if risk <= self.max_sl:
                            sig = SIGNAL_LONG
                            risk_p = risk
                            traded = True

                # Bearish Pullback to AVWAP
                # Previous or current bar rallied to or above AVWAP, but current close is red and below AVWAP
                elif row['macro_bias'] < 0 and (prev['high'] >= prev_avwap or row['high'] >= avwap):
                    if row['close'] < avwap and row['close'] < row['open']:
                        sl_p = max(row['high'], prev['high']) + 1.2 * self.pip_size
                        risk = max((sl_p - row['close']) / self.pip_size, self.min_sl)
                        if risk <= self.max_sl:
                            sig = SIGNAL_SHORT
                            risk_p = risk
                            traded = True

                if sig != 0:
                    exec_t = row['time_ms'] + dur_ms
                    idx = int(np.searchsorted(tick_ts, exec_t))
                    if idx < n_ticks and signals[idx] == 0:
                        signals[idx] = sig
                        sl_pips_arr[idx] = risk_p
                        tp_pips_arr[idx] = risk_p * self.target_rr

        return signals, sl_pips_arr, tp_pips_arr


# ==============================================================================
# DISC12: New York Anchored VWAP (AVWAP) Pullback Scalper (M15)
# ==============================================================================
class NewYorkAVWAPPullbackScalper(BaseStrategy):
    """
    Anchored VWAP resets at 13:00 UTC (US Open).
    Between 13:45 and 16:30 UTC, if price pulls back to retest US AVWAP and rejects,
    enters in direction of H1 Macro Bias.
    """
    def __init__(self, target_rr: float = 1.60, min_sl: float = 6.0, max_sl: float = 16.0, **kwargs):
        super().__init__(**kwargs)
        self.target_rr = target_rr
        self.min_sl = min_sl
        self.max_sl = max_sl
        self.pip_size = PIP_SIZE
        self.name = "NY_AVWAP_Pullback"

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

        m15 = enrich_phase7_indicators(m15, base_df)
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        for d, day_df in m15.groupby('date'):
            # Trade window: 13:45 to 16:30 UTC
            trade_bars = day_df[((day_df['hour'] == 13) & (day_df['minute'] >= 45)) | (day_df['hour'] == 14) | (day_df['hour'] == 15) | (day_df['hour'] == 16)]
            traded = False

            for idx_val, row in trade_bars.iterrows():
                if traded:
                    break

                i = m15.index.get_loc(idx_val)
                prev = m15.iloc[i-1]
                avwap = row['avwap_ny']
                prev_avwap = prev['avwap_ny']

                if pd.isna(avwap) or pd.isna(prev_avwap):
                    continue

                sig = 0
                risk_p = 0.0

                # Bullish Pullback to US AVWAP
                if row['macro_bias'] > 0 and (prev['low'] <= prev_avwap or row['low'] <= avwap):
                    if row['close'] > avwap and row['close'] > row['open']:
                        sl_p = min(row['low'], prev['low']) - 1.2 * self.pip_size
                        risk = max((row['close'] - sl_p) / self.pip_size, self.min_sl)
                        if risk <= self.max_sl:
                            sig = SIGNAL_LONG
                            risk_p = risk
                            traded = True

                # Bearish Pullback to US AVWAP
                elif row['macro_bias'] < 0 and (prev['high'] >= prev_avwap or row['high'] >= avwap):
                    if row['close'] < avwap and row['close'] < row['open']:
                        sl_p = max(row['high'], prev['high']) + 1.2 * self.pip_size
                        risk = max((sl_p - row['close']) / self.pip_size, self.min_sl)
                        if risk <= self.max_sl:
                            sig = SIGNAL_SHORT
                            risk_p = risk
                            traded = True

                if sig != 0:
                    exec_t = row['time_ms'] + dur_ms
                    idx = int(np.searchsorted(tick_ts, exec_t))
                    if idx < n_ticks and signals[idx] == 0:
                        signals[idx] = sig
                        sl_pips_arr[idx] = risk_p
                        tp_pips_arr[idx] = risk_p * self.target_rr

        return signals, sl_pips_arr, tp_pips_arr


# ==============================================================================
# DISC13: Kaufman Efficiency Ratio (KER) Dynamic Breakout (M15)
# ==============================================================================
class KaufmanEfficiencyBreakout(BaseStrategy):
    """
    Kaufman Efficiency Ratio (KER 10) bursts from noise (<= 0.30) to trend (>= 0.55).
    Enters breakout with H1 Macro Bias during 07:00-15:00 UTC.
    """
    def __init__(self, target_rr: float = 1.60, min_sl: float = 6.0, max_sl: float = 16.0, **kwargs):
        super().__init__(**kwargs)
        self.target_rr = target_rr
        self.min_sl = min_sl
        self.max_sl = max_sl
        self.pip_size = PIP_SIZE
        self.name = "Kaufman_Efficiency_Breakout"

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

        m15 = enrich_phase7_indicators(m15, base_df)
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

            # KER ignition: was low recently (<= 0.32 in last 3 bars), now high >= 0.55
            was_noisy = any(m15['ker_10'].iloc[i-k] <= 0.32 for k in [1, 2, 3])
            is_efficient = row['ker_10'] >= 0.55

            if was_noisy and is_efficient:
                # Bullish Impulse
                if row['macro_bias'] > 0 and row['close'] > row['ema_20'] and row['close'] > row['open']:
                    body_pips = (row['close'] - row['open']) / self.pip_size
                    if body_pips >= 0.6 * row['atr_pips']:
                        sl_p = min(row['low'], prev['low']) - 1.0 * self.pip_size
                        risk = max((row['close'] - sl_p) / self.pip_size, self.min_sl)
                        if risk <= self.max_sl:
                            sig = SIGNAL_LONG
                            risk_p = risk

                # Bearish Impulse
                elif row['macro_bias'] < 0 and row['close'] < row['ema_20'] and row['close'] < row['open']:
                    body_pips = (row['open'] - row['close']) / self.pip_size
                    if body_pips >= 0.6 * row['atr_pips']:
                        sl_p = max(row['high'], prev['high']) + 1.0 * self.pip_size
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
# DISC14: Connors 2-Period RSI Trend-Pullback Scalper (M15)
# ==============================================================================
class ConnorsRSI2PullbackScalper(BaseStrategy):
    """
    Larry Connors 2-Period RSI extreme pullback.
    Bullish: Macro Bullish, Close > EMA50, RSI(2) <= 10.0, current close > open.
    Bearish: Macro Bearish, Close < EMA50, RSI(2) >= 90.0, current close < open.
    Target: 1.5R.
    """
    def __init__(self, target_rr: float = 1.50, min_sl: float = 6.0, max_sl: float = 16.0, **kwargs):
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

        m15 = enrich_phase7_indicators(m15, base_df)
        time_ms = m15['time_ms'].values
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        for i in range(50, len(m15)):
            row = m15.iloc[i]
            prev = m15.iloc[i-1]

            if row['hour'] < 7 or row['hour'] > 16:
                continue

            sig = 0
            risk_p = 0.0

            # Bullish Connors Pullback:
            # Prev bar dipped to extreme RSI(2) <= 12.0 while above EMA50, current bar closes green
            if row['macro_bias'] > 0 and prev['close'] > prev['ema_50'] and prev['rsi_2'] <= 12.0:
                if row['close'] > row['open']:
                    sl_p = min(row['low'], prev['low']) - 1.0 * self.pip_size
                    risk = max((row['close'] - sl_p) / self.pip_size, self.min_sl)
                    if risk <= self.max_sl:
                        sig = SIGNAL_LONG
                        risk_p = risk

            # Bearish Connors Pullback:
            # Prev bar rallied to extreme RSI(2) >= 88.0 while below EMA50, current bar closes red
            elif row['macro_bias'] < 0 and prev['close'] < prev['ema_50'] and prev['rsi_2'] >= 88.0:
                if row['close'] < row['open']:
                    sl_p = max(row['high'], prev['high']) + 1.0 * self.pip_size
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
# DISC15: London-to-NY Momentum Handover (M15)
# ==============================================================================
class LondonToNYHandover(BaseStrategy):
    """
    If London morning (07:00-12:45 UTC) closed strong (top 25% of range) and Macro Bullish,
    enters long at US Open (13:00-13:45 UTC) on green breakout candle beyond London high.
    Inverse for Short. Target: 1.6R with Midpoint stop.
    """
    def __init__(self, target_rr: float = 1.60, min_sl: float = 6.0, max_sl: float = 16.0, **kwargs):
        super().__init__(**kwargs)
        self.target_rr = target_rr
        self.min_sl = min_sl
        self.max_sl = max_sl
        self.pip_size = PIP_SIZE
        self.name = "London_NY_Handover"

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

        m15 = enrich_phase7_indicators(m15, base_df)
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        for d, day_df in m15.groupby('date'):
            # London morning: 07:00 to 12:45 UTC (24 bars)
            lon = day_df[(day_df['hour'] >= 7) & (day_df['hour'] < 13)]
            if len(lon) < 18:
                continue

            lon_h = float(lon['high'].max())
            lon_l = float(lon['low'].min())
            lon_c = float(lon['close'].iloc[-1])
            lon_range = (lon_h - lon_l) / self.pip_size

            if lon_range < 12.0 or lon_range > 50.0:
                continue

            lon_mid = 0.5 * (lon_h + lon_l)
            is_strong_bull = lon_c >= (lon_l + 0.75 * (lon_h - lon_l))
            is_strong_bear = lon_c <= (lon_l + 0.25 * (lon_h - lon_l))

            # US Open entry window: 13:00 to 14:00 UTC
            us_window = day_df[day_df['hour'] == 13]
            traded = False

            for idx_val, row in us_window.iterrows():
                if traded:
                    break

                i = m15.index.get_loc(idx_val)
                prev = m15.iloc[i-1]
                sig = 0
                risk_p = 0.0

                # Bullish Handover Breakout
                if is_strong_bull and row['macro_bias'] > 0 and row['close'] > lon_h and row['close'] > row['open']:
                    risk = max((row['close'] - lon_mid) / self.pip_size, self.min_sl)
                    if risk <= self.max_sl:
                        sig = SIGNAL_LONG
                        risk_p = risk
                        traded = True

                # Bearish Handover Breakout
                elif is_strong_bear and row['macro_bias'] < 0 and row['close'] < lon_l and row['close'] < row['open']:
                    risk = max((lon_mid - row['close']) / self.pip_size, self.min_sl)
                    if risk <= self.max_sl:
                        sig = SIGNAL_SHORT
                        risk_p = risk
                        traded = True

                if sig != 0:
                    exec_t = row['time_ms'] + dur_ms
                    idx = int(np.searchsorted(tick_ts, exec_t))
                    if idx < n_ticks and signals[idx] == 0:
                        signals[idx] = sig
                        sl_pips_arr[idx] = risk_p
                        tp_pips_arr[idx] = risk_p * self.target_rr

        return signals, sl_pips_arr, tp_pips_arr


# ==============================================================================
# LABORATORY EXPERIMENT HARNESS
# ==============================================================================
PHASE7_STRATEGIES = [
    {"code": "DISC11", "name": "London AVWAP Pullback (M15)", "cls": LondonAVWAPPullbackScalper},
    {"code": "DISC12", "name": "New York AVWAP Pullback (M15)", "cls": NewYorkAVWAPPullbackScalper},
    {"code": "DISC13", "name": "Kaufman Efficiency Breakout (M15)", "cls": KaufmanEfficiencyBreakout},
    {"code": "DISC14", "name": "Connors RSI(2) Pullback (M15)", "cls": ConnorsRSI2PullbackScalper},
    {"code": "DISC15", "name": "London-to-NY Handover (M15)", "cls": LondonToNYHandover},
]


def run_week_experiment(week_num: int) -> Dict[str, Any]:
    ticks, meta = get_week_dataset(week_num)
    n_ticks = len(ticks)
    engine = BacktestEngine(pip_size=PIP_SIZE, use_slippage=True)

    week_trades = {}
    for item in PHASE7_STRATEGIES:
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
    print("             DISCOVERY LABORATORY PHASE 7: 52 WEEKS FULL TICK AUDIT (~26M TICKS)")
    print("         Evaluating London AVWAP, NY AVWAP, Kaufman Efficiency, Connors RSI, London-NY Handover")
    print("=" * 135)

    start_t = time.time()
    weeks = list(range(1, 53))

    strat_trades: Dict[str, List[Dict[str, Any]]] = {item["code"]: [] for item in PHASE7_STRATEGIES}
    total_ticks = 0

    with ProcessPoolExecutor() as executor:
        results = list(executor.map(run_week_experiment, weeks))

    for res in results:
        total_ticks += res["ticks"]
        for code, t_list in res["trades"].items():
            strat_trades[code].extend(t_list)

    elapsed = time.time() - start_t
    print(f"\n[+] Executed 52 weeks in {elapsed:.1f}s | Total Ticks Audited: {total_ticks:,}\n")

    print(f"{'Code':<8} | {'Strategy Name':<36} | {'Trades':<6} | {'Trd/Wk':<6} | {'Win Rate':<8} | {'Real RR':<7} | {'PF':<6} | {'Net Pips':<9} | {'Max DD%':<7} | {'Max Recov':<9}")
    print("-" * 135)

    for item in PHASE7_STRATEGIES:
        code = item["code"]
        name = item["name"]
        t_list = strat_trades[code]
        metrics = analyze_combination_performance(t_list, risk_pct=0.40)
        print(f"{code:<8} | {name:<36} | {metrics['total_trades']:<6} | {metrics['trades_per_week']:<6.1f} | {metrics['win_rate']:<7.1f}% | {metrics['realized_rr']:<7.2f} | {metrics['profit_factor']:<6.2f} | {metrics['net_pips']:<+9.1f} | {metrics['max_dd_pct']:<6.2f}% | {metrics['max_recovery_days']:<5.1f} days")

    print("=" * 135)


if __name__ == "__main__":
    main()
