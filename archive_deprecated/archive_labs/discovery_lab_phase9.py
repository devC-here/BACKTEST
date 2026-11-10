#!/usr/bin/env python3
"""
Prop Firm Strategy Discovery Laboratory Phase 9:
High-Frequency & Shorter-Duration Institutional Models on 52 Weeks (~26M Ticks):

1. DISC21: London Cash Open 15-Min ORB (08:00-08:15 UTC Range) (M5 Execution) [Zarattini et al. SSRN]
   - Opening range established in first 15 mins of London Cash Open (08:00-08:15 UTC).
   - Coiled range: 5 to 18 pips.
   - Breakout during 08:15-10:00 UTC in direction of H1 50-EMA macro trend.
   - Tight stop: 0.5 * (H + L) midpoint. Target: 1.8R.

2. DISC22: NY Cash Open 15-Min ORB (13:00-13:15 UTC Range) (M5 Execution) [Zarattini et al. SSRN]
   - Opening range established in first 15 mins of US Cash Open (13:00-13:15 UTC).
   - Coiled range: 5 to 18 pips.
   - Breakout during 13:15-15:00 UTC in direction of H1 50-EMA macro trend.
   - Tight stop: 0.5 * (H + L) midpoint. Target: 1.8R.

3. DISC23: Calibrated Keltner Relative Bandwidth Squeeze (M15)
   - Bandwidth <= 1.35 * rolling_min(30 bars).
   - Impulse body >= 0.60 * ATR breaking beyond KC Upper/Lower in direction of H1 Macro Bias.
   - Stop Loss: KC Middle line (EMA 20), Target: 1.65R.

4. DISC24: Dual Squeeze Momentum (BB Bandwidth + Keltner Compression Hybrid) (M15)
   - Bollinger Bandwidth AND Keltner Bandwidth both in bottom 25% of 40-bar range.
   - Sudden breakout bar breaking both KC Upper and BB Upper.
   - Target: 1.70R with 1.0 ATR stop.

5. DISC25: London Opening 3-Bar Momentum Velocity (M5)
   - Between 08:05 and 09:30 UTC, detects 3 consecutive directional M5 candles with expanding volume.
   - Enters on 3rd candle close in direction of expansion with stop at low/high of the 3-candle sequence.
   - Target: 1.7R.

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


def enrich_phase9_m15(df: pd.DataFrame, base_df: pd.DataFrame) -> pd.DataFrame:
    c = df['close']
    h = df['high']
    l = df['low']
    o = df['open']

    # 1. ATR (14)
    tr = np.maximum(h - l, np.maximum(abs(h - c.shift(1)), abs(l - c.shift(1))))
    df['tr'] = tr
    df['atr_14'] = tr.rolling(14).mean().bfill()
    df['atr_pips'] = df['atr_14'] / PIP_SIZE

    # 2. EMAs & SMAs
    df['ema_20'] = c.ewm(span=20, adjust=False).mean()
    df['sma_20'] = c.rolling(20).mean()
    rolling_std = c.rolling(20).std().fillna(1e-5)

    # 3. Bollinger Bands
    df['bb_upper'] = df['sma_20'] + 2.0 * rolling_std
    df['bb_lower'] = df['sma_20'] - 2.0 * rolling_std
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / np.maximum(df['sma_20'], 1e-6)
    df['bb_width_min30'] = df['bb_width'].rolling(30).min()
    df['bb_width_min40'] = df['bb_width'].rolling(40).min()

    # 4. Keltner Channels
    df['kc_upper'] = df['ema_20'] + 1.5 * df['atr_14']
    df['kc_lower'] = df['ema_20'] - 1.5 * df['atr_14']
    df['kc_width'] = (df['kc_upper'] - df['kc_lower']) / np.maximum(df['ema_20'], 1e-6)
    df['kc_width_min30'] = df['kc_width'].rolling(30).min()
    df['kc_width_min40'] = df['kc_width'].rolling(40).min()

    # 5. H1 50-EMA Macro Trend (zero-lookahead shift(1))
    h1 = base_df.resample('1h').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
    h1['ema50'] = h1['close'].ewm(span=50, adjust=False).mean()
    h1['macro_bias'] = np.where(h1['close'] > h1['ema50'], 1, -1)
    df['macro_bias'] = h1['macro_bias'].shift(1).reindex(df.index, method='ffill').fillna(0)

    # 6. Session Timings
    df['date'] = df.index.date
    df['hour'] = df.index.hour
    df['minute'] = df.index.minute
    df['time_ms'] = df.index.astype('datetime64[ms]').view('int64')

    return df


def enrich_phase9_m5(df: pd.DataFrame, base_df: pd.DataFrame) -> pd.DataFrame:
    c = df['close']
    h = df['high']
    l = df['low']
    o = df['open']

    # ATR (14)
    tr = np.maximum(h - l, np.maximum(abs(h - c.shift(1)), abs(l - c.shift(1))))
    df['tr'] = tr
    df['atr_14'] = tr.rolling(14).mean().bfill()
    df['atr_pips'] = df['atr_14'] / PIP_SIZE

    # H1 50-EMA Macro Trend (zero-lookahead shift(1))
    h1 = base_df.resample('1h').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
    h1['ema50'] = h1['close'].ewm(span=50, adjust=False).mean()
    h1['macro_bias'] = np.where(h1['close'] > h1['ema50'], 1, -1)
    df['macro_bias'] = h1['macro_bias'].shift(1).reindex(df.index, method='ffill').fillna(0)

    df['date'] = df.index.date
    df['hour'] = df.index.hour
    df['minute'] = df.index.minute
    df['time_ms'] = df.index.astype('datetime64[ms]').view('int64')

    vol = tr / PIP_SIZE + 10.0
    df['vol'] = vol

    return df


# ==============================================================================
# DISC21: London Cash Open 15-Min ORB (08:00-08:15 UTC Range) (M5 Execution)
# ==============================================================================
class London15MinORBM5(BaseStrategy):
    """
    Opening range is 08:00 to 08:15 UTC (3 bars on M5).
    Breakout window: 08:15 to 10:00 UTC.
    Coiled range: 5 to 18 pips.
    Stop: 0.5 * (H + L) Midpoint, Target: 1.8R.
    """
    def __init__(self, target_rr: float = 1.80, min_sl: float = 5.0, max_sl: float = 14.0, **kwargs):
        super().__init__(**kwargs)
        self.target_rr = target_rr
        self.min_sl = min_sl
        self.max_sl = max_sl
        self.pip_size = PIP_SIZE
        self.name = "London_15Min_ORB_M5"

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
        m5 = base_df.resample('5min').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
        if len(m5) < 60:
            return signals, sl_pips_arr, tp_pips_arr

        m5 = enrich_phase9_m5(m5, base_df)
        dur_ms = 5 * 60 * 1000
        tick_ts = ticks['timestamp']

        for d, day_df in m5.groupby('date'):
            # 08:00, 08:05, 08:10 bars (3 bars)
            orb_bars = day_df[(day_df['hour'] == 8) & (day_df['minute'] < 15)]
            if len(orb_bars) < 3:
                continue

            orb_h = float(orb_bars['high'].max())
            orb_l = float(orb_bars['low'].min())
            orb_mid = 0.5 * (orb_h + orb_l)
            range_pips = (orb_h - orb_l) / self.pip_size

            if range_pips < 5.0 or range_pips > 18.0:
                continue

            # Breakout window: 08:15 to 10:00 UTC
            trade_bars = day_df[((day_df['hour'] == 8) & (day_df['minute'] >= 15)) | (day_df['hour'] == 9)]
            traded = False

            for idx_val, row in trade_bars.iterrows():
                if traded:
                    break

                i = m5.index.get_loc(idx_val)
                prev = m5.iloc[i-1]
                sig = 0
                risk_p = 0.0

                # Bullish Breakout
                if prev['close'] <= orb_h and row['close'] > orb_h + 0.3 * self.pip_size and row['macro_bias'] > 0:
                    if row['close'] > row['open']:
                        risk = max((row['close'] - orb_mid) / self.pip_size, self.min_sl)
                        if risk <= self.max_sl:
                            sig = SIGNAL_LONG
                            risk_p = risk
                            traded = True

                # Bearish Breakout
                elif prev['close'] >= orb_l and row['close'] < orb_l - 0.3 * self.pip_size and row['macro_bias'] < 0:
                    if row['close'] < row['open']:
                        risk = max((orb_mid - row['close']) / self.pip_size, self.min_sl)
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
# DISC22: NY Cash Open 15-Min ORB (13:00-13:15 UTC Range) (M5 Execution)
# ==============================================================================
class NewYork15MinORBM5(BaseStrategy):
    """
    Opening range is 13:00 to 13:15 UTC (3 bars on M5).
    Breakout window: 13:15 to 15:00 UTC.
    Coiled range: 5 to 18 pips.
    Stop: 0.5 * (H + L) Midpoint, Target: 1.8R.
    """
    def __init__(self, target_rr: float = 1.80, min_sl: float = 5.0, max_sl: float = 14.0, **kwargs):
        super().__init__(**kwargs)
        self.target_rr = target_rr
        self.min_sl = min_sl
        self.max_sl = max_sl
        self.pip_size = PIP_SIZE
        self.name = "NY_15Min_ORB_M5"

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
        m5 = base_df.resample('5min').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
        if len(m5) < 60:
            return signals, sl_pips_arr, tp_pips_arr

        m5 = enrich_phase9_m5(m5, base_df)
        dur_ms = 5 * 60 * 1000
        tick_ts = ticks['timestamp']

        for d, day_df in m5.groupby('date'):
            # 13:00, 13:05, 13:10 bars (3 bars)
            orb_bars = day_df[(day_df['hour'] == 13) & (day_df['minute'] < 15)]
            if len(orb_bars) < 3:
                continue

            orb_h = float(orb_bars['high'].max())
            orb_l = float(orb_bars['low'].min())
            orb_mid = 0.5 * (orb_h + orb_l)
            range_pips = (orb_h - orb_l) / self.pip_size

            if range_pips < 5.0 or range_pips > 18.0:
                continue

            # Breakout window: 13:15 to 15:00 UTC
            trade_bars = day_df[((day_df['hour'] == 13) & (day_df['minute'] >= 15)) | (day_df['hour'] == 14)]
            traded = False

            for idx_val, row in trade_bars.iterrows():
                if traded:
                    break

                i = m5.index.get_loc(idx_val)
                prev = m5.iloc[i-1]
                sig = 0
                risk_p = 0.0

                # Bullish Breakout
                if prev['close'] <= orb_h and row['close'] > orb_h + 0.3 * self.pip_size and row['macro_bias'] > 0:
                    if row['close'] > row['open']:
                        risk = max((row['close'] - orb_mid) / self.pip_size, self.min_sl)
                        if risk <= self.max_sl:
                            sig = SIGNAL_LONG
                            risk_p = risk
                            traded = True

                # Bearish Breakout
                elif prev['close'] >= orb_l and row['close'] < orb_l - 0.3 * self.pip_size and row['macro_bias'] < 0:
                    if row['close'] < row['open']:
                        risk = max((orb_mid - row['close']) / self.pip_size, self.min_sl)
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
# DISC23: Calibrated Keltner Relative Bandwidth Squeeze (M15)
# ==============================================================================
class CalibratedKeltnerSqueeze(BaseStrategy):
    """
    Keltner Bandwidth <= 1.35 * rolling_min(30 bars).
    Impulse body >= 0.60 * ATR breaking beyond KC Upper/Lower.
    Target: 1.65R with KC EMA20 stop.
    """
    def __init__(self, target_rr: float = 1.65, min_sl: float = 6.0, max_sl: float = 16.0, **kwargs):
        super().__init__(**kwargs)
        self.target_rr = target_rr
        self.min_sl = min_sl
        self.max_sl = max_sl
        self.pip_size = PIP_SIZE
        self.name = "Calibrated_Keltner_Squeeze"

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

        m15 = enrich_phase9_m15(m15, base_df)
        time_ms = m15['time_ms'].values
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        for i in range(35, len(m15)):
            row = m15.iloc[i]
            prev = m15.iloc[i-1]

            if row['hour'] < 7 or row['hour'] > 16:
                continue

            sig = 0
            risk_p = 0.0

            # Was coiled in recent bars (lookback 3 bars)
            was_coiled = any(m15['kc_width'].iloc[i-k] <= 1.35 * m15['kc_width_min30'].iloc[i-k] for k in [1, 2, 3])
            if not was_coiled:
                continue

            # Bullish Breakout
            if row['macro_bias'] > 0 and row['close'] > row['kc_upper'] and row['close'] > row['open']:
                body_pips = (row['close'] - row['open']) / self.pip_size
                if body_pips >= 0.60 * row['atr_pips']:
                    sl_p = min(row['low'], row['ema_20']) - 1.0 * self.pip_size
                    risk = max((row['close'] - sl_p) / self.pip_size, self.min_sl)
                    if risk <= self.max_sl:
                        sig = SIGNAL_LONG
                        risk_p = risk

            # Bearish Breakout
            elif row['macro_bias'] < 0 and row['close'] < row['kc_lower'] and row['close'] < row['open']:
                body_pips = (row['open'] - row['close']) / self.pip_size
                if body_pips >= 0.60 * row['atr_pips']:
                    sl_p = max(row['high'], row['ema_20']) + 1.0 * self.pip_size
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
# DISC24: Dual Squeeze Momentum (BB + Keltner Hybrid) (M15)
# ==============================================================================
class DualSqueezeHybrid(BaseStrategy):
    """
    Both Bollinger Bandwidth and Keltner Bandwidth coiled <= 1.30 * min(35 bars).
    Breakout bar breaks both BB Upper and KC Upper in direction of Macro Bias.
    Target: 1.70R with 1.0 ATR stop.
    """
    def __init__(self, target_rr: float = 1.70, min_sl: float = 6.0, max_sl: float = 16.0, **kwargs):
        super().__init__(**kwargs)
        self.target_rr = target_rr
        self.min_sl = min_sl
        self.max_sl = max_sl
        self.pip_size = PIP_SIZE
        self.name = "Dual_Squeeze_Hybrid"

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

        m15 = enrich_phase9_m15(m15, base_df)
        time_ms = m15['time_ms'].values
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        for i in range(35, len(m15)):
            row = m15.iloc[i]
            prev = m15.iloc[i-1]

            if row['hour'] < 7 or row['hour'] > 16:
                continue

            sig = 0
            risk_p = 0.0

            # Dual coiling in recent bars
            was_dual_coiled = any(
                (m15['bb_width'].iloc[i-k] <= 1.35 * m15['bb_width_min30'].iloc[i-k]) and
                (m15['kc_width'].iloc[i-k] <= 1.35 * m15['kc_width_min30'].iloc[i-k])
                for k in [1, 2, 3]
            )
            if not was_dual_coiled:
                continue

            # Bullish Breakout
            if row['macro_bias'] > 0 and row['close'] > max(row['bb_upper'], row['kc_upper']) and row['close'] > row['open']:
                body_pips = (row['close'] - row['open']) / self.pip_size
                if body_pips >= 0.65 * row['atr_pips']:
                    sl_p = min(row['low'], prev['low']) - 1.0 * self.pip_size
                    risk = max((row['close'] - sl_p) / self.pip_size, self.min_sl)
                    if risk <= self.max_sl:
                        sig = SIGNAL_LONG
                        risk_p = risk

            # Bearish Breakout
            elif row['macro_bias'] < 0 and row['close'] < min(row['bb_lower'], row['kc_lower']) and row['close'] < row['open']:
                body_pips = (row['open'] - row['close']) / self.pip_size
                if body_pips >= 0.65 * row['atr_pips']:
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
# DISC25: London Opening 3-Bar Momentum Velocity (M5)
# ==============================================================================
class London3BarMomentumM5(BaseStrategy):
    """
    Between 08:05 and 09:30 UTC on M5, detects 3 consecutive directional candles with expanding volume.
    Enters on 3rd candle close with stop at low/high of the 3-candle sequence. Target: 1.7R.
    """
    def __init__(self, target_rr: float = 1.70, min_sl: float = 5.0, max_sl: float = 14.0, **kwargs):
        super().__init__(**kwargs)
        self.target_rr = target_rr
        self.min_sl = min_sl
        self.max_sl = max_sl
        self.pip_size = PIP_SIZE
        self.name = "London_3Bar_Momentum_M5"

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
        m5 = base_df.resample('5min').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
        if len(m5) < 60:
            return signals, sl_pips_arr, tp_pips_arr

        m5 = enrich_phase9_m5(m5, base_df)
        time_ms = m5['time_ms'].values
        dur_ms = 5 * 60 * 1000
        tick_ts = ticks['timestamp']

        for i in range(20, len(m5)):
            row = m5.iloc[i]
            b1 = m5.iloc[i-1]
            b2 = m5.iloc[i-2]

            # Active window: 08:05 to 09:30 UTC
            if not ((row['hour'] == 8 and row['minute'] >= 10) or (row['hour'] == 9 and row['minute'] <= 30)):
                continue

            sig = 0
            risk_p = 0.0

            # 3 consecutive green bars with increasing close and macro bullish
            if row['macro_bias'] > 0:
                if row['close'] > row['open'] and b1['close'] > b1['open'] and b2['close'] > b2['open']:
                    if row['close'] > b1['close'] > b2['close']:
                        # Total displacement over 3 bars >= 1.2 * ATR
                        disp = (row['close'] - b2['open']) / self.pip_size
                        if disp >= 1.2 * row['atr_pips']:
                            sl_p = min(row['low'], b1['low'], b2['low']) - 1.0 * self.pip_size
                            risk = max((row['close'] - sl_p) / self.pip_size, self.min_sl)
                            if risk <= self.max_sl:
                                sig = SIGNAL_LONG
                                risk_p = risk

            # 3 consecutive red bars with decreasing close and macro bearish
            elif row['macro_bias'] < 0:
                if row['close'] < row['open'] and b1['close'] < b1['open'] and b2['close'] < b2['open']:
                    if row['close'] < b1['close'] < b2['close']:
                        disp = (b2['open'] - row['close']) / self.pip_size
                        if disp >= 1.2 * row['atr_pips']:
                            sl_p = max(row['high'], b1['high'], b2['high']) + 1.0 * self.pip_size
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
PHASE9_STRATEGIES = [
    {"code": "DISC21", "name": "London 15M ORB (M5 Exec)", "cls": London15MinORBM5},
    {"code": "DISC22", "name": "NY 15M ORB (M5 Exec)", "cls": NewYork15MinORBM5},
    {"code": "DISC23", "name": "Calibrated Keltner Squeeze (M15)", "cls": CalibratedKeltnerSqueeze},
    {"code": "DISC24", "name": "Dual Squeeze Hybrid (M15)", "cls": DualSqueezeHybrid},
    {"code": "DISC25", "name": "London 3-Bar Momentum (M5)", "cls": London3BarMomentumM5},
]


def run_week_experiment(week_num: int) -> Dict[str, Any]:
    ticks, meta = get_week_dataset(week_num)
    n_ticks = len(ticks)
    engine = BacktestEngine(pip_size=PIP_SIZE, use_slippage=True)

    week_trades = {}
    for item in PHASE9_STRATEGIES:
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
    print("             DISCOVERY LABORATORY PHASE 9: 52 WEEKS FULL TICK AUDIT (~26M TICKS)")
    print("       Evaluating London 15M ORB, NY 15M ORB, Calibrated Keltner, Dual Squeeze, London 3-Bar M5")
    print("=" * 135)

    start_t = time.time()
    weeks = list(range(1, 53))

    strat_trades: Dict[str, List[Dict[str, Any]]] = {item["code"]: [] for item in PHASE9_STRATEGIES}
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

    for item in PHASE9_STRATEGIES:
        code = item["code"]
        name = item["name"]
        t_list = strat_trades[code]
        metrics = analyze_combination_performance(t_list, risk_pct=0.40)
        print(f"{code:<8} | {name:<36} | {metrics['total_trades']:<6} | {metrics['trades_per_week']:<6.1f} | {metrics['win_rate']:<7.1f}% | {metrics['realized_rr']:<7.2f} | {metrics['profit_factor']:<6.2f} | {metrics['net_pips']:<+9.1f} | {metrics['max_dd_pct']:<6.2f}% | {metrics['max_recovery_days']:<5.1f} days")

    print("=" * 135)


if __name__ == "__main__":
    main()
