#!/usr/bin/env python3
"""
Prop Firm Strategy Discovery Laboratory Phase 8:
Evaluating 5 High-Potential Session Breakout & Structural Coiling Models on 52 Weeks (~26M Ticks):

1. DISC16: Frankfurt DAX Cash Open Breakout (07:00 UTC) (M15)
   - Reference range 04:00-06:45 UTC (Late Asian consolidation). Range coiled (8 to 25 pips).
   - European/German Cash Open (07:00-08:30 UTC) impulse breakout beyond box.
   - H1 50-EMA Macro Trend alignment (shift(1)). Target: 1.6R with Midpoint stop.

2. DISC17: London Lunch Tight Coil Breakout (11:00-13:00 UTC Range) (M15)
   - Reference range 11:00-12:45 UTC (European lunch lull). Range coiled (8 to 22 pips).
   - US Pre-Market/Open influx (13:00-14:30 UTC) breakout beyond lunch box.
   - H1 50-EMA Macro Trend alignment. Target: 1.6R with Midpoint stop.

3. DISC18: Tokyo Cash Open Breakout (00:00 UTC) (M15)
   - Reference range 21:00-23:45 UTC (Late NY / Sydney quiet zone). Range coiled (6 to 20 pips).
   - Tokyo Cash Open (00:00-02:00 UTC) impulse breakout beyond box.
   - H1 50-EMA Macro Trend alignment. Target: 1.6R with Midpoint stop.

4. DISC19: Breakout-Pullback-Continuation (BPC) on Asian Range (M15)
   - Asian Range 00:00-06:00 UTC (10 to 30 pips).
   - Breakout occurs at 07:00-08:30 UTC.
   - Instead of chasing, waits for M15 pullback to retest the broken Asian level (acting as S/R) during 08:30-11:30 UTC!
   - Enters on confirmation bounce candle with tight stop just beyond retest swing. Target: 1.8R.

5. DISC20: Keltner Relative Bandwidth Squeeze (M15)
   - Keltner Channel (EMA 20, 1.5 * ATR 14).
   - Bandwidth = (KC_Upper - KC_Lower) / EMA20.
   - Relative coiling: KC_Bandwidth <= 1.25 * rolling_min(40 bars).
   - Trigger: Close breaks beyond KC Upper/Lower with body >= 0.7 * ATR in direction of H1 50-EMA.
   - Dynamic stop: KC Middle line (EMA 20) or 1.0 ATR. Target: 1.65R.

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


def enrich_phase8_indicators(df: pd.DataFrame, base_df: pd.DataFrame) -> pd.DataFrame:
    c = df['close']
    h = df['high']
    l = df['low']
    o = df['open']

    # 1. ATR (14)
    tr = np.maximum(h - l, np.maximum(abs(h - c.shift(1)), abs(l - c.shift(1))))
    df['tr'] = tr
    df['atr_14'] = tr.rolling(14).mean().bfill()
    df['atr_pips'] = df['atr_14'] / PIP_SIZE

    # 2. EMAs
    df['ema_20'] = c.ewm(span=20, adjust=False).mean()

    # 3. Keltner Channels
    df['kc_upper'] = df['ema_20'] + 1.5 * df['atr_14']
    df['kc_lower'] = df['ema_20'] - 1.5 * df['atr_14']
    kc_width = (df['kc_upper'] - df['kc_lower']) / np.maximum(df['ema_20'], 1e-6)
    df['kc_width'] = kc_width
    df['kc_width_min40'] = kc_width.rolling(40).min()

    # 4. H1 50-EMA Macro Trend (zero-lookahead shift(1))
    h1 = base_df.resample('1h').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
    h1['ema50'] = h1['close'].ewm(span=50, adjust=False).mean()
    h1['macro_bias'] = np.where(h1['close'] > h1['ema50'], 1, -1)
    df['macro_bias'] = h1['macro_bias'].shift(1).reindex(df.index, method='ffill').fillna(0)

    # 5. Session Timings
    df['date'] = df.index.date
    df['hour'] = df.index.hour
    df['minute'] = df.index.minute
    df['time_ms'] = df.index.astype('datetime64[ms]').view('int64')

    return df


# ==============================================================================
# DISC16: Frankfurt DAX Cash Open Breakout (07:00 UTC) (M15)
# ==============================================================================
class FrankfurtCashOpenBreakout(BaseStrategy):
    """
    Reference range 04:00-06:45 UTC (Late Asian consolidation).
    Breakout at European Cash Open (07:00-08:30 UTC).
    Target: 1.6R with Midpoint stop.
    """
    def __init__(self, target_rr: float = 1.60, min_sl: float = 6.0, max_sl: float = 16.0, **kwargs):
        super().__init__(**kwargs)
        self.target_rr = target_rr
        self.min_sl = min_sl
        self.max_sl = max_sl
        self.pip_size = PIP_SIZE
        self.name = "Frankfurt_Cash_Breakout"

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

        m15 = enrich_phase8_indicators(m15, base_df)
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        for d, day_df in m15.groupby('date'):
            # 04:00 to 06:45 UTC (12 bars)
            ref_bars = day_df[(day_df['hour'] >= 4) & (day_df['hour'] < 7)]
            if len(ref_bars) < 8:
                continue

            ref_h = float(ref_bars['high'].max())
            ref_l = float(ref_bars['low'].min())
            ref_mid = 0.5 * (ref_h + ref_l)
            range_pips = (ref_h - ref_l) / self.pip_size

            if range_pips < 8.0 or range_pips > 26.0:
                continue

            # Breakout window: 07:00 to 08:30 UTC
            open_bars = day_df[(day_df['hour'] == 7) | ((day_df['hour'] == 8) & (day_df['minute'] <= 30))]
            traded = False

            for idx_val, row in open_bars.iterrows():
                if traded:
                    break

                i = m15.index.get_loc(idx_val)
                prev = m15.iloc[i-1]
                sig = 0
                risk_p = 0.0

                # Bullish Breakout
                if prev['close'] <= ref_h and row['close'] > ref_h + 0.5 * self.pip_size and row['macro_bias'] > 0:
                    if row['close'] > row['open']:
                        risk = max((row['close'] - ref_mid) / self.pip_size, self.min_sl)
                        if risk <= self.max_sl:
                            sig = SIGNAL_LONG
                            risk_p = risk
                            traded = True

                # Bearish Breakout
                elif prev['close'] >= ref_l and row['close'] < ref_l - 0.5 * self.pip_size and row['macro_bias'] < 0:
                    if row['close'] < row['open']:
                        risk = max((ref_mid - row['close']) / self.pip_size, self.min_sl)
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
# DISC17: London Lunch Tight Coil Breakout (11:00-13:00 UTC Range) (M15)
# ==============================================================================
class LondonLunchBreakout(BaseStrategy):
    """
    Reference range 11:00-12:45 UTC (European lunch lull). Coiled 8 to 22 pips.
    Breakout at US Pre-Market/Open (13:00-14:30 UTC).
    Target: 1.6R with Midpoint stop.
    """
    def __init__(self, target_rr: float = 1.60, min_sl: float = 6.0, max_sl: float = 16.0, **kwargs):
        super().__init__(**kwargs)
        self.target_rr = target_rr
        self.min_sl = min_sl
        self.max_sl = max_sl
        self.pip_size = PIP_SIZE
        self.name = "London_Lunch_Breakout"

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

        m15 = enrich_phase8_indicators(m15, base_df)
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        for d, day_df in m15.groupby('date'):
            # 11:00 to 12:45 UTC (8 bars)
            lunch_bars = day_df[(day_df['hour'] >= 11) & (day_df['hour'] < 13)]
            if len(lunch_bars) < 6:
                continue

            lunch_h = float(lunch_bars['high'].max())
            lunch_l = float(lunch_bars['low'].min())
            lunch_mid = 0.5 * (lunch_h + lunch_l)
            range_pips = (lunch_h - lunch_l) / self.pip_size

            if range_pips < 8.0 or range_pips > 24.0:
                continue

            # Breakout window: 13:00 to 14:30 UTC
            us_bars = day_df[(day_df['hour'] == 13) | ((day_df['hour'] == 14) & (day_df['minute'] <= 30))]
            traded = False

            for idx_val, row in us_bars.iterrows():
                if traded:
                    break

                i = m15.index.get_loc(idx_val)
                prev = m15.iloc[i-1]
                sig = 0
                risk_p = 0.0

                # Bullish Breakout
                if prev['close'] <= lunch_h and row['close'] > lunch_h + 0.5 * self.pip_size and row['macro_bias'] > 0:
                    if row['close'] > row['open']:
                        risk = max((row['close'] - lunch_mid) / self.pip_size, self.min_sl)
                        if risk <= self.max_sl:
                            sig = SIGNAL_LONG
                            risk_p = risk
                            traded = True

                # Bearish Breakout
                elif prev['close'] >= lunch_l and row['close'] < lunch_l - 0.5 * self.pip_size and row['macro_bias'] < 0:
                    if row['close'] < row['open']:
                        risk = max((lunch_mid - row['close']) / self.pip_size, self.min_sl)
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
# DISC18: Tokyo Cash Open Breakout (00:00 UTC) (M15)
# ==============================================================================
class TokyoCashOpenBreakout(BaseStrategy):
    """
    Reference range 21:00-23:45 UTC (Late NY / Sydney quiet zone). Coiled 6 to 20 pips.
    Breakout at Tokyo Cash Open (00:00-02:00 UTC).
    Target: 1.6R with Midpoint stop.
    """
    def __init__(self, target_rr: float = 1.60, min_sl: float = 5.0, max_sl: float = 15.0, **kwargs):
        super().__init__(**kwargs)
        self.target_rr = target_rr
        self.min_sl = min_sl
        self.max_sl = max_sl
        self.pip_size = PIP_SIZE
        self.name = "Tokyo_Cash_Breakout"

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

        m15 = enrich_phase8_indicators(m15, base_df)
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        dates = sorted(list(set(m15['date'])))
        for d_idx in range(1, len(dates)):
            curr_date = dates[d_idx]
            prev_date = dates[d_idx - 1]

            prev_day_bars = m15[m15['date'] == prev_date]
            curr_day_bars = m15[m15['date'] == curr_date]

            # Late NY bars: 21:00 to 23:45 UTC of previous day
            late_ny = prev_day_bars[prev_day_bars['hour'] >= 21]
            if len(late_ny) < 8:
                continue

            ny_h = float(late_ny['high'].max())
            ny_l = float(late_ny['low'].min())
            ny_mid = 0.5 * (ny_h + ny_l)
            range_pips = (ny_h - ny_l) / self.pip_size

            if range_pips < 6.0 or range_pips > 20.0:
                continue

            # Tokyo Open: 00:00 to 02:00 UTC of current day
            tokyo_bars = curr_day_bars[curr_day_bars['hour'] < 2]
            traded = False

            for idx_val, row in tokyo_bars.iterrows():
                if traded:
                    break

                i = m15.index.get_loc(idx_val)
                prev = m15.iloc[i-1]
                sig = 0
                risk_p = 0.0

                # Bullish Breakout
                if prev['close'] <= ny_h and row['close'] > ny_h + 0.5 * self.pip_size and row['macro_bias'] > 0:
                    if row['close'] > row['open']:
                        risk = max((row['close'] - ny_mid) / self.pip_size, self.min_sl)
                        if risk <= self.max_sl:
                            sig = SIGNAL_LONG
                            risk_p = risk
                            traded = True

                # Bearish Breakout
                elif prev['close'] >= ny_l and row['close'] < ny_l - 0.5 * self.pip_size and row['macro_bias'] < 0:
                    if row['close'] < row['open']:
                        risk = max((ny_mid - row['close']) / self.pip_size, self.min_sl)
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
# DISC19: Breakout-Pullback-Continuation (BPC) on Asian Range (M15)
# ==============================================================================
class AsianBreakoutPullbackContinuation(BaseStrategy):
    """
    Asian Range 00:00-06:00 UTC. Breakout happens at 07:00-08:30 UTC.
    Instead of chasing, waits for pullback to retest broken Asian level between 08:30-11:30 UTC.
    Enters on confirmation bounce candle with tight stop just beyond retest swing. Target: 1.8R.
    """
    def __init__(self, target_rr: float = 1.80, min_sl: float = 5.0, max_sl: float = 14.0, **kwargs):
        super().__init__(**kwargs)
        self.target_rr = target_rr
        self.min_sl = min_sl
        self.max_sl = max_sl
        self.pip_size = PIP_SIZE
        self.name = "Asian_BPC_Retest"

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

        m15 = enrich_phase8_indicators(m15, base_df)
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        for d, day_df in m15.groupby('date'):
            asia = day_df[day_df['hour'] < 6]
            if len(asia) < 18:
                continue

            asia_h = float(asia['high'].max())
            asia_l = float(asia['low'].min())
            asia_range = (asia_h - asia_l) / self.pip_size

            if asia_range < 10.0 or asia_range > 35.0:
                continue

            # Look for initial breakout between 07:00 and 08:30 UTC
            morning_bars = day_df[(day_df['hour'] == 7) | ((day_df['hour'] == 8) & (day_df['minute'] <= 30))]
            broke_high = any(b['close'] > asia_h for _, b in morning_bars.iterrows())
            broke_low = any(b['close'] < asia_l for _, b in morning_bars.iterrows())

            if not (broke_high or broke_low):
                continue

            # Retest window: 08:45 to 11:30 UTC
            retest_window = day_df[((day_df['hour'] == 8) & (day_df['minute'] >= 45)) | (day_df['hour'] == 9) | (day_df['hour'] == 10) | (day_df['hour'] == 11)]
            traded = False

            for idx_val, row in retest_window.iterrows():
                if traded:
                    break

                i = m15.index.get_loc(idx_val)
                prev = m15.iloc[i-1]
                sig = 0
                risk_p = 0.0

                # Bullish Retest: Price broke high earlier, now pulled back to test within 3 pips of asia_h
                # and bounces (row close > open, close > prev high)
                if broke_high and row['macro_bias'] > 0:
                    if (prev['low'] <= asia_h + 3.0 * self.pip_size and prev['close'] >= asia_h - 2.0 * self.pip_size) or \
                       (row['low'] <= asia_h + 3.0 * self.pip_size and row['close'] >= asia_h - 2.0 * self.pip_size):
                        if row['close'] > row['open'] and row['close'] > prev['high']:
                            sl_p = min(row['low'], prev['low']) - 1.2 * self.pip_size
                            risk = max((row['close'] - sl_p) / self.pip_size, self.min_sl)
                            if risk <= self.max_sl:
                                sig = SIGNAL_LONG
                                risk_p = risk
                                traded = True

                # Bearish Retest: Price broke low earlier, now rallied to test within 3 pips of asia_l
                # and bounces downward (row close < open, close < prev low)
                elif broke_low and row['macro_bias'] < 0:
                    if (prev['high'] >= asia_l - 3.0 * self.pip_size and prev['close'] <= asia_l + 2.0 * self.pip_size) or \
                       (row['high'] >= asia_l - 3.0 * self.pip_size and row['close'] <= asia_l + 2.0 * self.pip_size):
                        if row['close'] < row['open'] and row['close'] < prev['low']:
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
# DISC20: Keltner Relative Bandwidth Squeeze (M15)
# ==============================================================================
class KeltnerRelativeBandwidthSqueeze(BaseStrategy):
    """
    Relative Volatility Contraction on Keltner Channel.
    Bandwidth <= 1.25 * rolling_min(40 bars).
    Impulse body >= 0.70 * ATR breaking beyond KC Upper/Lower in direction of H1 Macro Bias.
    Target: 1.65R.
    """
    def __init__(self, target_rr: float = 1.65, min_sl: float = 6.0, max_sl: float = 16.0, **kwargs):
        super().__init__(**kwargs)
        self.target_rr = target_rr
        self.min_sl = min_sl
        self.max_sl = max_sl
        self.pip_size = PIP_SIZE
        self.name = "Keltner_Relative_Squeeze"

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

        m15 = enrich_phase8_indicators(m15, base_df)
        time_ms = m15['time_ms'].values
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        for i in range(45, len(m15)):
            row = m15.iloc[i]
            prev = m15.iloc[i-1]

            if row['hour'] < 7 or row['hour'] > 16:
                continue

            sig = 0
            risk_p = 0.0

            # Was coiled in recent bars
            was_coiled = any(m15['kc_width'].iloc[i-k] <= 1.25 * m15['kc_width_min40'].iloc[i-k] for k in [1, 2, 3])
            if not was_coiled:
                continue

            # Bullish Breakout
            if row['macro_bias'] > 0 and row['close'] > row['kc_upper'] and row['close'] > row['open']:
                body_pips = (row['close'] - row['open']) / self.pip_size
                if body_pips >= 0.70 * row['atr_pips']:
                    sl_p = min(row['low'], row['ema_20']) - 1.0 * self.pip_size
                    risk = max((row['close'] - sl_p) / self.pip_size, self.min_sl)
                    if risk <= self.max_sl:
                        sig = SIGNAL_LONG
                        risk_p = risk

            # Bearish Breakout
            elif row['macro_bias'] < 0 and row['close'] < row['kc_lower'] and row['close'] < row['open']:
                body_pips = (row['open'] - row['close']) / self.pip_size
                if body_pips >= 0.70 * row['atr_pips']:
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
# LABORATORY EXPERIMENT HARNESS
# ==============================================================================
PHASE8_STRATEGIES = [
    {"code": "DISC16", "name": "Frankfurt Cash Open Breakout (M15)", "cls": FrankfurtCashOpenBreakout},
    {"code": "DISC17", "name": "London Lunch Coil Breakout (M15)", "cls": LondonLunchBreakout},
    {"code": "DISC18", "name": "Tokyo Cash Open Breakout (M15)", "cls": TokyoCashOpenBreakout},
    {"code": "DISC19", "name": "Asian BPC Retest Continuation (M15)", "cls": AsianBreakoutPullbackContinuation},
    {"code": "DISC20", "name": "Keltner Relative Squeeze (M15)", "cls": KeltnerRelativeBandwidthSqueeze},
]


def run_week_experiment(week_num: int) -> Dict[str, Any]:
    ticks, meta = get_week_dataset(week_num)
    n_ticks = len(ticks)
    engine = BacktestEngine(pip_size=PIP_SIZE, use_slippage=True)

    week_trades = {}
    for item in PHASE8_STRATEGIES:
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
    print("             DISCOVERY LABORATORY PHASE 8: 52 WEEKS FULL TICK AUDIT (~26M TICKS)")
    print("         Evaluating Frankfurt Cash Open, London Lunch Coil, Tokyo Open, Asian BPC, Keltner Squeeze")
    print("=" * 135)

    start_t = time.time()
    weeks = list(range(1, 53))

    strat_trades: Dict[str, List[Dict[str, Any]]] = {item["code"]: [] for item in PHASE8_STRATEGIES}
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

    for item in PHASE8_STRATEGIES:
        code = item["code"]
        name = item["name"]
        t_list = strat_trades[code]
        metrics = analyze_combination_performance(t_list, risk_pct=0.40)
        print(f"{code:<8} | {name:<36} | {metrics['total_trades']:<6} | {metrics['trades_per_week']:<6.1f} | {metrics['win_rate']:<7.1f}% | {metrics['realized_rr']:<7.2f} | {metrics['profit_factor']:<6.2f} | {metrics['net_pips']:<+9.1f} | {metrics['max_dd_pct']:<6.2f}% | {metrics['max_recovery_days']:<5.1f} days")

    print("=" * 135)


if __name__ == "__main__":
    main()
