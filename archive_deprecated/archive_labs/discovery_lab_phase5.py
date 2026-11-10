#!/usr/bin/env python3
"""
Prop Firm Strategy Discovery Laboratory Phase 5:
Evaluating 5 Brand New Institutional & Quantitative Models on 52 Weeks (~26M Ticks):

1. DISC01: Guppy Multiple Moving Average (GMMA) Value Pocket Scalper (M15)
   - 6 Short-Term EMAs (3,5,8,10,12,15) + 6 Long-Term EMAs (30,35,40,45,50,60)
   - Fanning trend alignment + pullback into long-term pocket without breaking bottom
   - Rejection candle trigger with 1.6R target

2. DISC02: Frankfurt/London Pre-Market ORB (06:00-08:00 UTC Range Breakout) (M15)
   - Range established 06:00-07:45 UTC (Frankfurt open & London pre-open)
   - London Cash Open (08:00-10:00 UTC) impulse breakout beyond box
   - Filtered by H1 50-EMA macro trend (zero lookahead shift(1))
   - Target: 1.6R with Midpoint stop

3. DISC03: New York Open (13:00 UTC) London-NY Overlap Breakout (M15)
   - Reference range 09:00-13:00 UTC (mid-London consolidation)
   - NY Cash Open (13:00-15:00 UTC) breakout beyond reference box
   - Filtered by H1 50-EMA macro trend
   - Target: 1.6R with Midpoint stop

4. DISC04: J. Welles Wilder's Classic RSI(14) Failure Swing (M15)
   - Bullish: RSI drops < 32, rallies to M, dips but stays > 32, then breaks above M peak!
   - Bearish: RSI rises > 68, pulls back to N, rallies but stays < 68, then breaks below N trough!
   - Dynamic ATR stop with 1.8R target

5. DISC05: Session VWAP 2.0-Sigma Statistical Extreme Rejection (M15)
   - Daily anchored VWAP with rolling Standard Deviation bands
   - When price reaches +/- 2.0 Sigma extreme and prints pin-bar rejection (wick >= 45%)
   - Target: 1.5R back to VWAP mean

All models evaluated strictly with zero-lookahead on authentic tick data.
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


def enrich_phase5_indicators(df: pd.DataFrame, base_df: pd.DataFrame) -> pd.DataFrame:
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

    # 2. GMMA Fast & Slow Groups
    fast_periods = [3, 5, 8, 10, 12, 15]
    slow_periods = [30, 35, 40, 45, 50, 60]
    for p in fast_periods:
        df[f'gmma_fast_{p}'] = c.ewm(span=p, adjust=False).mean()
    for p in slow_periods:
        df[f'gmma_slow_{p}'] = c.ewm(span=p, adjust=False).mean()

    df['gmma_fast_min'] = df[[f'gmma_fast_{p}' for p in fast_periods]].min(axis=1)
    df['gmma_fast_max'] = df[[f'gmma_fast_{p}' for p in fast_periods]].max(axis=1)
    df['gmma_slow_min'] = df[[f'gmma_slow_{p}' for p in slow_periods]].min(axis=1)
    df['gmma_slow_max'] = df[[f'gmma_slow_{p}' for p in slow_periods]].max(axis=1)

    # 3. RSI (14)
    c_diff = c.diff()
    gain = c_diff.clip(lower=0.0).ewm(alpha=1.0/14, adjust=False).mean()
    loss = (-c_diff.clip(upper=0.0)).ewm(alpha=1.0/14, adjust=False).mean()
    rs = gain / np.maximum(loss, 1e-9)
    df['rsi_14'] = 100.0 - (100.0 / (1.0 + rs))

    # 4. H1 50-EMA Macro Trend (zero-lookahead shift(1))
    h1 = base_df.resample('1h').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
    h1['ema50'] = h1['close'].ewm(span=50, adjust=False).mean()
    h1['macro_bias'] = np.where(h1['close'] > h1['ema50'], 1, -1)
    df['macro_bias'] = h1['macro_bias'].shift(1).reindex(df.index, method='ffill').fillna(0)

    # 5. Session VWAP & SD Bands (00:00 UTC daily reset)
    df['date'] = df.index.date
    df['hour'] = df.index.hour
    df['minute'] = df.index.minute
    df['time_ms'] = df.index.astype('datetime64[ms]').view('int64')

    vol = tr / PIP_SIZE + 10.0
    cum_vol = vol.groupby(df['date']).cumsum()
    cum_pv = (c * vol).groupby(df['date']).cumsum()
    vwap = cum_pv / np.maximum(cum_vol, 1e-9)
    df['vwap'] = vwap

    cum_p2_v = ((c ** 2) * vol).groupby(df['date']).cumsum()
    variance = (cum_p2_v / np.maximum(cum_vol, 1e-9)) - (vwap ** 2)
    std_dev = np.sqrt(np.maximum(variance, 1e-12))
    df['vwap_up2'] = vwap + 2.0 * std_dev
    df['vwap_low2'] = vwap - 2.0 * std_dev

    return df


# ==============================================================================
# DISC01: Guppy Multiple Moving Average (GMMA) Value Pocket Scalper (M15)
# ==============================================================================
class GuppyValuePocketScalper(BaseStrategy):
    def __init__(self, target_rr: float = 1.60, min_sl: float = 6.0, max_sl: float = 16.0, **kwargs):
        super().__init__(**kwargs)
        self.target_rr = target_rr
        self.min_sl = min_sl
        self.max_sl = max_sl
        self.pip_size = PIP_SIZE
        self.name = "Guppy_Value_Pocket_Scalper"

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
        if len(m15) < 65:
            return signals, sl_pips_arr, tp_pips_arr

        m15 = enrich_phase5_indicators(m15, base_df)
        time_ms = m15['time_ms'].values
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        for i in range(30, len(m15)):
            row = m15.iloc[i]
            prev = m15.iloc[i-1]

            if row['hour'] < 7 or row['hour'] > 15:
                continue

            sig = 0
            risk_p = 0.0

            # Bullish: Slow group fanning (slow_max > slow_min), Macro Bullish
            # Pullback: Low dipped into slow group pocket (low <= slow_max), but close held above slow_min
            # Trigger: Close > Open and Close > Fast EMA 8
            if row['macro_bias'] > 0 and row['gmma_slow_max'] > row['gmma_slow_min']:
                if prev['low'] <= prev['gmma_slow_max'] and row['close'] > row['gmma_slow_max'] and row['close'] > row['open']:
                    if row['close'] > row['gmma_fast_8']:
                        sl_p = min(prev['low'], row['low'], row['gmma_slow_min']) - 1.2 * self.pip_size
                        risk = max((row['close'] - sl_p) / self.pip_size, self.min_sl)
                        if risk <= self.max_sl:
                            sig = SIGNAL_LONG
                            risk_p = risk

            # Bearish: Slow group fanning down, Macro Bearish
            # Pullback: High reached into slow group pocket (high >= slow_min), but close held below slow_max
            # Trigger: Close < Open and Close < Fast EMA 8
            elif row['macro_bias'] < 0 and row['gmma_slow_max'] > row['gmma_slow_min']:
                if prev['high'] >= prev['gmma_slow_min'] and row['close'] < row['gmma_slow_min'] and row['close'] < row['open']:
                    if row['close'] < row['gmma_fast_8']:
                        sl_p = max(prev['high'], row['high'], row['gmma_slow_max']) + 1.2 * self.pip_size
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
# DISC02: Frankfurt/London Pre-Market ORB (06:00-08:00 Range Breakout) (M15)
# ==============================================================================
class FrankfurtLondonPreMarketORB(BaseStrategy):
    def __init__(self, target_rr: float = 1.60, min_sl: float = 6.0, max_sl: float = 16.0, **kwargs):
        super().__init__(**kwargs)
        self.target_rr = target_rr
        self.min_sl = min_sl
        self.max_sl = max_sl
        self.pip_size = PIP_SIZE
        self.name = "Frankfurt_London_ORB"

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

        m15 = enrich_phase5_indicators(m15, base_df)
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        for d, day_df in m15.groupby('date'):
            # Range window 06:00 to 07:45 UTC
            box_bars = day_df[(day_df['hour'] >= 6) & (day_df['hour'] < 8)]
            if len(box_bars) < 6:
                continue

            box_h = float(box_bars['high'].max())
            box_l = float(box_bars['low'].min())
            box_mid = 0.5 * (box_h + box_l)
            range_pips = (box_h - box_l) / self.pip_size

            # Filter: range must be coiled (10 to 30 pips)
            if range_pips < 10.0 or range_pips > 30.0:
                continue

            london_open = day_df[(day_df['hour'] >= 8) & (day_df['hour'] <= 11)]
            traded = False

            for idx_val, row in london_open.iterrows():
                if traded:
                    break

                i = m15.index.get_loc(idx_val)
                prev = m15.iloc[i-1]
                sig = 0
                risk_p = 0.0

                # Bullish Breakout
                if prev['close'] <= box_h and row['close'] > box_h + 0.5 * self.pip_size and row['macro_bias'] > 0:
                    if row['close'] > row['open']:
                        risk = max((row['close'] - box_mid) / self.pip_size, self.min_sl)
                        if risk <= self.max_sl:
                            sig = SIGNAL_LONG
                            risk_p = risk
                            traded = True

                # Bearish Breakout
                elif prev['close'] >= box_l and row['close'] < box_l - 0.5 * self.pip_size and row['macro_bias'] < 0:
                    if row['close'] < row['open']:
                        risk = max((box_mid - row['close']) / self.pip_size, self.min_sl)
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
# DISC03: New York Open (13:00 UTC) London-NY Overlap Breakout (M15)
# ==============================================================================
class NewYorkOpenOverlapBreakout(BaseStrategy):
    def __init__(self, target_rr: float = 1.60, min_sl: float = 6.0, max_sl: float = 16.0, **kwargs):
        super().__init__(**kwargs)
        self.target_rr = target_rr
        self.min_sl = min_sl
        self.max_sl = max_sl
        self.pip_size = PIP_SIZE
        self.name = "NY_Open_Overlap_Breakout"

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

        m15 = enrich_phase5_indicators(m15, base_df)
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        for d, day_df in m15.groupby('date'):
            # Pre-NY range window 09:00 to 12:45 UTC
            pre_ny = day_df[(day_df['hour'] >= 9) & (day_df['hour'] < 13)]
            if len(pre_ny) < 12:
                continue

            ref_h = float(pre_ny['high'].max())
            ref_l = float(pre_ny['low'].min())
            ref_mid = 0.5 * (ref_h + ref_l)
            range_pips = (ref_h - ref_l) / self.pip_size

            if range_pips < 12.0 or range_pips > 40.0:
                continue

            ny_open = day_df[(day_df['hour'] >= 13) & (day_df['hour'] <= 15)]
            traded = False

            for idx_val, row in ny_open.iterrows():
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
# DISC04: J. Welles Wilder's Classic RSI(14) Failure Swing (M15)
# ==============================================================================
class WilderRSIFailureSwing(BaseStrategy):
    def __init__(self, target_rr: float = 1.80, min_sl: float = 6.0, max_sl: float = 16.0, **kwargs):
        super().__init__(**kwargs)
        self.target_rr = target_rr
        self.min_sl = min_sl
        self.max_sl = max_sl
        self.pip_size = PIP_SIZE
        self.name = "Wilder_RSI_Failure_Swing"

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

        m15 = enrich_phase5_indicators(m15, base_df)
        time_ms = m15['time_ms'].values
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        rsi = m15['rsi_14'].values

        for i in range(15, len(m15)):
            row = m15.iloc[i]
            prev = m15.iloc[i-1]

            if row['hour'] < 7 or row['hour'] > 15:
                continue

            sig = 0
            risk_p = 0.0

            # Bullish Failure Swing:
            # 1. Prior dip went oversold (min in last 6 bars < 32)
            # 2. Bounced to peak M
            # 3. Pullback held above 32
            # 4. Current bar crosses above peak M
            recent_low_rsi = np.min(rsi[i-6:i-1])
            if recent_low_rsi < 32.0 and rsi[i-1] > 32.0:
                peak_rsi = np.max(rsi[i-5:i-1])
                if rsi[i] > peak_rsi and row['close'] > row['open']:
                    sl_p = min(m15['low'].iloc[i-3:i+1]) - 1.2 * self.pip_size
                    risk = max((row['close'] - sl_p) / self.pip_size, self.min_sl)
                    if risk <= self.max_sl:
                        sig = SIGNAL_LONG
                        risk_p = risk

            # Bearish Failure Swing:
            # 1. Prior rally went overbought (max in last 6 bars > 68)
            # 2. Pulled back to trough N
            # 3. Next bounce failed below 68
            # 4. Current bar crosses below trough N
            recent_high_rsi = np.max(rsi[i-6:i-1])
            if recent_high_rsi > 68.0 and rsi[i-1] < 68.0:
                trough_rsi = np.min(rsi[i-5:i-1])
                if rsi[i] < trough_rsi and row['close'] < row['open']:
                    sl_p = max(m15['high'].iloc[i-3:i+1]) + 1.2 * self.pip_size
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
# DISC05: Session VWAP 2.0-Sigma Statistical Extreme Rejection (M15)
# ==============================================================================
class SessionVWAP2SigmaExtreme(BaseStrategy):
    def __init__(self, target_rr: float = 1.50, min_sl: float = 6.0, max_sl: float = 16.0, **kwargs):
        super().__init__(**kwargs)
        self.target_rr = target_rr
        self.min_sl = min_sl
        self.max_sl = max_sl
        self.pip_size = PIP_SIZE
        self.name = "Session_VWAP_2Sigma_Extreme"

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

        m15 = enrich_phase5_indicators(m15, base_df)
        time_ms = m15['time_ms'].values
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        for i in range(20, len(m15)):
            row = m15.iloc[i]
            prev = m15.iloc[i-1]

            if row['hour'] < 7 or row['hour'] > 15:
                continue

            sig = 0
            risk_p = 0.0

            candle_range = row['high'] - row['low']
            if candle_range < 4.0 * self.pip_size:
                continue

            # Bullish: Tested below VWAP - 2.0 SD, rejected with lower wick >= 45% and green close
            lower_wick = min(row['open'], row['close']) - row['low']
            if prev['low'] <= prev['vwap_low2'] or row['low'] <= row['vwap_low2']:
                if row['close'] > row['vwap_low2'] and row['close'] > row['open'] and (lower_wick / candle_range) >= 0.45:
                    sl_p = row['low'] - 1.2 * self.pip_size
                    risk = max((row['close'] - sl_p) / self.pip_size, self.min_sl)
                    if risk <= self.max_sl:
                        sig = SIGNAL_LONG
                        risk_p = risk

            # Bearish: Tested above VWAP + 2.0 SD, rejected with upper wick >= 45% and red close
            upper_wick = row['high'] - max(row['open'], row['close'])
            if prev['high'] >= prev['vwap_up2'] or row['high'] >= row['vwap_up2']:
                if row['close'] < row['vwap_up2'] and row['close'] < row['open'] and (upper_wick / candle_range) >= 0.45:
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
# LABORATORY EXPERIMENT HARNESS
# ==============================================================================
PHASE5_STRATEGIES = [
    {"code": "DISC01", "name": "Guppy GMMA Pocket Scalper (M15)", "cls": GuppyValuePocketScalper},
    {"code": "DISC02", "name": "Frankfurt/London ORB (M15)", "cls": FrankfurtLondonPreMarketORB},
    {"code": "DISC03", "name": "NY Open Overlap Breakout (M15)", "cls": NewYorkOpenOverlapBreakout},
    {"code": "DISC04", "name": "Wilder RSI Failure Swing (M15)", "cls": WilderRSIFailureSwing},
    {"code": "DISC05", "name": "Session VWAP 2.0σ Rejection (M15)", "cls": SessionVWAP2SigmaExtreme},
]


def run_week_experiment(week_num: int) -> Dict[str, Any]:
    ticks, meta = get_week_dataset(week_num)
    n_ticks = len(ticks)
    engine = BacktestEngine(pip_size=PIP_SIZE, use_slippage=True)

    week_trades = {}
    for item in PHASE5_STRATEGIES:
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
    print("             DISCOVERY LABORATORY PHASE 5: 52 WEEKS FULL TICK AUDIT (~26M TICKS)")
    print("          Evaluating GMMA Pocket, Frankfurt/London ORB, NY Open Overlap, Wilder RSI, VWAP 2σ")
    print("=" * 135)

    start_t = time.time()
    weeks = list(range(1, 53))

    strat_trades: Dict[str, List[Dict[str, Any]]] = {item["code"]: [] for item in PHASE5_STRATEGIES}
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
    print("                                      PHASE 5 DISCOVERY SCORECARD")
    print("=" * 145)
    print(f"{'Code':<7} | {'Model Name':<42} | {'Trades':<6} | {'Trd/Wk':<6} | {'Win %':<6} | {'Real RR':<7} | {'PF':<5} | {'Net Pips':<9} | {'Return %':<8} | {'Max DD%':<7} | {'Daily DD%':<9} | {'Ulcer':<5} | {'Max Recov':<9}")
    print("-" * 145)

    for item in PHASE5_STRATEGIES:
        code = item["code"]
        name = item["name"]
        t_data = strat_trades[code]
        metrics = analyze_combination_performance(t_data, risk_pct=0.40)

        print(
            f"{code:<7} | {name:<42} | "
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
