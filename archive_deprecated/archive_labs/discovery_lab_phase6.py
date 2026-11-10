#!/usr/bin/env python3
"""
Prop Firm Strategy Discovery Laboratory Phase 6:
Evaluating 5 High-Potential Academic & Institutional Quantitative Models on 52 Weeks (~26M Ticks):

1. DISC06: London First-Hour Intraday Momentum (IMOM-London) [Gao et al. 2018, Elaut et al. 2018]
   - Opening range established 07:00-08:00 UTC (Frankfurt to London Cash Open).
   - Significant impulse displacement (|Delta P| >= 0.70 ATR).
   - M15 Pullback to EMA8/VWAP during 08:15-10:30 UTC followed by continuation.
   - Filtered by H1 50-EMA macro trend. Target: 1.6R.

2. DISC07: Pre-Fix Order Flow Trend Continuation (PFTC-16:00 WMR Fix) [Melvin & Prins 2015]
   - Institutional rebalancing into the 16:00 London WMR Benchmark Fix.
   - Reference range 13:00-14:30 UTC (US Open reaction).
   - Execution window 14:30-15:45 UTC riding the afternoon breakout trend into the Fix.
   - Target: 1.6R with Midpoint stop.

3. DISC08: Order Flow Imbalance (OFI) / Cumulative Tick Delta Momentum [Cont et al. 2014]
   - Bar Delta proxy: Vol * [(Close-Low) - (High-Close)] / (High-Low)
   - Rolling 20-bar Cumulative Volume Delta (CVD) breaking +/- 1.8 SD Bollinger Band.
   - 12-bar Donchian breakout confirmation during active London/NY hours (07:00-16:00 UTC).
   - Dynamic ATR stop with 1.6R target.

4. DISC09: Asian Liquidity Sweep & Judas Swing Reversal (ICT / Wyckoff Spring)
   - Asian Range 00:00-06:00 UTC (coiled 10 to 35 pips).
   - Sweep window 06:30-09:00 UTC sweeps Asian High/Low by 1.5 to 15.0 pips.
   - Sharp rejection closing back INSIDE the Asian range.
   - Stop Loss tight at sweep extreme, Target: 1.8R.

5. DISC10: Garman-Klass Volatility Compression to Expansion Breakout (GK-Breakout) [Garman & Klass 1980]
   - Garman-Klass variance: 0.5 * ln(H/L)^2 - (2*ln(2)-1) * ln(C/O)^2
   - Ratio of GK(5) / GK(30) compresses <= 0.70 then bursts >= 1.15.
   - 15-bar Donchian breakout in direction of H1 50-EMA macro trend.
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


def enrich_phase6_indicators(df: pd.DataFrame, base_df: pd.DataFrame) -> pd.DataFrame:
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

    # 2. EMAs (8, 20)
    df['ema_8'] = c.ewm(span=8, adjust=False).mean()
    df['ema_20'] = c.ewm(span=20, adjust=False).mean()

    # 3. Donchian Channels (12 and 15)
    df['donchian_h12'] = h.rolling(12).max().shift(1)
    df['donchian_l12'] = l.rolling(12).min().shift(1)
    df['donchian_h15'] = h.rolling(15).max().shift(1)
    df['donchian_l15'] = l.rolling(15).min().shift(1)

    # 4. H1 50-EMA Macro Trend (zero-lookahead shift(1))
    h1 = base_df.resample('1h').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
    h1['ema50'] = h1['close'].ewm(span=50, adjust=False).mean()
    h1['macro_bias'] = np.where(h1['close'] > h1['ema50'], 1, -1)
    df['macro_bias'] = h1['macro_bias'].shift(1).reindex(df.index, method='ffill').fillna(0)

    # 5. Session Timings & VWAP
    df['date'] = df.index.date
    df['hour'] = df.index.hour
    df['minute'] = df.index.minute
    df['time_ms'] = df.index.astype('datetime64[ms]').view('int64')

    vol = tr / PIP_SIZE + 10.0
    cum_vol = vol.groupby(df['date']).cumsum()
    cum_pv = (c * vol).groupby(df['date']).cumsum()
    df['vwap'] = cum_pv / np.maximum(cum_vol, 1e-9)

    # 6. Bar Delta & Cumulative Volume Delta (CVD)
    hl_diff = np.maximum(h - l, 1e-6)
    bar_delta_norm = ((c - l) - (h - c)) / hl_diff
    bar_delta = bar_delta_norm * vol
    df['bar_delta'] = bar_delta
    cvd_20 = bar_delta.rolling(20).sum()
    df['cvd_20'] = cvd_20
    cvd_mean = cvd_20.rolling(20).mean()
    cvd_std = cvd_20.rolling(20).std().fillna(1.0)
    df['cvd_upper'] = cvd_mean + 1.8 * cvd_std
    df['cvd_lower'] = cvd_mean - 1.8 * cvd_std

    # 7. Garman-Klass Volatility
    hl_ratio = np.maximum(h / np.maximum(l, 1e-6), 1.0)
    co_ratio = np.maximum(c / np.maximum(o, 1e-6), 1e-6)
    log_hl = np.log(hl_ratio)
    log_co = np.log(co_ratio)
    gk_var = 0.5 * (log_hl ** 2) - (2.0 * np.log(2.0) - 1.0) * (log_co ** 2)
    gk_vol = np.sqrt(np.maximum(gk_var, 1e-12))
    df['gk_vol'] = gk_vol
    gk_fast = gk_vol.rolling(5).mean()
    gk_slow = gk_vol.rolling(30).mean()
    df['gk_ratio'] = gk_fast / np.maximum(gk_slow, 1e-9)

    return df


# ==============================================================================
# DISC06: London First-Hour Intraday Momentum (IMOM-London) [Gao et al. 2018]
# ==============================================================================
class LondonFirstHourMomentum(BaseStrategy):
    """
    Measures London Open impulse (07:00-08:00 UTC).
    If impulse >= 0.70 ATR, enters on M15 pullback to EMA8/VWAP during 08:15-10:30 UTC.
    """
    def __init__(self, target_rr: float = 1.60, min_sl: float = 6.0, max_sl: float = 16.0, **kwargs):
        super().__init__(**kwargs)
        self.target_rr = target_rr
        self.min_sl = min_sl
        self.max_sl = max_sl
        self.pip_size = PIP_SIZE
        self.name = "London_First_Hour_Momentum"

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

        m15 = enrich_phase6_indicators(m15, base_df)
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        for d, day_df in m15.groupby('date'):
            # 07:00 to 07:45 UTC bars (4 bars)
            first_hr = day_df[day_df['hour'] == 7]
            if len(first_hr) < 4:
                continue

            open_p = float(first_hr['open'].iloc[0])
            close_p = float(first_hr['close'].iloc[-1])
            hr_high = float(first_hr['high'].max())
            hr_low = float(first_hr['low'].min())
            net_disp = (close_p - open_p) / self.pip_size
            atr_val = float(first_hr['atr_pips'].iloc[-1])

            if atr_val < 5.0:
                continue

            # Check significant displacement (>= 0.70 ATR)
            is_bullish = net_disp >= 0.70 * atr_val
            is_bearish = net_disp <= -0.70 * atr_val
            if not (is_bullish or is_bearish):
                continue

            trade_window = day_df[(day_df['hour'] >= 8) & (day_df['hour'] <= 10)]
            traded = False

            for idx_val, row in trade_window.iterrows():
                if traded:
                    break

                i = m15.index.get_loc(idx_val)
                prev = m15.iloc[i-1]
                sig = 0
                risk_p = 0.0

                # Bullish Pullback & Continuation
                if is_bullish and row['macro_bias'] > 0:
                    # Pullback: prev low reached into EMA8 or VWAP, current close is green and above EMA8
                    if prev['low'] <= max(prev['ema_8'], prev['vwap']) and row['close'] > row['open'] and row['close'] > row['ema_8']:
                        sl_p = min(prev['low'], row['low'], hr_low) - 1.0 * self.pip_size
                        risk = max((row['close'] - sl_p) / self.pip_size, self.min_sl)
                        if risk <= self.max_sl:
                            sig = SIGNAL_LONG
                            risk_p = risk
                            traded = True

                # Bearish Pullback & Continuation
                elif is_bearish and row['macro_bias'] < 0:
                    # Pullback: prev high reached into EMA8 or VWAP, current close is red and below EMA8
                    if prev['high'] >= min(prev['ema_8'], prev['vwap']) and row['close'] < row['open'] and row['close'] < row['ema_8']:
                        sl_p = max(prev['high'], row['high'], hr_high) + 1.0 * self.pip_size
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
# DISC07: Pre-Fix Trend Continuation (PFTC-16:00 WMR Fix) [Melvin & Prins 2015]
# ==============================================================================
class PreFixOrderFlowContinuation(BaseStrategy):
    """
    Rides institutional order flow into the 16:00 London WMR Benchmark Fix.
    Reference window: 13:00-14:30 UTC. Execution window: 14:30-15:45 UTC.
    """
    def __init__(self, target_rr: float = 1.60, min_sl: float = 6.0, max_sl: float = 16.0, **kwargs):
        super().__init__(**kwargs)
        self.target_rr = target_rr
        self.min_sl = min_sl
        self.max_sl = max_sl
        self.pip_size = PIP_SIZE
        self.name = "Pre_Fix_Order_Flow_Continuation"

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

        m15 = enrich_phase6_indicators(m15, base_df)
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        for d, day_df in m15.groupby('date'):
            # Pre-NY reference window 13:00 to 14:15 UTC (6 bars)
            ref_bars = day_df[(day_df['hour'] == 13) | ((day_df['hour'] == 14) & (day_df['minute'] <= 15))]
            if len(ref_bars) < 5:
                continue

            ref_h = float(ref_bars['high'].max())
            ref_l = float(ref_bars['low'].min())
            ref_mid = 0.5 * (ref_h + ref_l)
            range_p = (ref_h - ref_l) / self.pip_size

            if range_p < 8.0 or range_p > 35.0:
                continue

            # Execution window: 14:30 to 15:45 UTC (prior to 16:00 Fix)
            exec_bars = day_df[((day_df['hour'] == 14) & (day_df['minute'] >= 30)) | (day_df['hour'] == 15)]
            traded = False

            for idx_val, row in exec_bars.iterrows():
                if traded:
                    break

                i = m15.index.get_loc(idx_val)
                prev = m15.iloc[i-1]
                sig = 0
                risk_p = 0.0

                # Bullish Breakout into Fix
                if prev['close'] <= ref_h and row['close'] > ref_h + 0.5 * self.pip_size and row['macro_bias'] > 0:
                    if row['close'] > row['open']:
                        risk = max((row['close'] - ref_mid) / self.pip_size, self.min_sl)
                        if risk <= self.max_sl:
                            sig = SIGNAL_LONG
                            risk_p = risk
                            traded = True

                # Bearish Breakout into Fix
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
# DISC08: Order Flow Imbalance (OFI) / CVD Momentum [Cont et al. 2014]
# ==============================================================================
class OrderFlowDeltaMomentum(BaseStrategy):
    """
    Detects institutional accumulation/distribution via 20-bar Cumulative Volume Delta
    breaking +/- 1.8 SD Bollinger Band confirmed by Donchian breakout.
    """
    def __init__(self, target_rr: float = 1.60, min_sl: float = 6.0, max_sl: float = 16.0, **kwargs):
        super().__init__(**kwargs)
        self.target_rr = target_rr
        self.min_sl = min_sl
        self.max_sl = max_sl
        self.pip_size = PIP_SIZE
        self.name = "Order_Flow_Delta_Momentum"

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

        m15 = enrich_phase6_indicators(m15, base_df)
        time_ms = m15['time_ms'].values
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        for i in range(25, len(m15)):
            row = m15.iloc[i]
            prev = m15.iloc[i-1]

            # Active hours: London + NY (07:00 to 16:00 UTC)
            if row['hour'] < 7 or row['hour'] > 16:
                continue

            sig = 0
            risk_p = 0.0

            # Bullish CVD Momentum Breakout
            if row['cvd_20'] > row['cvd_upper'] and prev['cvd_20'] <= prev['cvd_upper']:
                if row['close'] > row['donchian_h12'] and row['macro_bias'] > 0 and row['close'] > row['open']:
                    sl_p = min(row['low'], prev['low']) - 1.0 * self.pip_size
                    risk = max((row['close'] - sl_p) / self.pip_size, self.min_sl)
                    if risk <= self.max_sl:
                        sig = SIGNAL_LONG
                        risk_p = risk

            # Bearish CVD Momentum Breakout
            elif row['cvd_20'] < row['cvd_lower'] and prev['cvd_20'] >= prev['cvd_lower']:
                if row['close'] < row['donchian_l12'] and row['macro_bias'] < 0 and row['close'] < row['open']:
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
# DISC09: Asian Liquidity Sweep & Judas Swing Reversal (ICT / Wyckoff Spring)
# ==============================================================================
class AsianLiquiditySweepReversal(BaseStrategy):
    """
    Detects false breakouts (stop runs) beyond Asian High/Low (00:00-06:00 UTC).
    If price sweeps by 1.5-15 pips and rejects back INSIDE the Asian range during 06:30-09:00 UTC,
    fades the sweep back toward range midpoint/opposite extreme.
    """
    def __init__(self, target_rr: float = 1.80, min_sl: float = 6.0, max_sl: float = 16.0, **kwargs):
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

        m15 = enrich_phase6_indicators(m15, base_df)
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        for d, day_df in m15.groupby('date'):
            # Asian Session: 00:00 to 05:45 UTC (24 bars)
            asia = day_df[day_df['hour'] < 6]
            if len(asia) < 18:
                continue

            asia_h = float(asia['high'].max())
            asia_l = float(asia['low'].min())
            asia_range = (asia_h - asia_l) / self.pip_size

            # Coiled Asian range: 10 to 35 pips
            if asia_range < 10.0 or asia_range > 35.0:
                continue

            # Sweep window: 06:30 to 09:00 UTC
            sweep_window = day_df[((day_df['hour'] == 6) & (day_df['minute'] >= 30)) | (day_df['hour'] == 7) | (day_df['hour'] == 8)]
            traded = False

            for idx_val, row in sweep_window.iterrows():
                if traded:
                    break

                i = m15.index.get_loc(idx_val)
                prev = m15.iloc[i-1]
                sig = 0
                risk_p = 0.0

                # Bearish Judas Swing: Sweeps Asian High, but closes back BELOW Asian High
                sweep_high_pips = (row['high'] - asia_h) / self.pip_size
                if 1.5 <= sweep_high_pips <= 15.0 and row['close'] < asia_h and row['close'] < row['open']:
                    sl_p = row['high'] + 1.5 * self.pip_size
                    risk = max((sl_p - row['close']) / self.pip_size, self.min_sl)
                    if risk <= self.max_sl:
                        sig = SIGNAL_SHORT
                        risk_p = risk
                        traded = True

                # Bullish Judas Swing: Sweeps Asian Low, but closes back ABOVE Asian Low
                sweep_low_pips = (asia_l - row['low']) / self.pip_size
                if not traded and 1.5 <= sweep_low_pips <= 15.0 and row['close'] > asia_l and row['close'] > row['open']:
                    sl_p = row['low'] - 1.5 * self.pip_size
                    risk = max((row['close'] - sl_p) / self.pip_size, self.min_sl)
                    if risk <= self.max_sl:
                        sig = SIGNAL_LONG
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
# DISC10: Garman-Klass Volatility Compression to Expansion Breakout (GK-Breakout)
# ==============================================================================
class GarmanKlassBreakout(BaseStrategy):
    """
    Garman-Klass Volatility Ratio (GK5 / GK30).
    Compresses <= 0.70 then bursts >= 1.15 with Donchian 15-bar channel breakout.
    """
    def __init__(self, target_rr: float = 1.70, min_sl: float = 6.0, max_sl: float = 16.0, **kwargs):
        super().__init__(**kwargs)
        self.target_rr = target_rr
        self.min_sl = min_sl
        self.max_sl = max_sl
        self.pip_size = PIP_SIZE
        self.name = "Garman_Klass_Breakout"

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

        m15 = enrich_phase6_indicators(m15, base_df)
        time_ms = m15['time_ms'].values
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        for i in range(35, len(m15)):
            row = m15.iloc[i]
            prev = m15.iloc[i-1]

            # Active hours: London + NY (07:00 to 16:00 UTC)
            if row['hour'] < 7 or row['hour'] > 16:
                continue

            sig = 0
            risk_p = 0.0

            # Volatility expansion burst from compressed state
            # Look back 3 bars for compression <= 0.75
            was_compressed = any(m15['gk_ratio'].iloc[i-k] <= 0.75 for k in [1, 2, 3])
            is_bursting = row['gk_ratio'] >= 1.15

            if was_compressed and is_bursting:
                # Bullish Breakout
                if row['close'] > row['donchian_h15'] and row['macro_bias'] > 0 and row['close'] > row['open']:
                    sl_p = min(row['low'], prev['low']) - 1.0 * self.pip_size
                    risk = max((row['close'] - sl_p) / self.pip_size, self.min_sl)
                    if risk <= self.max_sl:
                        sig = SIGNAL_LONG
                        risk_p = risk

                # Bearish Breakout
                elif row['close'] < row['donchian_l15'] and row['macro_bias'] < 0 and row['close'] < row['open']:
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
# LABORATORY EXPERIMENT HARNESS
# ==============================================================================
PHASE6_STRATEGIES = [
    {"code": "DISC06", "name": "London First-Hour Momentum (IMOM)", "cls": LondonFirstHourMomentum},
    {"code": "DISC07", "name": "Pre-Fix Order Flow Continuation", "cls": PreFixOrderFlowContinuation},
    {"code": "DISC08", "name": "Order Flow CVD Momentum (M15)", "cls": OrderFlowDeltaMomentum},
    {"code": "DISC09", "name": "Asian Liquidity Sweep Reversal", "cls": AsianLiquiditySweepReversal},
    {"code": "DISC10", "name": "Garman-Klass Breakout (M15)", "cls": GarmanKlassBreakout},
]


def run_week_experiment(week_num: int) -> Dict[str, Any]:
    ticks, meta = get_week_dataset(week_num)
    n_ticks = len(ticks)
    engine = BacktestEngine(pip_size=PIP_SIZE, use_slippage=True)

    week_trades = {}
    for item in PHASE6_STRATEGIES:
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
    print("             DISCOVERY LABORATORY PHASE 6: 52 WEEKS FULL TICK AUDIT (~26M TICKS)")
    print("       Evaluating IMOM London, Pre-Fix Continuation, OFI Delta, Judas Swing, Garman-Klass")
    print("=" * 135)

    start_t = time.time()
    weeks = list(range(1, 53))

    strat_trades: Dict[str, List[Dict[str, Any]]] = {item["code"]: [] for item in PHASE6_STRATEGIES}
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

    for item in PHASE6_STRATEGIES:
        code = item["code"]
        name = item["name"]
        t_list = strat_trades[code]
        metrics = analyze_combination_performance(t_list, risk_pct=0.40)
        print(f"{code:<8} | {name:<36} | {metrics['total_trades']:<6} | {metrics['trades_per_week']:<6.1f} | {metrics['win_rate']:<7.1f}% | {metrics['realized_rr']:<7.2f} | {metrics['profit_factor']:<6.2f} | {metrics['net_pips']:<+9.1f} | {metrics['max_dd_pct']:<6.2f}% | {metrics['max_recovery_days']:<5.1f} days")

    print("=" * 135)


if __name__ == "__main__":
    main()
