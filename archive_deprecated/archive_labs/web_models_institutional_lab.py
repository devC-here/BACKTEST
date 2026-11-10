#!/usr/bin/env python3
"""
Web-Discovered Institutional Forex Models Laboratory:
------------------------------------------------------
Validates 4 High-Expectancy Institutional setups discovered via quantitative web research:

1. WEB01: Toby Crabel's London Opening Range Breakout (ORB) with NR Midpoint Stop
   - Narrow London Open range (07:00-07:30 UTC <= 25 pips)
   - Breakout beyond OR High/Low
   - TIGHT STOP at OR Midpoint (Risk = ~0.5x OR, giving 1.6R - 2.0R with tight SL)
   - Filtered by H1 50-EMA macro trend (zero lookahead shift(1))

2. WEB02: Session VWAP Dynamic Standard Deviation (-1.5σ / +1.5σ) Trend Pullback
   - Anchored Session VWAP with rolling Standard Deviation bands
   - In Bullish H1 Trend: Buy when price dips below -1.5σ and closes back above with green candle
   - In Bearish H1 Trend: Sell when price rallies above +1.5σ and closes back below with red candle
   - Target = Mean (VWAP), Stop = Dip extreme

3. WEB03: London 4 PM Fix Rebalancing Fade (16:03 - 16:30 UTC)
   - Identifies institutional pre-fix extension (14:30 - 16:00 UTC displacement > 1.6x ATR)
   - Reversal fade entry at 16:03 UTC after fix window closes
   - Target = 50% retracement of pre-fix move

4. WEB04: Dual-Timeframe Elder Impulse Momentum Breakout (H1 + M5/M15)
   - H1 shifted(1) EMA13 + MACD Hist determines trend permission
   - M15 Pullback to Blue/Opposite resolves back to Green/Red
   - Entry on fresh impulse alignment with swing stop and 1.6R target

All tests execute on authentic Dukascopy tick data (~26M ticks) across 52 weeks.
Zero-lookahead strictly enforced.
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


# ==============================================================================
# WEB01: Toby Crabel London ORB with Midpoint Stop
# ==============================================================================
class CrabelLondonORBMidpoint(BaseStrategy):
    """
    Toby Crabel Opening Range Breakout:
    - Opening Range: 07:00 to 07:30 UTC (first 30 mins of London).
    - If OR Range <= 25 pips (volatility compression / coil).
    - Long: M15 close > OR High + 0.5 pip, and H1 50-EMA is Bullish.
      SL = OR Midpoint. TP = Entry + 1.60 * (Entry - SL).
    - Short: M15 close < OR Low - 0.5 pip, and H1 50-EMA is Bearish.
      SL = OR Midpoint. TP = Entry - 1.60 * (SL - Entry).
    - Max 1 trade per day. Entry cutoff: 11:30 UTC.
    """
    def __init__(self, target_rr: float = 1.60, max_or_pips: float = 25.0, **kwargs):
        super().__init__(**kwargs)
        self.target_rr = target_rr
        self.max_or_pips = max_or_pips
        self.pip_size = PIP_SIZE
        self.name = "Crabel_London_ORB_Midpoint"

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

        # H1 50 EMA Macro Trend with shift(1)
        h1 = base_df.resample('1h').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
        h1['ema50'] = h1['close'].ewm(span=50, adjust=False).mean()
        h1['macro_bias'] = np.where(h1['close'] > h1['ema50'], 1, -1)
        m15['macro_bias'] = h1['macro_bias'].shift(1).reindex(m15.index, method='ffill').fillna(0)

        m15['date'] = m15.index.date
        m15['hour'] = m15.index.hour
        m15['minute'] = m15.index.minute
        m15['time_ms'] = m15.index.astype('datetime64[ms]').view('int64')

        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        # Group by date to find 07:00-07:30 range
        for d, day_df in m15.groupby('date'):
            or_bars = day_df[(day_df['hour'] == 7) & (day_df['minute'] < 30)]
            if len(or_bars) < 2:
                continue

            or_high = float(or_bars['high'].max())
            or_low = float(or_bars['low'].min())
            or_range_pips = (or_high - or_low) / self.pip_size
            if or_range_pips < 6.0 or or_range_pips > self.max_or_pips:
                continue

            or_mid = 0.5 * (or_high + or_low)
            trade_taken = False

            # Evaluate bars between 07:30 and 11:30
            trade_bars = day_df[(day_df.index >= or_bars.index[-1] + pd.Timedelta(minutes=15)) & 
                                (day_df['hour'] <= 11)]

            for idx_val, bar in trade_bars.iterrows():
                if trade_taken:
                    break

                macro = bar['macro_bias']
                sig = 0
                risk_p = 0.0

                # Long Breakout
                if macro > 0 and bar['close'] > or_high + 0.5 * self.pip_size:
                    risk = max((bar['close'] - or_mid) / self.pip_size, 5.0)
                    if 5.0 <= risk <= 18.0:
                        sig = SIGNAL_LONG
                        risk_p = risk
                        trade_taken = True

                # Short Breakout
                elif macro < 0 and bar['close'] < or_low - 0.5 * self.pip_size:
                    risk = max((or_mid - bar['close']) / self.pip_size, 5.0)
                    if 5.0 <= risk <= 18.0:
                        sig = SIGNAL_SHORT
                        risk_p = risk
                        trade_taken = True

                if sig != 0:
                    exec_t = bar['time_ms'] + dur_ms
                    idx = int(np.searchsorted(tick_ts, exec_t))
                    if idx < n_ticks and signals[idx] == 0:
                        signals[idx] = sig
                        sl_pips_arr[idx] = risk_p
                        tp_pips_arr[idx] = risk_p * self.target_rr

        return signals, sl_pips_arr, tp_pips_arr


# ==============================================================================
# WEB02: Session VWAP Standard Deviation (-1.5σ / +1.5σ) Pullback
# ==============================================================================
class SessionVWAPDevPullback(BaseStrategy):
    """
    Session VWAP Dynamic Standard Deviation Value Pullback:
    - Session VWAP reset at 00:00 UTC.
    - Compute dynamic standard deviation bands: Upper = VWAP + 1.5*SD, Lower = VWAP - 1.5*SD.
    - In London/NY sessions (07:00 - 16:00 UTC).
    - If H1 Macro is Bullish:
      When M15 dips below Lower Band (<= VWAP - 1.5*SD) and closes back above Lower Band with green candle.
      Target = Mean (VWAP). SL = Dip low - 1.2 pips.
    - If H1 Macro is Bearish:
      When M15 rallies above Upper Band (>= VWAP + 1.5*SD) and closes back below Upper Band with red candle.
      Target = Mean (VWAP). SL = Rally high + 1.2 pips.
    """
    def __init__(self, sd_multiplier: float = 1.40, min_sl: float = 5.0, max_sl: float = 15.0, **kwargs):
        super().__init__(**kwargs)
        self.sd_multiplier = sd_multiplier
        self.min_sl = min_sl
        self.max_sl = max_sl
        self.pip_size = PIP_SIZE
        self.name = "Session_VWAP_Dev_Pullback"

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

        c = m15['close']
        h = m15['high']
        l = m15['low']
        o = m15['open']

        tr = np.maximum(h - l, np.maximum(abs(h - c.shift(1)), abs(l - c.shift(1))))
        vol = tr / self.pip_size + 10.0
        m15['date'] = m15.index.date
        m15['hour'] = m15.index.hour
        m15['time_ms'] = m15.index.astype('datetime64[ms]').view('int64')

        # Session VWAP & SD
        cum_vol = vol.groupby(m15['date']).cumsum()
        cum_pv = (c * vol).groupby(m15['date']).cumsum()
        vwap = cum_pv / np.maximum(cum_vol, 1e-9)
        m15['vwap'] = vwap

        # Variance
        cum_p2_v = ((c ** 2) * vol).groupby(m15['date']).cumsum()
        variance = (cum_p2_v / np.maximum(cum_vol, 1e-9)) - (vwap ** 2)
        std_dev = np.sqrt(np.maximum(variance, 1e-12))
        m15['upper_band'] = vwap + self.sd_multiplier * std_dev
        m15['lower_band'] = vwap - self.sd_multiplier * std_dev

        # Macro Bias: H1 50 EMA shift(1)
        h1 = base_df.resample('1h').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
        h1['ema50'] = h1['close'].ewm(span=50, adjust=False).mean()
        h1['bias'] = np.where(h1['close'] > h1['ema50'], 1, -1)
        m15['macro_bias'] = h1['bias'].shift(1).reindex(m15.index, method='ffill').fillna(0)

        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        for i in range(15, len(m15)):
            row = m15.iloc[i]
            prev = m15.iloc[i-1]

            if row['hour'] < 7 or row['hour'] > 15:
                continue

            sig = 0
            risk_p = 0.0
            tp_p = 0.0

            # Bullish Value Pullback: Macro Up, Low pierced lower band, closed back above with green candle
            if row['macro_bias'] > 0 and prev['low'] <= prev['lower_band']:
                if row['close'] > row['lower_band'] and row['close'] > row['open']:
                    sl_p = min(prev['low'], row['low']) - 1.2 * self.pip_size
                    risk = max((row['close'] - sl_p) / self.pip_size, self.min_sl)
                    dist_to_mean = (row['vwap'] - row['close']) / self.pip_size
                    if risk <= self.max_sl and dist_to_mean >= 6.0:
                        sig = SIGNAL_LONG
                        risk_p = risk
                        tp_p = max(dist_to_mean, risk * 1.30)

            # Bearish Value Pullback: Macro Down, High pierced upper band, closed back below with red candle
            elif row['macro_bias'] < 0 and prev['high'] >= prev['upper_band']:
                if row['close'] < row['upper_band'] and row['close'] < row['open']:
                    sl_p = max(prev['high'], row['high']) + 1.2 * self.pip_size
                    risk = max((sl_p - row['close']) / self.pip_size, self.min_sl)
                    dist_to_mean = (row['close'] - row['vwap']) / self.pip_size
                    if risk <= self.max_sl and dist_to_mean >= 6.0:
                        sig = SIGNAL_SHORT
                        risk_p = risk
                        tp_p = max(dist_to_mean, risk * 1.30)

            if sig != 0:
                exec_t = row['time_ms'] + dur_ms
                idx = int(np.searchsorted(tick_ts, exec_t))
                if idx < n_ticks and signals[idx] == 0:
                    signals[idx] = sig
                    sl_pips_arr[idx] = risk_p
                    tp_pips_arr[idx] = tp_p

        return signals, sl_pips_arr, tp_pips_arr


# ==============================================================================
# WEB03: London 4 PM Fix Mean-Reversion Fade (16:00 - 16:30 UTC)
# ==============================================================================
class London4PMFixFade(BaseStrategy):
    """
    London 4 PM Institutional Fix Rebalancing Fade:
    - Pre-fix move: Checks price movement from 14:15 to 16:00 UTC.
    - If pre-fix displacement >= 18 pips (strong institutional ramp into fix).
    - At 16:00-16:15 UTC, when price halts/reverses (pin bar or reversal bar).
    - Fades the spike back to the pre-fix mean (50% retracement).
    - TIGHT stop loss placed 2 pips beyond the 16:00 extreme wick.
    """
    def __init__(self, min_thrust_pips: float = 16.0, **kwargs):
        super().__init__(**kwargs)
        self.min_thrust_pips = min_thrust_pips
        self.pip_size = PIP_SIZE
        self.name = "London_4PM_Fix_Fade"

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

        m15['date'] = m15.index.date
        m15['hour'] = m15.index.hour
        m15['minute'] = m15.index.minute
        m15['time_ms'] = m15.index.astype('datetime64[ms]').view('int64')

        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        for d, day_df in m15.groupby('date'):
            # Pre-fix window 14:15 to 15:45
            pre_fix = day_df[(day_df['hour'] >= 14) & (day_df['hour'] <= 15)]
            fix_bar = day_df[(day_df['hour'] == 16) & (day_df['minute'] == 0)]
            if len(pre_fix) < 4 or len(fix_bar) == 0:
                continue

            start_price = float(pre_fix.iloc[0]['open'])
            fix_close = float(fix_bar.iloc[0]['close'])
            fix_high = float(fix_bar.iloc[0]['high'])
            fix_low = float(fix_bar.iloc[0]['low'])
            disp_pips = (fix_close - start_price) / self.pip_size

            # Bullish Thrust into 16:00 Fix -> FADE SHORT
            if disp_pips >= self.min_thrust_pips:
                # If fix candle showed upper rejection or closed red
                wick_top = (fix_high - max(fix_close, float(fix_bar.iloc[0]['open']))) / self.pip_size
                if wick_top >= 2.5 or fix_close < float(fix_bar.iloc[0]['open']):
                    sl_p = fix_high + 1.5 * self.pip_size
                    risk = max((sl_p - fix_close) / self.pip_size, 5.0)
                    target_pips = disp_pips * 0.50
                    if 5.0 <= risk <= 16.0 and target_pips >= risk * 1.10:
                        exec_t = fix_bar.iloc[0]['time_ms'] + dur_ms
                        idx = int(np.searchsorted(tick_ts, exec_t))
                        if idx < n_ticks and signals[idx] == 0:
                            signals[idx] = SIGNAL_SHORT
                            sl_pips_arr[idx] = risk
                            tp_pips_arr[idx] = target_pips

            # Bearish Thrust into 16:00 Fix -> FADE LONG
            elif disp_pips <= -self.min_thrust_pips:
                wick_bot = (min(fix_close, float(fix_bar.iloc[0]['open'])) - fix_low) / self.pip_size
                if wick_bot >= 2.5 or fix_close > float(fix_bar.iloc[0]['open']):
                    sl_p = fix_low - 1.5 * self.pip_size
                    risk = max((fix_close - sl_p) / self.pip_size, 5.0)
                    target_pips = abs(disp_pips) * 0.50
                    if 5.0 <= risk <= 16.0 and target_pips >= risk * 1.10:
                        exec_t = fix_bar.iloc[0]['time_ms'] + dur_ms
                        idx = int(np.searchsorted(tick_ts, exec_t))
                        if idx < n_ticks and signals[idx] == 0:
                            signals[idx] = SIGNAL_LONG
                            sl_pips_arr[idx] = risk
                            tp_pips_arr[idx] = target_pips

        return signals, sl_pips_arr, tp_pips_arr


# ==============================================================================
# WEB04: Dual-Timeframe Elder Impulse Momentum Alignment
# ==============================================================================
class DualTimeframeElderImpulse(BaseStrategy):
    """
    Elder Impulse System (M15 trigger, H1 shift(1) bias):
    - H1 Elder Impulse: EMA13 slope & MACD(12,26,9) Hist slope.
      H1 Green = EMA13 slope > 0 and Hist slope > 0 (Shifted 1 bar).
      H1 Red = EMA13 slope < 0 and Hist slope < 0 (Shifted 1 bar).
    - M15 Trigger:
      Long: H1 is Green, M15 was Neutral/Red, and current M15 bar turns Green!
      Short: H1 is Red, M15 was Neutral/Green, and current M15 bar turns Red!
    - Target RR: 1.50 : 1 with swing stop.
    """
    def __init__(self, target_rr: float = 1.50, min_sl: float = 6.0, max_sl: float = 15.0, **kwargs):
        super().__init__(**kwargs)
        self.target_rr = target_rr
        self.min_sl = min_sl
        self.max_sl = max_sl
        self.pip_size = PIP_SIZE
        self.name = "Dual_Timeframe_Elder_Impulse"

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
        if len(m15) < 50:
            return signals, sl_pips_arr, tp_pips_arr

        # M15 EMA13 & MACD Hist
        m15_ema13 = m15['close'].ewm(span=13, adjust=False).mean()
        m15_macd = m15['close'].ewm(span=12, adjust=False).mean() - m15['close'].ewm(span=26, adjust=False).mean()
        m15_sig = m15_macd.ewm(span=9, adjust=False).mean()
        m15_hist = m15_macd - m15_sig
        m15_ema_up = m15_ema13 > m15_ema13.shift(1)
        m15_hist_up = m15_hist > m15_hist.shift(1)
        m15['impulse'] = np.where(m15_ema_up & m15_hist_up, 1, np.where((~m15_ema_up) & (~m15_hist_up), -1, 0))

        # H1 Impulse with STRICT shift(1)
        h1 = base_df.resample('1h').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
        h1_ema13 = h1['close'].ewm(span=13, adjust=False).mean()
        h1_macd = h1['close'].ewm(span=12, adjust=False).mean() - h1['close'].ewm(span=26, adjust=False).mean()
        h1_sig = h1_macd.ewm(span=9, adjust=False).mean()
        h1_hist = h1_macd - h1_sig
        h1_ema_up = h1_ema13 > h1_ema13.shift(1)
        h1_hist_up = h1_hist > h1_hist.shift(1)
        h1['impulse'] = np.where(h1_ema_up & h1_hist_up, 1, np.where((~h1_ema_up) & (~h1_hist_up), -1, 0))
        m15['h1_impulse'] = h1['impulse'].shift(1).reindex(m15.index, method='ffill').fillna(0)

        m15['hour'] = m15.index.hour
        m15['time_ms'] = m15.index.astype('datetime64[ms]').view('int64')

        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        for i in range(25, len(m15)):
            row = m15.iloc[i]
            prev = m15.iloc[i-1]

            if row['hour'] < 7 or row['hour'] > 15:
                continue

            sig = 0
            risk_p = 0.0

            # Long: H1 is Green (+1), Prev M15 was not Green (<= 0), and Current M15 turns Green (+1)
            if row['h1_impulse'] == 1 and prev['impulse'] <= 0 and row['impulse'] == 1:
                sl_p = min(prev['low'], row['low']) - 1.2 * self.pip_size
                risk = max((row['close'] - sl_p) / self.pip_size, self.min_sl)
                if risk <= self.max_sl:
                    sig = SIGNAL_LONG
                    risk_p = risk

            # Short: H1 is Red (-1), Prev M15 was not Red (>= 0), and Current M15 turns Red (-1)
            elif row['h1_impulse'] == -1 and prev['impulse'] >= 0 and row['impulse'] == -1:
                sl_p = max(prev['high'], row['high']) + 1.2 * self.pip_size
                risk = max((sl_p - row['close']) / self.pip_size, self.min_sl)
                if risk <= self.max_sl:
                    sig = SIGNAL_SHORT
                    risk_p = risk

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
WEB_STRATEGIES = [
    {"code": "WEB01", "name": "Crabel London ORB Midpoint (M15)", "cls": CrabelLondonORBMidpoint},
    {"code": "WEB02", "name": "Session VWAP Dev Pullback (M15)", "cls": SessionVWAPDevPullback},
    {"code": "WEB03", "name": "London 4PM Fix Rebalancing Fade (M15)", "cls": London4PMFixFade},
    {"code": "WEB04", "name": "Dual-TF Elder Impulse Momentum (M15)", "cls": DualTimeframeElderImpulse},
]


def run_week_experiment(week_num: int) -> Dict[str, Any]:
    ticks, meta = get_week_dataset(week_num)
    n_ticks = len(ticks)
    engine = BacktestEngine(pip_size=PIP_SIZE, use_slippage=True)

    week_trades = {}
    for item in WEB_STRATEGIES:
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
    print("             WEB-DISCOVERED INSTITUTIONAL MODELS LABORATORY: 52 WEEKS FULL TICK AUDIT (~26M TICKS)")
    print("                      Evaluating Crabel ORB, VWAP SD Pullback, London Fix Fade, Elder Impulse")
    print("=" * 135)

    start_t = time.time()
    weeks = list(range(1, 53))

    strat_trades: Dict[str, List[Dict[str, Any]]] = {item["code"]: [] for item in WEB_STRATEGIES}
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
    print("                                      WEB-DISCOVERED MODELS SCORECARD")
    print("=" * 145)
    print(f"{'Code':<7} | {'Model Name':<42} | {'Trades':<6} | {'Trd/Wk':<6} | {'Win %':<6} | {'Real RR':<7} | {'PF':<5} | {'Net Pips':<9} | {'Return %':<8} | {'Max DD%':<7} | {'Daily DD%':<9} | {'Ulcer':<5} | {'Max Recov':<9}")
    print("-" * 145)

    for item in WEB_STRATEGIES:
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
