#!/usr/bin/env python3
"""
================================================================================
  RESEARCH LAB: "WISDOM OF THE CROWD" HIGH-WIN-RATE ALPHA DISCOVERY
================================================================================
Explores 4 distinct quantitative formulations of "Wisdom of the Crowd"
specifically designed to achieve Win Rate > 50%:

  1. WOTC_Indicator_Consensus:
     Multi-indicator voting quorum (Trend, Momentum, Oscillator, Volatility,
     Candlestick). Trade triggered ONLY when >= 4 out of 5 orthogonal crowds agree.
     
  2. WOTC_HeikenAshi_Ribbon:
     The classic Forex "Wisdom of the Crowd" template (50/100 EMA Baseline +
     Guppy Fast Ribbon expansion + Heiken Ashi pullbacks).
     
  3. WOTC_Retail_Trap_Fade:
     Contrarian fade of the trapped retail crowd at Asian High/Low liquidity sweeps
     during London Open (07:00 - 10:30 UTC).
     
  4. WOTC_Multi_Timeframe_Supermajority:
     Unanimous directional alignment across 3 distinct timeframe crowds:
     Macro (H4), Intraday (H1), and Micro Scalpers (M15).
================================================================================
"""

from __future__ import annotations
import sys
import os
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tick_engine.core.engine import BacktestEngine, TRADE_DTYPE
from tick_engine.core.dataloader import get_week_dataset
from tick_engine.strategies.base import BaseStrategy, SIGNAL_LONG, SIGNAL_SHORT
from data_engineering_suite.bars import build_time_bars

PIP_SIZE = 0.0001


# ==============================================================================
# 1. WOTC_Indicator_Consensus_Elite (Multi-Indicator Voting Quorum)
# ==============================================================================
class WOTCIndicatorConsensusStrategy(BaseStrategy):
    """
    Council of 5 orthogonal indicators voting on M15 bars:
      1. Trend Crowd: Close > EMA20 and EMA20 > EMA50
      2. Momentum Crowd: RSI(14) between 52 and 68 (bullish) or 32 and 48 (bearish)
      3. Oscillator Crowd: Stochastic(14,3,3) %K crossed %D in trend direction
      4. Volatility Crowd: Close near upper/lower Bollinger Band (20, 2)
      5. Candle Body Crowd: Rejection wick or strong directional body (>60% of range)
    Requires supermajority (>= 4 out of 5 votes).
    """
    def __init__(
        self,
        min_votes: int = 4,
        sl_pips: float = 10.0,
        tp_pips: float = 12.5,
        target_rr: float = 1.25,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.min_votes = int(min_votes)
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.target_rr = float(target_rr)
        self.allowed_hours = {7, 8, 9, 10, 11, 12, 13, 14, 15, 16}
        self.name = "WOTC_Indicator_Consensus"

    def generate_signals(self, ticks: np.ndarray):
        n_ticks = len(ticks)
        signals = np.zeros(n_ticks, dtype=np.int8)
        sl_pips_arr = np.zeros(n_ticks, dtype=np.float64)
        tp_pips_arr = np.zeros(n_ticks, dtype=np.float64)

        if n_ticks < 500:
            return signals, sl_pips_arr, tp_pips_arr

        m15 = build_time_bars(ticks, timeframe_min=15)
        if len(m15) < 60:
            return signals, sl_pips_arr, tp_pips_arr

        c = m15['close'].values
        h = m15['high'].values
        l = m15['low'].values
        o = m15['open'].values
        n_bars = len(m15)

        # 1. Trend Crowd: EMA20 & EMA50
        ema20 = m15['close'].ewm(span=20, adjust=False).mean().values
        ema50 = m15['close'].ewm(span=50, adjust=False).mean().values
        trend_long = (c > ema20) & (ema20 > ema50)
        trend_short = (c < ema20) & (ema20 < ema50)

        # 2. Momentum Crowd: RSI 14
        delta = pd.Series(c).diff()
        gain = delta.where(delta > 0, 0.0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
        rs = gain / (loss + 1e-9)
        rsi = (100 - (100 / (1 + rs))).values
        rsi_long = (rsi >= 52) & (rsi <= 68)
        rsi_short = (rsi <= 48) & (rsi >= 32)

        # 3. Oscillator Crowd: Fast Stochastic %K, %D (14, 3)
        low14 = pd.Series(l).rolling(14).min()
        high14 = pd.Series(h).rolling(14).max()
        stoch_k = ((pd.Series(c) - low14) / (high14 - low14 + 1e-9) * 100).rolling(3).mean().values
        stoch_d = pd.Series(stoch_k).rolling(3).mean().values
        stoch_long = (stoch_k > stoch_d) & (stoch_k < 80)
        stoch_short = (stoch_k < stoch_d) & (stoch_k > 20)

        # 4. Volatility Crowd: Bollinger Bands (20, 2)
        bb_mid = pd.Series(c).rolling(20).mean()
        bb_std = pd.Series(c).rolling(20).std()
        bb_upper = (bb_mid + 2.0 * bb_std).values
        bb_lower = (bb_mid - 2.0 * bb_std).values
        # Long: price above middle band and pointing towards upper
        bb_long = (c > bb_mid.values) & (c <= bb_upper)
        bb_short = (c < bb_mid.values) & (c >= bb_lower)

        # 5. Candlestick Crowd: Directional body strength
        bar_range = np.maximum(h - l, 1e-6)
        body = np.abs(c - o)
        body_pct = body / bar_range
        candle_long = (c > o) & (body_pct >= 0.50) & ((c - l) / bar_range >= 0.65)
        candle_short = (c < o) & (body_pct >= 0.50) & ((h - c) / bar_range >= 0.65)

        # Aggregate Votes
        votes_long = (trend_long.astype(int) + rsi_long.astype(int) + 
                      stoch_long.astype(int) + bb_long.astype(int) + candle_long.astype(int))
        votes_short = (trend_short.astype(int) + rsi_short.astype(int) + 
                       stoch_short.astype(int) + bb_short.astype(int) + candle_short.astype(int))

        tick_ts = ticks['timestamp']
        time_ms = m15.index.astype(np.int64) // 10**6
        dur_ms = 15 * 60 * 1000
        hours = m15.index.hour

        traded_today = False
        last_day = -1

        for i in range(50, n_bars - 1):
            cur_day = m15.index[i].dayofyear
            if cur_day != last_day:
                last_day = cur_day
                traded_today = False

            if traded_today:
                continue

            if hours[i] not in self.allowed_hours:
                continue

            exec_time = time_ms[i] + dur_ms
            t_idx = int(np.searchsorted(tick_ts, exec_time))
            if t_idx >= n_ticks:
                break

            spread = (ticks['ask'][t_idx] - ticks['bid'][t_idx]) / PIP_SIZE
            if spread > 0.65:
                continue

            # Long Consensus
            if votes_long[i] >= self.min_votes:
                signals[t_idx] = SIGNAL_LONG
                sl_pips_arr[t_idx] = self.sl_pips
                tp_pips_arr[t_idx] = self.tp_pips
                traded_today = True

            # Short Consensus
            elif votes_short[i] >= self.min_votes:
                signals[t_idx] = SIGNAL_SHORT
                sl_pips_arr[t_idx] = self.sl_pips
                tp_pips_arr[t_idx] = self.tp_pips
                traded_today = True

        return signals, sl_pips_arr, tp_pips_arr


# ==============================================================================
# 2. WOTC_HeikenAshi_Ribbon (Classic Wisdom of the Crowd Template)
# ==============================================================================
class WOTCHeikenAshiRibbonStrategy(BaseStrategy):
    """
    Classic Forex 'Wisdom of the Crowd' Template:
    - Baseline: 50 EMA + 100 EMA on M15.
    - Fast Ribbon: EMA 5, 8, 13 (Consensus of fast crowd).
    - Trigger: When Fast Ribbon expands above 50 EMA, price retraces to touch the
      fast ribbon, then prints a strong Heiken Ashi reversal bar in the trend direction.
    """
    def __init__(
        self,
        sl_pips: float = 9.0,
        tp_pips: float = 12.0,
        target_rr: float = 1.33,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.target_rr = float(target_rr)
        self.allowed_hours = {7, 8, 9, 10, 11, 12, 13, 14, 15, 16}
        self.name = "WOTC_HeikenAshi_Ribbon"

    def generate_signals(self, ticks: np.ndarray):
        n_ticks = len(ticks)
        signals = np.zeros(n_ticks, dtype=np.int8)
        sl_pips_arr = np.zeros(n_ticks, dtype=np.float64)
        tp_pips_arr = np.zeros(n_ticks, dtype=np.float64)

        if n_ticks < 500:
            return signals, sl_pips_arr, tp_pips_arr

        m15 = build_time_bars(ticks, timeframe_min=15)
        if len(m15) < 70:
            return signals, sl_pips_arr, tp_pips_arr

        c = m15['close'].values
        h = m15['high'].values
        l = m15['low'].values
        o = m15['open'].values
        n_bars = len(m15)

        # Baseline EMAs
        ema50 = m15['close'].ewm(span=50, adjust=False).mean().values
        ema100 = m15['close'].ewm(span=100, adjust=False).mean().values

        # Fast Ribbon
        ema5 = m15['close'].ewm(span=5, adjust=False).mean().values
        ema8 = m15['close'].ewm(span=8, adjust=False).mean().values
        ema13 = m15['close'].ewm(span=13, adjust=False).mean().values

        # Compute Heiken Ashi bars
        ha_close = (o + h + l + c) / 4.0
        ha_open = np.zeros(n_bars, dtype=np.float64)
        ha_open[0] = (o[0] + c[0]) / 2.0
        for b in range(1, n_bars):
            ha_open[b] = (ha_open[b-1] + ha_close[b-1]) / 2.0
        ha_high = np.maximum(h, np.maximum(ha_open, ha_close))
        ha_low = np.minimum(l, np.minimum(ha_open, ha_close))

        # Ribbon Alignment
        ribbon_bull = (ema5 > ema8) & (ema8 > ema13) & (ema13 > ema50) & (ema50 > ema100)
        ribbon_bear = (ema5 < ema8) & (ema8 < ema13) & (ema13 < ema50) & (ema50 < ema100)

        tick_ts = ticks['timestamp']
        time_ms = m15.index.astype(np.int64) // 10**6
        dur_ms = 15 * 60 * 1000
        hours = m15.index.hour

        traded_today = False
        last_day = -1

        for i in range(50, n_bars - 1):
            cur_day = m15.index[i].dayofyear
            if cur_day != last_day:
                last_day = cur_day
                traded_today = False

            if traded_today:
                continue

            if hours[i] not in self.allowed_hours:
                continue

            exec_time = time_ms[i] + dur_ms
            t_idx = int(np.searchsorted(tick_ts, exec_time))
            if t_idx >= n_ticks:
                break

            spread = (ticks['ask'][t_idx] - ticks['bid'][t_idx]) / PIP_SIZE
            if spread > 0.65:
                continue

            # Bullish Trigger: Ribbon aligned bull, previous bar dipped near ema13, current HA is solid green (open == low)
            if ribbon_bull[i] and (l[i-1] <= ema13[i-1] or l[i] <= ema13[i]):
                if ha_close[i] > ha_open[i] and (ha_open[i] - ha_low[i]) <= (0.8 * PIP_SIZE):
                    signals[t_idx] = SIGNAL_LONG
                    sl_pips_arr[t_idx] = self.sl_pips
                    tp_pips_arr[t_idx] = self.tp_pips
                    traded_today = True

            # Bearish Trigger: Ribbon aligned bear, dipped near ema13, current HA is solid red (open == high)
            elif ribbon_bear[i] and (h[i-1] >= ema13[i-1] or h[i] >= ema13[i]):
                if ha_close[i] < ha_open[i] and (ha_high[i] - ha_open[i]) <= (0.8 * PIP_SIZE):
                    signals[t_idx] = SIGNAL_SHORT
                    sl_pips_arr[t_idx] = self.sl_pips
                    tp_pips_arr[t_idx] = self.tp_pips
                    traded_today = True

        return signals, sl_pips_arr, tp_pips_arr


# ==============================================================================
# 3. WOTC_Retail_Trap_Fade (Fading the Trapped Crowd at Liquidity Sweeps)
# ==============================================================================
class WOTCRetailTrapFadeStrategy(BaseStrategy):
    """
    Contrarian Exploitation of Retail Breakout Traps:
    - Measures Asian Session Range (00:00 - 06:00 UTC).
    - During London Open (07:00 - 10:30 UTC), price sweeps Asian High or Low by 1.5 - 6.5 pips.
    - If bar immediately REJECTS back inside the Asian range with a pin bar or strong rejection wick:
      Retail breakout crowd is trapped!
    - We enter contrarian with a tight SL outside the sweep wick (7.0 - 10.0 pips) and target 1.25R.
    """
    def __init__(
        self,
        min_sweep: float = 1.5,
        max_sweep: float = 7.0,
        target_rr: float = 1.25,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.min_sweep = float(min_sweep)
        self.max_sweep = float(max_sweep)
        self.target_rr = float(target_rr)
        self.name = "WOTC_Retail_Trap_Fade"

    def generate_signals(self, ticks: np.ndarray):
        n_ticks = len(ticks)
        signals = np.zeros(n_ticks, dtype=np.int8)
        sl_pips_arr = np.zeros(n_ticks, dtype=np.float64)
        tp_pips_arr = np.zeros(n_ticks, dtype=np.float64)

        if n_ticks < 500:
            return signals, sl_pips_arr, tp_pips_arr

        m15 = build_time_bars(ticks, timeframe_min=15)
        if len(m15) < 40:
            return signals, sl_pips_arr, tp_pips_arr

        c = m15['close'].values
        h = m15['high'].values
        l = m15['low'].values
        o = m15['open'].values
        n_bars = len(m15)

        tick_ts = ticks['timestamp']
        time_ms = m15.index.astype(np.int64) // 10**6
        dur_ms = 15 * 60 * 1000

        m15['day'] = m15.index.date
        m15['hour'] = m15.index.hour
        m15['minute'] = m15.index.minute

        # Identify Asian range per day
        unique_days = m15['day'].unique()
        for day in unique_days:
            day_mask = m15['day'] == day
            asian_mask = day_mask & (m15['hour'] >= 0) & (m15['hour'] < 6)
            asian_bars = m15[asian_mask]

            if len(asian_bars) < 16:
                continue

            asian_high = asian_bars['high'].max()
            asian_low = asian_bars['low'].min()
            asian_range = (asian_high - asian_low) / PIP_SIZE

            # Asian range must be normal (12 to 38 pips)
            if asian_range < 12.0 or asian_range > 38.0:
                continue

            # Hunt for London sweeps between 07:00 and 10:30 UTC
            london_mask = day_mask & (
                ((m15['hour'] >= 7) & (m15['hour'] <= 10))
            )
            london_indices = np.where(london_mask.values)[0]
            traded = False

            for idx in london_indices:
                if traded:
                    break

                row_h = h[idx]
                row_l = l[idx]
                row_c = c[idx]
                row_o = o[idx]

                # Bearish Trap: Price swept Asian High, but closed back BELOW Asian High with upper wick
                sweep_up = (row_h - asian_high) / PIP_SIZE
                if self.min_sweep <= sweep_up <= self.max_sweep and row_c < asian_high:
                    upper_wick = row_h - max(row_o, row_c)
                    bar_len = max(row_h - row_l, 1e-6)
                    if (upper_wick / bar_len) >= 0.40 and row_c < row_o:
                        # Trap confirmed! Sell
                        exec_t = time_ms[idx] + dur_ms
                        t_idx = int(np.searchsorted(tick_ts, exec_t))
                        if t_idx < n_ticks:
                            spread = (ticks['ask'][t_idx] - ticks['bid'][t_idx]) / PIP_SIZE
                            if spread <= 0.65:
                                risk_pips = max(sweep_up + 2.5, 7.5)
                                risk_pips = min(risk_pips, 12.0)
                                signals[t_idx] = SIGNAL_SHORT
                                sl_pips_arr[t_idx] = risk_pips
                                tp_pips_arr[t_idx] = risk_pips * self.target_rr
                                traded = True

                # Bullish Trap: Price swept Asian Low, but closed back ABOVE Asian Low with lower wick
                sweep_dn = (asian_low - row_l) / PIP_SIZE
                if self.min_sweep <= sweep_dn <= self.max_sweep and row_c > asian_low:
                    lower_wick = min(row_o, row_c) - row_l
                    bar_len = max(row_h - row_l, 1e-6)
                    if (lower_wick / bar_len) >= 0.40 and row_c > row_o:
                        # Trap confirmed! Buy
                        exec_t = time_ms[idx] + dur_ms
                        t_idx = int(np.searchsorted(tick_ts, exec_t))
                        if t_idx < n_ticks:
                            spread = (ticks['ask'][t_idx] - ticks['bid'][t_idx]) / PIP_SIZE
                            if spread <= 0.65:
                                risk_pips = max(sweep_dn + 2.5, 7.5)
                                risk_pips = min(risk_pips, 12.0)
                                signals[t_idx] = SIGNAL_LONG
                                sl_pips_arr[t_idx] = risk_pips
                                tp_pips_arr[t_idx] = risk_pips * self.target_rr
                                traded = True

        return signals, sl_pips_arr, tp_pips_arr


# ==============================================================================
# 4. WOTC_Multi_Timeframe_Supermajority (H4 Macro + H1 Trend + M15 Entry)
# ==============================================================================
class WOTCMultiTimeframeSupermajorityStrategy(BaseStrategy):
    """
    Wisdom of 3 Timeframe Crowds:
    - H4 Macro Crowd: Price > H4 EMA50, H4 slope positive
    - H1 Intraday Crowd: Price > H1 EMA20, H1 RSI > 50
    - M15 Micro Execution: Pullback into M15 20-EMA during London/NY liquid hours (07:00 - 16:00 UTC)
      with immediate pin-bar rejection.
    - Sized with tight 9.0p SL, 12.0p TP (1.33R).
    """
    def __init__(
        self,
        sl_pips: float = 9.0,
        tp_pips: float = 12.0,
        target_rr: float = 1.33,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.target_rr = float(target_rr)
        self.allowed_hours = {7, 8, 9, 10, 11, 12, 13, 14, 15, 16}
        self.name = "WOTC_MTF_Supermajority"

    def generate_signals(self, ticks: np.ndarray):
        n_ticks = len(ticks)
        signals = np.zeros(n_ticks, dtype=np.int8)
        sl_pips_arr = np.zeros(n_ticks, dtype=np.float64)
        tp_pips_arr = np.zeros(n_ticks, dtype=np.float64)

        if n_ticks < 500:
            return signals, sl_pips_arr, tp_pips_arr

        m15 = build_time_bars(ticks, timeframe_min=15)
        if len(m15) < 60:
            return signals, sl_pips_arr, tp_pips_arr

        # Resample H1 and H4
        ts = pd.to_datetime(ticks['timestamp'], unit='ms')
        mid = 0.5 * (ticks['bid'] + ticks['ask'])
        base = pd.DataFrame({'close': mid, 'high': mid, 'low': mid, 'open': mid}, index=ts)

        h1 = base.resample('1h').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
        h4 = base.resample('4h').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()

        if len(h1) < 25 or len(h4) < 15:
            return signals, sl_pips_arr, tp_pips_arr

        # H4 Macro
        h4_ema50 = h4['close'].ewm(span=50, adjust=False).mean()
        h4_macro = np.where(h4['close'] > h4_ema50, 1, -1)
        h4_macro_m15 = pd.Series(h4_macro, index=h4.index).shift(1).reindex(m15.index, method='ffill').fillna(0).values

        # H1 Intraday
        h1_ema20 = h1['close'].ewm(span=20, adjust=False).mean()
        h1_delta = h1['close'].diff()
        h1_gain = h1_delta.where(h1_delta > 0, 0.0).rolling(14).mean()
        h1_loss = (-h1_delta.where(h1_delta < 0, 0.0)).rolling(14).mean()
        h1_rs = h1_gain / (h1_loss + 1e-9)
        h1_rsi = 100 - (100 / (1 + h1_rs))
        h1_bull = (h1['close'] > h1_ema20) & (h1_rsi > 50)
        h1_bear = (h1['close'] < h1_ema20) & (h1_rsi < 50)
        h1_state = np.where(h1_bull, 1, np.where(h1_bear, -1, 0))
        h1_state_m15 = pd.Series(h1_state, index=h1.index).shift(1).reindex(m15.index, method='ffill').fillna(0).values

        # M15 Micro
        m15_ema20 = m15['close'].ewm(span=20, adjust=False).mean().values
        c = m15['close'].values
        h = m15['high'].values
        l = m15['low'].values
        o = m15['open'].values
        n_bars = len(m15)

        tick_ts = ticks['timestamp']
        time_ms = m15.index.astype(np.int64) // 10**6
        dur_ms = 15 * 60 * 1000
        hours = m15.index.hour

        traded_today = False
        last_day = -1

        for i in range(30, n_bars - 1):
            cur_day = m15.index[i].dayofyear
            if cur_day != last_day:
                last_day = cur_day
                traded_today = False

            if traded_today:
                continue

            if hours[i] not in self.allowed_hours:
                continue

            exec_time = time_ms[i] + dur_ms
            t_idx = int(np.searchsorted(tick_ts, exec_time))
            if t_idx >= n_ticks:
                break

            spread = (ticks['ask'][t_idx] - ticks['bid'][t_idx]) / PIP_SIZE
            if spread > 0.65:
                continue

            bar_range = max(h[i] - l[i], 1e-6)

            # Triple Bullish Supermajority
            if h4_macro_m15[i] == 1 and h1_state_m15[i] == 1:
                # M15 dipped to or near EMA20 and printed bullish rejection wick
                if l[i] <= m15_ema20[i] + 1.5 * PIP_SIZE and c[i] > m15_ema20[i]:
                    lower_wick = min(o[i], c[i]) - l[i]
                    if (lower_wick / bar_range) >= 0.40 and c[i] >= o[i]:
                        signals[t_idx] = SIGNAL_LONG
                        sl_pips_arr[t_idx] = self.sl_pips
                        tp_pips_arr[t_idx] = self.tp_pips
                        traded_today = True

            # Triple Bearish Supermajority
            elif h4_macro_m15[i] == -1 and h1_state_m15[i] == -1:
                # M15 rose to or near EMA20 and printed bearish rejection wick
                if h[i] >= m15_ema20[i] - 1.5 * PIP_SIZE and c[i] < m15_ema20[i]:
                    upper_wick = h[i] - max(o[i], c[i])
                    if (upper_wick / bar_range) >= 0.40 and c[i] <= o[i]:
                        signals[t_idx] = SIGNAL_SHORT
                        sl_pips_arr[t_idx] = self.sl_pips
                        tp_pips_arr[t_idx] = self.tp_pips
                        traded_today = True

        return signals, sl_pips_arr, tp_pips_arr


# ==============================================================================
# AUDIT RUNNER ACROSS 52 WEEKS
# ==============================================================================
def _worker_eval(args):
    w, StratCls, kwargs = args
    ticks, _ = get_week_dataset(w)
    engine = BacktestEngine(pip_size=0.0001, use_slippage=True)
    strat = StratCls(**kwargs)
    trades = engine.run(ticks, strat)
    return trades


def screen_candidate(StratCls, **kwargs):
    weeks = [(w, StratCls, kwargs) for w in range(1, 53)]
    num_workers = min(8, os.cpu_count() or 4)
    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        results = list(ex.map(_worker_eval, weeks))

    trades = [t for r in results for t in r]
    n = len(trades)
    if n == 0:
        return {'name': StratCls().name, 'trades': 0, 'wr': 0, 'pips': 0, 'pf': 0, 'max_dd': 0, 'rr': 0}

    pips = sum(t['pips'] for t in trades)
    wins = [t for t in trades if t['pips'] > 0]
    losses = [t for t in trades if t['pips'] < 0]
    wr = len(wins) / n * 100.0
    gp = sum(t['pips'] for t in wins)
    gl = abs(sum(t['pips'] for t in losses))
    pf = gp / gl if gl > 0 else 99.0
    realized_rr = (gp / max(len(wins), 1)) / (gl / max(len(losses), 1)) if gl > 0 else 0.0

    equity = np.cumsum([t['pips'] for t in trades])
    peak = np.maximum.accumulate(equity)
    dd = peak - equity
    max_dd = dd.max()

    return {
        'name': StratCls().name,
        'trades': n,
        'wr': wr,
        'pips': pips,
        'pf': pf,
        'max_dd': max_dd,
        'rr': realized_rr
    }


if __name__ == "__main__":
    t0 = time.time()
    print("=" * 80)
    print("  SCREENING: 4 'WISDOM OF THE CROWD' HYPOTHESES ACROSS 52 WEEKS (~26M TICKS)")
    print("  TARGET CRITERIA: WIN RATE > 50%, POSITIVE EXPECTANCY, LOW DRAWDOWN")
    print("=" * 80)

    candidates = [
        (WOTCIndicatorConsensusStrategy, {'min_votes': 4, 'sl_pips': 10.0, 'tp_pips': 13.0}),
        (WOTCHeikenAshiRibbonStrategy, {'sl_pips': 9.0, 'tp_pips': 12.0}),
        (WOTCRetailTrapFadeStrategy, {'min_sweep': 1.5, 'max_sweep': 6.5, 'target_rr': 1.25}),
        (WOTCMultiTimeframeSupermajorityStrategy, {'sl_pips': 9.0, 'tp_pips': 12.5}),
    ]

    for StratCls, kwargs in candidates:
        sub_t0 = time.time()
        res = screen_candidate(StratCls, **kwargs)
        sub_dt = time.time() - sub_t0
        print(f"Strategy: {res['name']}")
        print(f"  Trades:   {res['trades']} ({res['trades']/52:.1f}/wk)")
        print(f"  Win Rate: {res['wr']:.1f}%  <--- [WR TARGET: > 50%]")
        print(f"  PF:       {res['pf']:.2f}")
        print(f"  Real RR:  {res['rr']:.2f} : 1")
        print(f"  Net Pips: {res['pips']:+.1f} pips")
        print(f"  Max DD:   {res['max_dd']:.1f} pips")
        print(f"  Duration: {sub_dt:.1f}s")
        print("-" * 60)

    print(f"All 4 candidate hypotheses screened in {time.time() - t0:.1f}s total.")
