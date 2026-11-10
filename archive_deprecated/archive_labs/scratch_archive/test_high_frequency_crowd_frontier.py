#!/usr/bin/env python3
"""
================================================================================
  HIGH-FREQUENCY HIGH-WIN-RATE LAB: WISDOM OF THE CROWD (WR > 50%, HIGH TRADES)
================================================================================
Targeting:
  1. High Trade Count: >= 150 - 400+ trades per year (750 - 2,000+ trades / 5Y)
  2. Win Rate strictly > 50% (52% - 65%+)
  3. Smooth Equity Curve (Low DD, high Profit Factor > 1.30, steady daily/weekly cash flow)
  4. Complete protection against spread drag (SL >= 6.0p, TP >= 8.0p, spread filter <= 0.55p)

Candidates Tested:
  1. WOTC_M5_Multi_Quorum_Scalper:
     5-Indicator Consensus on M5 (EMA Ribbon, RSI, MACD, Stoch, Candle) + H1 50-EMA trend.
     Cooldown: 6 bars (30 mins). Fires multiple times per day.

  2. WOTC_M5_Micro_Rejection_Continuous:
     Intraday H1 Structural Level + M5 50% Rejection Wick with a 45-min cooldown
     (allowing multiple intraday structural defenses per day instead of 1/day).

  3. WOTC_M5_Momentum_Pulse:
     M5 Fast Ribbon (EMA 5 > 8 > 13) + RSI Momentum cross + Tick Volume / Range expansion.
     Cooldown: 5 bars (25 mins).

  4. WOTC_M15_Multi_Signal_Engine:
     M15 Trend Continuation Quorum with a 2-bar cooldown (30 mins).
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
# 1. WOTC_M5_Multi_Quorum_Scalper (High Trade Count 5-Indicator Quorum)
# ==============================================================================
class WOTCM5MultiQuorumScalper(BaseStrategy):
    """
    Evaluates every completed M5 bar in active hours (07:00 - 17:00 UTC).
    Macro Filter: H1 50-EMA trend.
    Micro Council (5 indicators on M5):
      1. EMA Ribbon: EMA 8 > EMA 21 (bull) or EMA 8 < EMA 21 (bear)
      2. RSI: RSI(14) between 50 and 68 (bull) or 32 and 50 (bear)
      3. MACD: MACD Line > Signal Line and expanding
      4. Stoch: %K > %D and %K between 25 and 75
      5. Candle Body: Closes in outer 40% of range with directional body
    Requires >= 4 out of 5 votes.
    Cooldown: 6 bars (30 min) between trades.
    SL: 6.5p, TP: 8.5p (1.31R)
    """
    def __init__(
        self,
        sl_pips: float = 6.5,
        tp_pips: float = 8.5,
        cooldown_bars: int = 6,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.cooldown_bars = int(cooldown_bars)
        self.allowed_hours = {7, 8, 9, 10, 11, 12, 13, 14, 15, 16}
        self.name = f"WOTC_M5_Quorum_{self.sl_pips}SL_{self.tp_pips}TP_CD{self.cooldown_bars}"

    def generate_signals(self, ticks: np.ndarray):
        n_ticks = len(ticks)
        signals = np.zeros(n_ticks, dtype=np.int8)
        sl_pips_arr = np.zeros(n_ticks, dtype=np.float64)
        tp_pips_arr = np.zeros(n_ticks, dtype=np.float64)

        if n_ticks < 500:
            return signals, sl_pips_arr, tp_pips_arr

        m5 = build_time_bars(ticks, timeframe_min=5)
        if len(m5) < 70:
            return signals, sl_pips_arr, tp_pips_arr

        # H1 Macro Trend
        ts = pd.to_datetime(ticks['timestamp'], unit='ms')
        mid = 0.5 * (ticks['bid'] + ticks['ask'])
        base = pd.DataFrame({'close': mid}, index=ts)
        h1 = base.resample('1h').last().dropna()
        if len(h1) < 50:
            return signals, sl_pips_arr, tp_pips_arr

        h1['ema50'] = h1['close'].ewm(span=50, adjust=False).mean()
        h1['macro'] = np.where(h1['close'] > h1['ema50'], 1, -1)
        h1_macro_m5 = h1['macro'].reindex(m5.index, method='ffill').fillna(0).values

        c = m5['close'].values
        h = m5['high'].values
        l = m5['low'].values
        o = m5['open'].values
        n_bars = len(m5)

        # 1. EMA Ribbon 8 & 21
        ema8 = m5['close'].ewm(span=8, adjust=False).mean().values
        ema21 = m5['close'].ewm(span=21, adjust=False).mean().values
        v1_bull = (c > ema8) & (ema8 > ema21)
        v1_bear = (c < ema8) & (ema8 < ema21)

        # 2. RSI 14
        delta = pd.Series(c).diff()
        gain = delta.where(delta > 0, 0.0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
        rs = gain / (loss + 1e-9)
        rsi = (100 - (100 / (1 + rs))).values
        v2_bull = (rsi >= 50) & (rsi <= 68)
        v2_bear = (rsi <= 50) & (rsi >= 32)

        # 3. MACD (12, 26, 9)
        ema12 = pd.Series(c).ewm(span=12, adjust=False).mean()
        ema26 = pd.Series(c).ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        hist = (macd - signal).values
        v3_bull = (hist > 0) & (hist > np.roll(hist, 1))
        v3_bear = (hist < 0) & (hist < np.roll(hist, 1))

        # 4. Stochastic (14, 3)
        low14 = pd.Series(l).rolling(14).min()
        high14 = pd.Series(h).rolling(14).max()
        stoch_k = ((pd.Series(c) - low14) / (high14 - low14 + 1e-9) * 100).rolling(3).mean().values
        stoch_d = pd.Series(stoch_k).rolling(3).mean().values
        v4_bull = (stoch_k > stoch_d) & (stoch_k >= 30) & (stoch_k <= 75)
        v4_bear = (stoch_k < stoch_d) & (stoch_k <= 70) & (stoch_k >= 25)

        # 5. Candle Directional Body
        bar_range = np.maximum(h - l, 1e-6)
        v5_bull = (c > o) & ((c - l) / bar_range >= 0.60)
        v5_bear = (c < o) & ((h - c) / bar_range >= 0.60)

        votes_bull = v1_bull.astype(int) + v2_bull.astype(int) + v3_bull.astype(int) + v4_bull.astype(int) + v5_bull.astype(int)
        votes_bear = v1_bear.astype(int) + v2_bear.astype(int) + v3_bear.astype(int) + v4_bear.astype(int) + v5_bear.astype(int)

        tick_ts = ticks['timestamp']
        time_ms = m5.index.astype(np.int64) // 10**6
        dur_ms = 5 * 60 * 1000
        hours = m5.index.hour

        last_trade_idx = -999

        for i in range(50, n_bars - 1):
            if hours[i] not in self.allowed_hours:
                continue

            if (i - last_trade_idx) < self.cooldown_bars:
                continue

            exec_time = time_ms[i] + dur_ms
            t_idx = int(np.searchsorted(tick_ts, exec_time))
            if t_idx >= n_ticks:
                break

            spread = (ticks['ask'][t_idx] - ticks['bid'][t_idx]) / PIP_SIZE
            if spread > 0.55:
                continue

            # Bullish Quorum
            if h1_macro_m5[i] == 1 and votes_bull[i] >= 4:
                signals[t_idx] = SIGNAL_LONG
                sl_pips_arr[t_idx] = self.sl_pips
                tp_pips_arr[t_idx] = self.tp_pips
                last_trade_idx = i

            # Bearish Quorum
            elif h1_macro_m5[i] == -1 and votes_bear[i] >= 4:
                signals[t_idx] = SIGNAL_SHORT
                sl_pips_arr[t_idx] = self.sl_pips
                tp_pips_arr[t_idx] = self.tp_pips
                last_trade_idx = i

        return signals, sl_pips_arr, tp_pips_arr


# ==============================================================================
# 2. WOTC_M5_Micro_Rejection_Continuous (High-Frequency Structural Defense)
# ==============================================================================
class WOTCM5MicroRejectionContinuous(BaseStrategy):
    """
    Adapting DISC43's 52% WR mechanism with a shorter cooldown (8 bars = 40 mins)
    and evaluating 8-bar rolling support/resistance to capture multiple high-probability
    bounces across London and NY sessions.
    """
    def __init__(
        self,
        sl_pips: float = 6.0,
        tp_pips: float = 8.5,
        min_wick: float = 0.50,
        cooldown_bars: int = 8,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.min_wick = float(min_wick)
        self.cooldown_bars = int(cooldown_bars)
        self.name = f"WOTC_M5_Rejection_Continuous_{self.sl_pips}SL_{self.tp_pips}TP"

    def generate_signals(self, ticks: np.ndarray):
        n_ticks = len(ticks)
        signals = np.zeros(n_ticks, dtype=np.int8)
        sl_pips_arr = np.zeros(n_ticks, dtype=np.float64)
        tp_pips_arr = np.zeros(n_ticks, dtype=np.float64)

        if n_ticks < 500:
            return signals, sl_pips_arr, tp_pips_arr

        m5 = build_time_bars(ticks, timeframe_min=5)
        if len(m5) < 50:
            return signals, sl_pips_arr, tp_pips_arr

        ts = pd.to_datetime(ticks['timestamp'], unit='ms')
        mid = 0.5 * (ticks['bid'] + ticks['ask'])
        base = pd.DataFrame({'close': mid, 'high': mid, 'low': mid, 'open': mid}, index=ts)
        h1 = base.resample('1h').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
        if len(h1) < 15:
            return signals, sl_pips_arr, tp_pips_arr

        h1['ema50'] = h1['close'].ewm(span=50, adjust=False).mean()
        h1['macro'] = np.where(h1['close'] > h1['ema50'], 1, -1)
        h1['roll_h'] = h1['high'].rolling(10).max().shift(1).bfill()
        h1['roll_l'] = h1['low'].rolling(10).min().shift(1).bfill()

        h1_macro = h1['macro'].reindex(m5.index, method='ffill').fillna(0).values
        h1_supp = h1['roll_l'].reindex(m5.index, method='ffill').values
        h1_res = h1['roll_h'].reindex(m5.index, method='ffill').values

        c = m5['close'].values
        h = m5['high'].values
        l = m5['low'].values
        o = m5['open'].values
        spread = m5['spread_pips'].values
        hours = m5.index.hour.values

        time_ms = m5.index.astype('datetime64[ms]').view('int64')
        dur_ms = 5 * 60 * 1000
        tick_ts = ticks['timestamp']

        last_trade_idx = -999

        for i in range(15, len(m5)):
            hr = hours[i]
            if not (7 <= hr <= 16):
                continue
            if spread[i] > 0.55:
                continue

            if (i - last_trade_idx) < self.cooldown_bars:
                continue

            m = h1_macro[i]
            bar_rng = (h[i] - l[i]) / PIP_SIZE
            if bar_rng < 2.5:
                continue

            sig = 0
            if m > 0:
                dist_supp = abs(l[i] - h1_supp[i]) / PIP_SIZE
                lower_wick = (min(o[i], c[i]) - l[i]) / PIP_SIZE
                if dist_supp <= 4.5 and (lower_wick / bar_rng) >= self.min_wick and c[i] > o[i]:
                    sig = SIGNAL_LONG

            elif m < 0:
                dist_res = abs(h1_res[i] - h[i]) / PIP_SIZE
                upper_wick = (h[i] - max(o[i], c[i])) / PIP_SIZE
                if dist_res <= 4.5 and (upper_wick / bar_rng) >= self.min_wick and c[i] < o[i]:
                    sig = SIGNAL_SHORT

            if sig != 0:
                exec_t = time_ms[i] + dur_ms
                idx = int(np.searchsorted(tick_ts, exec_t))
                if idx < n_ticks and signals[idx] == 0:
                    signals[idx] = sig
                    sl_pips_arr[idx] = self.sl_pips
                    tp_pips_arr[idx] = self.tp_pips
                    last_trade_idx = i

        return signals, sl_pips_arr, tp_pips_arr


# ==============================================================================
# 3. WOTC_M5_Momentum_Pulse (Fast Ribbon + RSI Impulse + Range Expansion)
# ==============================================================================
class WOTCM5MomentumPulse(BaseStrategy):
    """
    Momentum Pulse Scalper:
    - H1 50-EMA Macro Trend
    - M5 Ribbon: EMA 5 > EMA 13 > EMA 34
    - Pullback into EMA 13, then bar closes green breaking above previous bar high with
      bar range > 1.25x average 10-bar range (momentum injection).
    - Cooldown: 6 bars (30 min).
    - SL: 6.5p, TP: 8.5p (1.31R).
    """
    def __init__(
        self,
        sl_pips: float = 6.5,
        tp_pips: float = 8.5,
        cooldown_bars: int = 6,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.cooldown_bars = int(cooldown_bars)
        self.allowed_hours = {7, 8, 9, 10, 11, 12, 13, 14, 15, 16}
        self.name = f"WOTC_M5_Momentum_Pulse_{self.sl_pips}SL_{self.tp_pips}TP"

    def generate_signals(self, ticks: np.ndarray):
        n_ticks = len(ticks)
        signals = np.zeros(n_ticks, dtype=np.int8)
        sl_pips_arr = np.zeros(n_ticks, dtype=np.float64)
        tp_pips_arr = np.zeros(n_ticks, dtype=np.float64)

        if n_ticks < 500:
            return signals, sl_pips_arr, tp_pips_arr

        m5 = build_time_bars(ticks, timeframe_min=5)
        if len(m5) < 60:
            return signals, sl_pips_arr, tp_pips_arr

        ts = pd.to_datetime(ticks['timestamp'], unit='ms')
        mid = 0.5 * (ticks['bid'] + ticks['ask'])
        base = pd.DataFrame({'close': mid}, index=ts)
        h1 = base.resample('1h').last().dropna()
        if len(h1) < 50:
            return signals, sl_pips_arr, tp_pips_arr

        h1['ema50'] = h1['close'].ewm(span=50, adjust=False).mean()
        h1['macro'] = np.where(h1['close'] > h1['ema50'], 1, -1)
        h1_macro_m5 = h1['macro'].reindex(m5.index, method='ffill').fillna(0).values

        c = m5['close'].values
        h = m5['high'].values
        l = m5['low'].values
        o = m5['open'].values
        n_bars = len(m5)

        ema5 = m5['close'].ewm(span=5, adjust=False).mean().values
        ema13 = m5['close'].ewm(span=13, adjust=False).mean().values
        ema34 = m5['close'].ewm(span=34, adjust=False).mean().values

        bar_range = h - l
        avg_range = pd.Series(bar_range).rolling(10).mean().values

        ribbon_bull = (ema5 > ema13) & (ema13 > ema34)
        ribbon_bear = (ema5 < ema13) & (ema13 < ema34)

        tick_ts = ticks['timestamp']
        time_ms = m5.index.astype(np.int64) // 10**6
        dur_ms = 5 * 60 * 1000
        hours = m5.index.hour

        last_trade_idx = -999

        for i in range(40, n_bars - 1):
            if hours[i] not in self.allowed_hours:
                continue

            if (i - last_trade_idx) < self.cooldown_bars:
                continue

            exec_time = time_ms[i] + dur_ms
            t_idx = int(np.searchsorted(tick_ts, exec_time))
            if t_idx >= n_ticks:
                break

            spread = (ticks['ask'][t_idx] - ticks['bid'][t_idx]) / PIP_SIZE
            if spread > 0.55:
                continue

            # Bullish Pulse: H1 Bull, M5 Ribbon Bull, previous bar dipped near ema13, current bar expands
            if h1_macro_m5[i] == 1 and ribbon_bull[i] and (l[i-1] <= ema13[i-1] or l[i] <= ema13[i]):
                if c[i] > o[i] and c[i] > h[i-1] and bar_range[i] >= 1.20 * avg_range[i]:
                    signals[t_idx] = SIGNAL_LONG
                    sl_pips_arr[t_idx] = self.sl_pips
                    tp_pips_arr[t_idx] = self.tp_pips
                    last_trade_idx = i

            # Bearish Pulse: H1 Bear, M5 Ribbon Bear, previous bar rose near ema13, current bar expands
            elif h1_macro_m5[i] == -1 and ribbon_bear[i] and (h[i-1] >= ema13[i-1] or h[i] >= ema13[i]):
                if c[i] < o[i] and c[i] < l[i-1] and bar_range[i] >= 1.20 * avg_range[i]:
                    signals[t_idx] = SIGNAL_SHORT
                    sl_pips_arr[t_idx] = self.sl_pips
                    tp_pips_arr[t_idx] = self.tp_pips
                    last_trade_idx = i

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
    strat_inst = StratCls(**kwargs)
    if n == 0:
        return {'name': strat_inst.name, 'trades': 0, 'wr': 0, 'pips': 0, 'pf': 0, 'max_dd': 0, 'rr': 0}

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
        'name': strat_inst.name,
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
    print("  SCREENING HIGH-FREQUENCY HIGH-WIN-RATE CANDIDATES (52 WEEKS / ~26M TICKS)")
    print("  TARGET: HIGH TRADE COUNT (150-400/YR) + WIN RATE > 50% + SMOOTH EQUITY")
    print("=" * 80)

    candidates = [
        (WOTCM5MultiQuorumScalper, {'sl_pips': 6.5, 'tp_pips': 8.5, 'cooldown_bars': 6}),
        (WOTCM5MultiQuorumScalper, {'sl_pips': 6.0, 'tp_pips': 7.5, 'cooldown_bars': 6}),
        (WOTCM5MicroRejectionContinuous, {'sl_pips': 6.0, 'tp_pips': 8.5, 'min_wick': 0.50, 'cooldown_bars': 8}),
        (WOTCM5MicroRejectionContinuous, {'sl_pips': 6.0, 'tp_pips': 8.0, 'min_wick': 0.50, 'cooldown_bars': 6}),
        (WOTCM5MomentumPulse, {'sl_pips': 6.5, 'tp_pips': 8.5, 'cooldown_bars': 6}),
    ]

    for StratCls, kwargs in candidates:
        sub_t0 = time.time()
        res = screen_candidate(StratCls, **kwargs)
        sub_dt = time.time() - sub_t0
        print(f"Strategy: {res['name']}")
        print(f"  Trades:   {res['trades']} ({res['trades']/52:.1f}/wk | ~{res['trades']} / yr) <--- [TRADE FREQUENCY]")
        print(f"  Win Rate: {res['wr']:.1f}%  <--- [WR TARGET: > 50%]")
        print(f"  PF:       {res['pf']:.2f}")
        print(f"  Real RR:  {res['rr']:.2f} : 1")
        print(f"  Net Pips: {res['pips']:+.1f} pips")
        print(f"  Max DD:   {res['max_dd']:.1f} pips")
        print(f"  Duration: {sub_dt:.1f}s")
        print("-" * 60)

    print(f"Screening complete in {time.time() - t0:.1f}s total.")
