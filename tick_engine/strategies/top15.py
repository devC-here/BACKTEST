"""
Top 15 High-Conviction Strategy Fleet for Tick Engine.
Bridges the 15 institutional models with the high-speed tick matching loop.
"""

from __future__ import annotations
import sys
import os
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd

from .base import BaseStrategy, SIGNAL_HOLD, SIGNAL_LONG, SIGNAL_SHORT, SIGNAL_FORCE_EXIT

# Ensure top_15_strategies is accessible for authentic evaluators
TOP15_DIR = Path(__file__).resolve().parent.parent.parent / "top_15_strategies"
if not TOP15_DIR.exists():
    TOP15_DIR = Path(__file__).resolve().parent.parent.parent / "legacy_labs" / "top_15_strategies"
if str(TOP15_DIR) not in sys.path:
    sys.path.insert(0, str(TOP15_DIR))

try:
    from evaluators import evaluate_strategy_detail
except ImportError:
    evaluate_strategy_detail = None



TOP_15_FLEET = [
    {"id": 8,  "code": "S08", "name": "Smart Money H1 Demand/Supply + 8 EMA", "tf": "M15", "allowed_hours": [0, 2, 3, 5, 6, 7, 8, 9, 12, 14, 15, 17]},
    {"id": 26, "code": "S26", "name": "Mamba FX Forex H4 S/R + 8/50 SMA",    "tf": "M15", "allowed_hours": [1, 2, 3, 7, 8, 9, 12, 13, 14, 15]},
    {"id": 28, "code": "S28", "name": "Chandelier Exit (22, 3.0) + ZLSMA",    "tf": "H1",  "allowed_hours": [2, 3, 7, 8, 9, 13, 14, 15]},
    {"id": 54, "code": "S54", "name": "Renko 10 SMA + OBV Volume Trend",      "tf": "M15", "allowed_hours": [2, 3, 7, 8, 9, 13, 14, 15]},
    {"id": 34, "code": "S34", "name": "Asia Forex Mentor Smoothed Heikin Ashi","tf": "H1", "allowed_hours": [2, 7, 8, 9, 13, 14, 15]},
    {"id": 1,  "code": "S01", "name": "ICT Market Structure Shift & Breaker",  "tf": "M15", "allowed_hours": [1, 2, 3, 7, 8, 9, 13, 14, 15]},
    {"id": 6,  "code": "S06", "name": "The Trading School Fib Golden Zone",     "tf": "M15", "allowed_hours": [1, 2, 7, 8, 9, 13, 14, 15]},
    {"id": 23, "code": "S23", "name": "Gomathi Shankar CPR + R1/PDH Break",    "tf": "M5",  "allowed_hours": [2, 3, 7, 8, 9, 13, 14, 15]},
    {"id": 43, "code": "S43", "name": "HalfTrend (2, 2.0) ATR Channel Break", "tf": "M15", "allowed_hours": [2, 3, 7, 8, 9, 13, 14, 15]},
    {"id": 18, "code": "S18", "name": "Multi-TF MACD Momentum",                 "tf": "M15", "allowed_hours": [2, 3, 7, 8, 9, 13, 14, 15]},
    {"id": 36, "code": "S36", "name": "Trade Like Berlin 'Final Boss' 7 EMA",   "tf": "M15", "allowed_hours": [2, 3, 7, 8, 9, 13, 14, 15]},
    {"id": 10, "code": "S10", "name": "20/200 Dual EMA Macro Trend Pullback",   "tf": "M5",  "allowed_hours": [2, 3, 7, 8, 9, 13, 14, 15]},
    {"id": 3,  "code": "S03", "name": "TG London Killzone EMA Stack + FVG",    "tf": "M30", "allowed_hours": [2, 3, 7, 8, 9, 13, 14, 15]},
    {"id": 14, "code": "S14", "name": "Rayner Teo 10 EMA + Inside Bar",        "tf": "H1",  "allowed_hours": [2, 7, 8, 9, 13, 14, 15]},
    {"id": 51, "code": "S51", "name": "Booming Bulls HalfTrend + DMI",          "tf": "M15", "allowed_hours": [2, 3, 7, 8, 9, 13, 14, 15]},
]

# Quick lookup by id or code
TOP_15_BY_ID = {s["id"]: s for s in TOP_15_FLEET}
TOP_15_BY_CODE = {s["code"].upper(): s for s in TOP_15_FLEET}

# Global weak cache for multi-TF candles so all 15 models reuse the same week calculation
_RATES_CACHE: Dict[Tuple[int, int, int], Dict[str, pd.DataFrame]] = {}


def enrich_timeframe_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes all technical indicators required by the Top 15 strategies in a vectorized manner.
    """
    c = df['close']
    h = df['high']
    l = df['low']
    o = df['open']

    # EMAs
    for p in [7, 8, 10, 13, 20, 21, 50, 200]:
        df[f'ema_{p}'] = c.ewm(span=p, adjust=False).mean()

    # SMAs
    df['sma_10'] = c.rolling(10).mean().bfill()
    df['sma_50'] = c.rolling(50).mean().bfill()

    # ATRs
    df['atr_14'] = (h - l).rolling(14).mean().bfill()
    df['atr_10'] = (h - l).rolling(10).mean().bfill()
    df['atr_22'] = (h - l).rolling(22).mean().bfill()

    # MACD
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    df['macd_line'] = ema12 - ema26
    df['macd_sig'] = df['macd_line'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd_line'] - df['macd_sig']

    # Chandelier Exit (22, 3.0)
    df['chandelier_long'] = h.rolling(22).max() - (3.0 * df['atr_22'])
    df['chandelier_short'] = l.rolling(22).min() + (3.0 * df['atr_22'])

    # Zero Lag SMA (32)
    lag = int((32 - 1) / 2)
    zlsma_data = c + (c - c.shift(lag))
    df['zlsma_32'] = zlsma_data.rolling(32).mean().fillna(c)

    # Heikin Ashi
    ha_c = (o + h + l + c) / 4.0
    n = len(df)
    ha_o = np.zeros(n)
    if n > 0:
        ha_o[0] = (o.iloc[0] + c.iloc[0]) / 2.0
        for i in range(1, n):
            ha_o[i] = (ha_o[i - 1] + ha_c.iloc[i - 1]) / 2.0
    df['ha_close'] = ha_c
    df['ha_open'] = pd.Series(ha_o, index=df.index)
    df['ha_high'] = pd.concat([h, df['ha_open'], df['ha_close']], axis=1).max(axis=1)
    df['ha_low'] = pd.concat([l, df['ha_open'], df['ha_close']], axis=1).min(axis=1)

    # DMI (+DI, -DI)
    up_move = h.diff()
    down_move = -l.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    tr_smooth = pd.Series(tr).ewm(alpha=1.0 / 14, adjust=False).mean()
    df['plus_di'] = (pd.Series(plus_dm, index=df.index).ewm(alpha=1.0 / 14, adjust=False).mean() / tr_smooth) * 100
    df['minus_di'] = (pd.Series(minus_dm, index=df.index).ewm(alpha=1.0 / 14, adjust=False).mean() / tr_smooth) * 100

    # CPR & Equilibrium
    high_roll = h.rolling(20).max()
    low_roll = l.rolling(20).min()
    df['cpr_p'] = (high_roll + low_roll + c) / 3.0
    df['cpr_r1'] = (2.0 * df['cpr_p']) - low_roll
    df['cpr_s1'] = (2.0 * df['cpr_p']) - high_roll
    df['Equilibrium'] = (high_roll + low_roll) / 2.0

    # Supertrend
    hl2 = (h + l) / 2.0
    df['supertrend'] = hl2 - (2.0 * df['atr_14'])

    df['hour'] = df.index.hour
    df['time'] = df.index.strftime('%Y-%m-%d %H:%M:%S')

    return df


def build_rates_by_timeframe(ticks: np.ndarray) -> Dict[str, pd.DataFrame]:
    """
    Builds multi-timeframe candle DataFrames (M5, M15, M30, H1, H4, D1)
    from tick data with vectorized indicator columns. Caches by tick footprint.
    """
    if len(ticks) == 0:
        return {}

    cache_key = (len(ticks), int(ticks['timestamp'][0]), int(ticks['timestamp'][-1]))
    if cache_key in _RATES_CACHE:
        return _RATES_CACHE[cache_key]

    ts = pd.to_datetime(ticks['timestamp'], unit='ms')
    mid = 0.5 * (ticks['bid'] + ticks['ask'])
    base_df = pd.DataFrame({'open': mid, 'high': mid, 'low': mid, 'close': mid}, index=ts)

    tf_map = {
        'M5': '5min',
        'M15': '15min',
        'M30': '30min',
        'H1': '1h',
        'H2': '2h',
        'H4': '4h',
        'D1': '1D'
    }

    rates = {}
    for name, freq in tf_map.items():
        res = base_df.resample(freq).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last'
        }).dropna()

        if len(res) == 0:
            continue

        rates[name] = enrich_timeframe_indicators(res)

    # Fallbacks for missing higher timeframes
    if 'H2' not in rates and 'H1' in rates:
        rates['H2'] = rates['H1']
    if 'H4' not in rates and 'H1' in rates:
        rates['H4'] = rates['H1']
    if 'D1' not in rates and 'H4' in rates:
        rates['D1'] = rates['H4']

    # Keep cache bounded
    if len(_RATES_CACHE) > 8:
        _RATES_CACHE.clear()

    _RATES_CACHE[cache_key] = rates
    return rates


class Top15Strategy(BaseStrategy):
    """
    Adapter strategy that runs authentic top-15 indicator logic on resampled candles
    and emits tick-level entry/exit decisions with authentic Stop Loss & Take Profit.
    """
    def __init__(
        self,
        strategy_id: int = 34,
        enforce_hours: bool = True,
        spread_pips: float = 0.8,
        min_lookback: int = 25,
        **kwargs
    ):
        self.strategy_id = int(strategy_id)
        info = TOP_15_BY_ID.get(self.strategy_id, {"code": f"S{strategy_id:02d}", "name": f"Strategy #{strategy_id}", "tf": "M15"})
        self.code = info["code"]
        self.name = f"{self.code} {info['name']}"
        self.exec_tf = info["tf"]
        self.allowed_hours = info.get("allowed_hours", [])
        self.enforce_hours = bool(enforce_hours)
        self.spread_pips = float(spread_pips)
        self.min_lookback = int(min_lookback)

        super().__init__(**kwargs)

    def generate_signals(self, ticks: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Evaluates authentic strategy criteria across resampled candles and maps signals
        to tick-level indices with individual SL and TP levels.
        """
        n_ticks = len(ticks)
        signals = np.zeros(n_ticks, dtype=np.int8)
        sl_pips_arr = np.zeros(n_ticks, dtype=np.float64)
        tp_pips_arr = np.zeros(n_ticks, dtype=np.float64)

        if evaluate_strategy_detail is None:
            raise ImportError("top_15_strategies.evaluators module not found")

        rates_by_tf = build_rates_by_timeframe(ticks)
        df_exec = rates_by_tf.get(self.exec_tf, rates_by_tf.get('M15', rates_by_tf.get('H1')))
        if df_exec is None or len(df_exec) <= self.min_lookback:
            return signals, sl_pips_arr, tp_pips_arr

        df_h1 = rates_by_tf.get('H1', df_exec)
        df_h2 = rates_by_tf.get('H2', df_h1)
        df_h4 = rates_by_tf.get('H4', df_h1)
        df_d1 = rates_by_tf.get('D1', df_h4)

        # Precompute timestamp arrays for fast binary search
        exec_times = df_exec.index.to_numpy()
        h1_times = df_h1.index.to_numpy()
        h2_times = df_h2.index.to_numpy()
        h4_times = df_h4.index.to_numpy()
        d1_times = df_d1.index.to_numpy()

        m5_df = rates_by_tf.get('M5', df_exec)
        m15_df = rates_by_tf.get('M15', df_exec)
        m30_df = rates_by_tf.get('M30', df_exec)
        m5_times = m5_df.index.to_numpy()
        m15_times = m15_df.index.to_numpy()
        m30_times = m30_df.index.to_numpy()

        TF_DURATIONS_MS = {
            'M5': 5 * 60 * 1000,
            'M15': 15 * 60 * 1000,
            'M30': 30 * 60 * 1000,
            'H1': 60 * 60 * 1000,
            'H2': 120 * 60 * 1000,
            'H4': 240 * 60 * 1000,
            'D1': 1440 * 60 * 1000,
        }
        bar_dur_ms = TF_DURATIONS_MS.get(self.exec_tf, 15 * 60 * 1000)

        # Datetime index in milliseconds
        if str(df_exec.index.dtype) == 'datetime64[ns]':
            bar_times_ms = df_exec.index.view('int64') // 10**6
        else:
            bar_times_ms = df_exec.index.view('int64')
        tick_ts = ticks['timestamp']

        meta = {
            "name": self.name,
            "short_name": self.code.lower(),
            "timeframe": self.exec_tf
        }

        n_bars = len(df_exec)
        for i in range(self.min_lookback, n_bars):
            t_cur = exec_times[i]
            bar_hour = int(df_exec['hour'].iloc[i])
            if self.enforce_hours and self.allowed_hours and bar_hour not in self.allowed_hours:
                continue

            sub_exec = df_exec.iloc[:i + 1]

            # Fast searchsorted bounds for higher timeframes
            idx_h1 = int(np.searchsorted(h1_times, t_cur, side='right'))
            sub_h1 = df_h1.iloc[:idx_h1] if idx_h1 > 0 else sub_exec

            idx_h2 = int(np.searchsorted(h2_times, t_cur, side='right'))
            sub_h2 = df_h2.iloc[:idx_h2] if idx_h2 > 0 else sub_h1

            idx_h4 = int(np.searchsorted(h4_times, t_cur, side='right'))
            sub_h4 = df_h4.iloc[:idx_h4] if idx_h4 > 0 else sub_h1

            idx_d1 = int(np.searchsorted(d1_times, t_cur, side='right'))
            sub_d1 = df_d1.iloc[:idx_d1] if idx_d1 > 0 else sub_h4

            idx_m5 = int(np.searchsorted(m5_times, t_cur, side='right'))
            idx_m15 = int(np.searchsorted(m15_times, t_cur, side='right'))
            idx_m30 = int(np.searchsorted(m30_times, t_cur, side='right'))

            sub_rates = {
                "M5": m5_df.iloc[:idx_m5] if idx_m5 > 0 else sub_exec,
                "M15": m15_df.iloc[:idx_m15] if idx_m15 > 0 else sub_exec,
                "M30": m30_df.iloc[:idx_m30] if idx_m30 > 0 else sub_exec,
                "H1": sub_h1,
                "H2": sub_h2,
                "H4": sub_h4,
                "D1": sub_d1
            }

            close_p = float(sub_exec['close'].iloc[-1])

            try:
                dir_eval, steps, metrics, sig, h2, h1 = evaluate_strategy_detail(
                    self.strategy_id,
                    meta,
                    sub_h1,
                    sub_h2,
                    sub_h4,
                    sub_d1,
                    live_price=close_p,
                    spread_pips=self.spread_pips,
                    rates_by_tf=sub_rates
                )
            except Exception:
                continue

            if sig and dir_eval in ["BUY", "SELL"]:
                # Map candle close timestamp to tick stream (candle open + duration)
                bar_close_t = bar_times_ms[i] + bar_dur_ms
                tick_idx = int(np.searchsorted(tick_ts, bar_close_t))
                if tick_idx < n_ticks:
                    signals[tick_idx] = 1 if dir_eval == "BUY" else -1
                    risk_p = float(sig.get("risk_pips", 15.0))
                    target_rr = float(sig.get("target_rr", 2.0))
                    sl_pips_arr[tick_idx] = max(3.0, risk_p)
                    tp_pips_arr[tick_idx] = max(6.0, risk_p * target_rr)

        return signals, sl_pips_arr, tp_pips_arr
