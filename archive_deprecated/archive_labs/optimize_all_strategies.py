#!/usr/bin/env python3
"""
Master Production Fleet Bayesian Optimizer (SOP 10-Block Cross-Validation).
Calibrates all strategies adhering strictly to User Institutional Constraints:
- Profit Factor (PF): Target 1.50 to 2.00 (penalizes PF > 2.00 to avoid curve-fitting)
- Realized Risk-to-Reward (RR): Target 1.50 to 2.50 : 1
- Substantial Trade Volume: Maximum active market opportunity without over-filtering
- In-Sample Training: 6 Random Blocks (36 Months / 157 Weeks / 67.8M Ticks)
- Out-of-Sample Testing: 4 Unseen Blocks (24 Months / 105 Weeks / 55.4M Ticks)
- Strict Parameter Freeze between In-Sample and Out-of-Sample
"""

from __future__ import annotations
import sys
import os
import time
import argparse
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import pandas as pd
import optuna

# Prevent bytecode generation
sys.dont_write_bytecode = True

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tick_engine.core.block_sampler import (
    BLOCK_UNITS,
    sample_train_test_blocks,
    get_weeks_for_blocks,
    format_block_summary
)
from tick_engine.core.dataloader import get_week_dataset
from tick_engine.core.engine import run_matching_loop, TRADE_DTYPE
from tick_engine.core.metrics import compute_metrics, print_scorecard

RAW_DIR = ROOT_DIR / "eurusd_weekly_data"
PROC_DIR = ROOT_DIR / "tick_engine" / "data" / "processed"
OPT_DIR = ROOT_DIR / "results" / "optimization"
OPT_DIR.mkdir(parents=True, exist_ok=True)


class PreloadedDataset:
    """Pre-loads and resamples candles into memory once for high-speed trials."""
    def __init__(self, week_numbers: List[int], desc: str = "Dataset"):
        self.weeks_data: List[Dict[str, Any]] = []
        t0 = time.time()
        print(f"[*] Pre-loading and caching {len(week_numbers)} weeks for {desc}...")
        for idx, w in enumerate(week_numbers):
            ticks, meta = get_week_dataset(w, raw_dir=RAW_DIR, processed_dir=PROC_DIR, auto_convert=False)
            n_ticks = len(ticks)
            ts = pd.to_datetime(ticks['timestamp'], unit='ms')
            mid = 0.5 * (ticks['bid'] + ticks['ask'])
            base_df = pd.DataFrame({'open': mid, 'high': mid, 'low': mid, 'close': mid}, index=ts)
            
            h1 = base_df.resample('1h').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
            m30 = base_df.resample('30min').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
            m15 = base_df.resample('15min').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
            
            self.weeks_data.append({
                'week': w, 'ticks': ticks, 'tick_ts': ticks['timestamp'], 'n_ticks': n_ticks,
                'h1': h1, 'm30': m30, 'm15': m15, 'meta': meta
            })
            if (idx + 1) % 40 == 0 or (idx + 1) == len(week_numbers):
                print(f"    Progress: {idx + 1}/{len(week_numbers)} weeks cached ({time.time() - t0:.1f}s)")
        print(f"[+] Loaded {len(self.weeks_data)} weeks in {time.time() - t0:.2f}s!")


def score_trial(metrics: Dict[str, Any], min_trades: int = 35) -> float:
    """
    Scoring function specifically designed for user constraints:
    - Target PF: 1.50 to 2.00 (sharp penalty if PF > 2.00 to avoid curve-fitting)
    - Target RR: 1.50 to 2.50 : 1
    - Substantial Trade Volume: strongly rewards high trade counts
    """
    trades = metrics['total_trades']
    pf = metrics['profit_factor']
    rr = metrics['win_loss_ratio']
    net = metrics['net_pips']
    max_dd = metrics['max_drawdown_pips']
    
    if trades < min_trades or net <= 0 or pf <= 1.05:
        return -100.0

    # Base reward: net pips scaled by trade frequency and drawdown control
    base_score = net * (trades / float(min_trades))**0.45 * (1.0 - min(0.5, max_dd / 600.0))
    
    # Target PF Golden Corridor: [1.50, 2.00]
    if 1.50 <= pf <= 2.00:
        pf_mult = 1.40
    elif pf > 2.00:
        # Sharp penalty for over-filtering / curve-fitting above 2.00
        pf_mult = max(0.15, 1.0 - (pf - 2.00) * 1.5)
    else:  # pf < 1.50
        pf_mult = max(0.20, 1.0 - (1.50 - pf) * 2.0)

    # Target RR Golden Corridor: [1.50, 2.50]
    if 1.50 <= rr <= 2.50:
        rr_mult = 1.30
    elif rr > 2.50:
        # Penalize holding too long
        rr_mult = max(0.20, 1.0 - (rr - 2.50) * 1.5)
    else:  # rr < 1.50
        rr_mult = max(0.20, 1.0 - (1.50 - rr) * 2.0)

    return float(base_score * pf_mult * rr_mult)


# ==========================================
# 1. S66 BHANUSHALI 44 MA PULLBACK H1
# ==========================================
def simulate_s66(dataset: PreloadedDataset, ma_pb: int, ma_macro: int, target_rr: float, touch_pips: float, buffer_pips: float, min_risk: float, max_risk: float, hours: List[int]):
    all_trades = []
    dur_ms = 60 * 60 * 1000
    pip_size = 0.0001
    for item in dataset.weeks_data:
        h1 = item['h1'].copy()
        if len(h1) < 55: continue
        ticks = item['ticks']; n_ticks = item['n_ticks']; tick_ts = item['tick_ts']
        c = h1['close']
        h1['ma_pb'] = c.rolling(ma_pb, min_periods=1).mean().bfill()
        h1['ma_macro'] = c.rolling(ma_macro, min_periods=1).mean().bfill()
        h1_hours = h1.index.hour; h1_days = h1.index.dayofweek
        time_ms = h1.index.astype('datetime64[ms]').view('int64')
        signals = np.zeros(n_ticks, dtype=np.int8)
        sl_arr = np.zeros(n_ticks, dtype=np.float64); tp_arr = np.zeros(n_ticks, dtype=np.float64)
        has_signal = False
        
        for i in range(45, len(h1)):
            if h1_days[i] == 4 and h1_hours[i] >= 16: continue
            if h1_hours[i] not in hours: continue
            row = h1.iloc[i]; prev = h1.iloc[i-1]
            close_p = row['close']; open_p = row['open']; mapb = row['ma_pb']; mamacro = row['ma_macro']
            
            if close_p > mamacro and mapb > mamacro:
                if (h1['low'].iloc[i-3:i] <= h1['ma_pb'].iloc[i-3:i] + touch_pips * pip_size).any():
                    if close_p > open_p and close_p > mapb and close_p > prev['high']:
                        pb_low = float(h1['low'].iloc[i-3:i+1].min())
                        risk = close_p - (pb_low - buffer_pips * pip_size)
                        if min_risk * pip_size <= risk <= max_risk * pip_size:
                            idx = int(np.searchsorted(tick_ts, time_ms[i] + dur_ms))
                            if idx < n_ticks and signals[idx] == 0:
                                signals[idx] = 1; sl_arr[idx] = risk / pip_size; tp_arr[idx] = (risk / pip_size) * target_rr; has_signal = True
            elif close_p < mamacro and mapb < mamacro:
                if (h1['high'].iloc[i-3:i] >= h1['ma_pb'].iloc[i-3:i] - touch_pips * pip_size).any():
                    if close_p < open_p and close_p < mapb and close_p < prev['low']:
                        pb_high = float(h1['high'].iloc[i-3:i+1].max())
                        risk = (pb_high + buffer_pips * pip_size) - close_p
                        if min_risk * pip_size <= risk <= max_risk * pip_size:
                            idx = int(np.searchsorted(tick_ts, time_ms[i] + dur_ms))
                            if idx < n_ticks and signals[idx] == 0:
                                signals[idx] = -1; sl_arr[idx] = risk / pip_size; tp_arr[idx] = (risk / pip_size) * target_rr; has_signal = True
                                
        if has_signal:
            tr = run_matching_loop(ticks, signals, pip_size=pip_size, sl_pips=sl_arr, tp_pips=tp_arr)
            if len(tr) > 0: all_trades.append(tr)
    combined = np.concatenate(all_trades) if all_trades else np.zeros(0, dtype=TRADE_DTYPE)
    return compute_metrics(combined), combined


# ==========================================
# 2. S10 DUAL EMA 20/200 PULLBACK H1
# ==========================================
def simulate_s10(dataset: PreloadedDataset, ema_fast: int, ema_trend: int, target_rr: float, touch_pips: float, buffer_pips: float, min_risk: float, max_risk: float, hours: List[int]):
    all_trades = []
    dur_ms = 60 * 60 * 1000
    pip_size = 0.0001
    for item in dataset.weeks_data:
        h1 = item['h1'].copy()
        if len(h1) < 45: continue
        ticks = item['ticks']; n_ticks = item['n_ticks']; tick_ts = item['tick_ts']
        c = h1['close']
        h1['ema_fast'] = c.ewm(span=ema_fast, adjust=False).mean()
        h1['ema_trend'] = c.ewm(span=ema_trend, adjust=False).mean()
        h1_hours = h1.index.hour; h1_days = h1.index.dayofweek
        time_ms = h1.index.astype('datetime64[ms]').view('int64')
        signals = np.zeros(n_ticks, dtype=np.int8)
        sl_arr = np.zeros(n_ticks, dtype=np.float64); tp_arr = np.zeros(n_ticks, dtype=np.float64)
        has_signal = False
        
        for i in range(25, len(h1)):
            if h1_days[i] == 4 and h1_hours[i] >= 16: continue
            if h1_hours[i] not in hours: continue
            row = h1.iloc[i]; prev = h1.iloc[i-1]
            close_p = row['close']; open_p = row['open']; emaf = row['ema_fast']; emat = row['ema_trend']
            
            if close_p > emat and emaf > emat:
                if (h1['low'].iloc[i-3:i] <= h1['ema_fast'].iloc[i-3:i] + touch_pips * pip_size).any():
                    if close_p > open_p and close_p > emaf and close_p > prev['high']:
                        pb_low = float(h1['low'].iloc[i-3:i+1].min())
                        risk = close_p - (pb_low - buffer_pips * pip_size)
                        if min_risk * pip_size <= risk <= max_risk * pip_size:
                            idx = int(np.searchsorted(tick_ts, time_ms[i] + dur_ms))
                            if idx < n_ticks and signals[idx] == 0:
                                signals[idx] = 1; sl_arr[idx] = risk / pip_size; tp_arr[idx] = (risk / pip_size) * target_rr; has_signal = True
            elif close_p < emat and emaf < emat:
                if (h1['high'].iloc[i-3:i] >= h1['ema_fast'].iloc[i-3:i] - touch_pips * pip_size).any():
                    if close_p < open_p and close_p < emaf and close_p < prev['low']:
                        pb_high = float(h1['high'].iloc[i-3:i+1].max())
                        risk = (pb_high + buffer_pips * pip_size) - close_p
                        if min_risk * pip_size <= risk <= max_risk * pip_size:
                            idx = int(np.searchsorted(tick_ts, time_ms[i] + dur_ms))
                            if idx < n_ticks and signals[idx] == 0:
                                signals[idx] = -1; sl_arr[idx] = risk / pip_size; tp_arr[idx] = (risk / pip_size) * target_rr; has_signal = True
                                
        if has_signal:
            tr = run_matching_loop(ticks, signals, pip_size=pip_size, sl_pips=sl_arr, tp_pips=tp_arr)
            if len(tr) > 0: all_trades.append(tr)
    combined = np.concatenate(all_trades) if all_trades else np.zeros(0, dtype=TRADE_DTYPE)
    return compute_metrics(combined), combined


# ==========================================
# 3. S14 INSIDE BAR M30 BREAKOUT
# ==========================================
def simulate_s14(dataset: PreloadedDataset, ema_fast: int, ema_trend: int, target_rr: float, sl_buffer: float, min_risk: float, max_risk: float, hours: List[int]):
    all_trades = []
    dur_ms = 30 * 60 * 1000
    pip_size = 0.0001
    for item in dataset.weeks_data:
        m30 = item['m30'].copy()
        if len(m30) < 40: continue
        ticks = item['ticks']; n_ticks = item['n_ticks']; tick_ts = item['tick_ts']
        c = m30['close']
        m30['ema_fast'] = c.ewm(span=ema_fast, adjust=False).mean()
        m30['ema_trend'] = c.ewm(span=ema_trend, adjust=False).mean()
        m30_hours = m30.index.hour; m30_days = m30.index.dayofweek
        time_ms = m30.index.astype('datetime64[ms]').view('int64')
        signals = np.zeros(n_ticks, dtype=np.int8)
        sl_arr = np.zeros(n_ticks, dtype=np.float64); tp_arr = np.zeros(n_ticks, dtype=np.float64)
        has_signal = False
        
        for i in range(25, len(m30)):
            if m30_days[i] == 4 and m30_hours[i] >= 15: continue
            if m30_hours[i] not in hours: continue
            ib = m30.iloc[i-1]; mother = m30.iloc[i-2]
            if not ((ib['high'] <= mother['high']) and (ib['low'] >= mother['low'])): continue
            row = m30.iloc[i]; close_p = row['close']; emaf = row['ema_fast']; emat = row['ema_trend']
            
            if (close_p > mother['high']) and (close_p > emaf) and (close_p > emat):
                risk = close_p - (ib['low'] - sl_buffer * pip_size)
                if min_risk * pip_size <= risk <= max_risk * pip_size:
                    idx = int(np.searchsorted(tick_ts, time_ms[i] + dur_ms))
                    if idx < n_ticks and signals[idx] == 0:
                        signals[idx] = 1; sl_arr[idx] = risk / pip_size; tp_arr[idx] = (risk / pip_size) * target_rr; has_signal = True
            elif (close_p < mother['low']) and (close_p < emaf) and (close_p < emat):
                risk = (ib['high'] + sl_buffer * pip_size) - close_p
                if min_risk * pip_size <= risk <= max_risk * pip_size:
                    idx = int(np.searchsorted(tick_ts, time_ms[i] + dur_ms))
                    if idx < n_ticks and signals[idx] == 0:
                        signals[idx] = -1; sl_arr[idx] = risk / pip_size; tp_arr[idx] = (risk / pip_size) * target_rr; has_signal = True
                        
        if has_signal:
            tr = run_matching_loop(ticks, signals, pip_size=pip_size, sl_pips=sl_arr, tp_pips=tp_arr)
            if len(tr) > 0: all_trades.append(tr)
    combined = np.concatenate(all_trades) if all_trades else np.zeros(0, dtype=TRADE_DTYPE)
    return compute_metrics(combined), combined


# ==========================================
# 4. S34 HEIKIN ASHI MOMENTUM H1
# ==========================================
def simulate_s34(dataset: PreloadedDataset, ema_fast: int, ema_trend: int, target_rr: float, sl_pips: float, flat_thresh: float, hours: List[int]):
    all_trades = []
    dur_ms = 60 * 60 * 1000
    pip_size = 0.0001
    for item in dataset.weeks_data:
        h1 = item['h1'].copy()
        if len(h1) < 40: continue
        ticks = item['ticks']; n_ticks = item['n_ticks']; tick_ts = item['tick_ts']
        c = h1['close']; o = h1['open']; h = h1['high']; l = h1['low']
        h1['ema_fast'] = c.ewm(span=ema_fast, adjust=False).mean()
        h1['ema_trend'] = c.ewm(span=ema_trend, adjust=False).mean()
        
        # Smoothed Heikin Ashi
        ha_close = (o + h + l + c) / 4.0
        n = len(h1)
        ha_open = np.zeros(n)
        ha_open[0] = (o.iloc[0] + c.iloc[0]) / 2.0
        for k in range(1, n):
            ha_open[k] = (ha_open[k-1] + ha_close.iloc[k-1]) / 2.0
        h1['ha_open'] = ha_open
        h1['ha_close'] = ha_close
        
        h1_hours = h1.index.hour; h1_days = h1.index.dayofweek
        time_ms = h1.index.astype('datetime64[ms]').view('int64')
        signals = np.zeros(n_ticks, dtype=np.int8)
        sl_arr = np.zeros(n_ticks, dtype=np.float64); tp_arr = np.zeros(n_ticks, dtype=np.float64)
        has_signal = False
        
        for i in range(25, len(h1)):
            if h1_days[i] == 4 and h1_hours[i] >= 16: continue
            if h1_hours[i] not in hours: continue
            row = h1.iloc[i]
            emaf = row['ema_fast']; emat = row['ema_trend']
            ha_c = row['ha_close']; ha_o = row['ha_open']
            
            # Bullish HA
            flat_bottom = abs(ha_o - row['low']) <= flat_thresh * pip_size
            if ha_c > ha_o and flat_bottom and row['close'] > row['open'] and row['close'] > emaf and row['close'] > emat:
                idx = int(np.searchsorted(tick_ts, time_ms[i] + dur_ms))
                if idx < n_ticks and signals[idx] == 0:
                    signals[idx] = 1; sl_arr[idx] = sl_pips; tp_arr[idx] = sl_pips * target_rr; has_signal = True
            # Bearish HA
            flat_top = abs(ha_o - row['high']) <= flat_thresh * pip_size
            if ha_c < ha_o and flat_top and row['close'] < row['open'] and row['close'] < emaf:
                idx = int(np.searchsorted(tick_ts, time_ms[i] + dur_ms))
                if idx < n_ticks and signals[idx] == 0:
                    signals[idx] = -1; sl_arr[idx] = sl_pips; tp_arr[idx] = sl_pips * target_rr; has_signal = True
                    
        if has_signal:
            tr = run_matching_loop(ticks, signals, pip_size=pip_size, sl_pips=sl_arr, tp_pips=tp_arr)
            if len(tr) > 0: all_trades.append(tr)
    combined = np.concatenate(all_trades) if all_trades else np.zeros(0, dtype=TRADE_DTYPE)
    return compute_metrics(combined), combined


# ==========================================
# 5. S01 M15 DISPLACEMENT & RETEST SCALPER
# ==========================================
def simulate_s01(dataset: PreloadedDataset, min_disp: float, target_rr: float, min_atr: float, sl_buffer: float, min_risk: float, max_risk: float, hours: List[int]):
    all_trades = []
    dur_ms = 15 * 60 * 1000
    pip_size = 0.0001
    for item in dataset.weeks_data:
        m15 = item['m15'].copy()
        if len(m15) < 60: continue
        ticks = item['ticks']; n_ticks = item['n_ticks']; tick_ts = item['tick_ts']
        c = m15['close']; h = m15['high']; l = m15['low']
        tr = np.maximum(h - l, np.maximum(abs(h - c.shift(1)), abs(l - c.shift(1))))
        m15['atr'] = tr.rolling(14).mean().bfill()
        m15['ema_50'] = c.ewm(span=50, adjust=False).mean()
        m15['ema_200'] = c.ewm(span=200, adjust=False).mean()
        
        m15_hours = m15.index.hour; m15_days = m15.index.dayofweek
        time_ms = m15.index.astype('datetime64[ms]').view('int64')
        signals = np.zeros(n_ticks, dtype=np.int8)
        sl_arr = np.zeros(n_ticks, dtype=np.float64); tp_arr = np.zeros(n_ticks, dtype=np.float64)
        has_signal = False
        
        for i in range(25, len(m15)):
            if m15_days[i] == 4 and m15_hours[i] >= 15: continue
            if m15_hours[i] not in hours: continue
            row = m15.iloc[i]
            c1 = m15.iloc[i-1]
            c2 = m15.iloc[i-2]
            if (row['atr'] / pip_size) < min_atr: continue
            
            h1_bull = row['ema_50'] > row['ema_200']
            h1_bear = row['ema_50'] < row['ema_200']
            
            # Bullish Displacement & Retest
            disp_pips = (c1['close'] - c1['open']) / pip_size
            if h1_bull and disp_pips >= min_disp and c1['close'] > c2['high']:
                retest_level = c2['high']
                if row['low'] <= retest_level and row['close'] > retest_level and row['close'] > row['open']:
                    sl_price = c1['low'] - sl_buffer * pip_size
                    risk = (row['close'] - sl_price) / pip_size
                    if min_risk <= risk <= max_risk:
                        idx = int(np.searchsorted(tick_ts, time_ms[i] + dur_ms))
                        if idx < n_ticks and signals[idx] == 0:
                            signals[idx] = 1; sl_arr[idx] = risk; tp_arr[idx] = risk * target_rr; has_signal = True
                            
            # Bearish Displacement & Retest
            disp_pips = (c1['open'] - c1['close']) / pip_size
            if h1_bear and disp_pips >= min_disp and c1['close'] < c2['low']:
                retest_level = c2['low']
                if row['high'] >= retest_level and row['close'] < retest_level and row['close'] < row['open']:
                    sl_price = c1['high'] + sl_buffer * pip_size
                    risk = (sl_price - row['close']) / pip_size
                    if min_risk <= risk <= max_risk:
                        idx = int(np.searchsorted(tick_ts, time_ms[i] + dur_ms))
                        if idx < n_ticks and signals[idx] == 0:
                            signals[idx] = -1; sl_arr[idx] = risk; tp_arr[idx] = risk * target_rr; has_signal = True
                            
        if has_signal:
            tr = run_matching_loop(ticks, signals, pip_size=pip_size, sl_pips=sl_arr, tp_pips=tp_arr)
            if len(tr) > 0: all_trades.append(tr)
    combined = np.concatenate(all_trades) if all_trades else np.zeros(0, dtype=TRADE_DTYPE)
    return compute_metrics(combined), combined


# ==========================================
# 6. HIGH-FREQUENCY M15 SCALPER ELITE
# ==========================================
def simulate_hf_m15(dataset: PreloadedDataset, target_rr: float, min_sl: float, max_sl: float, sweep_lookback: int, h1_ema_period: int, use_h1_filter: bool, hours: List[int]):
    all_trades = []
    dur_ms = 15 * 60 * 1000
    pip_size = 0.0001
    for item in dataset.weeks_data:
        m15 = item['m15'].copy()
        h1 = item['h1'].copy()
        if len(m15) < 40 or len(h1) < 20: continue
        ticks = item['ticks']; n_ticks = item['n_ticks']; tick_ts = item['tick_ts']
        
        h1_ema = h1['close'].ewm(span=h1_ema_period, adjust=False).mean()
        h1_trend = np.where(h1['close'] > h1_ema, 1, -1)
        h1_series = pd.Series(h1_trend, index=h1.index)
        m15['macro_trend'] = h1_series.reindex(m15.index, method='ffill').fillna(0)
        
        tr = np.maximum(m15['high'] - m15['low'], np.maximum(abs(m15['high'] - m15['close'].shift(1)), abs(m15['low'] - m15['close'].shift(1))))
        m15['atr'] = tr.rolling(14).mean().bfill()
        m15['ema_20'] = m15['close'].ewm(span=20, adjust=False).mean()
        
        m15_hours = m15.index.hour; m15_days = m15.index.dayofweek
        time_ms = m15.index.astype('datetime64[ms]').view('int64')
        signals = np.zeros(n_ticks, dtype=np.int8)
        sl_arr = np.zeros(n_ticks, dtype=np.float64); tp_arr = np.zeros(n_ticks, dtype=np.float64)
        has_signal = False
        
        for i in range(max(25, sweep_lookback), len(m15)):
            if m15_days[i] == 4 and m15_hours[i] >= 16: continue
            if m15_hours[i] not in hours: continue
            row = m15.iloc[i]; prev = m15.iloc[i-1]; prior = m15.iloc[i-2]
            sig = 0; risk_p = 0.0
            macro = row['macro_trend'] if use_h1_filter else 0
            atr_pips = row['atr'] / pip_size
            dyn_min_sl = max(min_sl, 0.8 * atr_pips)
            dyn_max_sl = max(max_sl, 1.8 * atr_pips)
            
            window = m15.iloc[i - sweep_lookback : i]
            roll_h = window['high'].max()
            roll_l = window['low'].min()
            
            sweep_l = (roll_l - row['low']) / pip_size
            if 0.8 <= sweep_l <= 6.0 and row['close'] > roll_l and row['close'] > row['open']:
                if not use_h1_filter or macro >= 0:
                    sl_p = row['low'] - 1.2 * pip_size
                    r = max((row['close'] - sl_p) / pip_size, dyn_min_sl)
                    if r <= dyn_max_sl:
                        sig = 1; risk_p = r
            sweep_h = (row['high'] - roll_h) / pip_size
            if 0.8 <= sweep_h <= 6.0 and row['close'] < roll_h and row['close'] < row['open']:
                if not use_h1_filter or macro <= 0:
                    sl_p = row['high'] + 1.2 * pip_size
                    r = max((sl_p - row['close']) / pip_size, dyn_min_sl)
                    if r <= dyn_max_sl:
                        sig = -1; risk_p = r
            if sig == 0:
                if prev['close'] > prev['open'] and (prev['close'] - prev['open']) / pip_size >= 3.5:
                    if row['open'] > prior['high'] and row['low'] <= prior['high'] and row['close'] > prior['high'] and row['close'] > row['ema_20']:
                        if not use_h1_filter or macro >= 0:
                            sl_p = prev['low'] - 1.2 * pip_size
                            r = max((row['close'] - sl_p) / pip_size, dyn_min_sl)
                            if r <= dyn_max_sl:
                                sig = 1; risk_p = r
                elif prev['close'] < prev['open'] and (prev['open'] - prev['close']) / pip_size >= 3.5:
                    if row['open'] < prior['low'] and row['high'] >= prior['low'] and row['close'] < prior['low'] and row['close'] < row['ema_20']:
                        if not use_h1_filter or macro <= 0:
                            sl_p = prev['high'] + 1.2 * pip_size
                            r = max((sl_p - row['close']) / pip_size, dyn_min_sl)
                            if r <= dyn_max_sl:
                                sig = -1; risk_p = r
            if sig != 0:
                idx = int(np.searchsorted(tick_ts, time_ms[i] + dur_ms))
                if idx < n_ticks and signals[idx] == 0:
                    signals[idx] = sig; sl_arr[idx] = risk_p; tp_arr[idx] = risk_p * target_rr; has_signal = True
        if has_signal:
            tr = run_matching_loop(ticks, signals, pip_size=pip_size, sl_pips=sl_arr, tp_pips=tp_arr)
            if len(tr) > 0: all_trades.append(tr)
    combined = np.concatenate(all_trades) if all_trades else np.zeros(0, dtype=TRADE_DTYPE)
    return compute_metrics(combined), combined


def optimize_model(strat_id: str, train_ds: PreloadedDataset, test_ds: PreloadedDataset, n_trials: int = 25):
    print("\n" + "=" * 90)
    print(f"OPTIMIZING {strat_id.upper()}: CONSTRAINING PF TO [1.50, 2.00] & RR TO [1.50, 2.50]".center(90))
    print("=" * 90)
    
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    trial_records = []
    
    def objective(trial: optuna.Trial) -> float:
        # Sample target RR strictly inside the 1.6 - 2.5 sweet spot
        target_rr = trial.suggest_float("target_rr", 1.8, 2.5, step=0.1)
        
        if strat_id == "s66":
            ma_pb = trial.suggest_int("ma_pb", 38, 52, step=2)
            ma_macro = trial.suggest_int("ma_macro", 160, 220, step=10)
            touch_pips = trial.suggest_float("touch_pips", 1.5, 3.5, step=0.5)
            buffer_pips = trial.suggest_float("buffer_pips", 1.5, 3.0, step=0.5)
            min_risk = trial.suggest_float("min_risk", 6.0, 10.0, step=1.0)
            max_risk = trial.suggest_float("max_risk", 18.0, 28.0, step=2.0)
            session = trial.suggest_categorical("session", ["full_day", "london_ny", "london_core"])
            if session == "full_day":
                hours = [6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
            elif session == "london_ny":
                hours = [6,7,8,9,10,11,12,13,14,15,16,17,18]
            else:
                hours = [7,8,9,10,11,12,13,14,15,16]
            m, _ = simulate_s66(train_ds, ma_pb, ma_macro, target_rr, touch_pips, buffer_pips, min_risk, max_risk, hours)
            min_tr = 35
            
        elif strat_id == "s10":
            ema_fast = trial.suggest_int("ema_fast", 16, 26, step=2)
            ema_trend = trial.suggest_int("ema_trend", 170, 240, step=10)
            touch_pips = trial.suggest_float("touch_pips", 1.5, 3.5, step=0.5)
            buffer_pips = trial.suggest_float("buffer_pips", 1.5, 2.5, step=0.5)
            min_risk = trial.suggest_float("min_risk", 6.0, 10.0, step=1.0)
            max_risk = trial.suggest_float("max_risk", 18.0, 26.0, step=2.0)
            session = trial.suggest_categorical("session", ["london_core", "london_ny", "full_day"])
            if session == "full_day":
                hours = [6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
            elif session == "london_ny":
                hours = [6,7,8,9,10,11,12,13,14,15,16,17,18]
            else:
                hours = [7,8,9,10,11,12,13,14,15,16]
            m, _ = simulate_s10(train_ds, ema_fast, ema_trend, target_rr, touch_pips, buffer_pips, min_risk, max_risk, hours)
            min_tr = 45
            
        elif strat_id == "s14":
            ema_fast = trial.suggest_int("ema_fast", 30, 60, step=10)
            ema_trend = trial.suggest_int("ema_trend", 180, 250, step=10)
            sl_buffer = trial.suggest_float("sl_buffer", 1.0, 2.5, step=0.5)
            min_risk = trial.suggest_float("min_risk", 5.0, 8.0, step=1.0)
            max_risk = trial.suggest_float("max_risk", 16.0, 22.0, step=2.0)
            session = trial.suggest_categorical("session", ["full_day", "london_ny"])
            hours = [6,7,8,9,10,11,12,13,14,15,16,17] if session == "full_day" else [7,8,9,13,14,15,16]
            m, _ = simulate_s14(train_ds, ema_fast, ema_trend, target_rr, sl_buffer, min_risk, max_risk, hours)
            min_tr = 45
            
        elif strat_id == "s34":
            ema_fast = trial.suggest_int("ema_fast", 40, 60, step=10)
            ema_trend = trial.suggest_int("ema_trend", 160, 220, step=20)
            sl_pips = trial.suggest_float("sl_pips", 12.0, 18.0, step=2.0)
            flat_thresh = trial.suggest_float("flat_thresh", 1.0, 2.5, step=0.5)
            session = trial.suggest_categorical("session", ["expanded_active", "core_hours"])
            hours = [7,8,9,10,11,12,13,14,15,16] if session == "expanded_active" else [8,9,13,14,15,16]
            m, _ = simulate_s34(train_ds, ema_fast, ema_trend, target_rr, sl_pips, flat_thresh, hours)
            min_tr = 40

        elif strat_id == "s01":
            min_disp = trial.suggest_float("min_disp", 3.0, 5.0, step=0.5)
            min_atr = trial.suggest_float("min_atr", 3.0, 5.0, step=0.5)
            sl_buffer = trial.suggest_float("sl_buffer", 1.0, 2.0, step=0.5)
            min_risk = trial.suggest_float("min_risk", 5.0, 7.0, step=1.0)
            max_risk = trial.suggest_float("max_risk", 12.0, 16.0, step=2.0)
            session = trial.suggest_categorical("session", ["london_ny", "active_window"])
            hours = [7,8,9,10,11,12,13,14,15,16] if session == "london_ny" else [6,7,8,9,10,11,12,13,14,15,16,17]
            m, _ = simulate_s01(train_ds, min_disp, target_rr, min_atr, sl_buffer, min_risk, max_risk, hours)
            min_tr = 50

        elif strat_id == "hf_m15":
            min_sl = trial.suggest_float("min_sl", 6.0, 10.0, step=1.0)
            max_sl = trial.suggest_float("max_sl", 14.0, 20.0, step=2.0)
            sweep_lookback = trial.suggest_int("sweep_lookback", 16, 24, step=4)
            h1_ema_period = trial.suggest_int("h1_ema_period", 40, 70, step=10)
            use_h1_filter = True # Mandatory macro alignment
            session = trial.suggest_categorical("session", ["expanded_eu_us", "london_ny"])
            hours = [6,7,8,9,10,11,12,13,14,15,16,17,18] if session == "expanded_eu_us" else [7,8,9,11,12,13,14,15,17]
            m, _ = simulate_hf_m15(train_ds, target_rr, min_sl, max_sl, sweep_lookback, h1_ema_period, use_h1_filter, hours)
            min_tr = 250
            
        val = score_trial(m, min_trades=min_tr)
        trial_records.append({
            "trial": trial.number, "params": trial.params,
            "train_trades": m.get('total_trades', 0),
            "train_pf": round(m.get('profit_factor', 0), 2),
            "train_rr": round(m.get('win_loss_ratio', 0), 2),
            "train_net": round(m.get('net_pips', 0), 1),
            "train_max_dd": round(m.get('max_drawdown_pips', 0), 1),
            "score": round(val, 1)
        })
        return val

    t0 = time.time()
    study.optimize(objective, n_trials=n_trials)
    print(f"[+] Optimization finished in {time.time() - t0:.1f}s!")
    
    best_p = study.best_params
    print(f"\n[BEST PARAMETERS FOR {strat_id.upper()}]:")
    for k, v in best_p.items(): print(f"  • {k}: {v}")
    
    # Evaluate In-Sample vs Out-of-Sample with best params
    target_rr = best_p["target_rr"]
    if strat_id == "s66":
        hours = [6,7,8,9,10,11,12,13,14,15,16,17,18,19,20] if best_p["session"] == "full_day" else ([6,7,8,9,10,11,12,13,14,15,16,17,18] if best_p["session"] == "london_ny" else [7,8,9,10,11,12,13,14,15,16])
        tr_m, tr_t = simulate_s66(train_ds, best_p["ma_pb"], best_p["ma_macro"], target_rr, best_p["touch_pips"], best_p["buffer_pips"], best_p["min_risk"], best_p["max_risk"], hours)
        te_m, te_t = simulate_s66(test_ds, best_p["ma_pb"], best_p["ma_macro"], target_rr, best_p["touch_pips"], best_p["buffer_pips"], best_p["min_risk"], best_p["max_risk"], hours)
    elif strat_id == "s10":
        hours = [6,7,8,9,10,11,12,13,14,15,16,17,18,19,20] if best_p["session"] == "full_day" else ([6,7,8,9,10,11,12,13,14,15,16,17,18] if best_p["session"] == "london_ny" else [7,8,9,10,11,12,13,14,15,16])
        tr_m, tr_t = simulate_s10(train_ds, best_p["ema_fast"], best_p["ema_trend"], target_rr, best_p["touch_pips"], best_p["buffer_pips"], best_p["min_risk"], best_p["max_risk"], hours)
        te_m, te_t = simulate_s10(test_ds, best_p["ema_fast"], best_p["ema_trend"], target_rr, best_p["touch_pips"], best_p["buffer_pips"], best_p["min_risk"], best_p["max_risk"], hours)
    elif strat_id == "s14":
        hours = [6,7,8,9,10,11,12,13,14,15,16,17] if best_p["session"] == "full_day" else [7,8,9,13,14,15,16]
        tr_m, tr_t = simulate_s14(train_ds, best_p["ema_fast"], best_p["ema_trend"], target_rr, best_p["sl_buffer"], best_p["min_risk"], best_p["max_risk"], hours)
        te_m, te_t = simulate_s14(test_ds, best_p["ema_fast"], best_p["ema_trend"], target_rr, best_p["sl_buffer"], best_p["min_risk"], best_p["max_risk"], hours)
    elif strat_id == "s34":
        hours = [7,8,9,10,11,12,13,14,15,16] if best_p["session"] == "expanded_active" else [8,9,13,14,15,16]
        tr_m, tr_t = simulate_s34(train_ds, best_p["ema_fast"], best_p["ema_trend"], target_rr, best_p["sl_pips"], best_p["flat_thresh"], hours)
        te_m, te_t = simulate_s34(test_ds, best_p["ema_fast"], best_p["ema_trend"], target_rr, best_p["sl_pips"], best_p["flat_thresh"], hours)
    elif strat_id == "s01":
        hours = [7,8,9,10,11,12,13,14,15,16] if best_p["session"] == "london_ny" else [6,7,8,9,10,11,12,13,14,15,16,17]
        tr_m, tr_t = simulate_s01(train_ds, best_p["min_disp"], target_rr, best_p["min_atr"], best_p["sl_buffer"], best_p["min_risk"], best_p["max_risk"], hours)
        te_m, te_t = simulate_s01(test_ds, best_p["min_disp"], target_rr, best_p["min_atr"], best_p["sl_buffer"], best_p["min_risk"], best_p["max_risk"], hours)
    elif strat_id == "hf_m15":
        hours = [6,7,8,9,10,11,12,13,14,15,16,17,18] if best_p["session"] == "expanded_eu_us" else [7,8,9,11,12,13,14,15,17]
        tr_m, tr_t = simulate_hf_m15(train_ds, target_rr, best_p["min_sl"], best_p["max_sl"], best_p["sweep_lookback"], best_p["h1_ema_period"], True, hours)
        te_m, te_t = simulate_hf_m15(test_ds, target_rr, best_p["min_sl"], best_p["max_sl"], best_p["sweep_lookback"], best_p["h1_ema_period"], True, hours)

    all_5y = np.concatenate([tr_t, te_t]) if len(tr_t) > 0 and len(te_t) > 0 else tr_t
    tot_m = compute_metrics(all_5y)
    
    summary = {
        "strategy": strat_id,
        "best_params": best_p,
        "train": {"trades": tr_m['total_trades'], "pf": round(tr_m['profit_factor'], 2), "rr": round(tr_m['win_loss_ratio'], 2), "net": round(tr_m['net_pips'], 1), "max_dd": round(tr_m['max_drawdown_pips'], 1)},
        "test": {"trades": te_m['total_trades'], "pf": round(te_m['profit_factor'], 2), "rr": round(te_m['win_loss_ratio'], 2), "net": round(te_m['net_pips'], 1), "max_dd": round(te_m['max_drawdown_pips'], 1)},
        "total_5y": {"trades": tot_m['total_trades'], "pf": round(tot_m['profit_factor'], 2), "rr": round(tot_m['win_loss_ratio'], 2), "net": round(tot_m['net_pips'], 1), "max_dd": round(tot_m['max_drawdown_pips'], 1)}
    }
    
    with open(OPT_DIR / f"{strat_id}_constrained_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
        
    print_scorecard(tr_m, title=f"{strat_id.upper()} - 3Y IN-SAMPLE TRAIN")
    print_scorecard(te_m, title=f"{strat_id.upper()} - 2Y UNSEEN TEST")
    print_scorecard(tot_m, title=f"{strat_id.upper()} - CONSOLIDATED 5-YEAR")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Multi-Model Constrained Optimizer")
    parser.add_argument("--strategy", type=str, default="all", choices=["all", "s66", "s10", "s14", "s34", "s01", "hf_m15"])
    parser.add_argument("--trials", type=int, default=25, help="Optuna trials per model")
    parser.add_argument("--seed", type=int, default=42, help="Seed for block sampler")
    args = parser.parse_args()
    
    train_blocks, test_blocks = sample_train_test_blocks(args.seed)
    train_weeks = get_weeks_for_blocks(train_blocks)
    test_weeks = get_weeks_for_blocks(test_blocks)
    
    train_ds = PreloadedDataset(train_weeks, desc="Train 157 Weeks (36 Mos)")
    test_ds = PreloadedDataset(test_weeks, desc="Test 105 Weeks (24 Mos)")
    
    targets = ["s66", "s10", "s14", "s34", "s01", "hf_m15"] if args.strategy == "all" else [args.strategy]
    results = {}
    for sid in targets:
        results[sid] = optimize_model(sid, train_ds, test_ds, n_trials=args.trials)
        
    print("\n" + "=" * 90)
    print("ALL STRATEGIES OPTIMIZED FOR PF [1.5, 2.0] & RR [1.5, 2.5]: MASTER FLEET SCORECARD".center(90))
    print("=" * 90)
    print(f"{'Strategy':<10} | {'Train Trades':<12} {'Train PF':<9} {'Train RR':<9} | {'Test Trades':<11} {'Test PF':<8} {'Test RR':<8} | {'5Y Net Pips':<11} {'5Y PF':<6}")
    print("-" * 90)
    for sid, r in results.items():
        tr = r["train"]; te = r["test"]; tot = r["total_5y"]
        print(f"{sid.upper():<10} | {tr['trades']:<12} {tr['pf']:<9} {tr['rr']:<9} | {te['trades']:<11} {te['pf']:<8} {te['rr']:<8} | {tot['net']:<11} {tot['pf']:<6}")
    print("=" * 90)


if __name__ == "__main__":
    main()
