#!/usr/bin/env python3
"""
Institutional Bayesian Optimization Suite for Elite EUR/USD Strategies.
Adheres strictly to SOP-TRD-VAL-5Y-002:
- 10-Block (6-Month Unit) Cross-Validation Architecture
- In-Sample Calibration: 6 Random Blocks (36 Months / 157 Weeks / 67.8M Ticks)
- Out-of-Sample Validation: 4 Unseen Blocks (24 Months / 105 Weeks / 55.4M Ticks)
- Strict Parameter Freeze between Train and Test
- Target Performance: Profit Factor >= 2.00 & Realized Risk-to-Reward >= 2.00 : 1

Powered by: Optuna 5.0 (TPE Bayesian Sampler) & DuckDB 1.5
"""

from __future__ import annotations
import sys
import os
import time
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import pandas as pd
import optuna
import duckdb

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
from tick_engine.core.engine import BacktestEngine, run_matching_loop, TRADE_DTYPE
from tick_engine.core.metrics import compute_metrics, print_scorecard

RAW_DIR = ROOT_DIR / "eurusd_weekly_data"
PROC_DIR = ROOT_DIR / "tick_engine" / "data" / "processed"
OPT_RESULTS_DIR = ROOT_DIR / "results" / "optimization"
OPT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)


class PreloadedDataset:
    """
    In-memory pre-resampled dataset of the 5-year tick library.
    Caches M15, M30, and H1 candles alongside raw ticks for zero-overhead trials.
    """
    def __init__(self, week_numbers: List[int], desc: str = "Dataset"):
        self.weeks_data: List[Dict[str, Any]] = []
        t0 = time.time()
        print(f"[*] Pre-loading and resampling {len(week_numbers)} weeks for {desc}...")
        
        for idx, w in enumerate(week_numbers):
            ticks, meta = get_week_dataset(w, raw_dir=RAW_DIR, processed_dir=PROC_DIR, auto_convert=False)
            n_ticks = len(ticks)
            
            # Resample M15, M30, H1 candles from mid prices
            ts = pd.to_datetime(ticks['timestamp'], unit='ms')
            mid = 0.5 * (ticks['bid'] + ticks['ask'])
            base_df = pd.DataFrame({'open': mid, 'high': mid, 'low': mid, 'close': mid}, index=ts)
            
            h1 = base_df.resample('1h').agg({
                'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'
            }).dropna()
            
            m30 = base_df.resample('30min').agg({
                'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'
            }).dropna()
            
            m15 = base_df.resample('15min').agg({
                'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'
            }).dropna()
            
            self.weeks_data.append({
                'week': w,
                'ticks': ticks,
                'tick_ts': ticks['timestamp'],
                'n_ticks': n_ticks,
                'h1': h1,
                'm30': m30,
                'm15': m15,
                'meta': meta
            })
            if (idx + 1) % 40 == 0 or (idx + 1) == len(week_numbers):
                print(f"    Progress: {idx + 1}/{len(week_numbers)} weeks cached ({time.time() - t0:.1f}s)")
                
        print(f"[+] Loaded {len(self.weeks_data)} weeks into memory in {time.time() - t0:.2f}s!")


def simulate_s66_on_dataset(
    dataset: PreloadedDataset,
    ma_pb: int,
    ma_macro: int,
    target_rr: float,
    touch_pips: float,
    buffer_pips: float,
    min_risk_pips: float,
    max_risk_pips: float,
    allowed_hours: List[int],
    filter_fridays: bool = True,
    pip_size: float = 0.0001
) -> Dict[str, Any]:
    """
    High-speed simulation of S66 Bhanushali 44 MA on pre-loaded dataset.
    """
    all_trades = []
    dur_ms = 60 * 60 * 1000  # H1 candle duration

    for item in dataset.weeks_data:
        h1 = item['h1'].copy()
        if len(h1) < 55:
            continue
            
        ticks = item['ticks']
        n_ticks = item['n_ticks']
        tick_ts = item['tick_ts']
        
        c = h1['close']
        h1['ma_pb'] = c.rolling(ma_pb, min_periods=1).mean().bfill()
        h1['ma_macro'] = c.rolling(ma_macro, min_periods=1).mean().bfill()
        h1['hour'] = h1.index.hour
        h1['weekday'] = h1.index.dayofweek
        h1['time_ms'] = h1.index.astype('datetime64[ms]').view('int64')
        
        signals = np.zeros(n_ticks, dtype=np.int8)
        sl_pips_arr = np.zeros(n_ticks, dtype=np.float64)
        tp_pips_arr = np.zeros(n_ticks, dtype=np.float64)
        has_signal = False
        
        for i in range(45, len(h1)):
            row = h1.iloc[i]
            prev = h1.iloc[i - 1]
            
            if filter_fridays and row['weekday'] == 4 and row['hour'] >= 16:
                continue
            if row['hour'] not in allowed_hours:
                continue
                
            close_p = row['close']
            open_p = row['open']
            mapb = row['ma_pb']
            mamacro = row['ma_macro']
            
            # Long Setup
            if close_p > mamacro and mapb > mamacro:
                recent_lows = h1['low'].iloc[i-3:i]
                recent_pb = h1['ma_pb'].iloc[i-3:i]
                if (recent_lows <= recent_pb + touch_pips * pip_size).any():
                    if close_p > open_p and close_p > mapb and close_p > prev['high']:
                        pb_low = float(h1['low'].iloc[i-3:i+1].min())
                        sl_price = pb_low - buffer_pips * pip_size
                        risk = close_p - sl_price
                        if min_risk_pips * pip_size <= risk <= max_risk_pips * pip_size:
                            exec_t = row['time_ms'] + dur_ms
                            idx = int(np.searchsorted(tick_ts, exec_t))
                            if idx < n_ticks:
                                signals[idx] = 1
                                sl_p = risk / pip_size
                                sl_pips_arr[idx] = sl_p
                                tp_pips_arr[idx] = sl_p * target_rr
                                has_signal = True
                                
            # Short Setup
            elif close_p < mamacro and mapb < mamacro:
                recent_highs = h1['high'].iloc[i-3:i]
                recent_pb = h1['ma_pb'].iloc[i-3:i]
                if (recent_highs >= recent_pb - touch_pips * pip_size).any():
                    if close_p < open_p and close_p < mapb and close_p < prev['low']:
                        pb_high = float(h1['high'].iloc[i-3:i+1].max())
                        sl_price = pb_high + buffer_pips * pip_size
                        risk = sl_price - close_p
                        if min_risk_pips * pip_size <= risk <= max_risk_pips * pip_size:
                            exec_t = row['time_ms'] + dur_ms
                            idx = int(np.searchsorted(tick_ts, exec_t))
                            if idx < n_ticks:
                                signals[idx] = -1
                                sl_p = risk / pip_size
                                sl_pips_arr[idx] = sl_p
                                tp_pips_arr[idx] = sl_p * target_rr
                                has_signal = True
                                
        if has_signal:
            trades = run_matching_loop(ticks, signals, pip_size=pip_size, sl_pips=sl_pips_arr, tp_pips=tp_pips_arr)
            if len(trades) > 0:
                all_trades.append(trades)
                
    if all_trades:
        combined = np.concatenate(all_trades)
        return compute_metrics(combined), combined
    return compute_metrics(np.zeros(0, dtype=TRADE_DTYPE)), np.zeros(0, dtype=TRADE_DTYPE)


def optimize_s66(train_ds: PreloadedDataset, test_ds: PreloadedDataset, n_trials: int = 40):
    print("=" * 100)
    print("   OPTIMIZATION: S66 BHANUSHALI 44 MA PULLBACK H1 (SOP 6/4 BLOCK CROSS-VALIDATION)   ".center(100))
    print("=" * 100)
    
    study = optuna.create_study(
        study_name="s66_44ma_sop_opt",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    
    trial_records = []
    
    def objective(trial: optuna.Trial) -> float:
        ma_pb = trial.suggest_int("ma_pb", 34, 55, step=2)
        ma_macro = trial.suggest_int("ma_macro", 150, 250, step=10)
        target_rr = trial.suggest_float("target_rr", 2.2, 3.4, step=0.2)
        touch_pips = trial.suggest_float("touch_pips", 1.0, 3.5, step=0.5)
        buffer_pips = trial.suggest_float("buffer_pips", 1.0, 3.0, step=0.5)
        min_risk_pips = trial.suggest_float("min_risk_pips", 5.0, 10.0, step=1.0)
        max_risk_pips = trial.suggest_float("max_risk_pips", 18.0, 30.0, step=2.0)
        
        session_choice = trial.suggest_categorical("session", ["london_ny", "full_day", "london_core"])
        if session_choice == "london_ny":
            allowed_hours = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
        elif session_choice == "london_core":
            allowed_hours = [7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        else:
            allowed_hours = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
            
        m, _ = simulate_s66_on_dataset(
            train_ds,
            ma_pb=ma_pb,
            ma_macro=ma_macro,
            target_rr=target_rr,
            touch_pips=touch_pips,
            buffer_pips=buffer_pips,
            min_risk_pips=min_risk_pips,
            max_risk_pips=max_risk_pips,
            allowed_hours=allowed_hours
        )
        
        trades = m['total_trades']
        pf = m['profit_factor']
        net = m['net_pips']
        max_dd = m['max_drawdown_pips']
        rr = m['win_loss_ratio']
        win_rate = m['win_rate_pct']
        
        trial_records.append({
            "trial": trial.number,
            "ma_pb": ma_pb,
            "ma_macro": ma_macro,
            "target_rr": target_rr,
            "touch_pips": touch_pips,
            "buffer_pips": buffer_pips,
            "min_risk_pips": min_risk_pips,
            "max_risk_pips": max_risk_pips,
            "session": session_choice,
            "train_trades": trades,
            "train_pf": round(pf, 3),
            "train_rr": round(rr, 3),
            "train_win_rate": round(win_rate, 2),
            "train_net_pips": round(net, 1),
            "train_max_dd": round(max_dd, 1)
        })
        
        # Penalize insufficient trades or negative return
        if trades < 40 or net <= 0 or pf <= 1.0:
            return -100.0
            
        # Target: Profit Factor >= 2.0 & RR >= 2.0 : 1
        score = pf * np.sqrt(max(1.0, rr)) * (1.0 - min(0.6, max_dd / 600.0))
        if pf >= 2.0 and rr >= 2.0:
            score += 2.0  # Quality gate bonus
        return float(score)

    print(f"[*] Commencing Bayesian Optimization ({n_trials} trials)...")
    t0 = time.time()
    study.optimize(objective, n_trials=n_trials)
    print(f"[+] Optimization finished in {time.time() - t0:.1f}s!")
    
    best_p = study.best_params
    print("\n" + "=" * 80)
    print(f"BEST IN-SAMPLE PARAMETERS (Trial {study.best_trial.number}):")
    for k, v in best_p.items():
        print(f"  • {k}: {v}")
    print("=" * 80)
    
    # Save trials table using DuckDB
    trials_df = pd.DataFrame(trial_records)
    trials_csv = OPT_RESULTS_DIR / "s66_optuna_trials.csv"
    trials_df.to_csv(trials_csv, index=False)
    
    # Now evaluate on In-Sample (Train) with Best Parameters
    session_choice = best_p["session"]
    if session_choice == "london_ny":
        allowed_hours = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
    elif session_choice == "london_core":
        allowed_hours = [7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    else:
        allowed_hours = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        
    train_m, train_trades = simulate_s66_on_dataset(
        train_ds,
        ma_pb=best_p["ma_pb"],
        ma_macro=best_p["ma_macro"],
        target_rr=best_p["target_rr"],
        touch_pips=best_p["touch_pips"],
        buffer_pips=best_p["buffer_pips"],
        min_risk_pips=best_p["min_risk_pips"],
        max_risk_pips=best_p["max_risk_pips"],
        allowed_hours=allowed_hours
    )
    
    print_scorecard(train_m, title="GATE 1: IN-SAMPLE TRAINING (6 RANDOM BLOCKS / 36 MONTHS)")
    
    # STRICT OUT-OF-SAMPLE TEST (4 UNSEEN BLOCKS / 24 MONTHS)
    print("\n" + "=" * 80)
    print("EXECUTING GATE 2: STRICT OUT-OF-SAMPLE BLIND TEST (4 UNSEEN BLOCKS / 24 MONTHS)")
    print("Zero Parameter Changes - Complete Parameter Freeze")
    print("=" * 80)
    
    test_m, test_trades = simulate_s66_on_dataset(
        test_ds,
        ma_pb=best_p["ma_pb"],
        ma_macro=best_p["ma_macro"],
        target_rr=best_p["target_rr"],
        touch_pips=best_p["touch_pips"],
        buffer_pips=best_p["buffer_pips"],
        min_risk_pips=best_p["min_risk_pips"],
        max_risk_pips=best_p["max_risk_pips"],
        allowed_hours=allowed_hours
    )
    
    print_scorecard(test_m, title="GATE 2: OUT-OF-SAMPLE TEST (4 UNSEEN BLOCKS / 24 MONTHS)")
    
    # Consolidated 5-Year Performance
    all_5y_trades = np.concatenate([train_trades, test_trades]) if len(train_trades) > 0 and len(test_trades) > 0 else train_trades
    all_5y_m = compute_metrics(all_5y_trades)
    print_scorecard(all_5y_m, title="CONSOLIDATED FULL 5-YEAR AUDIT (ALL 10 BLOCKS / 262 WEEKS)")
    
    # SOP Qualification Status
    gate1_pass = (train_m["profit_factor"] >= 2.0 and train_m["win_loss_ratio"] >= 2.0 and train_m["net_pips"] > 500)
    gate2_pass = (test_m["profit_factor"] >= 2.0 and test_m["win_loss_ratio"] >= 2.0 and test_m["net_pips"] > 300)
    
    summary = {
        "strategy": "S66_Bhanushali_44MA",
        "best_params": best_p,
        "train_metrics": train_m,
        "test_metrics": test_m,
        "total_5y_metrics": all_5y_m,
        "gate1_pass": bool(gate1_pass),
        "gate2_pass": bool(gate2_pass)
    }
    
    summary_path = OPT_RESULTS_DIR / "s66_optimization_summary.json"
    with open(summary_path, "w") as f:
        # Convert non-serializable values
        json_clean = {
            "strategy": summary["strategy"],
            "best_params": best_p,
            "gate1_pass": summary["gate1_pass"],
            "gate2_pass": summary["gate2_pass"],
            "train": {
                "trades": train_m["total_trades"],
                "pf": round(train_m["profit_factor"], 3),
                "rr": round(train_m["win_loss_ratio"], 3),
                "win_rate": round(train_m["win_rate_pct"], 2),
                "net_pips": round(train_m["net_pips"], 1),
                "max_dd": round(train_m["max_drawdown_pips"], 1)
            },
            "test": {
                "trades": test_m["total_trades"],
                "pf": round(test_m["profit_factor"], 3),
                "rr": round(test_m["win_loss_ratio"], 3),
                "win_rate": round(test_m["win_rate_pct"], 2),
                "net_pips": round(test_m["net_pips"], 1),
                "max_dd": round(test_m["max_drawdown_pips"], 1)
            },
            "total_5y": {
                "trades": all_5y_m["total_trades"],
                "pf": round(all_5y_m["profit_factor"], 3),
                "rr": round(all_5y_m["win_loss_ratio"], 3),
                "win_rate": round(all_5y_m["win_rate_pct"], 2),
                "net_pips": round(all_5y_m["net_pips"], 1),
                "max_dd": round(all_5y_m["max_drawdown_pips"], 1)
            }
        }
        import json
        json.dump(json_clean, f, indent=2)
        
    print(f"\n[+] Optimization summary saved to: {summary_path}")
    print(f"[+] All trial telemetry saved to: {trials_csv}")
    return summary


def simulate_s10_on_dataset(
    dataset: PreloadedDataset,
    ema_fast: int,
    ema_trend: int,
    target_rr: float,
    touch_pips: float,
    buffer_pips: float,
    min_risk_pips: float,
    max_risk_pips: float,
    allowed_hours: List[int],
    filter_fridays: bool = True,
    pip_size: float = 0.0001
) -> Tuple[Dict[str, Any], np.ndarray]:
    """
    High-speed simulation of S10 Dual EMA 20/200 Pullback H1 on preloaded dataset.
    """
    all_trades = []
    dur_ms = 60 * 60 * 1000

    for item in dataset.weeks_data:
        h1 = item['h1'].copy()
        if len(h1) < 45:
            continue
            
        ticks = item['ticks']
        n_ticks = item['n_ticks']
        tick_ts = item['tick_ts']
        
        c = h1['close']
        h1['ema_fast'] = c.ewm(span=ema_fast, adjust=False).mean()
        h1['ema_trend'] = c.ewm(span=ema_trend, adjust=False).mean()
        h1['hour'] = h1.index.hour
        h1['weekday'] = h1.index.dayofweek
        h1['time_ms'] = h1.index.astype('datetime64[ms]').view('int64')
        
        signals = np.zeros(n_ticks, dtype=np.int8)
        sl_pips_arr = np.zeros(n_ticks, dtype=np.float64)
        tp_pips_arr = np.zeros(n_ticks, dtype=np.float64)
        has_signal = False
        
        for i in range(25, len(h1)):
            row = h1.iloc[i]
            prev = h1.iloc[i - 1]
            
            if filter_fridays and row['weekday'] == 4 and row['hour'] >= 16:
                continue
            if row['hour'] not in allowed_hours:
                continue
                
            close_p = row['close']
            open_p = row['open']
            emaf = row['ema_fast']
            emat = row['ema_trend']
            
            # Long Setup: Macro uptrend, pullback touch to fast EMA, departure breakout
            if close_p > emat and emaf > emat:
                recent_lows = h1['low'].iloc[i-3:i]
                recent_emas = h1['ema_fast'].iloc[i-3:i]
                if (recent_lows <= recent_emas + touch_pips * pip_size).any():
                    if close_p > open_p and close_p > emaf and close_p > prev['high']:
                        pb_low = float(h1['low'].iloc[i-3:i+1].min())
                        sl_price = pb_low - buffer_pips * pip_size
                        risk = close_p - sl_price
                        if min_risk_pips * pip_size <= risk <= max_risk_pips * pip_size:
                            exec_t = row['time_ms'] + dur_ms
                            idx = int(np.searchsorted(tick_ts, exec_t))
                            if idx < n_ticks:
                                signals[idx] = 1
                                sl_p = risk / pip_size
                                sl_pips_arr[idx] = sl_p
                                tp_pips_arr[idx] = sl_p * target_rr
                                has_signal = True
                                
            # Short Setup: Macro downtrend, pullback touch to fast EMA, departure breakdown
            elif close_p < emat and emaf < emat:
                recent_highs = h1['high'].iloc[i-3:i]
                recent_emas = h1['ema_fast'].iloc[i-3:i]
                if (recent_highs >= recent_emas - touch_pips * pip_size).any():
                    if close_p < open_p and close_p < emaf and close_p < prev['low']:
                        pb_high = float(h1['high'].iloc[i-3:i+1].max())
                        sl_price = pb_high + buffer_pips * pip_size
                        risk = sl_price - close_p
                        if min_risk_pips * pip_size <= risk <= max_risk_pips * pip_size:
                            exec_t = row['time_ms'] + dur_ms
                            idx = int(np.searchsorted(tick_ts, exec_t))
                            if idx < n_ticks:
                                signals[idx] = -1
                                sl_p = risk / pip_size
                                sl_pips_arr[idx] = sl_p
                                tp_pips_arr[idx] = sl_p * target_rr
                                has_signal = True
                                
        if has_signal:
            trades = run_matching_loop(ticks, signals, pip_size=pip_size, sl_pips=sl_pips_arr, tp_pips=tp_pips_arr)
            if len(trades) > 0:
                all_trades.append(trades)
                
    if all_trades:
        combined = np.concatenate(all_trades)
        return compute_metrics(combined), combined
    return compute_metrics(np.zeros(0, dtype=TRADE_DTYPE)), np.zeros(0, dtype=TRADE_DTYPE)


def optimize_s10(train_ds: PreloadedDataset, test_ds: PreloadedDataset, n_trials: int = 40):
    print("=" * 100)
    print("   OPTIMIZATION: S10 DUAL EMA 20/200 PULLBACK H1 (SOP 6/4 BLOCK CROSS-VALIDATION)   ".center(100))
    print("=" * 100)
    
    study = optuna.create_study(
        study_name="s10_ema_sop_opt",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    
    trial_records = []
    
    def objective(trial: optuna.Trial) -> float:
        ema_fast = trial.suggest_int("ema_fast", 16, 28, step=2)
        ema_trend = trial.suggest_int("ema_trend", 150, 250, step=10)
        target_rr = trial.suggest_float("target_rr", 2.2, 3.4, step=0.2)
        touch_pips = trial.suggest_float("touch_pips", 1.0, 3.0, step=0.5)
        buffer_pips = trial.suggest_float("buffer_pips", 1.0, 3.0, step=0.5)
        min_risk_pips = trial.suggest_float("min_risk_pips", 5.0, 10.0, step=1.0)
        max_risk_pips = trial.suggest_float("max_risk_pips", 18.0, 28.0, step=2.0)
        
        session_choice = trial.suggest_categorical("session", ["london_ny", "london_core", "full_day"])
        if session_choice == "london_ny":
            allowed_hours = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
        elif session_choice == "london_core":
            allowed_hours = [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
        else:
            allowed_hours = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
            
        m, _ = simulate_s10_on_dataset(
            train_ds,
            ema_fast=ema_fast,
            ema_trend=ema_trend,
            target_rr=target_rr,
            touch_pips=touch_pips,
            buffer_pips=buffer_pips,
            min_risk_pips=min_risk_pips,
            max_risk_pips=max_risk_pips,
            allowed_hours=allowed_hours
        )
        
        trades = m['total_trades']
        pf = m['profit_factor']
        net = m['net_pips']
        max_dd = m['max_drawdown_pips']
        rr = m['win_loss_ratio']
        win_rate = m['win_rate_pct']
        
        trial_records.append({
            "trial": trial.number,
            "ema_fast": ema_fast,
            "ema_trend": ema_trend,
            "target_rr": target_rr,
            "touch_pips": touch_pips,
            "buffer_pips": buffer_pips,
            "min_risk_pips": min_risk_pips,
            "max_risk_pips": max_risk_pips,
            "session": session_choice,
            "train_trades": trades,
            "train_pf": round(pf, 3),
            "train_rr": round(rr, 3),
            "train_win_rate": round(win_rate, 2),
            "train_net_pips": round(net, 1),
            "train_max_dd": round(max_dd, 1)
        })
        
        if trades < 40 or net <= 0 or pf <= 1.0:
            return -100.0
            
        score = pf * np.sqrt(max(1.0, rr)) * (1.0 - min(0.6, max_dd / 600.0))
        if pf >= 2.0 and rr >= 2.0:
            score += 2.0
        return float(score)

    print(f"[*] Commencing Bayesian Optimization ({n_trials} trials)...")
    t0 = time.time()
    study.optimize(objective, n_trials=n_trials)
    print(f"[+] Optimization finished in {time.time() - t0:.1f}s!")
    
    best_p = study.best_params
    print("\n" + "=" * 80)
    print(f"BEST IN-SAMPLE PARAMETERS (Trial {study.best_trial.number}):")
    for k, v in best_p.items():
        print(f"  • {k}: {v}")
    print("=" * 80)
    
    trials_df = pd.DataFrame(trial_records)
    trials_csv = OPT_RESULTS_DIR / "s10_optuna_trials.csv"
    trials_df.to_csv(trials_csv, index=False)
    
    session_choice = best_p["session"]
    if session_choice == "london_ny":
        allowed_hours = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
    elif session_choice == "london_core":
        allowed_hours = [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
    else:
        allowed_hours = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        
    train_m, train_trades = simulate_s10_on_dataset(
        train_ds,
        ema_fast=best_p["ema_fast"],
        ema_trend=best_p["ema_trend"],
        target_rr=best_p["target_rr"],
        touch_pips=best_p["touch_pips"],
        buffer_pips=best_p["buffer_pips"],
        min_risk_pips=best_p["min_risk_pips"],
        max_risk_pips=best_p["max_risk_pips"],
        allowed_hours=allowed_hours
    )
    print_scorecard(train_m, title="S10 GATE 1: IN-SAMPLE TRAINING (6 RANDOM BLOCKS / 36 MONTHS)")
    
    print("\n" + "=" * 80)
    print("EXECUTING GATE 2: STRICT OUT-OF-SAMPLE BLIND TEST (4 UNSEEN BLOCKS / 24 MONTHS)")
    print("Zero Parameter Changes - Complete Parameter Freeze")
    print("=" * 80)
    
    test_m, test_trades = simulate_s10_on_dataset(
        test_ds,
        ema_fast=best_p["ema_fast"],
        ema_trend=best_p["ema_trend"],
        target_rr=best_p["target_rr"],
        touch_pips=best_p["touch_pips"],
        buffer_pips=best_p["buffer_pips"],
        min_risk_pips=best_p["min_risk_pips"],
        max_risk_pips=best_p["max_risk_pips"],
        allowed_hours=allowed_hours
    )
    print_scorecard(test_m, title="S10 GATE 2: OUT-OF-SAMPLE TEST (4 UNSEEN BLOCKS / 24 MONTHS)")
    
    all_5y_trades = np.concatenate([train_trades, test_trades]) if len(train_trades) > 0 and len(test_trades) > 0 else train_trades
    all_5y_m = compute_metrics(all_5y_trades)
    print_scorecard(all_5y_m, title="S10 CONSOLIDATED FULL 5-YEAR AUDIT (ALL 10 BLOCKS / 262 WEEKS)")
    
    summary_path = OPT_RESULTS_DIR / "s10_optimization_summary.json"
    import json
    with open(summary_path, "w") as f:
        json.dump({
            "strategy": "S10_DualEMA_Pullback",
            "best_params": best_p,
            "train": {"trades": train_m["total_trades"], "pf": round(train_m["profit_factor"], 3), "rr": round(train_m["win_loss_ratio"], 3), "net_pips": round(train_m["net_pips"], 1), "max_dd": round(train_m["max_drawdown_pips"], 1)},
            "test": {"trades": test_m["total_trades"], "pf": round(test_m["profit_factor"], 3), "rr": round(test_m["win_loss_ratio"], 3), "net_pips": round(test_m["net_pips"], 1), "max_dd": round(test_m["max_drawdown_pips"], 1)},
            "total_5y": {"trades": all_5y_m["total_trades"], "pf": round(all_5y_m["profit_factor"], 3), "rr": round(all_5y_m["win_loss_ratio"], 3), "net_pips": round(all_5y_m["net_pips"], 1), "max_dd": round(all_5y_m["max_drawdown_pips"], 1)}
        }, f, indent=2)
    return summary_path


def simulate_s14_on_dataset(
    dataset: PreloadedDataset,
    ema_fast: int,
    ema_trend: int,
    target_rr: float,
    sl_buffer_pips: float,
    min_risk_pips: float,
    max_risk_pips: float,
    allowed_hours: List[int],
    filter_fridays: bool = True,
    pip_size: float = 0.0001
) -> Tuple[Dict[str, Any], np.ndarray]:
    """
    High-speed simulation of S14 Inside Bar M30 Breakout on preloaded dataset.
    """
    all_trades = []
    dur_30m_ms = 30 * 60 * 1000

    for item in dataset.weeks_data:
        m30 = item['m30'].copy()
        if len(m30) < 40:
            continue
            
        ticks = item['ticks']
        n_ticks = item['n_ticks']
        tick_ts = item['tick_ts']
        
        c = m30['close']
        m30['ema_fast'] = c.ewm(span=ema_fast, adjust=False).mean()
        m30['ema_trend'] = c.ewm(span=ema_trend, adjust=False).mean()
        m30['hour'] = m30.index.hour
        m30['weekday'] = m30.index.dayofweek
        m30['time_ms'] = m30.index.astype('datetime64[ms]').view('int64')
        
        signals = np.zeros(n_ticks, dtype=np.int8)
        sl_pips_arr = np.zeros(n_ticks, dtype=np.float64)
        tp_pips_arr = np.zeros(n_ticks, dtype=np.float64)
        has_signal = False
        
        for i in range(25, len(m30)):
            row = m30.iloc[i]
            ib = m30.iloc[i - 1]
            mother = m30.iloc[i - 2]
            
            if filter_fridays and row['weekday'] == 4 and row['hour'] >= 15:
                continue
            if row['hour'] not in allowed_hours:
                continue
                
            is_inside = (ib['high'] <= mother['high']) and (ib['low'] >= mother['low'])
            if not is_inside:
                continue
                
            close_p = row['close']
            emaf = row['ema_fast']
            emat = row['ema_trend']
            
            # Long Setup
            if (close_p > mother['high']) and (close_p > emaf) and (close_p > emat):
                sl_price = ib['low'] - sl_buffer_pips * pip_size
                risk = close_p - sl_price
                if min_risk_pips * pip_size <= risk <= max_risk_pips * pip_size:
                    exec_t = row['time_ms'] + dur_30m_ms
                    idx = int(np.searchsorted(tick_ts, exec_t))
                    if idx < n_ticks:
                        signals[idx] = 1
                        sl_p = risk / pip_size
                        sl_pips_arr[idx] = sl_p
                        tp_pips_arr[idx] = sl_p * target_rr
                        has_signal = True
                        
            # Short Setup
            elif (close_p < mother['low']) and (close_p < emaf) and (close_p < emat):
                sl_price = ib['high'] + sl_buffer_pips * pip_size
                risk = sl_price - close_p
                if min_risk_pips * pip_size <= risk <= max_risk_pips * pip_size:
                    exec_t = row['time_ms'] + dur_30m_ms
                    idx = int(np.searchsorted(tick_ts, exec_t))
                    if idx < n_ticks:
                        signals[idx] = -1
                        sl_p = risk / pip_size
                        sl_pips_arr[idx] = sl_p
                        tp_pips_arr[idx] = sl_p * target_rr
                        has_signal = True
                        
        if has_signal:
            trades = run_matching_loop(ticks, signals, pip_size=pip_size, sl_pips=sl_pips_arr, tp_pips=tp_pips_arr)
            if len(trades) > 0:
                all_trades.append(trades)
                
    if all_trades:
        combined = np.concatenate(all_trades)
        return compute_metrics(combined), combined
    return compute_metrics(np.zeros(0, dtype=TRADE_DTYPE)), np.zeros(0, dtype=TRADE_DTYPE)


def optimize_s14(train_ds: PreloadedDataset, test_ds: PreloadedDataset, n_trials: int = 40):
    print("=" * 100)
    print("   OPTIMIZATION: S14 INSIDE BAR M30 BREAKOUT (SOP 6/4 BLOCK CROSS-VALIDATION)   ".center(100))
    print("=" * 100)
    
    study = optuna.create_study(
        study_name="s14_ib_sop_opt",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    
    trial_records = []
    
    def objective(trial: optuna.Trial) -> float:
        ema_fast = trial.suggest_int("ema_fast", 30, 70, step=10)
        ema_trend = trial.suggest_int("ema_trend", 150, 250, step=10)
        target_rr = trial.suggest_float("target_rr", 2.2, 3.4, step=0.2)
        sl_buffer_pips = trial.suggest_float("sl_buffer_pips", 1.0, 3.0, step=0.5)
        min_risk_pips = trial.suggest_float("min_risk_pips", 4.0, 8.0, step=1.0)
        max_risk_pips = trial.suggest_float("max_risk_pips", 16.0, 26.0, step=2.0)
        
        session_choice = trial.suggest_categorical("session", ["london_ny", "london_core", "full_day"])
        if session_choice == "london_ny":
            allowed_hours = [7, 8, 9, 13, 14, 15, 16]
        elif session_choice == "london_core":
            allowed_hours = [7, 8, 9, 10, 13, 14, 15]
        else:
            allowed_hours = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
            
        m, _ = simulate_s14_on_dataset(
            train_ds,
            ema_fast=ema_fast,
            ema_trend=ema_trend,
            target_rr=target_rr,
            sl_buffer_pips=sl_buffer_pips,
            min_risk_pips=min_risk_pips,
            max_risk_pips=max_risk_pips,
            allowed_hours=allowed_hours
        )
        
        trades = m['total_trades']
        pf = m['profit_factor']
        net = m['net_pips']
        max_dd = m['max_drawdown_pips']
        rr = m['win_loss_ratio']
        win_rate = m['win_rate_pct']
        
        trial_records.append({
            "trial": trial.number,
            "ema_fast": ema_fast,
            "ema_trend": ema_trend,
            "target_rr": target_rr,
            "sl_buffer_pips": sl_buffer_pips,
            "min_risk_pips": min_risk_pips,
            "max_risk_pips": max_risk_pips,
            "session": session_choice,
            "train_trades": trades,
            "train_pf": round(pf, 3),
            "train_rr": round(rr, 3),
            "train_win_rate": round(win_rate, 2),
            "train_net_pips": round(net, 1),
            "train_max_dd": round(max_dd, 1)
        })
        
        if trades < 35 or net <= 0 or pf <= 1.0:
            return -100.0
            
        score = pf * np.sqrt(max(1.0, rr)) * (1.0 - min(0.6, max_dd / 600.0))
        if pf >= 2.0 and rr >= 2.0:
            score += 2.0
        return float(score)

    print(f"[*] Commencing Bayesian Optimization ({n_trials} trials)...")
    t0 = time.time()
    study.optimize(objective, n_trials=n_trials)
    print(f"[+] Optimization finished in {time.time() - t0:.1f}s!")
    
    best_p = study.best_params
    print("\n" + "=" * 80)
    print(f"BEST IN-SAMPLE PARAMETERS (Trial {study.best_trial.number}):")
    for k, v in best_p.items():
        print(f"  • {k}: {v}")
    print("=" * 80)
    
    trials_df = pd.DataFrame(trial_records)
    trials_csv = OPT_RESULTS_DIR / "s14_optuna_trials.csv"
    trials_df.to_csv(trials_csv, index=False)
    
    session_choice = best_p["session"]
    if session_choice == "london_ny":
        allowed_hours = [7, 8, 9, 13, 14, 15, 16]
    elif session_choice == "london_core":
        allowed_hours = [7, 8, 9, 10, 13, 14, 15]
    else:
        allowed_hours = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
        
    train_m, train_trades = simulate_s14_on_dataset(
        train_ds,
        ema_fast=best_p["ema_fast"],
        ema_trend=best_p["ema_trend"],
        target_rr=best_p["target_rr"],
        sl_buffer_pips=best_p["sl_buffer_pips"],
        min_risk_pips=best_p["min_risk_pips"],
        max_risk_pips=best_p["max_risk_pips"],
        allowed_hours=allowed_hours
    )
    print_scorecard(train_m, title="S14 GATE 1: IN-SAMPLE TRAINING (6 RANDOM BLOCKS / 36 MONTHS)")
    
    print("\n" + "=" * 80)
    print("EXECUTING GATE 2: STRICT OUT-OF-SAMPLE BLIND TEST (4 UNSEEN BLOCKS / 24 MONTHS)")
    print("Zero Parameter Changes - Complete Parameter Freeze")
    print("=" * 80)
    
    test_m, test_trades = simulate_s14_on_dataset(
        test_ds,
        ema_fast=best_p["ema_fast"],
        ema_trend=best_p["ema_trend"],
        target_rr=best_p["target_rr"],
        sl_buffer_pips=best_p["sl_buffer_pips"],
        min_risk_pips=best_p["min_risk_pips"],
        max_risk_pips=best_p["max_risk_pips"],
        allowed_hours=allowed_hours
    )
    print_scorecard(test_m, title="S14 GATE 2: OUT-OF-SAMPLE TEST (4 UNSEEN BLOCKS / 24 MONTHS)")
    
    all_5y_trades = np.concatenate([train_trades, test_trades]) if len(train_trades) > 0 and len(test_trades) > 0 else train_trades
    all_5y_m = compute_metrics(all_5y_trades)
    print_scorecard(all_5y_m, title="S14 CONSOLIDATED FULL 5-YEAR AUDIT (ALL 10 BLOCKS / 262 WEEKS)")
    
    summary_path = OPT_RESULTS_DIR / "s14_optimization_summary.json"
    import json
    with open(summary_path, "w") as f:
        json.dump({
            "strategy": "S14_InsideBar_M30",
            "best_params": best_p,
            "train": {"trades": train_m["total_trades"], "pf": round(train_m["profit_factor"], 3), "rr": round(train_m["win_loss_ratio"], 3), "net_pips": round(train_m["net_pips"], 1), "max_dd": round(train_m["max_drawdown_pips"], 1)},
            "test": {"trades": test_m["total_trades"], "pf": round(test_m["profit_factor"], 3), "rr": round(test_m["win_loss_ratio"], 3), "net_pips": round(test_m["net_pips"], 1), "max_dd": round(test_m["max_drawdown_pips"], 1)},
            "total_5y": {"trades": all_5y_m["total_trades"], "pf": round(all_5y_m["profit_factor"], 3), "rr": round(all_5y_m["win_loss_ratio"], 3), "net_pips": round(all_5y_m["net_pips"], 1), "max_dd": round(all_5y_m["max_drawdown_pips"], 1)}
        }, f, indent=2)
    return summary_path


def simulate_hf_m15_on_dataset(
    dataset: PreloadedDataset,
    target_rr: float,
    min_sl: float,
    max_sl: float,
    sweep_lookback: int,
    h1_ema_period: int,
    use_h1_filter: bool,
    allowed_hours: List[int],
    filter_fridays: bool = True,
    pip_size: float = 0.0001
) -> Tuple[Dict[str, Any], np.ndarray]:
    """
    High-speed simulation of High-Frequency M15 Scalper Elite on preloaded dataset.
    """
    all_trades = []
    dur_ms = 15 * 60 * 1000

    for item in dataset.weeks_data:
        m15 = item['m15'].copy()
        h1 = item['h1'].copy()
        if len(m15) < 40 or len(h1) < 20:
            continue
            
        ticks = item['ticks']
        n_ticks = item['n_ticks']
        tick_ts = item['tick_ts']

        # H1 Macro Trend
        h1_ema = h1['close'].ewm(span=h1_ema_period, adjust=False).mean()
        h1_trend = np.where(h1['close'] > h1_ema, 1, -1)
        h1_series = pd.Series(h1_trend, index=h1.index)
        m15['macro_trend'] = h1_series.reindex(m15.index, method='ffill').fillna(0)

        # M15 ATR & Fast EMA
        tr = np.maximum(m15['high'] - m15['low'], np.maximum(abs(m15['high'] - m15['close'].shift(1)), abs(m15['low'] - m15['close'].shift(1))))
        m15['atr'] = tr.rolling(14).mean().bfill()
        m15['ema_20'] = m15['close'].ewm(span=20, adjust=False).mean()
        
        m15_hours = m15.index.hour
        m15_weekdays = m15.index.dayofweek
        time_ms = m15.index.astype('datetime64[ms]').view('int64')

        signals = np.zeros(n_ticks, dtype=np.int8)
        sl_pips_arr = np.zeros(n_ticks, dtype=np.float64)
        tp_pips_arr = np.zeros(n_ticks, dtype=np.float64)
        has_signal = False

        for i in range(max(25, sweep_lookback), len(m15)):
            hour = m15_hours[i]
            weekday = m15_weekdays[i]
            if hour not in allowed_hours:
                continue
            if filter_fridays and weekday == 4 and hour >= 16:
                continue

            row = m15.iloc[i]
            prev = m15.iloc[i - 1]
            prior = m15.iloc[i - 2]

            sig = 0
            risk_p = 0.0
            macro = row['macro_trend'] if use_h1_filter else 0
            atr_pips = row['atr'] / pip_size
            dyn_min_sl = max(min_sl, 0.8 * atr_pips)
            dyn_max_sl = max(max_sl, 1.8 * atr_pips)

            # Pattern A: 4-Hour Rolling Liquidity Sweep
            window = m15.iloc[i - sweep_lookback : i]
            roll_h = window['high'].max()
            roll_l = window['low'].min()

            # Bullish Sweep: pierced low, closed back inside & green
            sweep_l = (roll_l - row['low']) / pip_size
            if 0.8 <= sweep_l <= 6.0 and row['close'] > roll_l and row['close'] > row['open']:
                if not use_h1_filter or macro >= 0:
                    sl_p = row['low'] - 1.2 * pip_size
                    r = max((row['close'] - sl_p) / pip_size, dyn_min_sl)
                    if r <= dyn_max_sl:
                        sig = 1
                        risk_p = r

            # Bearish Sweep: pierced high, closed back inside & red
            sweep_h = (row['high'] - roll_h) / pip_size
            if 0.8 <= sweep_h <= 6.0 and row['close'] < roll_h and row['close'] < row['open']:
                if not use_h1_filter or macro <= 0:
                    sl_p = row['high'] + 1.2 * pip_size
                    r = max((sl_p - row['close']) / pip_size, dyn_min_sl)
                    if r <= dyn_max_sl:
                        sig = -1
                        risk_p = r

            # Pattern B: Fair Value Gap Immediate Mitigation
            if sig == 0:
                # Bullish FVG
                if prev['close'] > prev['open'] and (prev['close'] - prev['open']) / pip_size >= 3.5:
                    if row['open'] > prior['high'] and row['low'] <= prior['high'] and row['close'] > prior['high'] and row['close'] > row['ema_20']:
                        if not use_h1_filter or macro >= 0:
                            sl_p = prev['low'] - 1.2 * pip_size
                            r = max((row['close'] - sl_p) / pip_size, dyn_min_sl)
                            if r <= dyn_max_sl:
                                sig = 1
                                risk_p = r

                # Bearish FVG
                elif prev['close'] < prev['open'] and (prev['open'] - prev['close']) / pip_size >= 3.5:
                    if row['open'] < prior['low'] and row['high'] >= prior['low'] and row['close'] < prior['low'] and row['close'] < row['ema_20']:
                        if not use_h1_filter or macro <= 0:
                            sl_p = prev['high'] + 1.2 * pip_size
                            r = max((sl_p - row['close']) / pip_size, dyn_min_sl)
                            if r <= dyn_max_sl:
                                sig = -1
                                risk_p = r

            if sig != 0:
                exec_t = time_ms[i] + dur_ms
                idx = int(np.searchsorted(tick_ts, exec_t))
                if idx < n_ticks and signals[idx] == 0:
                    signals[idx] = sig
                    sl_pips_arr[idx] = risk_p
                    tp_pips_arr[idx] = risk_p * target_rr
                    has_signal = True

        if has_signal:
            trades = run_matching_loop(ticks, signals, pip_size=pip_size, sl_pips=sl_pips_arr, tp_pips=tp_pips_arr)
            if len(trades) > 0:
                all_trades.append(trades)

    if all_trades:
        combined = np.concatenate(all_trades)
        return compute_metrics(combined), combined
    return compute_metrics(np.zeros(0, dtype=TRADE_DTYPE)), np.zeros(0, dtype=TRADE_DTYPE)


def optimize_hf_m15(train_ds: PreloadedDataset, test_ds: PreloadedDataset, n_trials: int = 25):
    print("=" * 100)
    print("   OPTIMIZATION: HIGH-FREQUENCY M15 SCALPER (CONSTRAINED: RR 1.5-2.5, PF < 2.0, HIGH VOLUME)   ".center(100))
    print("=" * 100)

    study = optuna.create_study(
        study_name="hf_m15_sop_opt",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42)
    )

    trial_records = []

    def objective(trial: optuna.Trial) -> float:
        # User constraint: Target RR strictly between 1.5 and 2.5
        target_rr = trial.suggest_float("target_rr", 1.5, 2.5, step=0.1)
        min_sl = trial.suggest_float("min_sl", 5.0, 10.0, step=1.0)
        max_sl = trial.suggest_float("max_sl", 14.0, 24.0, step=2.0)
        sweep_lookback = trial.suggest_int("sweep_lookback", 12, 24, step=4)
        h1_ema_period = trial.suggest_int("h1_ema_period", 30, 70, step=10)
        use_h1_filter = trial.suggest_categorical("use_h1_filter", [True, False])

        session_choice = trial.suggest_categorical("session", ["london_ny", "expanded_eu_us", "all_active"])
        if session_choice == "london_ny":
            allowed_hours = [7, 8, 9, 11, 12, 13, 14, 15, 17]
        elif session_choice == "expanded_eu_us":
            allowed_hours = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
        else:
            allowed_hours = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

        m, _ = simulate_hf_m15_on_dataset(
            train_ds,
            target_rr=target_rr,
            min_sl=min_sl,
            max_sl=max_sl,
            sweep_lookback=sweep_lookback,
            h1_ema_period=h1_ema_period,
            use_h1_filter=use_h1_filter,
            allowed_hours=allowed_hours
        )

        trades = m['total_trades']
        pf = m['profit_factor']
        net = m['net_pips']
        max_dd = m['max_drawdown_pips']
        rr = m['win_loss_ratio']
        win_rate = m['win_rate_pct']

        trial_records.append({
            "trial": trial.number,
            "target_rr": round(target_rr, 2),
            "min_sl": min_sl,
            "max_sl": max_sl,
            "sweep_lookback": sweep_lookback,
            "h1_ema_period": h1_ema_period,
            "use_h1_filter": use_h1_filter,
            "session": session_choice,
            "train_trades": trades,
            "train_pf": round(pf, 3),
            "train_rr": round(rr, 3),
            "train_win_rate": round(win_rate, 2),
            "train_net_pips": round(net, 1),
            "train_max_dd": round(max_dd, 1)
        })

        # Mandatory substantial volume requirement: at least 400 trades in 36 months!
        if trades < 400 or net <= 0 or pf <= 1.05:
            return -100.0

        # Reward high net return scaled by high volume
        score = net * (trades / 1000.0)**0.5 * (1.0 - min(0.5, max_dd / 600.0))
        
        # Penalize if PF is >= 2.0 (user specifically requested PF below 2.0 to avoid over-filtering!)
        if pf >= 2.0:
            score *= 0.75
        # Penalize if RR is outside 1.5 - 2.5
        if rr > 2.5 or rr < 1.5:
            score *= 0.75

        return float(score)

    print(f"[*] Commencing Bayesian Optimization ({n_trials} trials)...")
    t0 = time.time()
    study.optimize(objective, n_trials=n_trials)
    print(f"[+] Optimization finished in {time.time() - t0:.1f}s!")

    best_p = study.best_params
    print("\n" + "=" * 80)
    print(f"BEST IN-SAMPLE PARAMETERS (Trial {study.best_trial.number}):")
    for k, v in best_p.items():
        print(f"  • {k}: {v}")
    print("=" * 80)

    trials_df = pd.DataFrame(trial_records)
    trials_csv = OPT_RESULTS_DIR / "hf_m15_optuna_trials.csv"
    trials_df.to_csv(trials_csv, index=False)

    session_choice = best_p["session"]
    if session_choice == "london_ny":
        allowed_hours = [7, 8, 9, 11, 12, 13, 14, 15, 17]
    elif session_choice == "expanded_eu_us":
        allowed_hours = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
    else:
        allowed_hours = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

    train_m, train_trades = simulate_hf_m15_on_dataset(
        train_ds,
        target_rr=best_p["target_rr"],
        min_sl=best_p["min_sl"],
        max_sl=best_p["max_sl"],
        sweep_lookback=best_p["sweep_lookback"],
        h1_ema_period=best_p["h1_ema_period"],
        use_h1_filter=best_p["use_h1_filter"],
        allowed_hours=allowed_hours
    )
    print_scorecard(train_m, title="HF M15 GATE 1: IN-SAMPLE TRAINING (6 RANDOM BLOCKS / 36 MONTHS)")

    print("\n" + "=" * 80)
    print("EXECUTING GATE 2: STRICT OUT-OF-SAMPLE BLIND TEST (4 UNSEEN BLOCKS / 24 MONTHS)")
    print("Zero Parameter Changes - Complete Parameter Freeze")
    print("=" * 80)

    test_m, test_trades = simulate_hf_m15_on_dataset(
        test_ds,
        target_rr=best_p["target_rr"],
        min_sl=best_p["min_sl"],
        max_sl=best_p["max_sl"],
        sweep_lookback=best_p["sweep_lookback"],
        h1_ema_period=best_p["h1_ema_period"],
        use_h1_filter=best_p["use_h1_filter"],
        allowed_hours=allowed_hours
    )
    print_scorecard(test_m, title="HF M15 GATE 2: OUT-OF-SAMPLE TEST (4 UNSEEN BLOCKS / 24 MONTHS)")

    all_5y_trades = np.concatenate([train_trades, test_trades]) if len(train_trades) > 0 and len(test_trades) > 0 else train_trades
    all_5y_m = compute_metrics(all_5y_trades)
    print_scorecard(all_5y_m, title="HF M15 CONSOLIDATED FULL 5-YEAR AUDIT (ALL 10 BLOCKS / 262 WEEKS)")

    summary_path = OPT_RESULTS_DIR / "hf_m15_optimization_summary.json"
    import json
    with open(summary_path, "w") as f:
        json.dump({
            "strategy": "HighFrequency_M15_Scalper",
            "best_params": best_p,
            "train": {"trades": train_m["total_trades"], "pf": round(train_m["profit_factor"], 3), "rr": round(train_m["win_loss_ratio"], 3), "net_pips": round(train_m["net_pips"], 1), "max_dd": round(train_m["max_drawdown_pips"], 1)},
            "test": {"trades": test_m["total_trades"], "pf": round(test_m["profit_factor"], 3), "rr": round(test_m["win_loss_ratio"], 3), "net_pips": round(test_m["net_pips"], 1), "max_dd": round(test_m["max_drawdown_pips"], 1)},
            "total_5y": {"trades": all_5y_m["total_trades"], "pf": round(all_5y_m["profit_factor"], 3), "rr": round(all_5y_m["win_loss_ratio"], 3), "net_pips": round(all_5y_m["net_pips"], 1), "max_dd": round(all_5y_m["max_drawdown_pips"], 1)}
        }, f, indent=2)
    return summary_path


def main():
    parser = argparse.ArgumentParser(description="Institutional Strategy Optimization Suite")
    parser.add_argument("--strategy", type=str, default="hf_m15", choices=["s66", "s10", "s14", "hf_m15"])
    parser.add_argument("--seed", type=int, default=42, help="Seed for 6/4 block sampler")
    parser.add_argument("--trials", type=int, default=25, help="Number of Optuna trials")
    args = parser.parse_args()
    
    train_blocks, test_blocks = sample_train_test_blocks(args.seed)
    train_weeks = get_weeks_for_blocks(train_blocks)
    test_weeks = get_weeks_for_blocks(test_blocks)
    
    print("=" * 100)
    print(f"SOP SAMPLING ARCHITECTURE (Seed {args.seed})")
    print(f"  • Training Blocks (6 Units / 36 Months / 3 Years): {format_block_summary(train_blocks)}")
    print(f"  • Testing Blocks  (4 Units / 24 Months / 2 Years): {format_block_summary(test_blocks)}")
    print("=" * 100)
    
    # Preload datasets
    train_ds = PreloadedDataset(train_weeks, desc="Train Split (6 Blocks)")
    test_ds = PreloadedDataset(test_weeks, desc="Test Split (4 Blocks)")
    
    if args.strategy == "s66":
        optimize_s66(train_ds, test_ds, n_trials=args.trials)
    elif args.strategy == "s10":
        optimize_s10(train_ds, test_ds, n_trials=args.trials)
    elif args.strategy == "s14":
        optimize_s14(train_ds, test_ds, n_trials=args.trials)
    elif args.strategy == "hf_m15":
        optimize_hf_m15(train_ds, test_ds, n_trials=args.trials)


if __name__ == "__main__":
    main()
