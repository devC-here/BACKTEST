#!/usr/bin/env python3
"""
Champion Improvement Laboratory:
Systematically optimizing and refining our best models on 52 Weeks (~26M Ticks):

1. IMPROVE COMB08 (Asian Breakout):
   - Add H1 50-EMA Macro Trend Filter (shift(1)).
   - Test Midpoint Stop vs 1.0 ATR Stop vs Original Low/High Stop.
   - Test Target RR: 1.6R, 1.8R, 2.0R, 2.2R.

2. IMPROVE COMB14 (Supertrend + VWAP):
   - Test Target RR: 1.6R, 1.8R, 2.0R, 2.2R (to boost Win Rate from 35.9% towards 45%).
   - Test minimum ATR filter (>= 5.5 pips) to filter out flat-market chop.

3. IMPROVE COMB12 (Bollinger Squeeze):
   - Fine-tune lookback (30 vs 35 vs 40) and squeeze_ratio (1.35 vs 1.40) to boost annual velocity
     from 50 trades towards 70-80 trades while preserving >= 50% Win Rate.

4. IMPROVE DISC23 (Keltner Squeeze):
   - Tune lookback (25 vs 30 vs 35) and body_atr_ratio (0.50 vs 0.60 vs 0.70) to scale trade count
     while preserving >= 50% Win Rate and 1.6+ PF.

5. IMPROVE DISC02 (Frankfurt ORB):
   - Test Target RR: 1.5R vs 1.6R vs 1.8R vs 2.0R.
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
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tick_engine.core.engine import BacktestEngine
from tick_engine.core.dataloader import get_week_dataset
from tick_engine.strategies.base import BaseStrategy, SIGNAL_LONG, SIGNAL_SHORT

PIP_SIZE = 0.0001


# ==============================================================================
# 1. IMPROVED COMB08: Asian Breakout with H1 Macro Bias & Dynamic Stop
# ==============================================================================
class ImprovedAsianBreakout(BaseStrategy):
    def __init__(
        self,
        target_rr: float = 1.80,
        stop_type: str = "midpoint",  # 'midpoint', 'atr', or 'swing'
        use_macro_bias: bool = True,
        min_adx: float = 16.0,
        min_sl: float = 6.0,
        max_sl: float = 16.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.target_rr = target_rr
        self.stop_type = stop_type
        self.use_macro_bias = use_macro_bias
        self.min_adx = min_adx
        self.min_sl = min_sl
        self.max_sl = max_sl
        self.pip_size = PIP_SIZE
        self.name = f"Improved_Asian_{stop_type}_rr{target_rr}_macro{use_macro_bias}"

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

        c = m15['close']
        h = m15['high']
        l = m15['low']

        # ATR 14
        tr = np.maximum(h - l, np.maximum(abs(h - c.shift(1)), abs(l - c.shift(1))))
        m15['atr_14'] = tr.rolling(14).mean().bfill()
        m15['atr_pips'] = m15['atr_14'] / self.pip_size

        # ADX 14
        up_m = h.diff()
        dn_m = -l.diff()
        p_dm = np.where((up_m > dn_m) & (up_m > 0), up_m, 0.0)
        m_dm = np.where((dn_m > up_m) & (dn_m > 0), dn_m, 0.0)
        smooth_tr = tr.ewm(alpha=1.0/14, adjust=False).mean()
        plus_di = (pd.Series(p_dm, index=m15.index).ewm(alpha=1.0/14, adjust=False).mean() / np.maximum(smooth_tr, 1e-9)) * 100
        minus_di = (pd.Series(m_dm, index=m15.index).ewm(alpha=1.0/14, adjust=False).mean() / np.maximum(smooth_tr, 1e-9)) * 100
        dx = (abs(plus_di - minus_di) / np.maximum(plus_di + minus_di, 1e-9)) * 100
        m15['adx_14'] = dx.ewm(alpha=1.0/14, adjust=False).mean()

        # H1 50-EMA Macro Trend
        h1 = base_df.resample('1h').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
        h1['ema50'] = h1['close'].ewm(span=50, adjust=False).mean()
        h1['macro_bias'] = np.where(h1['close'] > h1['ema50'], 1, -1)
        m15['macro_bias'] = h1['macro_bias'].shift(1).reindex(m15.index, method='ffill').fillna(0)

        m15['hour'] = m15.index.hour
        m15['date'] = m15.index.date
        time_ms = m15.index.astype('datetime64[ms]').view('int64')
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        daily_groups = m15.groupby('date')
        for current_date, day_df in daily_groups:
            asian = day_df[(day_df['hour'] >= 0) & (day_df['hour'] < 6)]
            if len(asian) < 8:
                continue
            asian_high = float(asian['high'].max())
            asian_low = float(asian['low'].min())
            asian_mid = 0.5 * (asian_high + asian_low)
            asian_range = (asian_high - asian_low) / self.pip_size

            if asian_range < 10.0 or asian_range > 35.0:
                continue

            london = day_df[(day_df['hour'] >= 7) & (day_df['hour'] <= 10)]
            traded_today = False

            for idx_m15, row in london.iterrows():
                if traded_today:
                    break
                i = m15.index.get_loc(idx_m15)
                prev = m15.iloc[i-1]
                sig = 0
                risk_p = 0.0

                # Macro filter check
                macro_ok_long = (row['macro_bias'] > 0) if self.use_macro_bias else True
                macro_ok_short = (row['macro_bias'] < 0) if self.use_macro_bias else True

                # Bullish Breakout
                if prev['close'] <= asian_high and row['close'] > asian_high + 0.3 * self.pip_size and macro_ok_long:
                    if row['adx_14'] >= self.min_adx and row['close'] > row['open']:
                        if self.stop_type == "midpoint":
                            sl_p = asian_mid
                        elif self.stop_type == "atr":
                            sl_p = row['close'] - 1.1 * row['atr_14']
                        else:  # swing
                            sl_p = row['low'] - 1.2 * self.pip_size

                        risk = max((row['close'] - sl_p) / self.pip_size, self.min_sl)
                        if risk <= self.max_sl:
                            sig = SIGNAL_LONG
                            risk_p = risk
                            traded_today = True

                # Bearish Breakout
                elif prev['close'] >= asian_low and row['close'] < asian_low - 0.3 * self.pip_size and macro_ok_short:
                    if row['adx_14'] >= self.min_adx and row['close'] < row['open']:
                        if self.stop_type == "midpoint":
                            sl_p = asian_mid
                        elif self.stop_type == "atr":
                            sl_p = row['close'] + 1.1 * row['atr_14']
                        else:  # swing
                            sl_p = row['high'] + 1.2 * self.pip_size

                        risk = max((sl_p - row['close']) / self.pip_size, self.min_sl)
                        if risk <= self.max_sl:
                            sig = SIGNAL_SHORT
                            risk_p = risk
                            traded_today = True

                if sig != 0:
                    exec_t = row['date']  # not used directly, use time_ms[i]
                    exec_time = time_ms[i] + dur_ms
                    idx = int(np.searchsorted(tick_ts, exec_time))
                    if idx < n_ticks and signals[idx] == 0:
                        signals[idx] = sig
                        sl_pips_arr[idx] = risk_p
                        tp_pips_arr[idx] = risk_p * self.target_rr

        return signals, sl_pips_arr, tp_pips_arr


# ==============================================================================
# 2. IMPROVED COMB14: Supertrend VWAP with RR Sweep & ATR Filter
# ==============================================================================
class ImprovedSupertrendVWAP(BaseStrategy):
    def __init__(
        self,
        target_rr: float = 1.80,
        min_atr_pips: float = 5.5,
        st_multiplier: float = 2.0,
        min_sl: float = 6.0,
        max_sl: float = 16.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.target_rr = target_rr
        self.min_atr_pips = min_atr_pips
        self.st_multiplier = st_multiplier
        self.min_sl = min_sl
        self.max_sl = max_sl
        self.pip_size = PIP_SIZE
        self.name = f"Improved_ST_VWAP_rr{target_rr}_atr{min_atr_pips}"

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

        c = m15['close']
        h = m15['high']
        l = m15['low']
        o = m15['open']

        tr = np.maximum(h - l, np.maximum(abs(h - c.shift(1)), abs(l - c.shift(1))))
        m15['atr_14'] = tr.rolling(14).mean().bfill()
        m15['atr_pips'] = m15['atr_14'] / self.pip_size

        # Supertrend
        hl2 = (h + l) / 2.0
        up_band = hl2 - (self.st_multiplier * m15['atr_14'])
        dn_band = hl2 + (self.st_multiplier * m15['atr_14'])
        n = len(m15)
        st = np.zeros(n)
        dir_st = np.ones(n, dtype=int)

        for i in range(1, n):
            if c.iloc[i-1] > up_band.iloc[i-1]:
                up_band.iloc[i] = max(up_band.iloc[i], up_band.iloc[i-1])
            if c.iloc[i-1] < dn_band.iloc[i-1]:
                dn_band.iloc[i] = min(dn_band.iloc[i], dn_band.iloc[i-1])
            if dir_st[i-1] == 1:
                if c.iloc[i] < up_band.iloc[i]:
                    dir_st[i] = -1
                    st[i] = dn_band.iloc[i]
                else:
                    dir_st[i] = 1
                    st[i] = up_band.iloc[i]
            else:
                if c.iloc[i] > dn_band.iloc[i]:
                    dir_st[i] = 1
                    st[i] = up_band.iloc[i]
                else:
                    dir_st[i] = -1
                    st[i] = dn_band.iloc[i]

        m15['st_dir'] = dir_st
        m15['st_val'] = st

        # Session VWAP
        m15['date'] = m15.index.date
        vol = tr / self.pip_size + 10.0
        cum_vol = vol.groupby(m15['date']).cumsum()
        cum_pv = (c * vol).groupby(m15['date']).cumsum()
        m15['vwap'] = cum_pv / np.maximum(cum_vol, 1e-9)

        # Macro Trend (H1 EMA 50)
        h1 = base_df.resample('1h').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
        h1['ema50'] = h1['close'].ewm(span=50, adjust=False).mean()
        h1['macro_bias'] = np.where(h1['close'] > h1['ema50'], 1, -1)
        m15['macro_bias'] = h1['macro_bias'].shift(1).reindex(m15.index, method='ffill').fillna(0)

        m15['hour'] = m15.index.hour
        time_ms = m15.index.astype('datetime64[ms]').view('int64')
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        for i in range(35, len(m15)):
            row = m15.iloc[i]
            prev = m15.iloc[i-1]

            if row['hour'] < 7 or row['hour'] > 15:
                continue

            # ATR filter
            if row['atr_pips'] < self.min_atr_pips:
                continue

            sig = 0
            risk_p = 0.0

            # Bullish Supertrend + Pullback to VWAP
            if row['st_dir'] == 1 and row['macro_bias'] > 0:
                if prev['low'] <= prev['vwap'] and row['close'] > row['vwap'] and row['close'] > row['open']:
                    sl_p = min(prev['low'], row['low'], row['st_val']) - 1.0 * self.pip_size
                    risk = max((row['close'] - sl_p) / self.pip_size, self.min_sl)
                    if risk <= self.max_sl:
                        sig = SIGNAL_LONG
                        risk_p = risk

            # Bearish Supertrend + Pullback to VWAP
            elif row['st_dir'] == -1 and row['macro_bias'] < 0:
                if prev['high'] >= prev['vwap'] and row['close'] < row['vwap'] and row['close'] < row['open']:
                    sl_p = max(prev['high'], row['high'], row['st_val']) + 1.0 * self.pip_size
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
# EXPERIMENTAL LAB TEST RIG
# ==============================================================================
CANDIDATE_CONFIGS = [
    # 1. Asian Breakout Variations
    {"code": "ASIA_V1", "name": "Asian Midpoint Stop (RR 1.8, Macro Filter)", "cls": ImprovedAsianBreakout, "kwargs": {"target_rr": 1.80, "stop_type": "midpoint", "use_macro_bias": True}},
    {"code": "ASIA_V2", "name": "Asian Midpoint Stop (RR 1.6, Macro Filter)", "cls": ImprovedAsianBreakout, "kwargs": {"target_rr": 1.60, "stop_type": "midpoint", "use_macro_bias": True}},
    {"code": "ASIA_V3", "name": "Asian ATR Stop (RR 1.8, Macro Filter)", "cls": ImprovedAsianBreakout, "kwargs": {"target_rr": 1.80, "stop_type": "atr", "use_macro_bias": True}},

    # 2. Supertrend VWAP Variations
    {"code": "ST_V1", "name": "Supertrend VWAP (RR 1.80, ATR >= 5.5p)", "cls": ImprovedSupertrendVWAP, "kwargs": {"target_rr": 1.80, "min_atr_pips": 5.5}},
    {"code": "ST_V2", "name": "Supertrend VWAP (RR 1.65, ATR >= 5.5p)", "cls": ImprovedSupertrendVWAP, "kwargs": {"target_rr": 1.65, "min_atr_pips": 5.5}},
    {"code": "ST_V3", "name": "Supertrend VWAP (RR 2.00, ATR >= 5.5p)", "cls": ImprovedSupertrendVWAP, "kwargs": {"target_rr": 2.00, "min_atr_pips": 5.5}},
]


def run_week_experiment(week_num: int) -> Dict[str, Any]:
    ticks, meta = get_week_dataset(week_num)
    n_ticks = len(ticks)
    engine = BacktestEngine(pip_size=PIP_SIZE, use_slippage=True)

    week_trades = {}
    for item in CANDIDATE_CONFIGS:
        strat = item["cls"](**item["kwargs"])
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
            "profit_factor": 0.0, "net_pips": 0.0, "max_dd_pct": 0.0, "max_recovery_days": 0.0
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

    for t in s_trades:
        r_pnl = (t["pips"] / avg_loss) * (balance * (risk_pct / 100.0))
        balance += r_pnl
        curve.append(balance)
        timestamps.append(t["exit_time"])

    c_arr = np.array(curve)
    peaks = np.maximum.accumulate(c_arr)
    dd_pct = (peaks - c_arr) / peaks * 100.0
    max_dd_pct = float(np.max(dd_pct))

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

    return {
        "total_trades": n_trades,
        "trades_per_week": n_trades / 52.0,
        "win_rate": win_rate,
        "realized_rr": realized_rr,
        "profit_factor": pf,
        "net_pips": net_pips,
        "max_dd_pct": max_dd_pct,
        "max_recovery_days": max_recov_days
    }


def main():
    print("=" * 135)
    print("             CHAMPION IMPROVEMENT LABORATORY: 52 WEEKS FULL TICK AUDIT (~26M TICKS)")
    print("             Refining Asian Breakout (Macro/Midpoint) & Supertrend VWAP (RR Sweeps/ATR)")
    print("=" * 135)

    start_t = time.time()
    weeks = list(range(1, 53))

    strat_trades: Dict[str, List[Dict[str, Any]]] = {item["code"]: [] for item in CANDIDATE_CONFIGS}
    total_ticks = 0

    with ProcessPoolExecutor() as executor:
        results = list(executor.map(run_week_experiment, weeks))

    for res in results:
        total_ticks += res["ticks"]
        for code, t_list in res["trades"].items():
            strat_trades[code].extend(t_list)

    elapsed = time.time() - start_t
    print(f"\n[+] Executed 52 weeks in {elapsed:.1f}s | Total Ticks Audited: {total_ticks:,}\n")

    print(f"{'Code':<8} | {'Configuration':<44} | {'Trades':<6} | {'Trd/Wk':<6} | {'Win %':<6} | {'Real RR':<7} | {'PF':<6} | {'Net Pips':<9} | {'Max DD%':<7} | {'Max Recov':<9}")
    print("-" * 135)

    for item in CANDIDATE_CONFIGS:
        code = item["code"]
        name = item["name"]
        t_list = strat_trades[code]
        metrics = analyze_combination_performance(t_list, risk_pct=0.40)
        print(f"{code:<8} | {name:<44} | {metrics['total_trades']:<6} | {metrics['trades_per_week']:<6.1f} | {metrics['win_rate']:<5.1f}% | {metrics['realized_rr']:<7.2f} | {metrics['profit_factor']:<6.2f} | {metrics['net_pips']:<+9.1f} | {metrics['max_dd_pct']:<6.2f}% | {metrics['max_recovery_days']:<5.1f} days")

    print("=" * 135)


if __name__ == "__main__":
    main()
