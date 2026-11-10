#!/usr/bin/env python3
"""
Squeeze Improvement Laboratory:
Systematically optimizing and refining our highest-win-rate models on 52 Weeks (~26M Ticks):

Testing COMB12 (Bollinger Squeeze) and DISC23 (Keltner Squeeze) calibrations:
- Lookback periods: 30, 35, 40
- Squeeze ratios: 1.30, 1.35, 1.40
- Target RR: 1.50, 1.60, 1.75, 1.90
- Squeeze Ensemble: Triggering if EITHER Bollinger Squeeze OR Keltner Squeeze fires!
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


def enrich_squeeze_data(df: pd.DataFrame, base_df: pd.DataFrame) -> pd.DataFrame:
    c = df['close']
    h = df['high']
    l = df['low']
    o = df['open']

    # ATR 14
    tr = np.maximum(h - l, np.maximum(abs(h - c.shift(1)), abs(l - c.shift(1))))
    df['tr'] = tr
    df['atr_14'] = tr.rolling(14).mean().bfill()
    df['atr_pips'] = df['atr_14'] / PIP_SIZE

    # Bollinger Bands (20, 2.0)
    df['sma_20'] = c.rolling(20).mean().bfill()
    std20 = c.rolling(20).std().bfill()
    df['bb_up'] = df['sma_20'] + (2.0 * std20)
    df['bb_low'] = df['sma_20'] - (2.0 * std20)
    df['bb_width'] = (df['bb_up'] - df['bb_low']) / PIP_SIZE
    df['bb_pct_b'] = (c - df['bb_low']) / np.maximum(df['bb_up'] - df['bb_low'], 1e-9)

    # Keltner Channels (20, 1.5 ATR)
    df['ema_20'] = c.ewm(span=20, adjust=False).mean()
    df['kc_up'] = df['ema_20'] + 1.5 * df['atr_14']
    df['kc_low'] = df['ema_20'] - 1.5 * df['atr_14']
    df['kc_width'] = (df['kc_up'] - df['kc_low']) / np.maximum(df['ema_20'], 1e-6)

    # Rolling Mins for Lookbacks
    for lb in [25, 30, 35, 40]:
        df[f'bb_w_min_{lb}'] = df['bb_width'].rolling(lb).min().shift(1).bfill()
        df[f'kc_w_min_{lb}'] = df['kc_width'].rolling(lb).min().shift(1).bfill()

    # Macro Trend (H1 EMA 50)
    h1 = base_df.resample('1h').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
    h1['ema50'] = h1['close'].ewm(span=50, adjust=False).mean()
    h1['macro_bias'] = np.where(h1['close'] > h1['ema50'], 1, -1)
    df['macro_bias'] = h1['macro_bias'].shift(1).reindex(df.index, method='ffill').fillna(0)

    df['hour'] = df.index.hour
    df['date'] = df.index.date
    df['time_ms'] = df.index.astype('datetime64[ms]').view('int64')

    return df


class ParameterizedSqueezeStrategy(BaseStrategy):
    def __init__(
        self,
        mode: str = "bollinger",  # 'bollinger', 'keltner', or 'ensemble'
        lookback: int = 40,
        squeeze_ratio: float = 1.35,
        target_rr: float = 1.60,
        body_atr_ratio: float = 0.80,
        min_sl: float = 6.0,
        max_sl: float = 16.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.mode = mode
        self.lookback = lookback
        self.squeeze_ratio = squeeze_ratio
        self.target_rr = target_rr
        self.body_atr_ratio = body_atr_ratio
        self.min_sl = min_sl
        self.max_sl = max_sl
        self.pip_size = PIP_SIZE
        self.name = f"{mode}_lb{lookback}_sq{squeeze_ratio}_rr{target_rr}"

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

        m15 = enrich_squeeze_data(m15, base_df)
        time_ms = m15['time_ms'].values
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        bb_min_col = f'bb_w_min_{self.lookback}'
        kc_min_col = f'kc_w_min_{self.lookback}'

        for i in range(45, len(m15)):
            row = m15.iloc[i]
            prev = m15.iloc[i-1]

            if row['hour'] < 7 or row['hour'] > 15:
                continue

            sig = 0
            risk_p = 0.0

            # Check Squeeze Condition based on Mode
            if self.mode == "bollinger":
                # Bollinger Squeeze
                was_coiled = any(m15['bb_width'].iloc[i-k] <= self.squeeze_ratio * m15[bb_min_col].iloc[i-k] for k in [1, 2, 3])
                if was_coiled:
                    body_pips = abs(row['close'] - row['open']) / self.pip_size
                    if body_pips >= self.body_atr_ratio * row['atr_pips']:
                        if row['bb_pct_b'] > 1.0 and prev['bb_pct_b'] <= 1.0 and row['close'] > row['open']:
                            sl_p = min(row['low'], prev['low']) - 1.0 * self.pip_size
                            risk = max((row['close'] - sl_p) / self.pip_size, self.min_sl)
                            if risk <= self.max_sl:
                                sig = SIGNAL_LONG
                                risk_p = risk
                        elif row['bb_pct_b'] < 0.0 and prev['bb_pct_b'] >= 0.0 and row['close'] < row['open']:
                            sl_p = max(row['high'], prev['high']) + 1.0 * self.pip_size
                            risk = max((sl_p - row['close']) / self.pip_size, self.min_sl)
                            if risk <= self.max_sl:
                                sig = SIGNAL_SHORT
                                risk_p = risk

            elif self.mode == "keltner":
                # Keltner Squeeze
                was_coiled = any(m15['kc_width'].iloc[i-k] <= self.squeeze_ratio * m15[kc_min_col].iloc[i-k] for k in [1, 2, 3])
                if was_coiled:
                    body_pips = abs(row['close'] - row['open']) / self.pip_size
                    if body_pips >= self.body_atr_ratio * row['atr_pips']:
                        if row['macro_bias'] > 0 and row['close'] > row['kc_up'] and row['close'] > row['open']:
                            sl_p = min(row['low'], row['ema_20']) - 1.0 * self.pip_size
                            risk = max((row['close'] - sl_p) / self.pip_size, self.min_sl)
                            if risk <= self.max_sl:
                                sig = SIGNAL_LONG
                                risk_p = risk
                        elif row['macro_bias'] < 0 and row['close'] < row['kc_low'] and row['close'] < row['open']:
                            sl_p = max(row['high'], row['ema_20']) + 1.0 * self.pip_size
                            risk = max((sl_p - row['close']) / self.pip_size, self.min_sl)
                            if risk <= self.max_sl:
                                sig = SIGNAL_SHORT
                                risk_p = risk

            elif self.mode == "ensemble":
                # Fires if EITHER Bollinger Squeeze OR Keltner Squeeze triggers
                bb_coiled = any(m15['bb_width'].iloc[i-k] <= self.squeeze_ratio * m15[bb_min_col].iloc[i-k] for k in [1, 2, 3])
                kc_coiled = any(m15['kc_width'].iloc[i-k] <= self.squeeze_ratio * m15[kc_min_col].iloc[i-k] for k in [1, 2, 3])

                body_pips = abs(row['close'] - row['open']) / self.pip_size
                if body_pips >= self.body_atr_ratio * row['atr_pips']:
                    # Bullish trigger
                    bb_long = bb_coiled and (row['bb_pct_b'] > 1.0 and prev['bb_pct_b'] <= 1.0 and row['close'] > row['open'])
                    kc_long = kc_coiled and (row['macro_bias'] > 0 and row['close'] > row['kc_up'] and row['close'] > row['open'])

                    bb_short = bb_coiled and (row['bb_pct_b'] < 0.0 and prev['bb_pct_b'] >= 0.0 and row['close'] < row['open'])
                    kc_short = kc_coiled and (row['macro_bias'] < 0 and row['close'] < row['kc_low'] and row['close'] < row['open'])

                    if bb_long or kc_long:
                        sl_p = min(row['low'], prev['low'], row['ema_20']) - 1.0 * self.pip_size
                        risk = max((row['close'] - sl_p) / self.pip_size, self.min_sl)
                        if risk <= self.max_sl:
                            sig = SIGNAL_LONG
                            risk_p = risk
                    elif bb_short or kc_short:
                        sl_p = max(row['high'], prev['high'], row['ema_20']) + 1.0 * self.pip_size
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


SQUEEZE_CONFIGS = [
    # Baseline Champions
    {"code": "BB_BASE", "name": "Bollinger Squeeze Baseline (LB40, RR 1.60)", "cls": ParameterizedSqueezeStrategy, "kwargs": {"mode": "bollinger", "lookback": 40, "squeeze_ratio": 1.35, "target_rr": 1.60, "body_atr_ratio": 0.80}},
    {"code": "KC_BASE", "name": "Keltner Squeeze Baseline (LB30, RR 1.65)", "cls": ParameterizedSqueezeStrategy, "kwargs": {"mode": "keltner", "lookback": 30, "squeeze_ratio": 1.35, "target_rr": 1.65, "body_atr_ratio": 0.60}},

    # Higher RR Tests
    {"code": "BB_RR175", "name": "Bollinger Squeeze (LB40, RR 1.75)", "cls": ParameterizedSqueezeStrategy, "kwargs": {"mode": "bollinger", "lookback": 40, "squeeze_ratio": 1.35, "target_rr": 1.75, "body_atr_ratio": 0.80}},
    {"code": "BB_RR190", "name": "Bollinger Squeeze (LB40, RR 1.90)", "cls": ParameterizedSqueezeStrategy, "kwargs": {"mode": "bollinger", "lookback": 40, "squeeze_ratio": 1.35, "target_rr": 1.90, "body_atr_ratio": 0.80}},

    # Higher Velocity Squeeze (Lookback 30, Squeeze 1.40)
    {"code": "BB_FAST", "name": "Bollinger Squeeze Fast (LB30, Sq1.40, RR1.65)", "cls": ParameterizedSqueezeStrategy, "kwargs": {"mode": "bollinger", "lookback": 30, "squeeze_ratio": 1.40, "target_rr": 1.65, "body_atr_ratio": 0.70}},
    {"code": "KC_FAST", "name": "Keltner Squeeze Fast (LB25, Sq1.40, RR1.65)", "cls": ParameterizedSqueezeStrategy, "kwargs": {"mode": "keltner", "lookback": 25, "squeeze_ratio": 1.40, "target_rr": 1.65, "body_atr_ratio": 0.55}},

    # Squeeze Ensemble (Combined Alpha)
    {"code": "SQ_ENS1", "name": "Squeeze Ensemble Unified (LB35, RR 1.65)", "cls": ParameterizedSqueezeStrategy, "kwargs": {"mode": "ensemble", "lookback": 35, "squeeze_ratio": 1.35, "target_rr": 1.65, "body_atr_ratio": 0.70}},
    {"code": "SQ_ENS2", "name": "Squeeze Ensemble Unified (LB30, RR 1.75)", "cls": ParameterizedSqueezeStrategy, "kwargs": {"mode": "ensemble", "lookback": 30, "squeeze_ratio": 1.40, "target_rr": 1.75, "body_atr_ratio": 0.65}},
]


def run_week_experiment(week_num: int) -> Dict[str, Any]:
    ticks, meta = get_week_dataset(week_num)
    n_ticks = len(ticks)
    engine = BacktestEngine(pip_size=PIP_SIZE, use_slippage=True)

    week_trades = {}
    for item in SQUEEZE_CONFIGS:
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
    print("             SQUEEZE IMPROVEMENT LABORATORY: 52 WEEKS FULL TICK AUDIT (~26M TICKS)")
    print("             Refining Bollinger Squeeze, Keltner Squeeze, and Unified Squeeze Ensemble")
    print("=" * 135)

    start_t = time.time()
    weeks = list(range(1, 53))

    strat_trades: Dict[str, List[Dict[str, Any]]] = {item["code"]: [] for item in SQUEEZE_CONFIGS}
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

    for item in SQUEEZE_CONFIGS:
        code = item["code"]
        name = item["name"]
        t_list = strat_trades[code]
        metrics = analyze_combination_performance(t_list, risk_pct=0.40)
        print(f"{code:<8} | {name:<44} | {metrics['total_trades']:<6} | {metrics['trades_per_week']:<6.1f} | {metrics['win_rate']:<5.1f}% | {metrics['realized_rr']:<7.2f} | {metrics['profit_factor']:<6.2f} | {metrics['net_pips']:<+9.1f} | {metrics['max_dd_pct']:<6.2f}% | {metrics['max_recovery_days']:<5.1f} days")

    print("=" * 135)


if __name__ == "__main__":
    main()
