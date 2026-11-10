#!/usr/bin/env python3
"""
================================================================================
  LAB: CALIBRATING GAP STRATEGIES TO CHAMPION STATUS (52 WEEKS TICK AUDIT)
================================================================================
Deep dive and optimization into the 4 key in-between gaps:
  1. London Lunch Trap Fade (10:30 - 12:30 UTC) -> Optimize RR & probe thresholds
  2. Late NY Session Exhaustion Fade (16:30 - 20:30 UTC) -> Add RSI filter & dev multiplier
  3. Asian Session Liquidity Fade (01:00 - 05:30 UTC) -> Calibrate channel sweep
  4. Friday Afternoon Weekly Mean-Reversion (12:00 - 18:00 UTC Fridays)
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

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tick_engine.core.engine import BacktestEngine
from tick_engine.core.dataloader import get_week_dataset
from tick_engine.strategies.base import BaseStrategy, SIGNAL_LONG, SIGNAL_SHORT

PIP_SIZE = 0.0001


# ==============================================================================
# CANDIDATE 1: LONDON LUNCH TRAP FADE ELITE (10:30 - 12:30 UTC)
# ==============================================================================
class LondonLunchTrapElite(BaseStrategy):
    """
    Fades midday liquidity traps between 10:30 and 12:30 UTC.
    Morning Range established between 08:00 and 10:00 UTC.
    """
    def __init__(
        self,
        target_rr: float = 1.60,
        min_probe: float = 2.0,
        max_probe: float = 10.0,
        min_sl: float = 5.0,
        max_sl: float = 14.0,
        use_rsi: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.target_rr = target_rr
        self.min_probe = min_probe
        self.max_probe = max_probe
        self.min_sl = min_sl
        self.max_sl = max_sl
        self.use_rsi = use_rsi
        self.name = f"London_Lunch_Trap_RR{target_rr}_Probe{min_probe}_{max_probe}"

    def generate_signals(self, ticks: np.ndarray):
        n_ticks = len(ticks)
        signals = np.zeros(n_ticks, dtype=np.int8)
        sl_pips_arr = np.zeros(n_ticks, dtype=np.float64)
        tp_pips_arr = np.zeros(n_ticks, dtype=np.float64)
        if n_ticks < 500:
            return signals, sl_pips_arr, tp_pips_arr

        ts = pd.to_datetime(ticks['timestamp'], unit='ms')
        mid = 0.5 * (ticks['bid'] + ticks['ask'])
        base = pd.DataFrame({'open': mid, 'high': mid, 'low': mid, 'close': mid}, index=ts)
        m15 = base.resample('15min').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
        if len(m15) < 35:
            return signals, sl_pips_arr, tp_pips_arr

        c = m15['close']
        delta = c.diff()
        gain = delta.clip(lower=0.0).ewm(alpha=1.0/14, adjust=False).mean()
        loss = (-delta.clip(upper=0.0)).ewm(alpha=1.0/14, adjust=False).mean()
        rs = gain / np.maximum(loss, 1e-9)
        m15['rsi'] = 100.0 - (100.0 / (1.0 + rs))

        m15['date'] = m15.index.date
        m15['hour'] = m15.index.hour
        m15['minute'] = m15.index.minute
        time_ms = m15.index.astype('datetime64[ms]').view('int64')
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        dates = m15['date'].unique()
        for d in dates:
            day_mask = m15['date'] == d
            morning_mask = day_mask & (m15['hour'] >= 8) & (m15['hour'] < 10)
            morning_bars = m15[morning_mask]
            if len(morning_bars) < 6:
                continue

            m_high = morning_bars['high'].max()
            m_low = morning_bars['low'].min()
            m_range = (m_high - m_low) / PIP_SIZE
            if m_range < 10.0 or m_range > 45.0:
                continue

            lunch_indices = np.where(day_mask & (
                ((m15['hour'] == 10) & (m15['minute'] >= 30)) |
                (m15['hour'] == 11) |
                ((m15['hour'] == 12) & (m15['minute'] <= 30))
            ))[0]

            traded = False
            for idx in lunch_indices:
                if traded:
                    break
                row = m15.iloc[idx]

                sig = 0
                risk_p = 0.0

                # Bull trap (fade short)
                probe_up = (row['high'] - m_high) / PIP_SIZE
                if self.min_probe <= probe_up <= self.max_probe and row['close'] < m_high and row['close'] < row['open']:
                    if not self.use_rsi or row['rsi'] >= 52.0:
                        sl_p = row['high'] + 1.2 * PIP_SIZE
                        risk = max(min((sl_p - row['close']) / PIP_SIZE, self.max_sl), self.min_sl)
                        sig = SIGNAL_SHORT
                        risk_p = risk
                        traded = True

                # Bear trap (fade long)
                probe_dn = (m_low - row['low']) / PIP_SIZE
                if self.min_probe <= probe_dn <= self.max_probe and row['close'] > m_low and row['close'] > row['open']:
                    if not self.use_rsi or row['rsi'] <= 48.0:
                        sl_p = row['low'] - 1.2 * PIP_SIZE
                        risk = max(min((row['close'] - sl_p) / PIP_SIZE, self.max_sl), self.min_sl)
                        sig = SIGNAL_LONG
                        risk_p = risk
                        traded = True

                if sig != 0:
                    exec_t = time_ms[idx] + dur_ms
                    t_idx = int(np.searchsorted(tick_ts, exec_t))
                    if t_idx < n_ticks and signals[t_idx] == 0:
                        signals[t_idx] = sig
                        sl_pips_arr[t_idx] = risk_p
                        tp_pips_arr[t_idx] = risk_p * self.target_rr

        return signals, sl_pips_arr, tp_pips_arr


# ==============================================================================
# CANDIDATE 2: LATE NY SESSION VWAP EXHAUSTION ELITE (16:30 - 20:30 UTC)
# ==============================================================================
class LateNYVWAPElite(BaseStrategy):
    """
    Fades overextended moves back to Session VWAP after London Close.
    Requires:
    - Distance from VWAP >= dev_atr_mult * ATR
    - RSI extreme (<= 38 for buy, >= 62 for sell)
    """
    def __init__(
        self,
        target_rr: float = 1.60,
        dev_atr_mult: float = 2.2,
        min_sl: float = 6.0,
        max_sl: float = 15.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.target_rr = target_rr
        self.dev_atr_mult = dev_atr_mult
        self.min_sl = min_sl
        self.max_sl = max_sl
        self.name = f"Late_NY_VWAP_Elite_Dev{dev_atr_mult}_RR{target_rr}"

    def generate_signals(self, ticks: np.ndarray):
        n_ticks = len(ticks)
        signals = np.zeros(n_ticks, dtype=np.int8)
        sl_pips_arr = np.zeros(n_ticks, dtype=np.float64)
        tp_pips_arr = np.zeros(n_ticks, dtype=np.float64)
        if n_ticks < 500:
            return signals, sl_pips_arr, tp_pips_arr

        ts = pd.to_datetime(ticks['timestamp'], unit='ms')
        mid = 0.5 * (ticks['bid'] + ticks['ask'])
        base = pd.DataFrame({'open': mid, 'high': mid, 'low': mid, 'close': mid}, index=ts)
        m15 = base.resample('15min').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
        if len(m15) < 35:
            return signals, sl_pips_arr, tp_pips_arr

        c = m15['close']
        h = m15['high']
        l = m15['low']

        tr = np.maximum(h - l, np.maximum(abs(h - c.shift(1)), abs(l - c.shift(1))))
        m15['atr'] = tr.rolling(14).mean().bfill()

        # RSI 14
        delta = c.diff()
        gain = delta.clip(lower=0.0).ewm(alpha=1.0/14, adjust=False).mean()
        loss = (-delta.clip(upper=0.0)).ewm(alpha=1.0/14, adjust=False).mean()
        rs = gain / np.maximum(loss, 1e-9)
        m15['rsi'] = 100.0 - (100.0 / (1.0 + rs))

        # Session VWAP
        m15['date'] = m15.index.date
        typical = (h + l + c) / 3.0
        vol = tr / PIP_SIZE + 10.0
        cum_vol = vol.groupby(m15['date']).cumsum()
        cum_pv = (typical * vol).groupby(m15['date']).cumsum()
        m15['vwap'] = cum_pv / np.maximum(cum_vol, 1e-9)

        m15['hour'] = m15.index.hour
        m15['minute'] = m15.index.minute
        time_ms = m15.index.astype('datetime64[ms]').view('int64')
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        allowed_hours = {16, 17, 18, 19, 20}

        for i in range(25, len(m15)):
            row = m15.iloc[i]
            prev = m15.iloc[i-1]

            if row['hour'] not in allowed_hours:
                continue
            if row['hour'] == 16 and row['minute'] < 30:
                continue
            if row['hour'] == 20 and row['minute'] > 30:
                continue

            vwap = row['vwap']
            atr = row['atr']
            dev = row['close'] - vwap

            sig = 0
            risk_p = 0.0

            # Bearish fade: Price >= +2.2x ATR above VWAP + RSI >= 62 + Red bar
            if dev >= self.dev_atr_mult * atr and row['rsi'] >= 62.0 and row['close'] < row['open']:
                sl_p = max(row['high'], prev['high']) + 1.2 * PIP_SIZE
                risk = max(min((sl_p - row['close']) / PIP_SIZE, self.max_sl), self.min_sl)
                sig = SIGNAL_SHORT
                risk_p = risk

            # Bullish fade: Price <= -2.2x ATR below VWAP + RSI <= 38 + Green bar
            elif dev <= -self.dev_atr_mult * atr and row['rsi'] <= 38.0 and row['close'] > row['open']:
                sl_p = min(row['low'], prev['low']) - 1.2 * PIP_SIZE
                risk = max(min((row['close'] - sl_p) / PIP_SIZE, self.max_sl), self.min_sl)
                sig = SIGNAL_LONG
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
# CANDIDATE 3: ASIAN TOKYO CORE LIQUIDITY FADE (01:00 - 05:30 UTC)
# ==============================================================================
class AsianTokyoLiquidityFade(BaseStrategy):
    """
    Operates strictly between 01:00 and 05:30 UTC (Core Tokyo Session).
    First hour (00:00 - 01:00 UTC) sets Tokyo Opening Box.
    If Tokyo Opening Box is tight (5 - 20 pips), any breakout attempt between
    01:00 and 05:30 UTC that sweeps beyond the box by 1.5 - 8 pips and fails
    is faded back into the box.
    """
    def __init__(
        self,
        target_rr: float = 1.60,
        min_sweep: float = 1.0,
        max_sweep: float = 8.0,
        min_sl: float = 5.0,
        max_sl: float = 12.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.target_rr = target_rr
        self.min_sweep = min_sweep
        self.max_sweep = max_sweep
        self.min_sl = min_sl
        self.max_sl = max_sl
        self.name = f"Asian_Tokyo_Fade_Sweep{min_sweep}_{max_sweep}"

    def generate_signals(self, ticks: np.ndarray):
        n_ticks = len(ticks)
        signals = np.zeros(n_ticks, dtype=np.int8)
        sl_pips_arr = np.zeros(n_ticks, dtype=np.float64)
        tp_pips_arr = np.zeros(n_ticks, dtype=np.float64)
        if n_ticks < 500:
            return signals, sl_pips_arr, tp_pips_arr

        ts = pd.to_datetime(ticks['timestamp'], unit='ms')
        mid = 0.5 * (ticks['bid'] + ticks['ask'])
        base = pd.DataFrame({'open': mid, 'high': mid, 'low': mid, 'close': mid}, index=ts)
        m15 = base.resample('15min').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
        if len(m15) < 35:
            return signals, sl_pips_arr, tp_pips_arr

        m15['date'] = m15.index.date
        m15['hour'] = m15.index.hour
        time_ms = m15.index.astype('datetime64[ms]').view('int64')
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        dates = m15['date'].unique()
        for d in dates:
            day_mask = m15['date'] == d
            tokyo_open_mask = day_mask & (m15['hour'] == 0)
            tokyo_open_bars = m15[tokyo_open_mask]
            if len(tokyo_open_bars) < 3:
                continue

            box_h = tokyo_open_bars['high'].max()
            box_l = tokyo_open_bars['low'].min()
            box_range = (box_h - box_l) / PIP_SIZE
            if box_range < 5.0 or box_range > 20.0:
                continue

            session_indices = np.where(day_mask & (m15['hour'] >= 1) & (m15['hour'] <= 5))[0]

            traded = False
            for idx in session_indices:
                if traded:
                    break
                row = m15.iloc[idx]

                sig = 0
                risk_p = 0.0

                # Swept Tokyo High and rejected back inside
                sweep_h = (row['high'] - box_h) / PIP_SIZE
                if self.min_sweep <= sweep_h <= self.max_sweep and row['close'] < box_h and row['close'] < row['open']:
                    sl_p = row['high'] + 1.0 * PIP_SIZE
                    risk = max(min((sl_p - row['close']) / PIP_SIZE, self.max_sl), self.min_sl)
                    sig = SIGNAL_SHORT
                    risk_p = risk
                    traded = True

                # Swept Tokyo Low and rejected back inside
                sweep_l = (box_l - row['low']) / PIP_SIZE
                if self.min_sweep <= sweep_l <= self.max_sweep and row['close'] > box_l and row['close'] > row['open']:
                    sl_p = row['low'] - 1.0 * PIP_SIZE
                    risk = max(min((row['close'] - sl_p) / PIP_SIZE, self.max_sl), self.min_sl)
                    sig = SIGNAL_LONG
                    risk_p = risk
                    traded = True

                if sig != 0:
                    exec_t = time_ms[idx] + dur_ms
                    t_idx = int(np.searchsorted(tick_ts, exec_t))
                    if t_idx < n_ticks and signals[t_idx] == 0:
                        signals[t_idx] = sig
                        sl_pips_arr[t_idx] = risk_p
                        tp_pips_arr[t_idx] = risk_p * self.target_rr

        return signals, sl_pips_arr, tp_pips_arr


# ==============================================================================
# CANDIDATE 4: FRIDAY AFTERNOON WEEKLY MEAN-REVERSION (13:00 - 18:00 UTC)
# ==============================================================================
class FridayWeeklyMeanReversion(BaseStrategy):
    """
    Operates strictly on Fridays between 13:00 and 18:00 UTC (where trend models are off).
    Fades weekly extreme extensions back towards the weekly midpoint before weekend close.
    """
    def __init__(
        self,
        target_rr: float = 1.60,
        min_sl: float = 6.0,
        max_sl: float = 16.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.target_rr = target_rr
        self.min_sl = min_sl
        self.max_sl = max_sl
        self.name = "Friday_Weekly_Mean_Reversion"

    def generate_signals(self, ticks: np.ndarray):
        n_ticks = len(ticks)
        signals = np.zeros(n_ticks, dtype=np.int8)
        sl_pips_arr = np.zeros(n_ticks, dtype=np.float64)
        tp_pips_arr = np.zeros(n_ticks, dtype=np.float64)
        if n_ticks < 500:
            return signals, sl_pips_arr, tp_pips_arr

        ts = pd.to_datetime(ticks['timestamp'], unit='ms')
        mid = 0.5 * (ticks['bid'] + ticks['ask'])
        base = pd.DataFrame({'open': mid, 'high': mid, 'low': mid, 'close': mid}, index=ts)
        m15 = base.resample('15min').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
        if len(m15) < 35:
            return signals, sl_pips_arr, tp_pips_arr

        m15['weekday'] = m15.index.weekday
        m15['hour'] = m15.index.hour
        time_ms = m15.index.astype('datetime64[ms]').view('int64')
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        # Weekly high and low up to Friday morning
        friday_mask = (m15['weekday'] == 4) & (m15['hour'] >= 13) & (m15['hour'] <= 18)
        friday_indices = np.where(friday_mask)[0]
        if len(friday_indices) == 0:
            return signals, sl_pips_arr, tp_pips_arr

        week_high = m15['high'].iloc[:friday_indices[0]].max()
        week_low = m15['low'].iloc[:friday_indices[0]].min()
        week_mid = 0.5 * (week_high + week_low)
        week_range = (week_high - week_low) / PIP_SIZE

        if week_range < 40.0:  # Need sufficient weekly expansion
            return signals, sl_pips_arr, tp_pips_arr

        traded = False
        for idx in friday_indices:
            if traded:
                break
            row = m15.iloc[idx]

            sig = 0
            risk_p = 0.0

            # Price tested weekly high on Friday afternoon and rejected
            if row['high'] >= week_high - 3.0 * PIP_SIZE and row['close'] < row['open']:
                sl_p = max(row['high'], week_high) + 1.2 * PIP_SIZE
                risk = max(min((sl_p - row['close']) / PIP_SIZE, self.max_sl), self.min_sl)
                sig = SIGNAL_SHORT
                risk_p = risk
                traded = True

            # Price tested weekly low on Friday afternoon and rejected
            elif row['low'] <= week_low + 3.0 * PIP_SIZE and row['close'] > row['open']:
                sl_p = min(row['low'], week_low) - 1.2 * PIP_SIZE
                risk = max(min((row['close'] - sl_p) / PIP_SIZE, self.max_sl), self.min_sl)
                sig = SIGNAL_LONG
                risk_p = risk
                traded = True

            if sig != 0:
                exec_t = time_ms[idx] + dur_ms
                t_idx = int(np.searchsorted(tick_ts, exec_t))
                if t_idx < n_ticks and signals[t_idx] == 0:
                    signals[t_idx] = sig
                    sl_pips_arr[t_idx] = risk_p
                    tp_pips_arr[t_idx] = risk_p * self.target_rr

        return signals, sl_pips_arr, tp_pips_arr


# ==============================================================================
# AUDIT RUNNER
# ==============================================================================
def eval_week(args):
    w, strat_cls, params = args
    ticks, _ = get_week_dataset(w)
    engine = BacktestEngine(pip_size=0.0001, use_slippage=True)
    strat = strat_cls(**params)
    trades = engine.run(ticks, strat)
    return trades


def run_test(strat_cls, params, label):
    t0 = time.time()
    weeks = list(range(1, 53))
    tasks = [(w, strat_cls, params) for w in weeks]
    num_workers = min(8, os.cpu_count() or 4)

    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        results = list(ex.map(eval_week, tasks))

    all_trades = [t for r in results for t in r]
    dt = time.time() - t0

    n = len(all_trades)
    if n == 0:
        print(f"[-] {label}: 0 trades.")
        return None

    pips = sum(t['pips'] for t in all_trades)
    wins = [t for t in all_trades if t['pips'] > 0]
    losses = [t for t in all_trades if t['pips'] < 0]
    wr = len(wins) / n * 100.0
    gp = sum(t['pips'] for t in wins)
    gl = abs(sum(t['pips'] for t in losses))
    pf = gp / gl if gl > 0 else 99.0
    realized_rr = (gp / max(len(wins), 1)) / (gl / max(len(losses), 1)) if gl > 0 else 0.0

    equity = np.cumsum([t['pips'] for t in all_trades])
    peak = np.maximum.accumulate(equity)
    dd = peak - equity
    max_dd = dd.max()

    print("=" * 80)
    print(f"AUDIT: {label} (Time: {dt:.1f}s)")
    print("=" * 80)
    print(f"Trades (52 Wks): {n}  (~{n/52:.1f}/wk, ~{n/250:.2f}/day)")
    print(f"Win Rate:        {wr:.1f}%")
    print(f"Realized RR:     {realized_rr:.2f} : 1")
    print(f"Profit Factor:   {pf:.2f}")
    print(f"Net Pips:        {pips:+.1f} pips")
    print(f"Max Drawdown:    {max_dd:.1f} pips")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    print("\n--- RUNNING CALIBRATION OF IN-BETWEEN GAP STRATEGIES ---\n")

    # 1. London Lunch Trap Fade (10:30 - 12:30 UTC)
    run_test(LondonLunchTrapElite, {"target_rr": 1.60, "min_probe": 2.0, "max_probe": 10.0, "use_rsi": True}, "1. London Lunch Trap Fade (RSI Filter)")
    run_test(LondonLunchTrapElite, {"target_rr": 1.75, "min_probe": 2.0, "max_probe": 10.0, "use_rsi": False}, "2. London Lunch Trap Fade (RR 1.75)")

    # 2. Late NY VWAP Elite (16:30 - 20:30 UTC)
    run_test(LateNYVWAPElite, {"target_rr": 1.60, "dev_atr_mult": 2.2}, "3. Late NY VWAP Exhaustion Elite (2.2x ATR)")
    run_test(LateNYVWAPElite, {"target_rr": 1.80, "dev_atr_mult": 2.5}, "4. Late NY VWAP Exhaustion Elite (2.5x ATR, RR 1.8)")

    # 3. Asian Tokyo Liquidity Fade (01:00 - 05:30 UTC)
    run_test(AsianTokyoLiquidityFade, {"target_rr": 1.60, "min_sweep": 1.0, "max_sweep": 8.0}, "5. Asian Tokyo Opening Box Sweep Fade (01:00 - 05:30 UTC)")

    # 4. Friday Afternoon Mean-Reversion (13:00 - 18:00 UTC)
    run_test(FridayWeeklyMeanReversion, {"target_rr": 1.60}, "6. Friday Afternoon Weekly Extreme Mean-Reversion")
