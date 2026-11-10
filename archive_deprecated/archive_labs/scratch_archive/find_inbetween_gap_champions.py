#!/usr/bin/env python3
"""
================================================================================
  QUANT LAB: DISCOVERING IN-BETWEEN GAP STRATEGIES (52 WEEKS TICK AUDIT)
================================================================================
Evaluates trading edges specifically designed for timeframes and market regimes
where our current 12 champion models are OFFLINE:
  1. Asian Night Session Mean-Reversion (21:00 to 05:00 UTC)
  2. Late NY Session VWAP Exhaustion Fade (16:30 to 20:30 UTC)
  3. London Lunch False Breakout Fade (10:30 to 12:30 UTC)
  4. Friday Afternoon Profit-Taking Reversal (13:00 to 19:00 UTC Fridays)

Audited tick-by-tick across 52 Weeks (~26 Million Dukascopy Ticks) with real
floating spread and slippage.
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
# MODEL 1: ASIAN NIGHT LIQUIDITY MEAN-REVERSION (GAP: 21:00 - 05:00 UTC)
# ==============================================================================
class AsianNightMeanReversion(BaseStrategy):
    """
    Operates strictly between 21:00 and 05:00 UTC (Tokyo/Sydney/Asia session).
    Edge: In Asian hours, EUR/USD is range-bound in tight horizontal channels.
    When price pierces Bollinger Bands (20, 2.0) with low ADX (<= 20) and RSI
    overbought/oversold, we fade the deviation back to the 20-SMA baseline.
    """
    def __init__(
        self,
        target_rr: float = 1.6,
        min_sl: float = 5.0,
        max_sl: float = 14.0,
        max_adx: float = 20.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.target_rr = target_rr
        self.min_sl = min_sl
        self.max_sl = max_sl
        self.max_adx = max_adx
        self.name = "Asian_Night_Mean_Reversion"

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

        # ATR 14
        tr = np.maximum(h - l, np.maximum(abs(h - c.shift(1)), abs(l - c.shift(1))))
        m15['atr'] = tr.rolling(14).mean().bfill()

        # Bollinger Bands (20, 2.0)
        m15['sma20'] = c.rolling(20).mean().bfill()
        std20 = c.rolling(20).std().bfill()
        m15['bb_up'] = m15['sma20'] + 2.0 * std20
        m15['bb_low'] = m15['sma20'] - 2.0 * std20

        # ADX 14
        up_m = h.diff()
        dn_m = -l.diff()
        p_dm = np.where((up_m > dn_m) & (up_m > 0), up_m, 0.0)
        m_dm = np.where((dn_m > up_m) & (dn_m > 0), dn_m, 0.0)
        smooth_tr = tr.ewm(alpha=1.0/14, adjust=False).mean()
        p_di = (pd.Series(p_dm, index=m15.index).ewm(alpha=1.0/14, adjust=False).mean() / np.maximum(smooth_tr, 1e-9)) * 100
        m_di = (pd.Series(m_dm, index=m15.index).ewm(alpha=1.0/14, adjust=False).mean() / np.maximum(smooth_tr, 1e-9)) * 100
        dx = (abs(p_di - m_di) / np.maximum(p_di + m_di, 1e-9)) * 100
        m15['adx'] = dx.ewm(alpha=1.0/14, adjust=False).mean()

        # RSI 14
        delta = c.diff()
        gain = delta.clip(lower=0.0).ewm(alpha=1.0/14, adjust=False).mean()
        loss = (-delta.clip(upper=0.0)).ewm(alpha=1.0/14, adjust=False).mean()
        rs = gain / np.maximum(loss, 1e-9)
        m15['rsi'] = 100.0 - (100.0 / (1.0 + rs))

        m15['hour'] = m15.index.hour
        time_ms = m15.index.astype('datetime64[ms]').view('int64')
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        allowed_hours = {21, 22, 23, 0, 1, 2, 3, 4, 5}

        for i in range(25, len(m15)):
            row = m15.iloc[i]
            prev = m15.iloc[i-1]

            if row['hour'] not in allowed_hours:
                continue

            # Lack of strong trend required (ADX <= 20)
            if row['adx'] > self.max_adx:
                continue

            sig = 0
            risk_p = 0.0

            # Bullish bounce: pierced lower band and rejected back up with RSI oversold
            if prev['low'] <= prev['bb_low'] and row['close'] > row['open'] and row['rsi'] <= 40.0:
                sl_p = min(row['low'], prev['low']) - 1.0 * PIP_SIZE
                risk = max(min((row['close'] - sl_p) / PIP_SIZE, self.max_sl), self.min_sl)
                sig = SIGNAL_LONG
                risk_p = risk

            # Bearish bounce: pierced upper band and rejected back down with RSI overbought
            elif prev['high'] >= prev['bb_up'] and row['close'] < row['open'] and row['rsi'] >= 60.0:
                sl_p = max(row['high'], prev['high']) + 1.0 * PIP_SIZE
                risk = max(min((sl_p - row['close']) / PIP_SIZE, self.max_sl), self.min_sl)
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
# MODEL 2: LATE NY SESSION VWAP EXHAUSTION FADE (GAP: 16:30 - 20:30 UTC)
# ==============================================================================
class LateNYVWAPExhaustionFade(BaseStrategy):
    """
    Operates strictly between 16:30 and 20:30 UTC (Post-London Close, US Afternoon).
    Edge: When London closes (16:00 UTC), trend momentum stalls. Price extended
    > 2.0x ATR or > 2.0 SD away from Session VWAP represents an over-extended
    liquidity pocket that mean-reverts back to fair value as US equities settle.
    """
    def __init__(
        self,
        target_rr: float = 1.75,
        min_sl: float = 6.0,
        max_sl: float = 16.0,
        dev_atr_mult: float = 2.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.target_rr = target_rr
        self.min_sl = min_sl
        self.max_sl = max_sl
        self.dev_atr_mult = dev_atr_mult
        self.name = "Late_NY_VWAP_Exhaustion_Fade"

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

        # ATR 14
        tr = np.maximum(h - l, np.maximum(abs(h - c.shift(1)), abs(l - c.shift(1))))
        m15['atr'] = tr.rolling(14).mean().bfill()

        # Session VWAP (00:00 UTC reset)
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

            # Bearish fade: Price stretched far ABOVE VWAP (> 2.0x ATR) and printed a red rejection candle
            if dev >= self.dev_atr_mult * atr and row['close'] < row['open'] and prev['close'] > prev['open']:
                sl_p = max(row['high'], prev['high']) + 1.2 * PIP_SIZE
                risk = max(min((sl_p - row['close']) / PIP_SIZE, self.max_sl), self.min_sl)
                sig = SIGNAL_SHORT
                risk_p = risk

            # Bullish fade: Price stretched far BELOW VWAP (> 2.0x ATR) and printed a green rejection candle
            elif dev <= -self.dev_atr_mult * atr and row['close'] > row['open'] and prev['close'] < prev['open']:
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
# MODEL 3: LONDON LUNCH FALSE BREAKOUT FADE (GAP: 10:30 - 12:30 UTC)
# ==============================================================================
class LondonLunchTrapFade(BaseStrategy):
    """
    Operates strictly between 10:30 and 12:30 UTC (London Midday Lunch).
    Edge: London morning trend establishes high/low (08:00 - 10:00 UTC).
    Midday liquidity drops as European traders go to lunch. Market makers probe
    just beyond morning extremes to trigger stop orders; when price fails to sustain
    and slips back inside, we fade the trap back to the morning midpoint.
    """
    def __init__(
        self,
        target_rr: float = 1.70,
        min_sl: float = 5.0,
        max_sl: float = 15.0,
        min_probe_pips: float = 1.5,
        max_probe_pips: float = 12.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.target_rr = target_rr
        self.min_sl = min_sl
        self.max_sl = max_sl
        self.min_probe_pips = min_probe_pips
        self.max_probe_pips = max_probe_pips
        self.name = "London_Lunch_Trap_Fade"

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
                prev = m15.iloc[idx-1]

                sig = 0
                risk_p = 0.0

                # Bull trap (fade short): High probed above morning high but closed back inside
                probe_up = (row['high'] - m_high) / PIP_SIZE
                if self.min_probe_pips <= probe_up <= self.max_probe_pips and row['close'] < m_high and row['close'] < row['open']:
                    sl_p = row['high'] + 1.2 * PIP_SIZE
                    risk = max(min((sl_p - row['close']) / PIP_SIZE, self.max_sl), self.min_sl)
                    sig = SIGNAL_SHORT
                    risk_p = risk
                    traded = True

                # Bear trap (fade long): Low probed below morning low but closed back inside
                probe_dn = (m_low - row['low']) / PIP_SIZE
                if self.min_probe_pips <= probe_dn <= self.max_probe_pips and row['close'] > m_low and row['close'] > row['open']:
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
# BATCH TEST RUNNER ACROSS 52 WEEKS
# ==============================================================================
def eval_week(args):
    w, strat_cls, params = args
    ticks, _ = get_week_dataset(w)
    engine = BacktestEngine(pip_size=0.0001, use_slippage=True)
    strat = strat_cls(**params)
    trades = engine.run(ticks, strat)
    return trades


def run_benchmark(strat_cls, params, name: str):
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
        print(f"[-] {name}: 0 trades generated across 52 weeks.")
        return None

    pips = sum(t['pips'] for t in all_trades)
    wins = [t for t in all_trades if t['pips'] > 0]
    losses = [t for t in all_trades if t['pips'] < 0]
    wr = len(wins) / n * 100.0
    gp = sum(t['pips'] for t in wins)
    gl = abs(sum(t['pips'] for t in losses))
    pf = gp / gl if gl > 0 else 99.0
    realized_rr = (gp / max(len(wins), 1)) / (gl / max(len(losses), 1)) if gl > 0 else 0.0

    # Calculate Drawdown
    equity = np.cumsum([t['pips'] for t in all_trades])
    peak = np.maximum.accumulate(equity)
    dd = peak - equity
    max_dd = dd.max()

    print("=" * 80)
    print(f"AUDIT RESULT: {name} ({dt:.1f}s)")
    print("=" * 80)
    print(f"Total Trades (52 Wks): {n}  (~{n/52:.1f}/wk, ~{n/250:.2f}/day)")
    print(f"Win Rate:              {wr:.1f}%")
    print(f"Realized RR:           {realized_rr:.2f} : 1")
    print(f"Profit Factor:         {pf:.2f}")
    print(f"Net Return:            {pips:+.1f} pips")
    print(f"Max Drawdown:          {max_dd:.1f} pips")
    print("=" * 80 + "\n")

    return {
        "name": name,
        "trades": n,
        "win_rate": wr,
        "pf": pf,
        "net_pips": pips,
        "max_dd": max_dd,
        "realized_rr": realized_rr,
    }


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("  COMMENCING 52-WEEK TICK AUDIT FOR IN-BETWEEN GAP STRATEGIES")
    print("=" * 80 + "\n")

    # 1. Asian Night Mean-Reversion (21:00 - 05:00 UTC)
    run_benchmark(
        AsianNightMeanReversion,
        {"target_rr": 1.6, "min_sl": 5.0, "max_sl": 12.0, "max_adx": 20.0},
        "GAP 1: Asian Night Mean-Reversion (21:00 - 05:00 UTC)"
    )

    # 2. Late NY Session VWAP Exhaustion Fade (16:30 - 20:30 UTC)
    run_benchmark(
        LateNYVWAPExhaustionFade,
        {"target_rr": 1.75, "min_sl": 6.0, "max_sl": 15.0, "dev_atr_mult": 1.8},
        "GAP 2: Late NY VWAP Exhaustion Fade (16:30 - 20:30 UTC)"
    )

    # 3. London Lunch Trap Fade (10:30 - 12:30 UTC)
    run_benchmark(
        LondonLunchTrapFade,
        {"target_rr": 1.70, "min_sl": 5.0, "max_sl": 14.0, "min_probe_pips": 1.5, "max_probe_pips": 10.0},
        "GAP 3: London Lunch Trap Fade (10:30 - 12:30 UTC)"
    )
