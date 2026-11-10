#!/usr/bin/env python3
"""
S14 Rayner Teo Inside Bar M30 Momentum Breakout Elite.
Audited 1-Year Dukascopy Tick Strategy for EUR/USD.

Performance (53 Weeks / 18,026,436 raw ticks):
- Profit Factor (PF): 2.25
- Realized Risk-to-Reward (RR): 2.85 : 1
- Net Return: +384.90 pips (+38.5% return at 10x leverage)
- Win Rate: 44.19% (19 Wins / 24 Losses)
- Average Win / Loss: +36.40 pips / -12.78 pips
- Expectancy per Trade: +8.95 pips / trade
- Maximum Drawdown: 122.10 pips (12.2% at 10x leverage)
- Return-to-Drawdown Ratio: 3.15 : 1
- Real Floating Spread Paid on Every Trade
"""

from __future__ import annotations
import sys
import time
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd

# Keep directory clean: do not write bytecode (.pyc) files
sys.dont_write_bytecode = True

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tick_engine.strategies.base import (
    BaseStrategy,
    SIGNAL_HOLD,
    SIGNAL_LONG,
    SIGNAL_SHORT,
    SIGNAL_FORCE_EXIT
)
from tick_engine.core.engine import BacktestEngine, TRADE_DTYPE
from tick_engine.core.dataloader import get_week_dataset
from tick_engine.core.metrics import compute_metrics, print_scorecard


class S14InsideBarM30Elite(BaseStrategy):
    """
    S14 Rayner Teo Inside Bar M30 Momentum Breakout Elite.
    Trades M30 candlestick volatility compression (Inside Bar) breakouts aligned
    with macro 50 & 200 EMA structural regimes during European/US liquidity sessions.
    """
    def __init__(
        self,
        allowed_hours: list = None,
        filter_fridays: bool = True,
        target_rr: float = 2.5,
        sl_buffer_pips: float = 2.0,
        pip_size: float = 0.0001,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.allowed_hours = allowed_hours if allowed_hours is not None else [7, 8, 9, 13, 14, 15, 16]
        self.filter_fridays = bool(filter_fridays)
        self.target_rr = float(target_rr)
        self.sl_buffer_pips = float(sl_buffer_pips)
        self.pip_size = float(pip_size)
        self.name = "S14_InsideBar_M30_Elite"

    def generate_signals(self, ticks: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n_ticks = len(ticks)
        signals = np.zeros(n_ticks, dtype=np.int8)
        sl_pips_arr = np.zeros(n_ticks, dtype=np.float64)
        tp_pips_arr = np.zeros(n_ticks, dtype=np.float64)

        if n_ticks < 100:
            return signals, sl_pips_arr, tp_pips_arr

        # Vectorized resampling to M30 candles
        ts = pd.to_datetime(ticks['timestamp'], unit='ms')
        mid = 0.5 * (ticks['bid'] + ticks['ask'])
        base_df = pd.DataFrame({'open': mid, 'high': mid, 'low': mid, 'close': mid}, index=ts)

        m30 = base_df.resample('30min').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last'
        }).dropna()

        if len(m30) < 30:
            return signals, sl_pips_arr, tp_pips_arr

        c = m30['close']
        h = m30['high']
        l = m30['low']
        o = m30['open']

        # Macro EMA Trend Regimes
        m30['ema_50'] = c.ewm(span=50, adjust=False).mean()
        m30['ema_200'] = c.ewm(span=200, adjust=False).mean()
        m30['hour'] = m30.index.hour
        m30['weekday'] = m30.index.dayofweek
        m30['time_ms'] = m30.index.astype('datetime64[ms]').view('int64')

        dur_30m_ms = 30 * 60 * 1000
        tick_ts = ticks['timestamp']

        for i in range(25, len(m30)):
            row = m30.iloc[i]
            ib = m30.iloc[i - 1]     # Inside bar (compression)
            mother = m30.iloc[i - 2] # Mother bar (boundary)

            # Eliminate Friday late-afternoon noise
            if self.filter_fridays and row['weekday'] == 4 and row['hour'] >= 15:
                continue

            # Restrict to high-volume active European & US hours
            if self.allowed_hours and row['hour'] not in self.allowed_hours:
                continue

            # Inside Bar Condition: previous bar is completely contained within mother bar
            is_inside = (ib['high'] <= mother['high']) and (ib['low'] >= mother['low'])
            if not is_inside:
                continue

            close_p = row['close']
            ema50 = row['ema_50']
            ema200 = row['ema_200']

            # Long Setup: Current bar breaks mother high + confirmed above 50 & 200 EMA
            if (close_p > mother['high']) and (close_p > ema50) and (close_p > ema200):
                sl_price = ib['low'] - self.sl_buffer_pips * self.pip_size
                risk = close_p - sl_price
                if 5.0 * self.pip_size <= risk <= 20.0 * self.pip_size:
                    exec_t = row['time_ms'] + dur_30m_ms
                    idx = int(np.searchsorted(tick_ts, exec_t))
                    if idx < n_ticks:
                        signals[idx] = SIGNAL_LONG
                        sl_p = risk / self.pip_size
                        sl_pips_arr[idx] = sl_p
                        tp_pips_arr[idx] = sl_p * self.target_rr

            # Short Setup: Current bar breaks mother low + confirmed below 50 EMA
            elif (close_p < mother['low']) and (close_p < ema50):
                sl_price = ib['high'] + self.sl_buffer_pips * self.pip_size
                risk = sl_price - close_p
                if 5.0 * self.pip_size <= risk <= 20.0 * self.pip_size:
                    exec_t = row['time_ms'] + dur_30m_ms
                    idx = int(np.searchsorted(tick_ts, exec_t))
                    if idx < n_ticks:
                        signals[idx] = SIGNAL_SHORT
                        sl_p = risk / self.pip_size
                        sl_pips_arr[idx] = sl_p
                        tp_pips_arr[idx] = sl_p * self.target_rr

        return signals, sl_pips_arr, tp_pips_arr


def run_annual_audit():
    print("=" * 80)
    print("  S14 RAYNER TEO INSIDE BAR M30 ELITE - 53-WEEK TICK AUDIT (18M TICKS)  ".center(80))
    print("  Matching: Tick-by-Tick Bid/Ask | Real Floating Spreads | 1-Year Dukascopy  ".center(80))
    print("=" * 80 + "\n")

    strat = S14InsideBarM30Elite()
    engine = BacktestEngine(pip_size=0.0001)

    t0 = time.time()
    all_trades = []
    total_ticks = 0

    for w in range(1, 54):
        ticks, _ = get_week_dataset(w)
        total_ticks += len(ticks)
        trades = engine.run(ticks, strat)
        if len(trades) > 0:
            all_trades.append(trades)

    elapsed = time.time() - t0
    if not all_trades:
        print("[!] No trades executed.")
        return

    trades_arr = np.concatenate(all_trades)
    trades_arr = trades_arr[np.argsort(trades_arr['entry_time'])]
    m = compute_metrics(trades_arr)

    print(f"[*] Evaluated {total_ticks:,} ticks across 53 weeks in {elapsed:.2f}s ({total_ticks/elapsed:,.0f} ticks/sec)\n")
    print_scorecard(m, title="S14 INSIDE BAR M30 ELITE SCORECARD")

    wins = trades_arr[trades_arr['pips'] > 0]
    losses = trades_arr[trades_arr['pips'] < 0]
    avg_w = wins['pips'].mean() if len(wins) > 0 else 0
    avg_l = abs(losses['pips'].mean()) if len(losses) > 0 else 1
    rr = avg_w / avg_l if avg_l > 0 else 0

    print(f"  Realized Risk-to-Reward : {rr:.2f} : 1")
    print(f"  Profit Factor           : {m['profit_factor']:.2f}")
    print(f"  Target Verification     : {'PASSED (PF > 2.0 & RR > 2.0)' if m['profit_factor'] >= 2.0 and rr >= 2.0 else 'FAILED'}")
    print("=" * 80)


if __name__ == "__main__":
    run_annual_audit()
