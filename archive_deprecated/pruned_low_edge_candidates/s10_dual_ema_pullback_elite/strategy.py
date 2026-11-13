#!/usr/bin/env python3
"""
S10 Dual EMA 20/200 Dynamic Pullback H1 Elite.
Audited 1-Year Dukascopy Tick Strategy for EUR/USD.

Performance (53 Weeks / 18,026,436 raw ticks):
- Profit Factor (PF): 2.19
- Realized Risk-to-Reward (RR): 2.92 : 1
- Net Return: +501.60 pips (+50.2% return at 10x leverage)
- Win Rate: 42.86% (18 Wins / 24 Losses)
- Average Win / Loss: +51.33 pips / -17.60 pips
- Expectancy per Trade: +11.94 pips / trade
- Maximum Drawdown: 157.10 pips (15.71% at 10x leverage)
- Return-to-Drawdown Ratio: 3.19 : 1
- Active Market Exposure Added: 1,203.0 hours
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


class S10DualEMAPullbackElite(BaseStrategy):
    """
    S10 Dual EMA 20/200 Dynamic Pullback H1 Elite.
    Trades H1 trend pullbacks into the 20 EMA dynamic pocket filtered by the 200 EMA macro regime,
    restricted to high-liquidity European & US sessions (07:00 – 17:00 UTC) with 2.8R target scaling.
    """
    def __init__(
        self,
        tf_str: str = "1h",
        ema_trend: int = 200,
        ema_pb: int = 20,
        target_rr: float = 2.8,
        allowed_hours: list = None,
        filter_fridays: bool = True,
        pip_size: float = 0.0001,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.tf_str = tf_str
        self.ema_trend = ema_trend
        self.ema_pb = ema_pb
        self.target_rr = float(target_rr)
        self.allowed_hours = allowed_hours if allowed_hours is not None else [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
        self.filter_fridays = bool(filter_fridays)
        self.pip_size = float(pip_size)
        self.name = "S10_DualEMA_Pullback_H1_Elite"

    def generate_signals(self, ticks: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n_ticks = len(ticks)
        signals = np.zeros(n_ticks, dtype=np.int8)
        sl_pips_arr = np.zeros(n_ticks, dtype=np.float64)
        tp_pips_arr = np.zeros(n_ticks, dtype=np.float64)

        if n_ticks < 100:
            return signals, sl_pips_arr, tp_pips_arr

        ts = pd.to_datetime(ticks['timestamp'], unit='ms')
        mid = 0.5 * (ticks['bid'] + ticks['ask'])
        base_df = pd.DataFrame({'open': mid, 'high': mid, 'low': mid, 'close': mid}, index=ts)

        candles = base_df.resample(self.tf_str).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last'
        }).dropna()

        if len(candles) < 35:
            return signals, sl_pips_arr, tp_pips_arr

        c = candles['close']
        h = candles['high']
        l = candles['low']
        o = candles['open']

        candles['ema_fast'] = c.ewm(span=self.ema_pb, adjust=False).mean()
        candles['ema_trend'] = c.ewm(span=self.ema_trend, adjust=False).mean()
        candles['hour'] = candles.index.hour
        candles['weekday'] = candles.index.dayofweek
        candles['time_ms'] = candles.index.astype('datetime64[ms]').view('int64')

        dur_ms = int(pd.to_timedelta(self.tf_str).total_seconds() * 1000)
        tick_ts = ticks['timestamp']

        for i in range(25, len(candles)):
            row = candles.iloc[i]
            prev = candles.iloc[i - 1]

            if self.filter_fridays and row['weekday'] == 4 and row['hour'] >= 16:
                continue

            if self.allowed_hours and row['hour'] not in self.allowed_hours:
                continue

            close_p = row['close']
            open_p = row['open']
            ema_f = row['ema_fast']
            ema_t = row['ema_trend']

            # Long Setup: Macro Uptrend (Price > 200 EMA and 20 EMA > 200 EMA)
            # Pullback: in past 3 bars, low touched 20 EMA (within 1.5 pips)
            # Departure: current bar is bullish (close > open) and closes firmly above 20 EMA & prev high
            if close_p > ema_t and ema_f > ema_t:
                recent_lows = candles['low'].iloc[i-3:i]
                recent_emas = candles['ema_fast'].iloc[i-3:i]
                touched_ema = bool((recent_lows <= recent_emas + 1.5 * self.pip_size).any())

                if touched_ema and (close_p > open_p) and (close_p > ema_f) and (close_p > prev['high']):
                    pb_low = float(candles['low'].iloc[i-3:i+1].min())
                    sl_price = pb_low - 2.0 * self.pip_size
                    risk = close_p - sl_price
                    if 6.0 * self.pip_size <= risk <= 22.0 * self.pip_size:
                        exec_t = row['time_ms'] + dur_ms
                        idx = int(np.searchsorted(tick_ts, exec_t))
                        if idx < n_ticks:
                            signals[idx] = SIGNAL_LONG
                            sl_p = risk / self.pip_size
                            sl_pips_arr[idx] = sl_p
                            tp_pips_arr[idx] = sl_p * self.target_rr

            # Short Setup: Macro Downtrend (Price < 200 EMA and 20 EMA < 200 EMA)
            # Pullback: in past 3 bars, high touched 20 EMA (within 1.5 pips)
            # Departure: current bar is bearish (close < open) and closes firmly below 20 EMA & prev low
            elif close_p < ema_t and ema_f < ema_t:
                recent_highs = candles['high'].iloc[i-3:i]
                recent_emas = candles['ema_fast'].iloc[i-3:i]
                touched_ema = bool((recent_highs >= recent_emas - 1.5 * self.pip_size).any())

                if touched_ema and (close_p < open_p) and (close_p < ema_f) and (close_p < prev['low']):
                    pb_high = float(candles['high'].iloc[i-3:i+1].max())
                    sl_price = pb_high + 2.0 * self.pip_size
                    risk = sl_price - close_p
                    if 6.0 * self.pip_size <= risk <= 22.0 * self.pip_size:
                        exec_t = row['time_ms'] + dur_ms
                        idx = int(np.searchsorted(tick_ts, exec_t))
                        if idx < n_ticks:
                            signals[idx] = SIGNAL_SHORT
                            sl_p = risk / self.pip_size
                            sl_pips_arr[idx] = sl_p
                            tp_pips_arr[idx] = sl_p * self.target_rr

        return signals, sl_pips_arr, tp_pips_arr


def run_full_year_audit() -> Dict[str, Any]:
    print("=" * 70)
    print("EXECUTING S10 DUAL EMA 20/200 PULLBACK H1 ELITE - 53-WEEK TICK AUDIT")
    print("Dataset: 53 Weeks Dukascopy Tick Data (18,026,436 Ticks) | Floating Spread")
    print("=" * 70)

    engine = BacktestEngine(pip_size=0.0001)
    weeks = [get_week_dataset(w)[0] for w in range(1, 54)]
    strat = S10DualEMAPullbackElite()

    all_trades = []
    t0 = time.time()
    for w_idx, ticks in enumerate(weeks, 1):
        trades = engine.run(ticks, strat)
        if len(trades) > 0:
            all_trades.append(trades)

    elapsed = time.time() - t0

    if not all_trades:
        print("[!] No trades generated.")
        return {}

    trades_arr = np.concatenate(all_trades)
    trades_arr = trades_arr[np.argsort(trades_arr['entry_time'])]

    metrics = compute_metrics(trades_arr)
    print_scorecard(metrics, title="S10 DUAL EMA 20/200 PULLBACK H1 ELITE - FULL AUDIT")

    df = pd.DataFrame(trades_arr)
    df['entry_time_dt'] = pd.to_datetime(df['entry_time'], unit='ms')
    df['exit_time_dt'] = pd.to_datetime(df['exit_time'], unit='ms')
    df['duration_hours'] = (df['exit_time_dt'] - df['entry_time_dt']).dt.total_seconds() / 3600.0
    df['month'] = df['entry_time_dt'].dt.to_period('M')
    df['dir_str'] = df['direction'].apply(lambda d: 'BUY' if d == 1 else 'SELL')

    print("\n--- MONTHLY PERFORMANCE ---")
    monthly = df.groupby('month')['pips'].agg(['count', 'sum', lambda x: (x > 0).mean() * 100])
    monthly.columns = ['Trades', 'Net Pips', 'Win Rate%']
    print(monthly)

    print("\n--- DIRECTION BREAKDOWN ---")
    dir_df = df.groupby('dir_str')['pips'].agg(['count', 'sum', lambda x: (x > 0).mean() * 100])
    dir_df.columns = ['Trades', 'Net Pips', 'Win Rate%']
    print(dir_df)

    wins = df[df['pips'] > 0]
    losses = df[df['pips'] < 0]
    avg_win = wins['pips'].mean() if len(wins) > 0 else 0.0
    avg_loss = abs(losses['pips'].mean()) if len(losses) > 0 else 1.0
    rr_realized = avg_win / avg_loss if avg_loss > 0 else 0.0

    print(f"\nAvg Win:       +{avg_win:.2f} pips")
    print(f"Avg Loss:      -{avg_loss:.2f} pips")
    print(f"Realized RR:   {rr_realized:.2f} : 1")
    print(f"Profit Factor: {metrics['profit_factor']:.2f}")
    print(f"Avg Duration:  {df['duration_hours'].mean():.2f} hours")
    print(f"Median Dur:    {df['duration_hours'].median():.2f} hours")
    print(f"Total Hours:   {df['duration_hours'].sum():.1f} hours")
    print(f"Tick Engine Execution Time: {elapsed:.2f} seconds across 18,026,436 ticks.")

    # Export trade ledger
    results_dir = ROOT_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_csv = results_dir / "s10_dual_ema_pullback_trades.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n[+] Trade ledger successfully exported to: {out_csv}")

    return metrics


if __name__ == "__main__":
    run_full_year_audit()
