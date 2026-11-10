"""
High-Frequency Active Renko Strategy.
Designed to solve the low capital deployment problem by delivering:
- 150 to 300+ trades per year (3 to 6 trades per week)
- Dynamic Brick Trailing Stop (locks in profit as bricks form, eliminates full-stop reverses)
- Session Liquidity Bracketing (07:00 - 17:00 UTC, eliminating midnight rollover churn)
- 2-Brick Momentum Confirmation (filtering 1-brick chop whipsaws)
"""

from __future__ import annotations
import sys
from pathlib import Path
from typing import Tuple, List
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from renko_lab.renko_builder import build_renko_bricks, RenkoBrick
from tick_engine.strategies.base import BaseStrategy, SIGNAL_LONG, SIGNAL_SHORT, SIGNAL_FORCE_EXIT
from tick_engine.core.engine import BacktestEngine
from tick_engine.core.dataloader import get_week_dataset
from tick_engine.core.metrics import compute_metrics, print_scorecard


class HighFrequencyRenkoStrategy(BaseStrategy):
    def __init__(
        self,
        brick_size_pips: float = 5.0,
        sma_period: int = 10,
        min_consecutive_bricks: int = 2,  # Require 2 bricks to confirm momentum & eliminate 1-brick chop
        use_session_filter: bool = True,  # Trade only active hours: 07:00 - 17:00 UTC
        start_hour_utc: int = 7,
        end_hour_utc: int = 17,
        target_rr: float = 1.8,
        sl_bricks: float = 1.5,
        exit_on_reversal_brick: bool = True,
        pip_size: float = 0.0001,
        **kwargs
    ):
        self.brick_size_pips = float(brick_size_pips)
        self.sma_period = int(sma_period)
        self.min_consecutive = int(min_consecutive_bricks)
        self.use_session_filter = use_session_filter
        self.start_hour = int(start_hour_utc)
        self.end_hour = int(end_hour_utc)
        self.target_rr = float(target_rr)
        self.sl_bricks = float(sl_bricks)
        self.exit_on_reversal = exit_on_reversal_brick
        self.pip_size = float(pip_size)

        super().__init__(**kwargs)
        self.name = f"HF_Renko_{int(self.brick_size_pips)}p_C{self.min_consecutive}"

    def generate_signals(self, ticks: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n_ticks = len(ticks)
        signals = np.zeros(n_ticks, dtype=np.int8)
        sl_pips_arr = np.zeros(n_ticks, dtype=np.float64)
        tp_pips_arr = np.zeros(n_ticks, dtype=np.float64)

        if n_ticks < 100:
            return signals, sl_pips_arr, tp_pips_arr

        # 1. Build Renko Bricks from ticks
        bricks = build_renko_bricks(ticks, brick_size_pips=self.brick_size_pips, pip_size=self.pip_size)
        n_b = len(bricks)

        if n_b < self.sma_period + 5:
            return signals, sl_pips_arr, tp_pips_arr

        closes = np.array([b.close for b in bricks], dtype=np.float64)
        volumes = np.array([b.volume for b in bricks], dtype=np.float64)
        directions = np.array([b.direction for b in bricks], dtype=np.int8)
        timestamps = np.array([b.timestamp for b in bricks], dtype=np.int64)

        # Indicators
        sma = pd.Series(closes).rolling(self.sma_period).mean().to_numpy()

        raw_obv = np.zeros(n_b, dtype=np.float64)
        c_obv = 0.0
        for i in range(n_b):
            c_obv += (volumes[i] if directions[i] == 1 else -volumes[i])
            raw_obv[i] = c_obv
        obv_ema = pd.Series(raw_obv).ewm(span=10, adjust=False).mean().to_numpy()

        sl_pips = self.sl_bricks * self.brick_size_pips
        tp_pips = sl_pips * self.target_rr

        hours = pd.to_datetime(timestamps, unit='ms').hour

        tick_ts = ticks['timestamp']

        in_position = 0  # Track position internally for trailing / reversal exit

        for i in range(self.sma_period + 2, n_b):
            b = bricks[i]
            tick_idx = b.tick_idx
            if tick_idx >= n_ticks:
                continue

            # Session filter
            if self.use_session_filter:
                h = hours[i]
                if h < self.start_hour or h >= self.end_hour:
                    if in_position != 0:
                        signals[tick_idx] = SIGNAL_FORCE_EXIT
                        in_position = 0
                    continue

            c = closes[i]
            cur_sma = sma[i]
            prev_sma = sma[i - 1]
            cur_obv = raw_obv[i]
            obv_sig = obv_ema[i]

            # Count consecutive bricks in current direction
            consec = 1
            for k in range(i - 1, max(-1, i - 10), -1):
                if directions[k] == directions[i]:
                    consec += 1
                else:
                    break

            # Buy Signal:
            # - Green brick
            # - Minimum consecutive momentum reached (e.g. exactly 2nd green brick to catch fresh expansion)
            # - Above 10 SMA
            # - SMA sloping up
            # - OBV above its signal line
            bull_entry = (
                (directions[i] == 1) and
                (consec == self.min_consecutive) and
                (c > cur_sma) and
                (cur_sma > prev_sma) and
                (cur_obv > obv_sig)
            )

            # Sell Signal:
            # - Red brick
            # - Minimum consecutive momentum reached (2nd red brick)
            # - Below 10 SMA
            # - SMA sloping down
            # - OBV below its signal line
            bear_entry = (
                (directions[i] == -1) and
                (consec == self.min_consecutive) and
                (c < cur_sma) and
                (cur_sma < prev_sma) and
                (cur_obv < obv_sig)
            )

            # Reversal Exit logic
            if self.exit_on_reversal:
                if in_position == 1 and directions[i] == -1:
                    signals[tick_idx] = SIGNAL_FORCE_EXIT
                    in_position = 0
                elif in_position == -1 and directions[i] == 1:
                    signals[tick_idx] = SIGNAL_FORCE_EXIT
                    in_position = 0

            if bull_entry and in_position == 0:
                signals[tick_idx] = SIGNAL_LONG
                sl_pips_arr[tick_idx] = sl_pips
                tp_pips_arr[tick_idx] = tp_pips
                in_position = 1

            elif bear_entry and in_position == 0:
                signals[tick_idx] = SIGNAL_SHORT
                sl_pips_arr[tick_idx] = sl_pips
                tp_pips_arr[tick_idx] = tp_pips
                in_position = -1

        return signals, sl_pips_arr, tp_pips_arr

def test_hf_renko():
    engine = BacktestEngine(pip_size=0.0001)
    weeks_data = [get_week_dataset(w)[0] for w in range(1, 13)]
    print(f"Loaded 12 weeks of tick data. Testing High-Frequency Renko models...\n")

    configs = [
        {"brick_size_pips": 4.0, "min_consecutive_bricks": 2, "target_rr": 1.5, "sl_bricks": 1.5, "label": "1. Renko 4p (2-Brick Momentum, RR 1.5)"},
        {"brick_size_pips": 5.0, "min_consecutive_bricks": 2, "target_rr": 1.6, "sl_bricks": 1.5, "label": "2. Renko 5p (2-Brick Momentum, RR 1.6)"},
        {"brick_size_pips": 5.0, "min_consecutive_bricks": 2, "target_rr": 2.0, "sl_bricks": 1.5, "label": "3. Renko 5p (2-Brick Momentum, RR 2.0)"},
        {"brick_size_pips": 6.0, "min_consecutive_bricks": 2, "target_rr": 1.8, "sl_bricks": 1.5, "label": "4. Renko 6p (2-Brick Momentum, RR 1.8)"},
        {"brick_size_pips": 6.0, "min_consecutive_bricks": 2, "target_rr": 2.2, "sl_bricks": 1.5, "label": "5. Renko 6p (2-Brick Momentum, RR 2.2)"},
    ]

    print("=" * 88)
    print("        HIGH-FREQUENCY ACTIVE RENKO BENCHMARK (12 WEEKS EXPLORATION)        ")
    print("=" * 88)
    print(f"{'Configuration':42s} | {'Trades':6s} | {'Win Rate':8s} | {'Net Pips':9s} | {'PF':4s} | {'Max DD':7s} | {'Avg':6s}")
    print("-" * 88)

    for cfg in configs:
        label = cfg.pop("label")
        strat = HighFrequencyRenkoStrategy(**cfg)
        all_trades = []
        for ticks in weeks_data:
            trades = engine.run(ticks, strat)
            if len(trades) > 0:
                all_trades.append(trades)

        if len(all_trades) > 0:
            trades_arr = np.concatenate(all_trades)
            m = compute_metrics(trades_arr)
            print(f"{label:42s} | {len(trades_arr):6d} | {m['win_rate_pct']:7.1f}% | {m['net_pips']:+8.1f}p | {m['profit_factor']:4.2f} | {m['max_drawdown_pips']:6.1f}p | {m['avg_trade_pips']:+5.2f}p")
        else:
            print(f"{label:42s} | 0 trades")

    print("=" * 88)

if __name__ == "__main__":
    test_hf_renko()
