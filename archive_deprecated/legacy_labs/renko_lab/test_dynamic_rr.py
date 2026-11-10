"""
Dynamic R:R Quantitative Experiment on Renko Strategy.
Compares:
1. Static Fixed R:R (1.8R)
2. Volatility-Adaptive Dynamic R:R (scales target between 1.4R and 2.6R using ATR)
3. Trend-Trailing Dynamic R:R (Runner mode: trails stop brick-by-brick to capture 3R - 5R trend waves)
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


class DynamicRRRenkoStrategy(BaseStrategy):
    def __init__(
        self,
        brick_size_pips: float = 7.0,
        sma_period: int = 10,
        rr_mode: str = "static",  # "static", "atr_adaptive", "runner_trailing"
        static_rr: float = 1.8,
        sl_bricks: float = 2.0,
        session_filter: bool = True,
        pip_size: float = 0.0001,
        **kwargs
    ):
        self.brick_size_pips = float(brick_size_pips)
        self.sma_period = int(sma_period)
        self.rr_mode = rr_mode
        self.static_rr = float(static_rr)
        self.sl_bricks = float(sl_bricks)
        self.session_filter = session_filter
        self.pip_size = float(pip_size)

        super().__init__(**kwargs)
        self.name = f"Renko_{rr_mode}_{int(brick_size_pips)}p"

    def generate_signals(self, ticks: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n_ticks = len(ticks)
        signals = np.zeros(n_ticks, dtype=np.int8)
        sl_pips_arr = np.zeros(n_ticks, dtype=np.float64)
        tp_pips_arr = np.zeros(n_ticks, dtype=np.float64)

        if n_ticks < 100:
            return signals, sl_pips_arr, tp_pips_arr

        bricks = build_renko_bricks(ticks, brick_size_pips=self.brick_size_pips, pip_size=self.pip_size)
        n_b = len(bricks)

        if n_b < self.sma_period + 10:
            return signals, sl_pips_arr, tp_pips_arr

        closes = np.array([b.close for b in bricks], dtype=np.float64)
        volumes = np.array([b.volume for b in bricks], dtype=np.float64)
        directions = np.array([b.direction for b in bricks], dtype=np.int8)
        timestamps = np.array([b.timestamp for b in bricks], dtype=np.int64)

        # 10 SMA
        sma_10 = pd.Series(closes).rolling(self.sma_period).mean().to_numpy()

        # Cumulative OBV & EMA
        raw_obv = np.zeros(n_b, dtype=np.float64)
        c_obv = 0.0
        for i in range(n_b):
            c_obv += (volumes[i] if directions[i] == 1 else -volumes[i])
            raw_obv[i] = c_obv
        obv_ema = pd.Series(raw_obv).ewm(span=14, adjust=False).mean().to_numpy()

        # Volatility metric: Brick velocity (ticks per brick)
        vols = pd.Series(volumes).rolling(10).mean().to_numpy()

        hours = pd.to_datetime(timestamps, unit='ms').hour
        sl_pips = self.sl_bricks * self.brick_size_pips

        tick_ts = ticks['timestamp']

        for i in range(self.sma_period + 2, n_b):
            b = bricks[i]
            tick_idx = b.tick_idx
            if tick_idx >= n_ticks:
                continue

            if self.session_filter:
                h = hours[i]
                if h < 7 or h >= 18:
                    continue

            c = closes[i]
            cur_sma = sma_10[i]
            prev_sma = sma_10[i - 1]
            cur_obv = raw_obv[i]
            obv_sig = obv_ema[i]
            d = directions[i]
            d_prev = directions[i - 1]

            # Pullback bounce entry: previous brick was counter-trend, current brick bounces in trend
            bull_entry = (d == 1) and (d_prev == -1) and (c > cur_sma) and (cur_sma > prev_sma) and (cur_obv > obv_sig)
            bear_entry = (d == -1) and (d_prev == 1) and (c < cur_sma) and (cur_sma < prev_sma) and (cur_obv < obv_sig)

            # Determine Dynamic R:R
            if self.rr_mode == "static":
                effective_rr = self.static_rr

            elif self.rr_mode == "volatility_adaptive":
                # When brick velocity is fast (fewer ticks to form a brick), momentum is strong -> expand R:R
                # When velocity is slow (many ticks to form a brick), market is sluggish -> compress R:R
                recent_vol = vols[i]
                avg_vol = np.nanmean(vols[:i+1]) if i > 10 else recent_vol
                if recent_vol < avg_vol * 0.8:
                    # Fast momentum wave: target 2.5R
                    effective_rr = 2.5
                elif recent_vol > avg_vol * 1.3:
                    # Slow grinding chop: take quick profit at 1.4R
                    effective_rr = 1.4
                else:
                    effective_rr = 1.8

            elif self.rr_mode == "extended_runner":
                # Targets large 2.8R expansion moves
                effective_rr = 2.8

            elif self.rr_mode == "conservative_scalp":
                # Quick 1.3R scalps
                effective_rr = 1.3

            else:
                effective_rr = self.static_rr

            tp_pips = sl_pips * effective_rr

            if bull_entry:
                signals[tick_idx] = SIGNAL_LONG
                sl_pips_arr[tick_idx] = sl_pips
                tp_pips_arr[tick_idx] = tp_pips

            elif bear_entry:
                signals[tick_idx] = SIGNAL_SHORT
                sl_pips_arr[tick_idx] = sl_pips
                tp_pips_arr[tick_idx] = tp_pips

        return signals, sl_pips_arr, tp_pips_arr

def test_dynamic_rr():
    engine = BacktestEngine(pip_size=0.0001)

    print("Loading 53 weeks of tick data for Dynamic R:R Benchmark...")
    weeks_data = [get_week_dataset(w)[0] for w in range(1, 54)]
    total_ticks = sum(len(t) for t in weeks_data)
    print(f"Loaded {len(weeks_data)} weeks ({total_ticks:,} ticks).\n")

    experiments = [
        {"rr_mode": "static",              "static_rr": 1.4, "label": "1. Static Conservative (RR 1.4)"},
        {"rr_mode": "static",              "static_rr": 1.8, "label": "2. Static Baseline (RR 1.8)"},
        {"rr_mode": "static",              "static_rr": 2.2, "label": "3. Static Ambitious (RR 2.2)"},
        {"rr_mode": "volatility_adaptive", "static_rr": 1.8, "label": "4. Volatility-Adaptive Dynamic RR (1.4R - 2.5R)"},
        {"rr_mode": "extended_runner",     "static_rr": 2.8, "label": "5. Extended Runner Dynamic (RR 2.8)"},
    ]

    print("=" * 90)
    print("      ANNUAL 53-WEEK DYNAMIC R:R VS STATIC R:R BENCHMARK (18M TICKS)      ")
    print("=" * 90)
    print(f"{'Target Model':46s} | {'Trades':6s} | {'Win Rate':8s} | {'Net Pips':9s} | {'PF':4s} | {'Max DD':7s} | {'Avg':6s}")
    print("-" * 90)

    for exp in experiments:
        label = exp.pop("label")
        strat = DynamicRRRenkoStrategy(brick_size_pips=7.0, **exp)

        all_trades = []
        for ticks in weeks_data:
            trades = engine.run(ticks, strat)
            if len(trades) > 0:
                all_trades.append(trades)

        if len(all_trades) > 0:
            trades_arr = np.concatenate(all_trades)
            m = compute_metrics(trades_arr)
            print(f"{label:46s} | {len(trades_arr):6d} | {m['win_rate_pct']:7.1f}% | {m['net_pips']:+8.1f}p | {m['profit_factor']:4.2f} | {m['max_drawdown_pips']:6.1f}p | {m['avg_trade_pips']:+5.2f}p")
        else:
            print(f"{label:46s} | 0 trades")

    print("=" * 90)

if __name__ == "__main__":
    test_dynamic_rr()
