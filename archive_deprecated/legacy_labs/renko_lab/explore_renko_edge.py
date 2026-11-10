"""
Quantitative Research: Unlocking Statistical Edge in Renko + SMA + OBV.
Diagnoses why naive Renko loses and tests institutional enhancements:
1. True OBV Signal Line (OBV EMA) to eliminate the tautology flaw
2. Dual Trend Alignment (10 SMA vs 50 SMA)
3. Pullback Bounce Entries (buying near the SMA rather than chasing extended bricks)
4. Session Liquidity Filters (avoiding Asian/Rollover consolidation chop)
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
from tick_engine.core.metrics import compute_metrics


class AdvancedRenkoStrategy(BaseStrategy):
    def __init__(
        self,
        brick_size_pips: float = 6.0,
        sma_fast_period: int = 10,
        sma_slow_period: int = 30,
        obv_ema_period: int = 14,
        mode: str = "pullback_bounce",  # "naive", "obv_signal", "trend_filter", "pullback_bounce"
        target_rr: float = 2.0,
        sl_bricks: float = 2.0,
        session_filter: bool = True,    # Trade only London/NY active hours (07:00 - 18:00 UTC)
        pip_size: float = 0.0001,
        **kwargs
    ):
        self.brick_size_pips = float(brick_size_pips)
        self.fast_period = int(sma_fast_period)
        self.slow_period = int(sma_slow_period)
        self.obv_ema_period = int(obv_ema_period)
        self.mode = mode
        self.target_rr = float(target_rr)
        self.sl_bricks = float(sl_bricks)
        self.session_filter = session_filter
        self.pip_size = float(pip_size)

        super().__init__(**kwargs)
        self.name = f"AdvRenko_{mode}_{int(brick_size_pips)}p"

    def generate_signals(self, ticks: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n_ticks = len(ticks)
        signals = np.zeros(n_ticks, dtype=np.int8)
        sl_pips_arr = np.zeros(n_ticks, dtype=np.float64)
        tp_pips_arr = np.zeros(n_ticks, dtype=np.float64)

        if n_ticks < 100:
            return signals, sl_pips_arr, tp_pips_arr

        bricks = build_renko_bricks(ticks, brick_size_pips=self.brick_size_pips, pip_size=self.pip_size)
        n_b = len(bricks)

        min_req = max(self.slow_period, self.obv_ema_period) + 5
        if n_b < min_req:
            return signals, sl_pips_arr, tp_pips_arr

        closes = np.array([b.close for b in bricks], dtype=np.float64)
        volumes = np.array([b.volume for b in bricks], dtype=np.float64)
        directions = np.array([b.direction for b in bricks], dtype=np.int8)
        ts_arr = np.array([b.timestamp for b in bricks], dtype=np.int64)

        # 1. Moving Averages
        sma_fast = pd.Series(closes).rolling(self.fast_period).mean().to_numpy()
        sma_slow = pd.Series(closes).rolling(self.slow_period).mean().to_numpy()

        # 2. Cumulative OBV and OBV Signal Line
        raw_obv = np.zeros(n_b, dtype=np.float64)
        c_obv = 0.0
        for i in range(n_b):
            c_obv += (volumes[i] if directions[i] == 1 else -volumes[i])
            raw_obv[i] = c_obv

        obv_signal = pd.Series(raw_obv).ewm(span=self.obv_ema_period, adjust=False).mean().to_numpy()

        sl_pips = self.sl_bricks * self.brick_size_pips
        tp_pips = sl_pips * self.target_rr

        # Datetime hours for session filter
        hours = pd.to_datetime(ts_arr, unit='ms').hour

        for i in range(min_req, n_b):
            b = bricks[i]
            tick_idx = b.tick_idx
            if tick_idx >= n_ticks:
                continue

            # Optional session filter: London + NY (07:00 to 18:00 UTC)
            if self.session_filter:
                h = hours[i]
                if h < 7 or h >= 18:
                    continue

            c = closes[i]
            c_prev = closes[i - 1]
            fast_cur = sma_fast[i]
            fast_prev = sma_fast[i - 1]
            slow_cur = sma_slow[i]
            obv_cur = raw_obv[i]
            obv_sig = obv_signal[i]

            # Mode 1: Naive (Baseline)
            if self.mode == "naive":
                long_cond = (b.direction == 1) and (c > fast_cur) and (fast_cur > fast_prev) and (obv_cur > raw_obv[i-1])
                short_cond = (b.direction == -1) and (c < fast_cur) and (fast_cur < fast_prev) and (obv_cur < raw_obv[i-1])

            # Mode 2: OBV Signal Line (Solves the tautology bug)
            elif self.mode == "obv_signal":
                # Demands OBV be strictly above its 14-period EMA
                long_cond = (b.direction == 1) and (c > fast_cur) and (fast_cur > fast_prev) and (obv_cur > obv_sig)
                short_cond = (b.direction == -1) and (c < fast_cur) and (fast_cur < fast_prev) and (obv_cur < obv_sig)

            # Mode 3: Trend Filter (10 SMA vs 30 SMA regime)
            elif self.mode == "trend_filter":
                trend_up = fast_cur > slow_cur
                trend_down = fast_cur < slow_cur
                long_cond = (b.direction == 1) and (c > fast_cur) and trend_up and (obv_cur > obv_sig)
                short_cond = (b.direction == -1) and (c < fast_cur) and trend_down and (obv_cur < obv_sig)

            # Mode 4: Pullback Bounce (Buy only the FIRST green brick after touching SMA)
            elif self.mode == "pullback_bounce":
                trend_up = fast_cur > slow_cur
                trend_down = fast_cur < slow_cur

                # Bullish Bounce: previous brick was red (pullback towards SMA), current brick is green, bouncing off SMA
                bull_bounce = (bricks[i - 1].direction == -1) and (b.direction == 1) and (c > fast_cur)
                # Bearish Bounce: previous brick was green (pullback towards SMA), current brick is red, dropping off SMA
                bear_bounce = (bricks[i - 1].direction == 1) and (b.direction == -1) and (c < fast_cur)

                long_cond = bull_bounce and trend_up and (obv_cur > obv_sig)
                short_cond = bear_bounce and trend_down and (obv_cur < obv_sig)

            else:
                long_cond = False
                short_cond = False

            if long_cond:
                signals[tick_idx] = SIGNAL_LONG
                sl_pips_arr[tick_idx] = sl_pips
                tp_pips_arr[tick_idx] = tp_pips
            elif short_cond:
                signals[tick_idx] = SIGNAL_SHORT
                sl_pips_arr[tick_idx] = sl_pips
                tp_pips_arr[tick_idx] = tp_pips

        return signals, sl_pips_arr, tp_pips_arr

def run_suite():
    engine = BacktestEngine(pip_size=0.0001)
    weeks_data = [get_week_dataset(w)[0] for w in range(1, 10)]

    modes = [
        ("naive", 6.0, 1.5, "1. Naive Renko 6p (Baseline)"),
        ("obv_signal", 6.0, 1.5, "2. + OBV EMA Signal Line Filter"),
        ("trend_filter", 6.0, 1.5, "3. + Dual SMA Trend Filter (10/30)"),
        ("pullback_bounce", 6.0, 1.5, "4. + Pullback Bounce Entry"),
        ("pullback_bounce", 6.0, 2.0, "5. Pullback Bounce (R:R 2.0 Target)"),
        ("pullback_bounce", 8.0, 2.0, "6. Pullback Bounce 8p Brick (R:R 2.0)"),
        ("pullback_bounce", 10.0, 2.2, "7. Pullback Bounce 10p Brick (R:R 2.2)"),
    ]

    print("=" * 90)
    print("        RENKO + SMA + OBV QUANTITATIVE EVOLUTION SUITE (WEEKS 1-9)        ")
    print("=" * 90)
    print(f"{'Evolutionary Step':42s} | {'Trades':6s} | {'WR %':5s} | {'Net Pips':9s} | {'PF':4s} | {'Max DD':7s} | {'Avg':6s}")
    print("-" * 90)

    for mode_id, bs, rr, label in modes:
        strat = AdvancedRenkoStrategy(
            brick_size_pips=bs,
            mode=mode_id,
            target_rr=rr,
            session_filter=True
        )

        all_trades = []
        for ticks in weeks_data:
            trades = engine.run(ticks, strat)
            if len(trades) > 0:
                all_trades.append(trades)

        if len(all_trades) > 0:
            trades_arr = np.concatenate(all_trades)
            m = compute_metrics(trades_arr)
            print(f"{label:42s} | {len(trades_arr):6d} | {m['win_rate_pct']:4.1f}% | {m['net_pips']:+8.1f}p | {m['profit_factor']:4.2f} | {m['max_drawdown_pips']:6.1f}p | {m['avg_trade_pips']:+5.2f}p")
        else:
            print(f"{label:42s} | 0 trades")

    print("=" * 90)

if __name__ == "__main__":
    run_suite()
