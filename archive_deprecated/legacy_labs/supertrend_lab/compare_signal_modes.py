"""
Comparison of Supertrend Signal Modes:
1. Raw Instance Count: How many signals actually occur vs how many were executed.
2. Mode A: Position Holding (1 position at a time until opposite flip)
3. Mode B: Flip-Only Trades (Enter only on the trend change)
4. Mode C: Every Instance as Independent Trade (SL at ST band, TP 1.5R / 2.0R)
5. Mode D: 24h vs Session Filtering
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from supertrend_lab.supertrend_engine import (
    resample_ticks_to_candles,
    calculate_supertrend,
    identify_strong_candles
)
from tick_engine.strategies.base import (
    BaseStrategy,
    SIGNAL_HOLD,
    SIGNAL_LONG,
    SIGNAL_SHORT,
    SIGNAL_FORCE_EXIT
)
from tick_engine.core.engine import BacktestEngine
from tick_engine.core.dataloader import get_week_dataset
from tick_engine.core.metrics import compute_metrics, print_scorecard

class MultiModeSupertrendStrategy(BaseStrategy):
    def __init__(
        self,
        mode: str = "every_instance", # "hold_reversal", "flip_only", "every_instance"
        timeframe: str = "5min",
        target_rr: float = 2.0,
        session_filter: bool = False,
        **kwargs
    ):
        self.mode = mode
        self.timeframe = timeframe
        self.target_rr = target_rr
        self.session_filter = session_filter
        super().__init__(**kwargs)
        self.name = f"ST_{mode}_RR{target_rr}_{'sess' if session_filter else '24h'}"

    def generate_signals(self, ticks: np.ndarray):
        n_ticks = len(ticks)
        signals = np.zeros(n_ticks, dtype=np.int8)
        sl_pips_arr = np.zeros(n_ticks, dtype=np.float64)
        tp_pips_arr = np.zeros(n_ticks, dtype=np.float64)

        if n_ticks < 100:
            return signals, sl_pips_arr, tp_pips_arr

        bids = ticks['bid']
        asks = ticks['ask']

        candles = resample_ticks_to_candles(ticks, timeframe=self.timeframe)
        n_c = len(candles)
        if n_c < 20:
            return signals, sl_pips_arr, tp_pips_arr

        st_line, trend, atr = calculate_supertrend(candles, atr_period=10, multiplier=3.0)
        strong_bull, strong_bear = identify_strong_candles(candles, atr_values=atr, min_body_ratio=0.55, min_atr_factor=0.50)

        tick_indices = candles['tick_idx'].to_numpy(dtype=np.int64)
        hours = candles.index.hour.to_numpy()

        pos = 0

        for i in range(15, n_c):
            t_idx = tick_indices[i]
            if t_idx + 1 >= n_ticks:
                break
            exec_idx = t_idx + 1

            h = hours[i]
            if self.session_filter and (h < 7 or h >= 18):
                continue

            cur_tr = trend[i]
            prev_tr = trend[i - 1]
            st_val = st_line[i]

            is_flip_long = (prev_tr == -1 and cur_tr == 1)
            is_flip_short = (prev_tr == 1 and cur_tr == -1)

            is_bull_instance = (cur_tr == 1 and strong_bull[i])
            is_bear_instance = (cur_tr == -1 and strong_bear[i])

            sl_long = max(((asks[exec_idx] - st_val) / 0.0001) + 1.0, 5.0)
            sl_short = max(((st_val - bids[exec_idx]) / 0.0001) + 1.0, 5.0)

            sl_long = min(sl_long, 25.0)
            sl_short = min(sl_short, 25.0)

            if self.mode == "hold_reversal":
                if pos == 1 and is_bear_instance:
                    signals[exec_idx] = SIGNAL_SHORT
                    sl_pips_arr[exec_idx] = sl_short
                    tp_pips_arr[exec_idx] = 0.0
                    pos = -1
                elif pos == -1 and is_bull_instance:
                    signals[exec_idx] = SIGNAL_LONG
                    sl_pips_arr[exec_idx] = sl_long
                    tp_pips_arr[exec_idx] = 0.0
                    pos = 1
                elif pos == 0:
                    if is_bull_instance:
                        signals[exec_idx] = SIGNAL_LONG
                        sl_pips_arr[exec_idx] = sl_long
                        tp_pips_arr[exec_idx] = 0.0
                        pos = 1
                    elif is_bear_instance:
                        signals[exec_idx] = SIGNAL_SHORT
                        sl_pips_arr[exec_idx] = sl_short
                        tp_pips_arr[exec_idx] = 0.0
                        pos = -1

            elif self.mode == "flip_only":
                # Only take trades on the flip event
                if is_flip_long and strong_bull[i]:
                    signals[exec_idx] = SIGNAL_LONG
                    sl_pips_arr[exec_idx] = sl_long
                    tp_pips_arr[exec_idx] = sl_long * self.target_rr
                elif is_flip_short and strong_bear[i]:
                    signals[exec_idx] = SIGNAL_SHORT
                    sl_pips_arr[exec_idx] = sl_short
                    tp_pips_arr[exec_idx] = sl_short * self.target_rr

            elif self.mode == "every_instance":
                # Treat EVERY occurrence of (ST trend + strong candle) as a trade
                if is_bull_instance:
                    signals[exec_idx] = SIGNAL_LONG
                    sl_pips_arr[exec_idx] = sl_long
                    tp_pips_arr[exec_idx] = sl_long * self.target_rr
                elif is_bear_instance:
                    signals[exec_idx] = SIGNAL_SHORT
                    sl_pips_arr[exec_idx] = sl_short
                    tp_pips_arr[exec_idx] = sl_short * self.target_rr

        return signals, sl_pips_arr, tp_pips_arr

def run_comparison():
    engine = BacktestEngine(pip_size=0.0001)
    print("Loading 10 weeks for Signal Mode Comparison...")
    weeks = [get_week_dataset(w)[0] for w in range(1, 11)]

    experiments = [
        {"mode": "hold_reversal",  "session_filter": True,  "target_rr": 0.0, "label": "1. Hold-to-Reversal (Session 7-18h)"},
        {"mode": "hold_reversal",  "session_filter": False, "target_rr": 0.0, "label": "2. Hold-to-Reversal (24h All Sessions)"},
        {"mode": "flip_only",      "session_filter": False, "target_rr": 2.0, "label": "3. Flip-Only Events (24h, RR 2.0)"},
        {"mode": "flip_only",      "session_filter": True,  "target_rr": 2.0, "label": "4. Flip-Only Events (Session, RR 2.0)"},
        {"mode": "every_instance", "session_filter": False, "target_rr": 1.5, "label": "5. Every Instance 24h (RR 1.5)"},
        {"mode": "every_instance", "session_filter": False, "target_rr": 2.0, "label": "6. Every Instance 24h (RR 2.0)"},
        {"mode": "every_instance", "session_filter": True,  "target_rr": 2.0, "label": "7. Every Instance Session (RR 2.0)"},
    ]

    print("=" * 96)
    print("                SUPERTREND SIGNAL MODES & TRADE FREQUENCY COMPARISON               ")
    print("=" * 96)
    print(f"{'Mode Configuration':42s} | {'Trades':6s} | {'Win Rate':8s} | {'Net Pips':9s} | {'PF':4s} | {'Max DD':7s}")
    print("-" * 96)

    for exp in experiments:
        label = exp.pop("label")
        strat = MultiModeSupertrendStrategy(**exp)
        trades_list = []
        for t in weeks:
            tr = engine.run(t, strat)
            if len(tr) > 0:
                trades_list.append(tr)
        if len(trades_list) > 0:
            all_tr = np.concatenate(trades_list)
            m = compute_metrics(all_tr)
            print(f"{label:42s} | {len(all_tr):6d} | {m['win_rate_pct']:7.1f}% | {m['net_pips']:+8.1f}p | {m['profit_factor']:4.2f} | {m['max_drawdown_pips']:6.1f}p")
        else:
            print(f"{label:42s} | 0 trades")

    print("=" * 96)

if __name__ == "__main__":
    run_comparison()
