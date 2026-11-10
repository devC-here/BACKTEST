"""
Supertrend Dynamic Support/Resistance Pullback Bounce Strategy.
Instead of buying breakouts (which buy at exhaustion in FX),
this buys the PULLBACK BOUNCE off the Supertrend dynamic support line.
Entry:
- Trend is already Bullish (+1)
- Candle pulls back to within 4 pips of the Supertrend line
- A strong bullish rejection candle forms
- Tight SL placed 1.5 pips below the Supertrend line
- Target R:R of 2.0R to 2.5R
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

class SupertrendPullbackBounceStrategy(BaseStrategy):
    def __init__(
        self,
        timeframe: str = "5min",
        atr_period: int = 10,
        multiplier: float = 3.0,
        max_dist_to_st_pips: float = 5.0,
        min_body_ratio: float = 0.55,
        target_rr: float = 2.2,
        sl_buffer_pips: float = 1.5,
        session_filter: bool = True,
        pip_size: float = 0.0001,
        **kwargs
    ):
        self.timeframe = timeframe
        self.atr_period = atr_period
        self.multiplier = multiplier
        self.max_dist_to_st_pips = max_dist_to_st_pips
        self.min_body_ratio = min_body_ratio
        self.target_rr = target_rr
        self.sl_buffer_pips = sl_buffer_pips
        self.session_filter = session_filter
        self.pip_size = pip_size

        super().__init__(**kwargs)
        self.name = f"ST_Bounce_{timeframe}_RR{target_rr}"

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
        if n_c < self.atr_period + 10:
            return signals, sl_pips_arr, tp_pips_arr

        st_line, trend, atr = calculate_supertrend(candles, atr_period=self.atr_period, multiplier=self.multiplier)
        strong_bull, strong_bear = identify_strong_candles(candles, atr_values=atr, min_body_ratio=self.min_body_ratio)

        lows = candles['low'].to_numpy(dtype=np.float64)
        highs = candles['high'].to_numpy(dtype=np.float64)
        closes = candles['close'].to_numpy(dtype=np.float64)
        tick_indices = candles['tick_idx'].to_numpy(dtype=np.int64)
        hours = candles.index.hour.to_numpy()

        for i in range(self.atr_period + 5, n_c):
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

            # We want an established trend (trend was already in place, not a flip)
            if cur_tr == 1 and prev_tr == 1:
                # Bullish Pullback: Low of candle came close to Supertrend line
                dist_pips = (lows[i] - st_val) / self.pip_size
                if 0 <= dist_pips <= self.max_dist_to_st_pips and strong_bull[i]:
                    sl_dist = ((asks[exec_idx] - st_val) / self.pip_size) + self.sl_buffer_pips
                    if 4.0 <= sl_dist <= 15.0:
                        signals[exec_idx] = SIGNAL_LONG
                        sl_pips_arr[exec_idx] = sl_dist
                        tp_pips_arr[exec_idx] = sl_dist * self.target_rr

            elif cur_tr == -1 and prev_tr == -1:
                # Bearish Pullback: High of candle came close to Supertrend line
                dist_pips = (st_val - highs[i]) / self.pip_size
                if 0 <= dist_pips <= self.max_dist_to_st_pips and strong_bear[i]:
                    sl_dist = ((st_val - bids[exec_idx]) / self.pip_size) + self.sl_buffer_pips
                    if 4.0 <= sl_dist <= 15.0:
                        signals[exec_idx] = SIGNAL_SHORT
                        sl_pips_arr[exec_idx] = sl_dist
                        tp_pips_arr[exec_idx] = sl_dist * self.target_rr

        return signals, sl_pips_arr, tp_pips_arr

def test_bounce():
    engine = BacktestEngine(pip_size=0.0001)
    print("Loading 20 weeks for Supertrend Pullback Bounce Suite...")
    weeks = [get_week_dataset(w)[0] for w in range(1, 21)]

    experiments = [
        {"timeframe": "5min",  "max_dist_to_st_pips": 4.0, "target_rr": 2.0, "label": "1. M5 Pullback Bounce (RR 2.0, Dist <= 4p)"},
        {"timeframe": "5min",  "max_dist_to_st_pips": 4.0, "target_rr": 2.4, "label": "2. M5 Pullback Bounce (RR 2.4, Dist <= 4p)"},
        {"timeframe": "5min",  "max_dist_to_st_pips": 6.0, "target_rr": 2.2, "label": "3. M5 Pullback Bounce (RR 2.2, Dist <= 6p)"},
        {"timeframe": "15min", "max_dist_to_st_pips": 6.0, "target_rr": 2.0, "label": "4. M15 Pullback Bounce (RR 2.0, Dist <= 6p)"},
        {"timeframe": "15min", "max_dist_to_st_pips": 8.0, "target_rr": 2.2, "label": "5. M15 Pullback Bounce (RR 2.2, Dist <= 8p)"},
    ]

    print("=" * 92)
    print("           SUPERTREND PULLBACK BOUNCE BENCHMARK (20 WEEKS)           ")
    print("=" * 92)
    print(f"{'Strategy Architecture':42s} | {'Trades':6s} | {'Win Rate':8s} | {'Net Pips':9s} | {'PF':4s} | {'Max DD':7s}")
    print("-" * 92)

    for exp in experiments:
        label = exp.pop("label")
        strat = SupertrendPullbackBounceStrategy(**exp)
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

    print("=" * 92)

if __name__ == "__main__":
    test_bounce()
