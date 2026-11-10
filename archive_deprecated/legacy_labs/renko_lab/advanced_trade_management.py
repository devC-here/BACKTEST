"""
Advanced Trade Management for Renko Strategies.
Tests:
1. Baseline Fixed SL/TP
2. Reversal Exit (Close Long on Red Brick, Close Short on Green Brick)
3. Breakeven Protection (Move SL to +1 pip once +1 brick in profit)
4. Dynamic Brick Trailing Stop (Trail SL behind previous brick low/high)
5. SMA-10 Trailing Stop (Trail SL behind the 10 SMA line)
6. Combined Institutional Renko (BE + 10 SMA Trail + Macro 10/30 Trend)
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from renko_lab.renko_builder import build_renko_bricks
from tick_engine.core.engine import (
    BacktestEngine, SIGNAL_HOLD, SIGNAL_LONG, SIGNAL_SHORT, SIGNAL_FORCE_EXIT
)
from tick_engine.core.dataloader import get_week_dataset
from tick_engine.core.metrics import compute_metrics, print_scorecard

def simulate_renko_managed_signals(
    ticks: np.ndarray,
    brick_size_pips: float = 7.0,
    sl_bricks: float = 1.5,
    target_rr: float = 2.0,
    exit_mode: str = "fixed",     # "fixed", "color_reversal", "breakeven", "brick_trail", "sma_trail", "combined"
    obv_ema_period: int = 14,
    pip_size: float = 0.0001
):
    """
    Simulates signals and dynamic exits on tick data for Renko bricks.
    """
    n_ticks = len(ticks)
    ts = ticks['timestamp']
    bids = ticks['bid']
    asks = ticks['ask']

    bricks = build_renko_bricks(ticks, brick_size_pips=brick_size_pips, pip_size=pip_size)
    n_b = len(bricks)

    signals = np.zeros(n_ticks, dtype=np.int8)
    sl_pips_arr = np.zeros(n_ticks, dtype=np.float64)
    tp_pips_arr = np.zeros(n_ticks, dtype=np.float64)

    min_req = 35
    if n_b < min_req:
        return signals, sl_pips_arr, tp_pips_arr

    closes = np.array([b.close for b in bricks], dtype=np.float64)
    volumes = np.array([b.volume for b in bricks], dtype=np.float64)
    directions = np.array([b.direction for b in bricks], dtype=np.int8)
    brick_ticks = np.array([b.tick_idx for b in bricks], dtype=np.int64)
    brick_ts = np.array([b.timestamp for b in bricks], dtype=np.int64)

    # 1. 10 SMA and 30 SMA
    sma_fast = pd.Series(closes).rolling(10).mean().to_numpy()
    sma_slow = pd.Series(closes).rolling(30).mean().to_numpy()

    # 2. OBV and OBV Signal EMA
    raw_obv = np.zeros(n_b, dtype=np.float64)
    c_obv = 0.0
    for i in range(n_b):
        c_obv += (volumes[i] if directions[i] == 1 else -volumes[i])
        raw_obv[i] = c_obv
    obv_ema = pd.Series(raw_obv).ewm(span=obv_ema_period, adjust=False).mean().to_numpy()

    # Session hours
    hours = pd.to_datetime(brick_ts, unit='ms').hour

    sl_dist_pips = sl_bricks * brick_size_pips
    tp_dist_pips = sl_dist_pips * target_rr if target_rr > 0 else 999.0

    # Simulation state machine
    pos = 0          # 0 = Flat, 1 = Long, -1 = Short
    entry_p = 0.0
    entry_brick_idx = 0
    cur_sl_price = 0.0
    cur_tp_price = 0.0
    be_active = False

    for i in range(min_req, n_b):
        b = bricks[i]
        t_idx = b.tick_idx
        if t_idx >= n_ticks:
            continue

        c = closes[i]
        c_prev = closes[i - 1]
        fast = sma_fast[i]
        slow = sma_slow[i]
        obv = raw_obv[i]
        obv_sig = obv_ema[i]
        d = directions[i]
        d_prev = directions[i - 1]
        h = hours[i]

        in_session = (7 <= h < 18)

        # A. Check active position management at this brick
        if pos == 1:
            # 1. Fixed TP / SL handled by engine, but we check dynamic rules:
            if exit_mode == "color_reversal":
                if d == -1:  # Red brick formed
                    signals[t_idx] = SIGNAL_FORCE_EXIT
                    pos = 0
                    continue

            elif exit_mode == "breakeven":
                # If brick moved +1 brick higher than entry, lock BE
                if c >= entry_p + (brick_size_pips * pip_size):
                    be_active = True
                if be_active and c <= entry_p + (0.5 * pip_size):
                    signals[t_idx] = SIGNAL_FORCE_EXIT
                    pos = 0
                    continue

            elif exit_mode == "brick_trail":
                # Trailing stop behind previous brick open
                trail_stop = closes[i - 1] - (brick_size_pips * pip_size)
                if c <= trail_stop:
                    signals[t_idx] = SIGNAL_FORCE_EXIT
                    pos = 0
                    continue

            elif exit_mode == "sma_trail":
                # Trailing stop behind 10 SMA
                if c < fast:
                    signals[t_idx] = SIGNAL_FORCE_EXIT
                    pos = 0
                    continue

            elif exit_mode == "combined":
                # 1. Breakeven after +1 brick
                if c >= entry_p + (brick_size_pips * pip_size):
                    be_active = True
                if be_active and c <= entry_p + (0.5 * pip_size):
                    signals[t_idx] = SIGNAL_FORCE_EXIT
                    pos = 0
                    continue
                # 2. Exit on 2 consecutive red bricks or close below 10 SMA
                if (d == -1 and d_prev == -1) or (c < fast):
                    signals[t_idx] = SIGNAL_FORCE_EXIT
                    pos = 0
                    continue

        elif pos == -1:
            if exit_mode == "color_reversal":
                if d == 1:  # Green brick formed
                    signals[t_idx] = SIGNAL_FORCE_EXIT
                    pos = 0
                    continue

            elif exit_mode == "breakeven":
                if c <= entry_p - (brick_size_pips * pip_size):
                    be_active = True
                if be_active and c >= entry_p - (0.5 * pip_size):
                    signals[t_idx] = SIGNAL_FORCE_EXIT
                    pos = 0
                    continue

            elif exit_mode == "brick_trail":
                trail_stop = closes[i - 1] + (brick_size_pips * pip_size)
                if c >= trail_stop:
                    signals[t_idx] = SIGNAL_FORCE_EXIT
                    pos = 0
                    continue

            elif exit_mode == "sma_trail":
                if c > fast:
                    signals[t_idx] = SIGNAL_FORCE_EXIT
                    pos = 0
                    continue

            elif exit_mode == "combined":
                if c <= entry_p - (brick_size_pips * pip_size):
                    be_active = True
                if be_active and c >= entry_p - (0.5 * pip_size):
                    signals[t_idx] = SIGNAL_FORCE_EXIT
                    pos = 0
                    continue
                if (d == 1 and d_prev == 1) or (c > fast):
                    signals[t_idx] = SIGNAL_FORCE_EXIT
                    pos = 0
                    continue

        # B. Check Entry Signal if Flat
        if pos == 0 and in_session:
            trend_up = fast > slow
            trend_down = fast < slow

            # Pullback bounce: counter-trend brick followed by bounce brick
            bull_bounce = (d_prev == -1) and (d == 1) and (c > fast) and trend_up and (obv > obv_sig)
            bear_bounce = (d_prev == 1) and (d == -1) and (c < fast) and trend_down and (obv < obv_sig)

            if bull_bounce:
                signals[t_idx] = SIGNAL_LONG
                sl_pips_arr[t_idx] = sl_dist_pips
                tp_pips_arr[t_idx] = tp_dist_pips if target_rr > 0 else 0.0
                pos = 1
                entry_p = asks[t_idx]
                entry_brick_idx = i
                be_active = False

            elif bear_bounce:
                signals[t_idx] = SIGNAL_SHORT
                sl_pips_arr[t_idx] = sl_dist_pips
                tp_pips_arr[t_idx] = tp_dist_pips if target_rr > 0 else 0.0
                pos = -1
                entry_p = bids[t_idx]
                entry_brick_idx = i
                be_active = False

    return signals, sl_pips_arr, tp_pips_arr

class ManagedRenkoWrapper:
    def __init__(self, brick_size_pips=7.0, target_rr=2.0, exit_mode="fixed", sl_bricks=1.5):
        self.brick_size_pips = brick_size_pips
        self.target_rr = target_rr
        self.exit_mode = exit_mode
        self.sl_bricks = sl_bricks

    def generate_signals(self, ticks: np.ndarray):
        return simulate_renko_managed_signals(
            ticks,
            brick_size_pips=self.brick_size_pips,
            sl_bricks=self.sl_bricks,
            target_rr=self.target_rr,
            exit_mode=self.exit_mode
        )

def run_tests():
    engine = BacktestEngine(pip_size=0.0001)
    print("Loading 15 weeks for Trade Management Suite...")
    weeks = [get_week_dataset(w)[0] for w in range(1, 16)]
    
    experiments = [
        {"exit_mode": "fixed",          "target_rr": 1.8, "brick_size_pips": 7.0, "label": "1. Baseline Fixed (SL 1.5b, TP 1.8R)"},
        {"exit_mode": "color_reversal", "target_rr": 2.5, "brick_size_pips": 7.0, "label": "2. Color Reversal Exit (Early Cut)"},
        {"exit_mode": "breakeven",      "target_rr": 2.0, "brick_size_pips": 7.0, "label": "3. Breakeven Rule (+1b in profit)"},
        {"exit_mode": "brick_trail",    "target_rr": 0.0, "brick_size_pips": 7.0, "label": "4. Brick Trailing Stop (Open Runner)"},
        {"exit_mode": "sma_trail",      "target_rr": 0.0, "brick_size_pips": 7.0, "label": "5. SMA-10 Trailing Stop"},
        {"exit_mode": "combined",       "target_rr": 2.5, "brick_size_pips": 7.0, "label": "6. Combined (BE + SMA Trail + 2.5R)"},
        {"exit_mode": "combined",       "target_rr": 2.2, "brick_size_pips": 8.0, "label": "7. Combined 8p Brick (BE + SMA Trail)"},
    ]

    print("=" * 86)
    print(f"{'Exit Management Strategy':40s} | {'Trades':6s} | {'Win Rate':8s} | {'Net Pips':9s} | {'PF':4s} | {'Max DD':7s}")
    print("-" * 86)

    for exp in experiments:
        label = exp.pop("label")
        strat = ManagedRenkoWrapper(**exp)
        trades_list = []
        for t in weeks:
            tr = engine.run(t, strat)
            if len(tr) > 0:
                trades_list.append(tr)
        if len(trades_list) > 0:
            all_tr = np.concatenate(trades_list)
            m = compute_metrics(all_tr)
            print(f"{label:40s} | {len(all_tr):6d} | {m['win_rate_pct']:7.1f}% | {m['net_pips']:+8.1f}p | {m['profit_factor']:4.2f} | {m['max_drawdown_pips']:6.1f}p")
        else:
            print(f"{label:40s} | 0 trades")

    print("=" * 86)

if __name__ == "__main__":
    run_tests()
