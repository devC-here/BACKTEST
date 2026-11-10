"""
Hunting for High Profit Factor (PF >= 1.50) Renko Strategy Configurations.
Tests:
1. Brick size scaling (7p, 9p, 11p, 14p) to reduce spread drag from 13% down to 6%.
2. Session focus: Prime London Open (07:00 - 12:00 UTC) vs Full London/NY.
3. 2-Brick Momentum Confirmation (avoiding 1-brick fakeouts).
4. Volume Expansion Filter (Volume > 1.3x rolling volume).
5. ATR/Volatility adaptive regime.
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
from tick_engine.core.metrics import compute_metrics

class HighPFHunterStrategy:
    def __init__(
        self,
        brick_size_pips: float = 10.0,
        target_rr: float = 2.0,
        sl_bricks: float = 1.2,
        entry_type: str = "pullback_bounce", # "pullback_bounce", "momentum_2brick", "london_breakout"
        session_window: tuple = (7, 13),      # London Prime (07:00 to 13:00 UTC)
        obv_filter: bool = True,
        trend_filter: bool = True
    ):
        self.brick_size_pips = brick_size_pips
        self.target_rr = target_rr
        self.sl_bricks = sl_bricks
        self.entry_type = entry_type
        self.session_window = session_window
        self.obv_filter = obv_filter
        self.trend_filter = trend_filter

    def generate_signals(self, ticks: np.ndarray):
        n_ticks = len(ticks)
        ts = ticks['timestamp']
        bids = ticks['bid']
        asks = ticks['ask']

        bricks = build_renko_bricks(ticks, brick_size_pips=self.brick_size_pips, pip_size=0.0001)
        n_b = len(bricks)

        signals = np.zeros(n_ticks, dtype=np.int8)
        sl_pips_arr = np.zeros(n_ticks, dtype=np.float64)
        tp_pips_arr = np.zeros(n_ticks, dtype=np.float64)

        if n_b < 35:
            return signals, sl_pips_arr, tp_pips_arr

        closes = np.array([b.close for b in bricks], dtype=np.float64)
        volumes = np.array([b.volume for b in bricks], dtype=np.float64)
        directions = np.array([b.direction for b in bricks], dtype=np.int8)
        brick_ts = np.array([b.timestamp for b in bricks], dtype=np.int64)

        # 10 SMA & 30 SMA
        sma_10 = pd.Series(closes).rolling(10).mean().to_numpy()
        sma_30 = pd.Series(closes).rolling(30).mean().to_numpy()

        # Cumulative OBV and EMA(14)
        raw_obv = np.zeros(n_b, dtype=np.float64)
        c_obv = 0.0
        for i in range(n_b):
            c_obv += (volumes[i] if directions[i] == 1 else -volumes[i])
            raw_obv[i] = c_obv
        obv_sig = pd.Series(raw_obv).ewm(span=14, adjust=False).mean().to_numpy()

        # Rolling average brick volume
        vol_roll = pd.Series(volumes).rolling(15).mean().to_numpy()

        hours = pd.to_datetime(brick_ts, unit='ms').hour

        sl_pips = self.sl_bricks * self.brick_size_pips
        tp_pips = sl_pips * self.target_rr

        h_start, h_end = self.session_window

        for i in range(35, n_b):
            b = bricks[i]
            t_idx = b.tick_idx
            if t_idx >= n_ticks:
                continue

            h = hours[i]
            if not (h_start <= h < h_end):
                continue

            c = closes[i]
            d = directions[i]
            d_prev = directions[i - 1]
            d_prev2 = directions[i - 2]
            f_cur = sma_10[i]
            s_cur = sma_30[i]
            cur_obv = raw_obv[i]
            sig_obv = obv_sig[i]
            cur_vol = volumes[i]
            avg_vol = vol_roll[i] if not np.isnan(vol_roll[i]) else cur_vol

            # Filters
            trend_ok_long = (f_cur > s_cur) if self.trend_filter else True
            trend_ok_short = (f_cur < s_cur) if self.trend_filter else True

            obv_ok_long = (cur_obv > sig_obv) if self.obv_filter else True
            obv_ok_short = (cur_obv < sig_obv) if self.obv_filter else True

            # Entry Logic
            long_sig = False
            short_sig = False

            if self.entry_type == "pullback_bounce":
                # 1 counter-trend pullback brick followed by 1 trend bounce brick
                long_sig = (d_prev == -1) and (d == 1) and (c > f_cur) and trend_ok_long and obv_ok_long
                short_sig = (d_prev == 1) and (d == -1) and (c < f_cur) and trend_ok_short and obv_ok_short

            elif self.entry_type == "momentum_2brick":
                # 2 consecutive bricks in trend direction with expanding volume
                vol_exp = cur_vol > (avg_vol * 1.1)
                long_sig = (d_prev == 1) and (d == 1) and (d_prev2 == -1) and (c > f_cur) and trend_ok_long and obv_ok_long and vol_exp
                short_sig = (d_prev == -1) and (d == -1) and (d_prev2 == 1) and (c < f_cur) and trend_ok_short and obv_ok_short and vol_exp

            elif self.entry_type == "london_breakout":
                # First breakout brick of London session above SMA
                long_sig = (d == 1) and (c > f_cur) and (f_cur > sma_10[i-1]) and trend_ok_long and obv_ok_long
                short_sig = (d == -1) and (c < f_cur) and (f_cur < sma_10[i-1]) and trend_ok_short and obv_ok_short

            if long_sig:
                signals[t_idx] = SIGNAL_LONG
                sl_pips_arr[t_idx] = sl_pips
                tp_pips_arr[t_idx] = tp_pips
            elif short_sig:
                signals[t_idx] = SIGNAL_SHORT
                sl_pips_arr[t_idx] = sl_pips
                tp_pips_arr[t_idx] = tp_pips

        return signals, sl_pips_arr, tp_pips_arr

def run_hunter():
    engine = BacktestEngine(pip_size=0.0001)
    print("Loading 25 weeks of tick data for High-PF Hunter Suite...")
    weeks = [get_week_dataset(w)[0] for w in range(1, 26)]
    print(f"Loaded 25 weeks ({sum(len(w) for w in weeks):,} ticks).\n")

    experiments = [
        {"brick_size_pips": 7.0,  "sl_bricks": 1.2, "target_rr": 1.8, "entry_type": "pullback_bounce",  "session_window": (7, 18), "label": "1. Baseline 7p Pullback (7-18 UTC)"},
        {"brick_size_pips": 7.0,  "sl_bricks": 1.2, "target_rr": 1.8, "entry_type": "pullback_bounce",  "session_window": (7, 12), "label": "2. 7p London Prime Only (7-12 UTC)"},
        {"brick_size_pips": 10.0, "sl_bricks": 1.2, "target_rr": 1.8, "entry_type": "pullback_bounce",  "session_window": (7, 18), "label": "3. 10p Brick (Lower Spread Drag)"},
        {"brick_size_pips": 10.0, "sl_bricks": 1.2, "target_rr": 2.0, "entry_type": "pullback_bounce",  "session_window": (7, 12), "label": "4. 10p London Prime (RR 2.0)"},
        {"brick_size_pips": 12.0, "sl_bricks": 1.2, "target_rr": 2.2, "entry_type": "pullback_bounce",  "session_window": (7, 14), "label": "5. 12p Brick London (RR 2.2)"},
        {"brick_size_pips": 8.0,  "sl_bricks": 1.2, "target_rr": 2.0, "entry_type": "momentum_2brick",   "session_window": (7, 14), "label": "6. 8p 2-Brick Momentum + Vol Exp"},
        {"brick_size_pips": 10.0, "sl_bricks": 1.2, "target_rr": 2.2, "entry_type": "momentum_2brick",   "session_window": (7, 14), "label": "7. 10p 2-Brick Momentum + Vol Exp"},
        {"brick_size_pips": 14.0, "sl_bricks": 1.0, "target_rr": 2.5, "entry_type": "pullback_bounce",  "session_window": (7, 14), "label": "8. 14p Macro Brick (RR 2.5)"},
    ]

    print("=" * 90)
    print("                 HIGH PROFIT FACTOR RENKO EXPLORATION (25 WEEKS)                 ")
    print("=" * 90)
    print(f"{'Strategy Architecture':42s} | {'Trades':6s} | {'Win Rate':8s} | {'Net Pips':9s} | {'PF':4s} | {'Max DD':7s}")
    print("-" * 90)

    for exp in experiments:
        label = exp.pop("label")
        strat = HighPFHunterStrategy(**exp)

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

    print("=" * 90)

if __name__ == "__main__":
    run_hunter()
