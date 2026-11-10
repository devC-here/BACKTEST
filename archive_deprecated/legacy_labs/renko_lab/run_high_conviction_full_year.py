"""
Full 53-Week Annual Benchmark of High-Conviction, Selective Renko Strategy.
Focuses on High Profit Factor (PF >= 1.50), strict drawdown control (< 10% at 10x lev),
and high-quality trade selection over raw trade volume.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from renko_lab.renko_builder import build_renko_bricks
from tick_engine.core.engine import BacktestEngine, SIGNAL_LONG, SIGNAL_SHORT
from tick_engine.core.dataloader import get_week_dataset
from tick_engine.core.metrics import compute_metrics

class HighConvictionRenkoStrategy:
    def __init__(
        self,
        brick_size_pips: float = 9.0,
        target_rr: float = 2.2,
        sl_bricks: float = 1.1,
        entry_mode: str = "momentum_vol",  # "momentum_vol" or "pullback_bounce"
        session_window: tuple = (7, 15),    # London + Early NY (07:00 to 15:00 UTC)
        vol_multiplier: float = 1.15,
        obv_ema_period: int = 14,
        pip_size: float = 0.0001
    ):
        self.brick_size_pips = brick_size_pips
        self.target_rr = target_rr
        self.sl_bricks = sl_bricks
        self.entry_mode = entry_mode
        self.session_window = session_window
        self.vol_multiplier = vol_multiplier
        self.obv_ema_period = obv_ema_period
        self.pip_size = pip_size

    def generate_signals(self, ticks: np.ndarray):
        n_ticks = len(ticks)
        ts = ticks['timestamp']
        bids = ticks['bid']
        asks = ticks['ask']

        bricks = build_renko_bricks(ticks, brick_size_pips=self.brick_size_pips, pip_size=self.pip_size)
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
        obv_sig = pd.Series(raw_obv).ewm(span=self.obv_ema_period, adjust=False).mean().to_numpy()

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

            trend_up = f_cur > s_cur
            trend_down = f_cur < s_cur
            obv_up = cur_obv > sig_obv
            obv_down = cur_obv < sig_obv

            long_sig = False
            short_sig = False

            if self.entry_mode == "momentum_vol":
                # 2 consecutive bricks in trend direction with expanding volume
                vol_expansion = cur_vol > (avg_vol * self.vol_multiplier)
                long_sig = (d_prev == 1) and (d == 1) and (d_prev2 == -1) and (c > f_cur) and trend_up and obv_up and vol_expansion
                short_sig = (d_prev == -1) and (d == -1) and (d_prev2 == 1) and (c < f_cur) and trend_down and obv_down and vol_expansion

            elif self.entry_mode == "pullback_bounce":
                # 1 counter-trend pullback brick followed by 1 bounce brick with volume expansion
                vol_expansion = cur_vol > (avg_vol * self.vol_multiplier)
                long_sig = (d_prev == -1) and (d == 1) and (c > f_cur) and trend_up and obv_up and vol_expansion
                short_sig = (d_prev == 1) and (d == -1) and (c < f_cur) and trend_down and obv_down and vol_expansion

            if long_sig:
                signals[t_idx] = SIGNAL_LONG
                sl_pips_arr[t_idx] = sl_pips
                tp_pips_arr[t_idx] = tp_pips
            elif short_sig:
                signals[t_idx] = SIGNAL_SHORT
                sl_pips_arr[t_idx] = sl_pips
                tp_pips_arr[t_idx] = tp_pips

        return signals, sl_pips_arr, tp_pips_arr

def run_annual_high_conviction_benchmark():
    engine = BacktestEngine(pip_size=0.0001)
    print("Loading 53 weeks of authentic tick data (18M ticks)...")
    weeks = [get_week_dataset(w)[0] for w in range(1, 54)]
    total_ticks = sum(len(w) for w in weeks)
    print(f"Loaded 53 weeks ({total_ticks:,} ticks).\n")

    grid = [
        {"brick_size_pips": 8.0,  "target_rr": 2.0, "sl_bricks": 1.1, "entry_mode": "momentum_vol",   "vol_multiplier": 1.10, "session_window": (7, 16), "label": "1. 8p 2-Brick Momentum (RR 2.0, 7-16h)"},
        {"brick_size_pips": 8.0,  "target_rr": 2.2, "sl_bricks": 1.1, "entry_mode": "momentum_vol",   "vol_multiplier": 1.15, "session_window": (7, 14), "label": "2. 8p 2-Brick Momentum (RR 2.2, 7-14h)"},
        {"brick_size_pips": 9.0,  "target_rr": 2.2, "sl_bricks": 1.1, "entry_mode": "momentum_vol",   "vol_multiplier": 1.15, "session_window": (7, 15), "label": "3. 9p 2-Brick Momentum (RR 2.2, 7-15h)"},
        {"brick_size_pips": 10.0, "target_rr": 2.2, "sl_bricks": 1.0, "entry_mode": "momentum_vol",   "vol_multiplier": 1.15, "session_window": (7, 15), "label": "4. 10p 2-Brick Momentum (RR 2.2, 7-15h)"},
        {"brick_size_pips": 8.0,  "target_rr": 2.0, "sl_bricks": 1.1, "entry_mode": "pullback_bounce", "vol_multiplier": 1.15, "session_window": (7, 14), "label": "5. 8p Pullback Bounce + Vol (RR 2.0, 7-14h)"},
        {"brick_size_pips": 9.0,  "target_rr": 2.2, "sl_bricks": 1.1, "entry_mode": "pullback_bounce", "vol_multiplier": 1.20, "session_window": (7, 14), "label": "6. 9p Pullback Bounce + Vol (RR 2.2, 7-14h)"},
        {"brick_size_pips": 10.0, "target_rr": 2.0, "sl_bricks": 1.0, "entry_mode": "pullback_bounce", "vol_multiplier": 1.15, "session_window": (7, 13), "label": "7. 10p Pullback London Prime (RR 2.0, 7-13h)"},
    ]

    print("=" * 96)
    print("      ANNUAL 53-WEEK HIGH-CONVICTION RENKO BENCHMARK (18,026,436 REAL TICKS)      ")
    print("=" * 96)
    print(f"{'Strategy Architecture':46s} | {'Trades':6s} | {'Win Rate':8s} | {'Net Pips':9s} | {'PF':4s} | {'Max DD':7s} | {'Avg':6s}")
    print("-" * 96)

    best_m = None
    best_lbl = ""

    for exp in grid:
        label = exp.pop("label")
        strat = HighConvictionRenkoStrategy(**exp)

        trades_list = []
        for t in weeks:
            tr = engine.run(t, strat)
            if len(tr) > 0:
                trades_list.append(tr)

        if len(trades_list) > 0:
            all_tr = np.concatenate(trades_list)
            m = compute_metrics(all_tr)
            print(f"{label:46s} | {len(all_tr):6d} | {m['win_rate_pct']:7.1f}% | {m['net_pips']:+8.1f}p | {m['profit_factor']:4.2f} | {m['max_drawdown_pips']:6.1f}p | {m['avg_trade_pips']:+5.2f}p")
            if best_m is None or m['profit_factor'] > best_m['profit_factor']:
                best_m = m
                best_lbl = label
        else:
            print(f"{label:46s} | 0 trades")

    print("=" * 96)

if __name__ == "__main__":
    run_annual_high_conviction_benchmark()
