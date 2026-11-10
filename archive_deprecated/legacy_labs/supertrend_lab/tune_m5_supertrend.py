"""
Tuning and Optimizing M5 Supertrend + Strong Candle Strategy.
Tests:
1. Pure Reversal vs Fixed Asymmetric R:R (1.5R, 1.8R, 2.2R, 2.5R)
2. Breakeven rule once +1.0R in profit
3. Macro EMA filter (H1 50 EMA alignment)
4. Strong candle body threshold (50%, 65%, 75%)
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

class TunedSupertrendStrategy(BaseStrategy):
    def __init__(
        self,
        timeframe: str = "5min",
        atr_period: int = 10,
        multiplier: float = 3.0,
        min_body_ratio: float = 0.60,
        min_atr_factor: float = 0.50,
        sl_type: str = "supertrend_band", # "supertrend_band", "atr_multiple"
        sl_atr_mult: float = 1.5,
        target_rr: float = 2.0,           # 0.0 for pure reversal
        session_filter: bool = True,
        macro_ema_period: int = 50,       # Macro trend filter on M5 (approx 4h trend)
        pip_size: float = 0.0001,
        **kwargs
    ):
        self.timeframe = timeframe
        self.atr_period = atr_period
        self.multiplier = multiplier
        self.min_body_ratio = min_body_ratio
        self.min_atr_factor = min_atr_factor
        self.sl_type = sl_type
        self.sl_atr_mult = sl_atr_mult
        self.target_rr = target_rr
        self.session_filter = session_filter
        self.macro_ema_period = macro_ema_period
        self.pip_size = pip_size

        super().__init__(**kwargs)
        self.name = f"TunedST_{timeframe}_RR{target_rr}"

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
        if n_c < max(self.macro_ema_period, self.atr_period) + 10:
            return signals, sl_pips_arr, tp_pips_arr

        # Supertrend
        st_line, trend, atr = calculate_supertrend(
            candles,
            atr_period=self.atr_period,
            multiplier=self.multiplier
        )

        # Strong Candles
        strong_bull, strong_bear = identify_strong_candles(
            candles,
            atr_values=atr,
            min_body_ratio=self.min_body_ratio,
            min_atr_factor=self.min_atr_factor
        )

        # Macro EMA on candle closes
        closes = candles['close'].to_numpy(dtype=np.float64)
        macro_ema = pd.Series(closes).ewm(span=self.macro_ema_period, adjust=False).mean().to_numpy()

        candle_tick_indices = candles['tick_idx'].to_numpy(dtype=np.int64)
        candle_hours = candles.index.hour.to_numpy()

        pos_state = 0  # 0 = Flat, 1 = Long, -1 = Short

        for i in range(max(self.macro_ema_period, self.atr_period) + 5, n_c):
            t_idx = candle_tick_indices[i]
            if t_idx + 1 >= n_ticks:
                break
            exec_idx = t_idx + 1

            h = candle_hours[i]
            in_session = (7 <= h < 18) if self.session_filter else True

            cur_trend = trend[i]
            c = closes[i]
            ema_val = macro_ema[i]
            st_val = st_line[i]
            atr_val = atr[i]

            macro_bull = c > ema_val
            macro_bear = c < ema_val

            # Entry condition
            buy_cond = (cur_trend == 1) and strong_bull[i] and macro_bull and in_session
            sell_cond = (cur_trend == -1) and strong_bear[i] and macro_bear and in_session

            # Stop Loss calculation
            if self.sl_type == "supertrend_band":
                sl_long_pips = (abs(asks[exec_idx] - st_val) / self.pip_size) + 1.0
                sl_short_pips = (abs(st_val - bids[exec_idx]) / self.pip_size) + 1.0
            else:
                sl_long_pips = (self.sl_atr_mult * atr_val / self.pip_size)
                sl_short_pips = (self.sl_atr_mult * atr_val / self.pip_size)

            # Cap SL between 5p and 20p
            sl_long_pips = np.clip(sl_long_pips, 5.0, 20.0)
            sl_short_pips = np.clip(sl_short_pips, 5.0, 20.0)

            # Take Profit calculation
            tp_long_pips = sl_long_pips * self.target_rr if self.target_rr > 0 else 0.0
            tp_short_pips = sl_short_pips * self.target_rr if self.target_rr > 0 else 0.0

            # Signal Generation
            if self.target_rr == 0.0:
                # Pure Reversal Mode
                if pos_state == 1 and sell_cond:
                    signals[exec_idx] = SIGNAL_SHORT
                    sl_pips_arr[exec_idx] = sl_short_pips
                    tp_pips_arr[exec_idx] = 0.0
                    pos_state = -1
                elif pos_state == -1 and buy_cond:
                    signals[exec_idx] = SIGNAL_LONG
                    sl_pips_arr[exec_idx] = sl_long_pips
                    tp_pips_arr[exec_idx] = 0.0
                    pos_state = 1
                elif pos_state == 0:
                    if buy_cond:
                        signals[exec_idx] = SIGNAL_LONG
                        sl_pips_arr[exec_idx] = sl_long_pips
                        tp_pips_arr[exec_idx] = 0.0
                        pos_state = 1
                    elif sell_cond:
                        signals[exec_idx] = SIGNAL_SHORT
                        sl_pips_arr[exec_idx] = sl_short_pips
                        tp_pips_arr[exec_idx] = 0.0
                        pos_state = -1
            else:
                # Target R:R Mode
                if buy_cond:
                    signals[exec_idx] = SIGNAL_LONG
                    sl_pips_arr[exec_idx] = sl_long_pips
                    tp_pips_arr[exec_idx] = tp_long_pips
                elif sell_cond:
                    signals[exec_idx] = SIGNAL_SHORT
                    sl_pips_arr[exec_idx] = sl_short_pips
                    tp_pips_arr[exec_idx] = tp_short_pips

        return signals, sl_pips_arr, tp_pips_arr

def run_tuning():
    engine = BacktestEngine(pip_size=0.0001)
    print("Loading 20 weeks of tick data for Supertrend Tuning Suite...")
    weeks = [get_week_dataset(w)[0] for w in range(1, 21)]
    print(f"Loaded 20 weeks ({sum(len(w) for w in weeks):,} ticks).\n")

    grid = [
        {"target_rr": 0.0, "min_body_ratio": 0.55, "label": "1. M5 Pure Reversal (Macro 50 EMA Filter)"},
        {"target_rr": 1.5, "min_body_ratio": 0.55, "label": "2. M5 Target RR 1.5 (ST Band SL)"},
        {"target_rr": 1.8, "min_body_ratio": 0.55, "label": "3. M5 Target RR 1.8 (ST Band SL)"},
        {"target_rr": 2.0, "min_body_ratio": 0.55, "label": "4. M5 Target RR 2.0 (ST Band SL)"},
        {"target_rr": 2.2, "min_body_ratio": 0.55, "label": "5. M5 Target RR 2.2 (ST Band SL)"},
        {"target_rr": 2.0, "min_body_ratio": 0.70, "label": "6. M5 Target RR 2.0 (Strict Body 70%)"},
        {"target_rr": 2.2, "min_body_ratio": 0.70, "label": "7. M5 Target RR 2.2 (Strict Body 70%)"},
        {"target_rr": 2.0, "min_body_ratio": 0.60, "sl_type": "atr_multiple", "sl_atr_mult": 1.2, "label": "8. M5 Target RR 2.0 (Tight 1.2x ATR SL)"},
    ]

    print("=" * 92)
    print("                 M5 SUPERTREND + STRONG CANDLE TUNING (20 WEEKS)                 ")
    print("=" * 92)
    print(f"{'Strategy Architecture':42s} | {'Trades':6s} | {'Win Rate':8s} | {'Net Pips':9s} | {'PF':4s} | {'Max DD':7s}")
    print("-" * 92)

    for exp in grid:
        label = exp.pop("label")
        strat = TunedSupertrendStrategy(**exp)

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
    run_tuning()
