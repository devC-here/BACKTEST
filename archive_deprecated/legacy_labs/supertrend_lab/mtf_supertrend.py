"""
Multi-Timeframe Supertrend Strategy (H1 Macro Filter + M5 Trigger).
Solves intraday range chop by:
1. Requiring H1 Supertrend alignment (H1 Trend == +1 for Longs, -1 for Shorts)
2. Taking M5 Supertrend + Strong Candle triggers in the H1 direction
3. Exiting on M5 reversal or H1 trend termination
4. Optional ADX filter (ADX > 20) to filter dead consolidation
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

def compute_adx(candles: pd.DataFrame, period: int = 14) -> np.ndarray:
    high = candles['high'].to_numpy(dtype=np.float64)
    low = candles['low'].to_numpy(dtype=np.float64)
    close = candles['close'].to_numpy(dtype=np.float64)
    n = len(candles)

    up_move = high[1:] - high[:-1]
    down_move = low[:-1] - low[1:]

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        tr[i] = max(hl, max(hc, lc))

    tr_s = pd.Series(tr).ewm(span=period, adjust=False).mean().to_numpy()
    pdm_s = pd.Series(np.insert(plus_dm, 0, 0.0)).ewm(span=period, adjust=False).mean().to_numpy()
    mdm_s = pd.Series(np.insert(minus_dm, 0, 0.0)).ewm(span=period, adjust=False).mean().to_numpy()

    p_di = 100.0 * (pdm_s / np.maximum(tr_s, 1e-7))
    m_di = 100.0 * (mdm_s / np.maximum(tr_s, 1e-7))

    dx = 100.0 * np.abs(p_di - m_di) / np.maximum(p_di + m_di, 1e-7)
    adx = pd.Series(dx).ewm(span=period, adjust=False).mean().to_numpy()
    return adx

class MTFSupertrendStrategy(BaseStrategy):
    def __init__(
        self,
        base_tf: str = "5min",
        macro_tf: str = "1h",
        base_atr: int = 10,
        base_mult: float = 3.0,
        macro_atr: int = 10,
        macro_mult: float = 3.0,
        min_body_ratio: float = 0.55,
        use_adx: bool = False,
        adx_threshold: float = 20.0,
        session_filter: bool = True,
        exit_mode: str = "base_flip", # "base_flip", "macro_flip", "pure_reversal"
        pip_size: float = 0.0001,
        **kwargs
    ):
        self.base_tf = base_tf
        self.macro_tf = macro_tf
        self.base_atr = base_atr
        self.base_mult = base_mult
        self.macro_atr = macro_atr
        self.macro_mult = macro_mult
        self.min_body_ratio = min_body_ratio
        self.use_adx = use_adx
        self.adx_threshold = adx_threshold
        self.session_filter = session_filter
        self.exit_mode = exit_mode
        self.pip_size = pip_size

        super().__init__(**kwargs)
        self.name = f"MTF_ST_{base_tf}_{macro_tf}_{exit_mode}"

    def generate_signals(self, ticks: np.ndarray):
        n_ticks = len(ticks)
        signals = np.zeros(n_ticks, dtype=np.int8)
        sl_pips_arr = np.zeros(n_ticks, dtype=np.float64)
        tp_pips_arr = np.zeros(n_ticks, dtype=np.float64)

        if n_ticks < 100:
            return signals, sl_pips_arr, tp_pips_arr

        # 1. Base Candles (M5)
        base_candles = resample_ticks_to_candles(ticks, timeframe=self.base_tf)
        if len(base_candles) < 30:
            return signals, sl_pips_arr, tp_pips_arr

        # 2. Macro Candles (H1)
        macro_candles = resample_ticks_to_candles(ticks, timeframe=self.macro_tf)
        if len(macro_candles) < self.macro_atr + 5:
            return signals, sl_pips_arr, tp_pips_arr

        # 3. Supertrend Calculations
        _, base_trend, base_atr = calculate_supertrend(base_candles, atr_period=self.base_atr, multiplier=self.base_mult)
        _, macro_trend, _ = calculate_supertrend(macro_candles, atr_period=self.macro_atr, multiplier=self.macro_mult)

        # 4. Strong Candles on Base TF
        strong_bull, strong_bear = identify_strong_candles(base_candles, atr_values=base_atr, min_body_ratio=self.min_body_ratio)

        # 5. ADX on Base TF (optional)
        adx_vals = compute_adx(base_candles, period=14) if self.use_adx else np.zeros(len(base_candles))

        # Align macro trend to base candles without lookahead bias
        macro_df = pd.DataFrame({'macro_trend': macro_trend}, index=macro_candles.index)
        # Shift macro trend by 1 candle to ensure strictly completed candle reference
        macro_df['macro_trend'] = macro_df['macro_trend'].shift(1)
        base_aligned = base_candles.join(macro_df, how='left')
        base_aligned['macro_trend'] = base_aligned['macro_trend'].ffill().fillna(0).to_numpy()

        aligned_macro_trend = base_aligned['macro_trend'].to_numpy(dtype=np.int8)
        base_tick_indices = base_candles['tick_idx'].to_numpy(dtype=np.int64)
        candle_hours = base_candles.index.hour.to_numpy()

        pos = 0  # 0 = Flat, 1 = Long, -1 = Short

        for i in range(self.base_atr + 5, len(base_candles)):
            t_idx = base_tick_indices[i]
            if t_idx + 1 >= n_ticks:
                break
            exec_idx = t_idx + 1

            h = candle_hours[i]
            in_session = (7 <= h < 18) if self.session_filter else True

            b_tr = base_trend[i]
            m_tr = aligned_macro_trend[i]
            adx_ok = (adx_vals[i] >= self.adx_threshold) if self.use_adx else True

            # Entry Criteria
            long_cond = (m_tr == 1) and (b_tr == 1) and strong_bull[i] and adx_ok and in_session
            short_cond = (m_tr == -1) and (b_tr == -1) and strong_bear[i] and adx_ok and in_session

            # Manage Exits
            if pos == 1:
                if self.exit_mode == "base_flip" and b_tr == -1:
                    signals[exec_idx] = SIGNAL_FORCE_EXIT
                    pos = 0
                    continue
                elif self.exit_mode == "macro_flip" and m_tr == -1:
                    signals[exec_idx] = SIGNAL_FORCE_EXIT
                    pos = 0
                    continue
                elif self.exit_mode == "pure_reversal" and short_cond:
                    signals[exec_idx] = SIGNAL_SHORT
                    sl_pips_arr[exec_idx] = 12.0
                    pos = -1
                    continue

            elif pos == -1:
                if self.exit_mode == "base_flip" and b_tr == 1:
                    signals[exec_idx] = SIGNAL_FORCE_EXIT
                    pos = 0
                    continue
                elif self.exit_mode == "macro_flip" and m_tr == 1:
                    signals[exec_idx] = SIGNAL_FORCE_EXIT
                    pos = 0
                    continue
                elif self.exit_mode == "pure_reversal" and long_cond:
                    signals[exec_idx] = SIGNAL_LONG
                    sl_pips_arr[exec_idx] = 12.0
                    pos = 1
                    continue

            # Manage Entries if Flat
            if pos == 0:
                if long_cond:
                    signals[exec_idx] = SIGNAL_LONG
                    sl_pips_arr[exec_idx] = 12.0
                    pos = 1
                elif short_cond:
                    signals[exec_idx] = SIGNAL_SHORT
                    sl_pips_arr[exec_idx] = 12.0
                    pos = -1

        return signals, sl_pips_arr, tp_pips_arr

def run_mtf_suite():
    engine = BacktestEngine(pip_size=0.0001)
    print("Loading 20 weeks of tick data for MTF Supertrend Suite...")
    weeks = [get_week_dataset(w)[0] for w in range(1, 21)]
    print(f"Loaded 20 weeks ({sum(len(w) for w in weeks):,} ticks).\n")

    experiments = [
        {"base_tf": "5min",  "macro_tf": "1h", "exit_mode": "base_flip",     "use_adx": False, "label": "1. M5 Trigger / H1 Macro (Exit M5 Flip)"},
        {"base_tf": "5min",  "macro_tf": "1h", "exit_mode": "macro_flip",    "use_adx": False, "label": "2. M5 Trigger / H1 Macro (Exit H1 Flip)"},
        {"base_tf": "5min",  "macro_tf": "1h", "exit_mode": "pure_reversal", "use_adx": False, "label": "3. M5 Trigger / H1 Macro (Pure Reversal)"},
        {"base_tf": "5min",  "macro_tf": "1h", "exit_mode": "base_flip",     "use_adx": True,  "label": "4. M5/H1 + ADX > 20 Filter"},
        {"base_tf": "15min", "macro_tf": "1h", "exit_mode": "base_flip",     "use_adx": False, "label": "5. M15 Trigger / H1 Macro (Exit M15 Flip)"},
        {"base_tf": "15min", "macro_tf": "4h", "exit_mode": "base_flip",     "use_adx": False, "label": "6. M15 Trigger / H4 Macro (Exit M15 Flip)"},
    ]

    print("=" * 96)
    print("                  MULTI-TIMEFRAME SUPERTREND BENCHMARK (20 WEEKS)                  ")
    print("=" * 96)
    print(f"{'Strategy Architecture':46s} | {'Trades':6s} | {'Win Rate':8s} | {'Net Pips':9s} | {'PF':4s} | {'Max DD':7s}")
    print("-" * 96)

    for exp in experiments:
        label = exp.pop("label")
        strat = MTFSupertrendStrategy(**exp)

        trades_list = []
        for t in weeks:
            tr = engine.run(t, strat)
            if len(tr) > 0:
                trades_list.append(tr)

        if len(trades_list) > 0:
            all_tr = np.concatenate(trades_list)
            m = compute_metrics(all_tr)
            print(f"{label:46s} | {len(all_tr):6d} | {m['win_rate_pct']:7.1f}% | {m['net_pips']:+8.1f}p | {m['profit_factor']:4.2f} | {m['max_drawdown_pips']:6.1f}p")
        else:
            print(f"{label:46s} | 0 trades")

    print("=" * 96)

if __name__ == "__main__":
    run_mtf_suite()
