#!/usr/bin/env python3
"""
Optimization and Holdout Calibration for M30 Engulfing Trend Resumption
Tests variations on:
  - 70% Discovery Partition
  - 30% Quarantined Out-of-Sample Holdout Partition
  - 100% Full 5-Year Cycle
"""

from pathlib import Path
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tick_engine.core.engine import BacktestEngine
from tick_engine.core.dataloader import get_holdout_test_weeks, get_discovery_weeks, get_week_dataset
from tick_engine.strategies.base import BaseStrategy, SIGNAL_LONG, SIGNAL_SHORT
from data_engineering_suite.bars import build_time_bars


class M30EngulfingOptimized(BaseStrategy):
    def __init__(
        self,
        asset: str = "USDJPY",
        sl_pips: float = 14.0,
        tp_pips: float = 28.0,
        pip_size: float = 0.01,
        max_spread: float = 0.8,
        min_body_ratio: float = 0.0,
        require_ema_touch: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.asset = asset
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.pip_size = float(pip_size)
        self.max_spread = float(max_spread)
        self.min_body_ratio = float(min_body_ratio)
        self.require_ema_touch = bool(require_ema_touch)
        self.name = f"{asset}_M30_Engulfing"

    def generate_signals(self, ticks: np.ndarray):
        n = len(ticks)
        signals = np.zeros(n, dtype=np.int8)
        sl_arr = np.zeros(n, dtype=np.float64)
        tp_arr = np.zeros(n, dtype=np.float64)
        if n < 500:
            return signals, sl_arr, tp_arr

        m30 = build_time_bars(ticks, timeframe_min=30, pip_size=self.pip_size)
        h4 = build_time_bars(ticks, timeframe_min=240, pip_size=self.pip_size)
        if len(m30) < 25 or len(h4) < 10:
            return signals, sl_arr, tp_arr

        h4_c = pd.Series(h4['close'].values)
        h4_ema20 = h4_c.ewm(span=20, adjust=False).mean().values
        h4_trend = np.where(h4_c.values > h4_ema20, 1, -1)
        h4_df = pd.DataFrame({'h4_bias': h4_trend}, index=h4.index)

        m30_df = pd.merge_asof(
            m30.sort_index(),
            h4_df.sort_index(),
            left_index=True,
            right_index=True,
            direction='backward'
        )

        c = m30_df['close'].values
        h = m30_df['high'].values
        l = m30_df['low'].values
        o = m30_df['open'].values
        spread = m30_df['spread_pips'].values
        h4_bias = m30_df['h4_bias'].fillna(0).values.astype(int)
        hours = m30_df.index.hour.values
        dates = m30_df.index.date
        time_ms = m30_df.index.astype('datetime64[ms]').view('int64')
        dur_ms = 30 * 60 * 1000
        tick_ts = ticks['timestamp']

        c_s = pd.Series(c)
        m30_ema20 = c_s.ewm(span=20, adjust=False).mean().values

        last_trade_date = None
        tol = 2.0 * self.pip_size

        for i in range(10, len(m30_df)):
            if not (7 <= hours[i] <= 17):
                continue

            dt = dates[i]
            if dt == last_trade_date:
                continue

            if spread[i] > self.max_spread:
                continue

            macro = h4_bias[i]
            if macro == 0:
                continue

            bar_rng = h[i] - l[i]
            if bar_rng < 2.5 * self.pip_size:
                continue

            # Bullish Engulfing: prev bar bearish, current bar engulfs high & low, closes bullish
            is_bull_engulf = (c[i-1] < o[i-1]) and (h[i] > h[i-1]) and (l[i] <= l[i-1]) and (c[i] > h[i-1]) and (c[i] > o[i])

            # Bearish Engulfing: prev bar bullish, current bar engulfs high & low, closes bearish
            is_bear_engulf = (c[i-1] > o[i-1]) and (l[i] < l[i-1]) and (h[i] >= h[i-1]) and (c[i] < l[i-1]) and (c[i] < o[i])

            sig = 0
            if macro == 1 and is_bull_engulf:
                if self.min_body_ratio > 0:
                    body = c[i] - o[i]
                    if body / bar_rng < self.min_body_ratio:
                        continue
                if self.require_ema_touch:
                    # Low touched or was near M30 EMA20
                    if l[i] > m30_ema20[i] + tol and l[i-1] > m30_ema20[i] + tol:
                        continue
                sig = SIGNAL_LONG

            elif macro == -1 and is_bear_engulf:
                if self.min_body_ratio > 0:
                    body = o[i] - c[i]
                    if body / bar_rng < self.min_body_ratio:
                        continue
                if self.require_ema_touch:
                    # High touched or was near M30 EMA20
                    if h[i] < m30_ema20[i] - tol and h[i-1] < m30_ema20[i] - tol:
                        continue
                sig = SIGNAL_SHORT

            if sig != 0:
                bar_close_ms = time_ms[i] + dur_ms
                idx = np.searchsorted(tick_ts, bar_close_ms)
                if idx < n:
                    signals[idx] = sig
                    sl_arr[idx] = self.sl_pips
                    tp_arr[idx] = self.tp_pips
                    last_trade_date = dt

        return signals, sl_arr, tp_arr


models = [
    {
        "name": "UJ_M30_ENG_BASE",
        "asset": "USDJPY",
        "strat": M30EngulfingOptimized(asset="USDJPY", sl_pips=14.0, tp_pips=28.0, pip_size=0.01, max_spread=0.8, min_body_ratio=0.0),
        "sl_pips": 14.0
    },
    {
        "name": "UJ_M30_ENG_BODY55",
        "asset": "USDJPY",
        "strat": M30EngulfingOptimized(asset="USDJPY", sl_pips=12.0, tp_pips=24.0, pip_size=0.01, max_spread=0.8, min_body_ratio=0.55),
        "sl_pips": 12.0
    },
    {
        "name": "UJ_M30_ENG_22R",
        "asset": "USDJPY",
        "strat": M30EngulfingOptimized(asset="USDJPY", sl_pips=12.0, tp_pips=26.4, pip_size=0.01, max_spread=0.8, min_body_ratio=0.50),
        "sl_pips": 12.0
    },
    {
        "name": "UJ_M30_ENG_EMA",
        "asset": "USDJPY",
        "strat": M30EngulfingOptimized(asset="USDJPY", sl_pips=12.0, tp_pips=24.0, pip_size=0.01, max_spread=0.8, min_body_ratio=0.50, require_ema_touch=True),
        "sl_pips": 12.0
    },
    {
        "name": "EU_M30_ENG_BODY55",
        "asset": "EURUSD",
        "strat": M30EngulfingOptimized(asset="EURUSD", sl_pips=9.0, tp_pips=18.0, pip_size=0.0001, max_spread=0.6, min_body_ratio=0.55),
        "sl_pips": 9.0
    },
    {
        "name": "EG_M30_ENG_BODY55",
        "asset": "EURGBP",
        "strat": M30EngulfingOptimized(asset="EURGBP", sl_pips=8.0, tp_pips=16.0, pip_size=0.0001, max_spread=0.8, min_body_ratio=0.55),
        "sl_pips": 8.0
    }
]

def run_task(args):
    m_idx, w, split = args
    m = models[m_idx]
    engine = BacktestEngine(pip_size=m["strat"].pip_size, use_slippage=True)
    try:
        ticks, _ = get_week_dataset(w, raw_dir=f"data/{m['asset']}")
        if ticks is None or len(ticks) < 500:
            return m_idx, split, []
        raw_trades = engine.run(ticks, m["strat"])
        pips = [float(r["pips"]) for r in raw_trades]
        return m_idx, split, pips
    except Exception:
        return m_idx, split, []

def calc_stats(pips, sl_pips, years):
    if not pips:
        return 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    arr = np.array(pips, dtype=np.float64)
    trades = len(arr)
    wins = arr[arr > 0]
    losses = arr[arr < 0]
    wr = len(wins) / trades * 100.0 if trades > 0 else 0.0
    gross_win = float(np.sum(wins)) if len(wins) > 0 else 0.0
    gross_loss = float(abs(np.sum(losses))) if len(losses) > 0 else 0.0
    pf = gross_win / gross_loss if gross_loss > 0 else 999.0
    net_pips = float(np.sum(arr))
    net_r = net_pips / sl_pips if sl_pips > 0 else 0.0
    ann_ret = (net_r / years) * 1.0

    equity = np.cumsum(arr)
    peak = np.maximum.accumulate(equity)
    dd = peak - equity
    max_dd = float(np.max(dd)) if len(dd) > 0 else 0.0
    ret_dd = net_pips / max_dd if max_dd > 0 else 999.0

    return trades, wr, pf, net_pips, net_r, ann_ret, max_dd, ret_dd

def main():
    tasks = []
    for idx, m in enumerate(models):
        disc_w = get_discovery_weeks(raw_dir=f"data/{m['asset']}")
        hold_w = get_holdout_test_weeks(raw_dir=f"data/{m['asset']}")
        for w in disc_w:
            tasks.append((idx, w, "DISC"))
        for w in hold_w:
            tasks.append((idx, w, "HOLD"))

    res = {i: {"DISC": [], "HOLD": []} for i in range(len(models))}
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(run_task, t) for t in tasks]
        for fut in as_completed(futures):
            idx, split, pips = fut.result()
            res[idx][split].extend(pips)

    print("=" * 125)
    print(f"{'Model Name':<20} {'Partition':<12} {'Trades':<8} {'WR%':<7} {'PF':<7} {'Net Pips':<11} {'Net R':<9} {'Annual Ret%':<13} {'Max DD(p)':<10} {'Ret/DD'}")
    print("-" * 125)

    for idx, m in enumerate(models):
        d_pips = res[idx]["DISC"]
        h_pips = res[idx]["HOLD"]
        all_pips = d_pips + h_pips

        tr_d, wr_d, pf_d, np_d, r_d, ann_d, dd_d, rdd_d = calc_stats(d_pips, m["sl_pips"], 3.5)
        tr_h, wr_h, pf_h, np_h, r_h, ann_h, dd_h, rdd_h = calc_stats(h_pips, m["sl_pips"], 1.5)
        tr_a, wr_a, pf_a, np_a, r_a, ann_a, dd_a, rdd_a = calc_stats(all_pips, m["sl_pips"], 5.0)

        print(f"{m['name']:<20} {'70% Disc':<12} {tr_d:<8} {wr_d:<7.1f} {pf_d:<7.2f} {np_d:<11.1f} {r_d:<9.1f} {ann_d:<13.1f} {dd_d:<10.1f} {rdd_d:.2f}x")
        print(f"{m['name']:<20} {'30% Holdout':<12} {tr_h:<8} {wr_h:<7.1f} {pf_h:<7.2f} {np_h:<11.1f} {r_h:<9.1f} {ann_h:<13.1f} {dd_h:<10.1f} {rdd_h:.2f}x")
        print(f"{m['name']:<20} {'100% 5-Year':<12} {tr_a:<8} {wr_a:<7.1f} {pf_a:<7.2f} {np_a:<11.1f} {r_a:<9.1f} {ann_a:<13.1f} {dd_a:<10.1f} {rdd_a:.2f}x")
        print("-" * 125)

if __name__ == "__main__":
    main()

