#!/usr/bin/env python3
"""
================================================================================
  CAMARILLA H4/L4 EXPANSION CROSS-PAIR SCREENER
================================================================================
Tests Camarilla H4/L4 Breakout Expansion across low-representation currency pairs:
  - AUDCAD
  - NZDCAD
  - GBPCHF
  - EURNZD
================================================================================
"""

import sys
import json
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from tick_engine.core.engine import run_matching_loop
from tick_engine.core.dataloader import get_week_dataset, get_complete_split_weeks
from tick_engine.core.config import PIP_SIZES, get_friction_pips, WARMUP_TICKS
from data_engineering_suite.bars import build_time_bars
from scripts.hunt_diversified_cross_pair_alpha import check_anti_redundancy


def eval_camarilla_week(ticks, pip_size, sl_pips, tp_pips, start_hr, end_hr, min_body, max_spread, comm, slip):
    if len(ticks) <= WARMUP_TICKS + 500:
        return np.empty(0)

    m15 = build_time_bars(ticks, timeframe_min=15, pip_size=pip_size)
    h4 = build_time_bars(ticks, timeframe_min=240, pip_size=pip_size)
    if len(m15) < 35 or len(h4) < 10:
        return np.empty(0)

    h4_c = pd.Series(h4['close'].values)
    h4_ema20 = h4_c.ewm(span=20, adjust=False).mean().values
    h4_bias = np.where(h4_c.values > h4_ema20, 1, -1)
    h4_df = pd.DataFrame({'h4_bias': h4_bias}, index=h4.index)

    m15_df = pd.merge_asof(
        m15.sort_index(),
        h4_df.sort_index(),
        left_index=True,
        right_index=True,
        direction='backward'
    )

    c = m15_df['close'].values
    h = m15_df['high'].values
    l = m15_df['low'].values
    o = m15_df['open'].values
    spread = m15_df['spread_pips'].values
    h4_trend = m15_df['h4_bias'].fillna(0).values.astype(int)
    hours = m15_df.index.hour.values
    dates = m15_df.index.date
    time_ms = m15_df.index.astype('datetime64[ms]').view('int64')
    dur_ms = 15 * 60 * 1000
    tick_ts = ticks['timestamp']
    n_ticks = len(ticks)

    # Camarilla from Daily
    daily = m15.resample('1D').agg({'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
    if len(daily) < 2:
        return np.empty(0)

    rng_d = (daily['high'] - daily['low']).values
    c_d = daily['close'].values
    h4_cam = c_d + rng_d * (1.1 / 2.0)
    l4_cam = c_d - rng_d * (1.1 / 2.0)

    cam_df = pd.DataFrame({'h4_cam': h4_cam, 'l4_cam': l4_cam}, index=daily.index).shift(1).bfill()
    m15_df = pd.merge_asof(
        m15_df.sort_index(),
        cam_df.sort_index(),
        left_index=True,
        right_index=True,
        direction='backward'
    )

    h4_lvl = m15_df['h4_cam'].bfill().values
    l4_lvl = m15_df['l4_cam'].bfill().values

    signals_buf = np.zeros(n_ticks, dtype=np.int8)
    sl_buf = np.full(n_ticks, sl_pips, dtype=np.float64)
    tp_buf = np.full(n_ticks, tp_pips, dtype=np.float64)

    last_trade_date = None
    fired_indices = []

    for i in range(15, len(m15_df)):
        hr = hours[i]
        if not (start_hr <= hr <= end_hr):
            continue
        dt = dates[i]
        if dt == last_trade_date:
            continue
        if spread[i] > max_spread:
            continue

        bar_rng = h[i] - l[i]
        if bar_rng < 2.0 * pip_size:
            continue

        sig = 0
        macro = h4_trend[i]

        # Long breakout of H4 Camarilla
        if macro == 1 and o[i] < h4_lvl[i] and c[i] > h4_lvl[i]:
            if (c[i] - o[i]) / bar_rng >= min_body:
                sig = 1

        # Short breakout of L4 Camarilla
        elif macro == -1 and o[i] > l4_lvl[i] and c[i] < l4_lvl[i]:
            if (o[i] - c[i]) / bar_rng >= min_body:
                sig = -1

        if sig != 0:
            idx = np.searchsorted(tick_ts, time_ms[i] + dur_ms)
            if idx < n_ticks and signals_buf[idx] == 0:
                signals_buf[idx] = sig
                fired_indices.append(idx)
                last_trade_date = dt

    if not fired_indices:
        return np.empty(0)

    min_entry_ts = ticks['timestamp'][WARMUP_TICKS]
    tr = run_matching_loop(
        ticks=ticks,
        signals=signals_buf,
        pip_size=pip_size,
        sl_pips=sl_buf,
        tp_pips=tp_buf,
        use_slippage=True,
        close_at_end=True,
        commission_pips=comm,
        entry_slippage_pips=slip,
        min_entry_timestamp=min_entry_ts,
        enforce_subsequent_tick=True
    )
    return tr


def test_symbol_camarilla(symbol: str, sl_tp_grid: list, start_hr: int = 7, end_hr: int = 12, max_spread: float = 2.5):
    pip_size = PIP_SIZES[symbol]
    comm, slip = get_friction_pips(symbol)
    hold_weeks = get_complete_split_weeks(symbol, "holdout_test_30")
    disc_weeks = get_complete_split_weeks(symbol, "discovery_70")[::3]

    print(f"\n=======================================================")
    print(f"  Testing Camarilla Expansion on {symbol}")
    print(f"=======================================================")

    raw_dir = ROOT_DIR / "data" / symbol
    print(f"Loading {len(disc_weeks)} discovery weeks...")
    disc_ticks = []
    for w in disc_weeks:
        try:
            t, _ = get_week_dataset(w, raw_dir, warmup_prev_week_ticks=WARMUP_TICKS)
            disc_ticks.append(t)
        except Exception:
            pass

    print(f"Loading {len(hold_weeks)} holdout weeks...")
    hold_ticks = []
    for w in hold_weeks:
        try:
            t, _ = get_week_dataset(w, raw_dir, warmup_prev_week_ticks=WARMUP_TICKS)
            hold_ticks.append(t)
        except Exception:
            pass

    found = []
    for sl, tp in sl_tp_grid:
        for min_b in [0.45, 0.50, 0.55]:
            # Disc
            tr_d = [eval_camarilla_week(t, pip_size, sl, tp, start_hr, end_hr, min_b, max_spread, comm, slip) for t in disc_ticks]
            tr_d = [t for t in tr_d if len(t) > 0]
            if not tr_d: continue
            comb_d = np.concatenate(tr_d)
            if len(comb_d) < 20: continue
            p_d = comb_d['pips']
            w_d, l_d = float(np.sum(p_d[p_d > 0])), float(abs(np.sum(p_d[p_d < 0])))
            pf_d = w_d / l_d if l_d > 0 else 99.0
            net_d = w_d - l_d
            if pf_d < 1.25 or net_d <= 0: continue

            # Hold
            tr_h = [eval_camarilla_week(t, pip_size, sl, tp, start_hr, end_hr, min_b, max_spread, comm, slip) for t in hold_ticks]
            tr_h = [t for t in tr_h if len(t) > 0]
            if not tr_h: continue
            comb_h = np.concatenate(tr_h)
            if len(comb_h) < 15: continue
            p_h = comb_h['pips']
            w_h, l_h = float(np.sum(p_h[p_h > 0])), float(abs(np.sum(p_h[p_h < 0])))
            pf_h = w_h / l_h if l_h > 0 else 99.0
            net_h = w_h - l_h
            exp_h = net_h / len(comb_h)

            p_sorted = np.sort(p_h)
            p_ex3 = float(np.sum(p_sorted[:-3]))
            is_distinct, msg = check_anti_redundancy(symbol, comb_h, max_overlap_pct=50.0, max_corr=0.50)

            if pf_h >= 1.20 and net_h > 0 and exp_h >= 1.50 and p_ex3 > 0 and is_distinct:
                found.append({
                    "symbol": symbol,
                    "archetype": "Camarilla_Breakout_Expansion",
                    "sl_pips": sl,
                    "tp_pips": tp,
                    "min_body": min_b,
                    "start_hr": start_hr,
                    "end_hr": end_hr,
                    "max_spread": max_spread,
                    "disc_trades": len(comb_d),
                    "disc_pips": round(net_d, 1),
                    "disc_pf": round(pf_d, 2),
                    "hold_trades": len(comb_h),
                    "hold_pips": round(net_h, 1),
                    "hold_pf": round(pf_h, 2),
                    "hold_exp": round(exp_h, 2),
                    "hold_pips_ex3": round(p_ex3, 1)
                })
                print(f"  FOUND: SL {sl:.0f}/TP {tp:.0f} | MinBody {min_b:.2f} | Hold PnL: {net_h:>+7.1f}p | Hold PF: {pf_h:.2f} | Trades: {len(comb_h)} | Exp: {exp_h:.2f}p")

    print(f"Total Camarilla champions found for {symbol}: {len(found)}")
    return found


if __name__ == "__main__":
    for sym in ["AUDCAD", "NZDCAD", "GBPCHF", "EURNZD"]:
        grid = [(14.0, 28.0), (16.0, 32.0), (18.0, 36.0), (20.0, 42.0)]
        res = test_symbol_camarilla(sym, grid)
        if res:
            with open(ROOT_DIR / "results" / f"new_camarilla_{sym.lower()}.json", "w", encoding="utf-8") as f:
                json.dump(res, f, indent=2)

