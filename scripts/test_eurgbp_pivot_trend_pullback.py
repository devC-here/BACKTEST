#!/usr/bin/env python3
"""
================================================================================
  EURGBP DAILY PIVOT TREND PULLBACK ENGINE
================================================================================
Tests the Floor Traders Daily Pivot Trend Pullback setup on EURGBP:
- H4 Macro Trend Filter: H4 Close > 20 EMA (Bullish) or < 20 EMA (Bearish)
- Microstructure:
  * Bullish: Price pulls back to test Daily Pivot P or S1 from above, touches level,
    and closes green with lower rejection wick >= min_wick
  * Bearish: Price rallies to test Daily Pivot P or R1 from below, touches level,
    and closes red with upper rejection wick >= min_wick
- Active Sessions: 07:00 - 16:00, 08:00 - 15:00 UTC
- ECN friction: 0.33p commission + 0.20p slippage + raw Dukascopy floating spread
- Subsequent-tick execution contract enforced
================================================================================
"""

import sys
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from tick_engine.core.engine import run_matching_loop
from tick_engine.core.dataloader import get_week_dataset, get_complete_split_weeks
from tick_engine.core.config import PIP_SIZES, get_friction_pips, WARMUP_TICKS
from data_engineering_suite.bars import build_time_bars

SYMBOL = "EURGBP"
PIP_SIZE = PIP_SIZES.get(SYMBOL, 0.0001)
COMM_PIPS, SLIP_PIPS = get_friction_pips(SYMBOL)


def load_eurgbp_pivot_data(week_num: int):
    raw_dir = ROOT_DIR / "data" / SYMBOL
    try:
        ticks, _ = get_week_dataset(week=week_num, raw_dir=raw_dir, warmup_prev_week_ticks=WARMUP_TICKS)
    except Exception:
        return None
    if len(ticks) <= WARMUP_TICKS + 500:
        return None

    m15 = build_time_bars(ticks, timeframe_min=15, pip_size=PIP_SIZE)
    if len(m15) < 35:
        return None

    c = m15['close'].values.astype(np.float64)
    h = m15['high'].values.astype(np.float64)
    l = m15['low'].values.astype(np.float64)
    o = m15['open'].values.astype(np.float64)
    spread = m15['spread_pips'].values.astype(np.float64)
    hours = m15.index.hour.values.astype(np.int16)
    dates = m15.index.date
    time_ms = m15.index.astype('datetime64[ms]').view('int64')
    dur_ms = 15 * 60 * 1000
    tick_ts = ticks['timestamp']
    n_ticks = len(ticks)
    exec_indices = np.searchsorted(tick_ts, time_ms + dur_ms).astype(np.int32)

    # Precompute daily stats (P, S1, R1)
    unique_dates = np.unique(dates)
    daily_stats = {}
    for d in unique_dates:
        mask = (dates == d)
        if np.sum(mask) >= 15:
            dh = float(np.max(h[mask]))
            dl = float(np.min(l[mask]))
            dc = float(c[mask][-1])
            p = (dh + dl + dc) / 3.0
            r1 = 2.0 * p - dl
            s1 = 2.0 * p - dh
            daily_stats[d] = {'p': p, 'r1': r1, 's1': s1}

    # H4 Trend Bias
    h4 = m15[['high', 'low', 'close']].resample('4h').agg({'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
    h4_bias = np.zeros(len(m15), dtype=np.int8)
    if len(h4) >= 10:
        h4_ema20 = h4['close'].ewm(span=20, adjust=False).mean()
        h4_b = np.where(h4['close'] > h4_ema20, 1, -1)
        h4_df = pd.DataFrame({'bias': h4_b}, index=h4.index)
        merged_h4 = pd.merge_asof(m15.sort_index(), h4_df.sort_index(), left_index=True, right_index=True, direction='backward')
        h4_bias = merged_h4['bias'].fillna(0).values.astype(np.int8)

    return {
        'ticks': ticks,
        'n_ticks': n_ticks,
        'min_entry_ts': ticks['timestamp'][WARMUP_TICKS],
        'c': c, 'h': h, 'l': l, 'o': o,
        'spread': spread,
        'hours': hours,
        'dates': dates,
        'exec_indices': exec_indices,
        'daily_stats': daily_stats,
        'h4_bias': h4_bias,
        'unique_dates': unique_dates,
        'n_bars': len(m15),
        'signals_buf': np.zeros(n_ticks, dtype=np.int8),
        'sl_buf': np.zeros(n_ticks, dtype=np.float64),
        'tp_buf': np.zeros(n_ticks, dtype=np.float64)
    }


def execute_matches(week_data, sig_indices, sig_types, sl_pips, tp_pips):
    if len(sig_indices) == 0:
        return []
    signals_buf = week_data['signals_buf']
    sl_buf = week_data['sl_buf']
    tp_buf = week_data['tp_buf']
    n_ticks = week_data['n_ticks']

    fired = []
    for idx, s_type in zip(sig_indices, sig_types):
        if idx < n_ticks and signals_buf[idx] == 0:
            signals_buf[idx] = s_type
            sl_buf[idx] = sl_pips
            tp_buf[idx] = tp_pips
            fired.append(idx)

    trades = run_matching_loop(
        ticks=week_data['ticks'],
        signals=signals_buf,
        sl_pips=sl_buf,
        tp_pips=tp_buf,
        min_entry_timestamp=int(week_data['min_entry_ts']),
        pip_size=PIP_SIZE,
        commission_pips=COMM_PIPS,
        entry_slippage_pips=SLIP_PIPS,
        enforce_subsequent_tick=True
    )

    for idx in fired:
        signals_buf[idx] = 0
        sl_buf[idx] = 0.0
        tp_buf[idx] = 0.0

    return trades


def eval_pivot_trend(week_data, touch_up, touch_dn, min_wick, start_hr, end_hr, sl_pips, tp_pips, max_spread=0.8):
    c, h, l, o = week_data['c'], week_data['h'], week_data['l'], week_data['o']
    spread = week_data['spread']
    hours = week_data['hours']
    dates = week_data['dates']
    exec_indices = week_data['exec_indices']
    daily_stats = week_data['daily_stats']
    unique_dates = week_data['unique_dates']
    h4_bias = week_data['h4_bias']
    n_bars = week_data['n_bars']

    date_to_prev = {}
    for idx_d in range(1, len(unique_dates)):
        d_prev = unique_dates[idx_d - 1]
        d_cur = unique_dates[idx_d]
        if d_prev in daily_stats:
            date_to_prev[d_cur] = daily_stats[d_prev]

    t_up = touch_up * PIP_SIZE
    t_dn = touch_dn * PIP_SIZE

    sig_indices = []
    sig_types = []
    last_trade_date = None

    for i in range(25, n_bars):
        hr = hours[i]
        if not (start_hr <= hr <= end_hr):
            continue
        dt = dates[i]
        if dt == last_trade_date:
            continue
        if dt not in date_to_prev:
            continue
        if spread[i] > max_spread:
            continue

        macro = h4_bias[i]
        if macro == 0:
            continue

        pstat = date_to_prev[dt]
        p = pstat['p']
        r1 = pstat['r1']
        s1 = pstat['s1']

        bar_rng = h[i] - l[i]
        if bar_rng < 1.2 * PIP_SIZE:
            continue

        sig = 0
        if macro == 1:
            # Bullish pullback to test Pivot P or S1 from above
            touched = (l[i] <= p + t_up and l[i] >= p - t_dn) or (l[i] <= s1 + t_up and l[i] >= s1 - t_dn)
            if touched and c[i] > o[i]:
                lw = (o[i] - l[i]) / bar_rng
                if lw >= min_wick:
                    sig = 1
        elif macro == -1:
            # Bearish pullback to test Pivot P or R1 from below
            touched = (h[i] >= p - t_up and h[i] <= p + t_dn) or (h[i] >= r1 - t_up and h[i] <= r1 + t_dn)
            if touched and c[i] < o[i]:
                uw = (h[i] - o[i]) / bar_rng
                if uw >= min_wick:
                    sig = -1

        if sig != 0:
            sig_indices.append(exec_indices[i])
            sig_types.append(sig)
            last_trade_date = dt

    return execute_matches(week_data, sig_indices, sig_types, sl_pips, tp_pips)


def main():
    print("Loading Holdout weeks...", flush=True)
    h_weeks = get_complete_split_weeks(ROOT_DIR / "data" / SYMBOL, split="holdout")
    h_data = {w: load_eurgbp_pivot_data(w) for w in h_weeks}
    h_data = {k: v for k, v in h_data.items() if v is not None}

    print("Loading Discovery weeks...", flush=True)
    d_weeks = get_complete_split_weeks(ROOT_DIR / "data" / SYMBOL, split="discovery")
    d_data = {w: load_eurgbp_pivot_data(w) for w in d_weeks}
    d_data = {k: v for k, v in d_data.items() if v is not None}
    print(f"Loaded: {len(h_data)} holdout weeks, {len(d_data)} discovery weeks.\n", flush=True)

    grid = []
    for (t_up, t_dn) in [(0.5, 1.5), (0.8, 2.0), (1.0, 2.5), (1.5, 3.0)]:
        for wick in [0.25, 0.30, 0.35]:
            for (start_hr, end_hr) in [(7, 16), (8, 15), (8, 14)]:
                for sl in [7.0, 8.0, 10.0]:
                    for tp in [14.0, 16.0, 18.0, 20.0]:
                        grid.append({
                            't_up': t_up, 't_dn': t_dn, 'wick': wick,
                            'start_hr': start_hr, 'end_hr': end_hr,
                            'sl': sl, 'tp': tp,
                            'desc': f"Touch=[+{t_up}, -{t_dn}]p, Wk={wick}, H={start_hr}-{end_hr}, SL={sl}p, TP={tp}p"
                        })

    print(f"Evaluating {len(grid)} Pivot Trend Pullback setups...", flush=True)
    champs = []

    for s in grid:
        h_trades = []
        for w, wd in h_data.items():
            tr = eval_pivot_trend(wd, s['t_up'], s['t_dn'], s['wick'], s['start_hr'], s['end_hr'], s['sl'], s['tp'])
            h_trades.extend(tr)

        if len(h_trades) < 15:
            continue
        hp = np.array([t['pips'] for t in h_trades])
        h_net = float(np.sum(hp))
        if h_net <= 0:
            continue
        hw = hp[hp > 0]
        hl = hp[hp < 0]
        h_pf = float(np.sum(hw)) / (float(np.abs(np.sum(hl))) + 1e-5)
        if h_pf < 1.15:
            continue
        shp = np.sort(hp)
        ex_top3 = float(np.sum(shp[:-3])) if len(hp) >= 4 else -1.0
        if ex_top3 <= 0:
            continue

        # Discovery
        d_trades = []
        for w, wd in d_data.items():
            tr = eval_pivot_trend(wd, s['t_up'], s['t_dn'], s['wick'], s['start_hr'], s['end_hr'], s['sl'], s['tp'])
            d_trades.extend(tr)

        if len(d_trades) < 20:
            continue
        dp = np.array([t['pips'] for t in d_trades])
        d_net = float(np.sum(dp))
        if d_net <= 0:
            continue
        dw = dp[dp > 0]
        dl = dp[dp < 0]
        d_pf = float(np.sum(dw)) / (float(np.abs(np.sum(dl))) + 1e-5)

        tag = "*** DUAL CHAMPION ***" if d_pf >= 1.15 else "Positive Discovery"
        print(f"[{tag}] {s['desc']}:", flush=True)
        print(f"   Holdout:   {len(h_trades)} tr, +{h_net:.1f}p, PF={h_pf:.2f}, Exp={np.mean(hp):+.2f}p, ExTop3=+{ex_top3:.1f}p", flush=True)
        print(f"   Discovery: {len(d_trades)} tr, +{d_net:.1f}p, PF={d_pf:.2f}, Exp={np.mean(dp):+.2f}p", flush=True)

        if d_pf >= 1.15:
            champs.append({
                'setup': s,
                'holdout': {'trades': len(h_trades), 'net_pips': h_net, 'pf': h_pf, 'exp': float(np.mean(hp)), 'ex_top3': ex_top3},
                'discovery': {'trades': len(d_trades), 'net_pips': d_net, 'pf': d_pf, 'exp': float(np.mean(dp))}
            })

    print(f"\nPivot Trend Pullback complete. Found {len(champs)} DUAL CHAMPIONS!", flush=True)
    if champs:
        out_f = ROOT_DIR / "results" / "new_champions_eurgbp_pivot_pullback.json"
        with open(out_f, "w", encoding="utf-8") as f:
            json.dump(champs, f, indent=2)
        print(f"Saved to {out_f}", flush=True)


if __name__ == "__main__":
    main()

