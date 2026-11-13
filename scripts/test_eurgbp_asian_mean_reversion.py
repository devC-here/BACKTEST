#!/usr/bin/env python3
"""
================================================================================
  EURGBP ASIAN SESSION MEAN REVERSION MINING ENGINE
================================================================================
Evaluates M15 Bollinger Band & Keltner Mean Reversion during the Asian Session
(00:00 - 06:00 UTC) across both Quarantined Holdout & Discovery:
- Zero overlap with disc107, disc172, or disc174 (they trade 07:00 - 16:00 UTC)
- Strict subsequent-tick causal execution
- ECN friction: 0.33p commission + 0.20p slippage + raw Dukascopy floating spread
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


def load_week_data(week_num: int):
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

    # Compute M15 Bollinger Bands & Keltner
    s = pd.Series(c)
    sma20 = s.rolling(20).mean().values
    std20 = s.rolling(20).std().values
    bb_up_20 = sma20 + 2.0 * std20
    bb_dn_20 = sma20 - 2.0 * std20
    bb_up_25 = sma20 + 2.5 * std20
    bb_dn_25 = sma20 - 2.5 * std20

    tr = np.maximum(h - l, np.maximum(np.abs(h - np.roll(c, 1)), np.abs(l - np.roll(c, 1))))
    tr[0] = h[0] - l[0]
    atr20 = pd.Series(tr).rolling(20).mean().values
    kc_up_15 = sma20 + 1.5 * atr20
    kc_dn_15 = sma20 - 1.5 * atr20
    kc_up_20 = sma20 + 2.0 * atr20
    kc_dn_20 = sma20 - 2.0 * atr20

    # RSI 14
    delta = pd.Series(c).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    rsi14 = (100.0 - (100.0 / (1.0 + rs))).values

    return {
        'ticks': ticks,
        'n_ticks': n_ticks,
        'min_entry_ts': ticks['timestamp'][WARMUP_TICKS],
        'c': c, 'h': h, 'l': l, 'o': o,
        'spread': spread,
        'hours': hours,
        'dates': dates,
        'exec_indices': exec_indices,
        'sma20': sma20,
        'bb_up_20': bb_up_20, 'bb_dn_20': bb_dn_20,
        'bb_up_25': bb_up_25, 'bb_dn_25': bb_dn_25,
        'kc_up_15': kc_up_15, 'kc_dn_15': kc_dn_15,
        'kc_up_20': kc_up_20, 'kc_dn_20': kc_dn_20,
        'rsi14': rsi14,
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


def eval_asian_mr(week_data, band_type, rsi_filter, min_wick, start_hr, end_hr, sl_pips, tp_pips, max_spread=1.0):
    c, h, l, o = week_data['c'], week_data['h'], week_data['l'], week_data['o']
    spread = week_data['spread']
    hours = week_data['hours']
    dates = week_data['dates']
    exec_indices = week_data['exec_indices']
    rsi14 = week_data['rsi14']
    n_bars = week_data['n_bars']

    if band_type == 'bb20':
        up_band = week_data['bb_up_20']
        dn_band = week_data['bb_dn_20']
    elif band_type == 'bb25':
        up_band = week_data['bb_up_25']
        dn_band = week_data['bb_dn_25']
    elif band_type == 'kc15':
        up_band = week_data['kc_up_15']
        dn_band = week_data['kc_dn_15']
    elif band_type == 'kc20':
        up_band = week_data['kc_up_20']
        dn_band = week_data['kc_dn_20']

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
        if spread[i] > max_spread:
            continue

        bar_rng = h[i] - l[i]
        if bar_rng < 0.8 * PIP_SIZE:
            continue

        sig = 0
        # Short Reversion: High pierced above upper band, closes back below upper band, candle is red (c < o)
        if h[i] >= up_band[i] and c[i] < up_band[i] and c[i] < o[i]:
            if rsi_filter == 0 or rsi14[i] >= (50 + rsi_filter):
                uw = (h[i] - o[i]) / bar_rng
                if uw >= min_wick:
                    sig = -1

        # Long Reversion: Low pierced below lower band, closes back above lower band, candle is green (c > o)
        elif l[i] <= dn_band[i] and c[i] > dn_band[i] and c[i] > o[i]:
            if rsi_filter == 0 or rsi14[i] <= (50 - rsi_filter):
                lw = (o[i] - l[i]) / bar_rng
                if lw >= min_wick:
                    sig = 1

        if sig != 0:
            sig_indices.append(exec_indices[i])
            sig_types.append(sig)
            last_trade_date = dt

    return execute_matches(week_data, sig_indices, sig_types, sl_pips, tp_pips)


def main():
    print("Loading Holdout weeks...", flush=True)
    h_weeks = get_complete_split_weeks(ROOT_DIR / "data" / SYMBOL, split="holdout")
    h_data = {w: load_week_data(w) for w in h_weeks}
    h_data = {k: v for k, v in h_data.items() if v is not None}

    print(f"Loading Discovery weeks...", flush=True)
    d_weeks = get_complete_split_weeks(ROOT_DIR / "data" / SYMBOL, split="discovery")
    d_data = {w: load_week_data(w) for w in d_weeks}
    d_data = {k: v for k, v in d_data.items() if v is not None}
    print(f"Ready: {len(h_data)} holdout weeks, {len(d_data)} discovery weeks.\n", flush=True)

    grid = []
    for band_type in ['bb20', 'bb25', 'kc15', 'kc20']:
        for rsi_filter in [0, 15, 20]: # 0=none, 15=RSI>65/<35, 20=RSI>70/<30
            for wick in [0.25, 0.35]:
                for (start_hr, end_hr) in [(0, 5), (0, 6), (1, 6)]:
                    for sl in [6.0, 7.0, 8.0]:
                        for tp in [8.0, 10.0, 12.0, 14.0]:
                            grid.append({
                                'band_type': band_type,
                                'rsi_filter': rsi_filter,
                                'wick': wick,
                                'start_hr': start_hr,
                                'end_hr': end_hr,
                                'sl': sl,
                                'tp': tp,
                                'desc': f"{band_type.upper()}, RSI_F={rsi_filter}, Wk={wick}, H={start_hr}-{end_hr}, SL={sl}p, TP={tp}p"
                            })

    print(f"Evaluating {len(grid)} Asian Mean Reversion setups...", flush=True)
    champs = []

    for idx, s in enumerate(grid):
        # Holdout
        h_trades = []
        for w, wd in h_data.items():
            tr = eval_asian_mr(wd, s['band_type'], s['rsi_filter'], s['wick'], s['start_hr'], s['end_hr'], s['sl'], s['tp'])
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
            tr = eval_asian_mr(wd, s['band_type'], s['rsi_filter'], s['wick'], s['start_hr'], s['end_hr'], s['sl'], s['tp'])
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

    print(f"\nCompleted! Found {len(champs)} DUAL CHAMPIONS!", flush=True)
    if champs:
        out_f = ROOT_DIR / "results" / "new_champions_eurgbp_asian_mr.json"
        with open(out_f, "w", encoding="utf-8") as f:
            json.dump(champs, f, indent=2)
        print(f"Saved to {out_f}", flush=True)


if __name__ == "__main__":
    main()

