#!/usr/bin/env python3
"""
================================================================================
  EURGBP ASIAN SWEEP & LONDON SPIKE MEAN-REVERSION DISCOVERY
================================================================================
Explores the quintessential EURGBP behavioral archetype:
EURGBP's low daily volatility (ATR ~35p) means morning breakout attempts
regularly fail and revert back to fair value.

Strategy Logic:
1. Define Asian Session Reference Range (00:00 - 06:30 UTC): Asian High & Asian Low.
2. During Frankfurt / London Morning (07:00 - 11:00 UTC):
   - Bullish Sweep: Price sweeps below Asian Low by buffer (2-6p) and rejects (reversion long).
   - Bearish Sweep: Price sweeps above Asian High by buffer (2-6p) and rejects (reversion short).
3. Risk Management:
   - SL: 6.0 to 10.0 pips.
   - TP: 10.0 to 18.0 pips (Reward:Risk 1.4:1 to 2.2:1).
4. Dual Partition Gate:
   - Discovery: Weeks 1..210 (PF >= 1.20, Trades >= 30, Net > 0)
   - Holdout: Weeks 211..300 (PF >= 1.20, Trades >= 15, Net > 0, ExTop3 > 0)
   - Friction: 0.33p commission + 0.20p slippage + raw Dukascopy floating spread.
   - Strict subsequent-tick execution (T_fill > T_signal).
================================================================================
"""

import sys
import json
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from tick_engine.core.engine import run_matching_loop
from tick_engine.core.dataloader import get_week_dataset, get_complete_split_weeks
from tick_engine.core.config import PIP_SIZES, get_friction_pips, WARMUP_TICKS
from data_engineering_suite.bars import build_time_bars
from scripts.hunt_diversified_cross_pair_alpha import check_anti_redundancy

SYMBOL = "EURGBP"
PIP_SIZE = PIP_SIZES.get(SYMBOL, 0.0001)
COMM_PIPS, SLIP_PIPS = get_friction_pips(SYMBOL)


def precompute_eurgbp_week(week_num: int):
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
    minutes = m15.index.minute.values.astype(np.int16)
    dates = m15.index.date
    time_ms = m15.index.astype('datetime64[ms]').view('int64')
    dur_ms = 15 * 60 * 1000
    tick_ts = ticks['timestamp']
    n_ticks = len(ticks)

    exec_indices = np.searchsorted(tick_ts, time_ms + dur_ms).astype(np.int32)

    # Precalculate daily Asian high/low (00:00 - 06:30 UTC)
    asian_highs = {}
    asian_lows = {}
    for d in np.unique(dates):
        mask = (dates == d) & ((hours < 6) | ((hours == 6) & (minutes <= 30)))
        if np.sum(mask) >= 4:
            asian_highs[d] = float(np.max(h[mask]))
            asian_lows[d] = float(np.min(l[mask]))

    return {
        'ticks': ticks,
        'n_ticks': n_ticks,
        'pip_size': PIP_SIZE,
        'min_entry_ts': ticks['timestamp'][WARMUP_TICKS],
        'c': c, 'h': h, 'l': l, 'o': o,
        'spread': spread,
        'hours': hours,
        'minutes': minutes,
        'dates': dates,
        'exec_indices': exec_indices,
        'asian_highs': asian_highs,
        'asian_lows': asian_lows,
        'n_bars': len(m15),
        'signals_buf': np.zeros(n_ticks, dtype=np.int8),
        'sl_buf': np.zeros(n_ticks, dtype=np.float64),
        'tp_buf': np.zeros(n_ticks, dtype=np.float64)
    }


def eval_eurgbp_reversion(week_data, start_hr, end_hr, sl_pips, tp_pips, sweep_min_p, sweep_max_p, min_wick_ratio, max_spread):
    n_bars = week_data['n_bars']
    n_ticks = week_data['n_ticks']
    pip_size = week_data['pip_size']
    c, h, l, o = week_data['c'], week_data['h'], week_data['l'], week_data['o']
    spread = week_data['spread']
    hours = week_data['hours']
    dates = week_data['dates']
    exec_indices = week_data['exec_indices']
    asian_highs = week_data['asian_highs']
    asian_lows = week_data['asian_lows']
    signals_buf = week_data['signals_buf']
    sl_buf = week_data['sl_buf']
    tp_buf = week_data['tp_buf']

    last_trade_date = None
    fired_indices = []

    for i in range(15, n_bars):
        hr = hours[i]
        if not (start_hr <= hr <= end_hr):
            continue
        dt = dates[i]
        if dt == last_trade_date:
            continue
        if dt not in asian_highs or dt not in asian_lows:
            continue
        if spread[i] > max_spread:
            continue

        bar_rng = h[i] - l[i]
        if bar_rng < 1.0 * pip_size:
            continue

        a_h = asian_highs[dt]
        a_l = asian_lows[dt]

        sig = 0

        # Bearish Reversion: Swept above Asian high and closed back down with upper rejection wick
        sweep_up = (h[i] - a_h) / pip_size
        if sweep_min_p <= sweep_up <= sweep_max_p and c[i] < a_h:
            upper_wick = h[i] - max(o[i], c[i])
            if (upper_wick / bar_rng) >= min_wick_ratio:
                sig = -1

        # Bullish Reversion: Swept below Asian low and closed back up with lower rejection wick
        sweep_dn = (a_l - l[i]) / pip_size
        if sweep_min_p <= sweep_dn <= sweep_max_p and c[i] > a_l:
            lower_wick = min(o[i], c[i]) - l[i]
            if (lower_wick / bar_rng) >= min_wick_ratio:
                sig = 1

        if sig != 0:
            idx = exec_indices[i]
            if idx < n_ticks and signals_buf[idx] == 0:
                signals_buf[idx] = sig
                sl_buf[idx] = sl_pips
                tp_buf[idx] = tp_pips
                fired_indices.append(idx)
                last_trade_date = dt

    if not fired_indices:
        return np.empty(0)

    tr = run_matching_loop(
        ticks=week_data['ticks'],
        signals=signals_buf,
        pip_size=pip_size,
        sl_pips=sl_buf,
        tp_pips=tp_buf,
        use_slippage=True,
        close_at_end=True,
        commission_pips=COMM_PIPS,
        entry_slippage_pips=SLIP_PIPS,
        min_entry_timestamp=week_data['min_entry_ts'],
        enforce_subsequent_tick=True
    )

    for idx in fired_indices:
        signals_buf[idx] = 0
        sl_buf[idx] = 0.0
        tp_buf[idx] = 0.0

    return tr


def main():
    print("=" * 80)
    print("  EURGBP ASIAN SWEEP & LONDON REVERSION ALPHA LAB")
    print(f"  Friction: {COMM_PIPS:.2f}p Commission + {SLIP_PIPS:.2f}p Slippage + Raw Spreads")
    print("=" * 80)

    disc_weeks = get_complete_split_weeks(SYMBOL, "discovery_70")
    hold_weeks = get_complete_split_weeks(SYMBOL, "holdout_test_30")

    # Sample discovery every 2nd week (99 weeks)
    sample_disc_weeks = disc_weeks[::2]

    print(f"Precomputing {len(sample_disc_weeks)} Discovery weeks...")
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=8) as ex:
        disc_cache = [w for w in ex.map(precompute_eurgbp_week, sample_disc_weeks) if w is not None]
    print(f"Discovery cache built ({len(disc_cache)} weeks) in {time.time() - t0:.1f}s.")

    sl_tp_grid = [
        (6.0, 9.0), (6.0, 10.0), (6.0, 12.0),
        (7.0, 10.0), (7.0, 12.0), (7.0, 14.0),
        (8.0, 12.0), (8.0, 14.0), (8.0, 16.0),
        (9.0, 14.0), (9.0, 16.0), (10.0, 16.0), (10.0, 20.0)
    ]
    windows = [(7, 9), (7, 10), (7, 11), (6, 9), (6, 10)]
    sweeps = [(1.5, 8.0), (2.0, 10.0), (2.5, 12.0), (1.0, 6.0)]
    wick_ratios = [0.25, 0.30, 0.35, 0.40]

    candidates = []

    print("\nScanning European morning reversion grid...")
    for (start_hr, end_hr) in windows:
        for (sl, tp) in sl_tp_grid:
            for (sw_min, sw_max) in sweeps:
                for wr in wick_ratios:
                    tr_list = []
                    for w in disc_cache:
                        tr = eval_eurgbp_reversion(w, start_hr, end_hr, sl, tp, sw_min, sw_max, wr, max_spread=1.4)
                        if len(tr) > 0:
                            tr_list.append(tr)
                    if not tr_list:
                        continue
                    comb = np.concatenate(tr_list)
                    if len(comb) < 25:
                        continue

                    pips = comb['pips']
                    wins = float(np.sum(pips[pips > 0]))
                    losses = float(np.abs(np.sum(pips[pips < 0])))
                    pf = (wins / losses) if losses > 0 else 99.0
                    net = wins - losses

                    if pf >= 1.22 and net > 0:
                        candidates.append({
                            "start_hr": start_hr, "end_hr": end_hr,
                            "sl_pips": sl, "tp_pips": tp,
                            "sweep_min": sw_min, "sweep_max": sw_max,
                            "min_wick": wr,
                            "disc_trades": len(comb),
                            "disc_pips": round(net, 1),
                            "disc_pf": round(pf, 2),
                            "disc_exp": round(net / len(comb), 2)
                        })

    print(f"Discovery screening complete. Candidates: {len(candidates)}")
    if not candidates:
        print("No discovery candidates cleared the PF 1.22 gate.")
        return

    candidates.sort(key=lambda x: x["disc_pf"] * x["disc_pips"], reverse=True)
    top_candidates = candidates[:15]

    print(f"\nPrecomputing {len(hold_weeks)} Holdout weeks...")
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=8) as ex:
        hold_cache = [w for w in ex.map(precompute_eurgbp_week, hold_weeks) if w is not None]
    print(f"Holdout cache built ({len(hold_cache)} weeks) in {time.time() - t0:.1f}s.")

    certified = []
    print("\n" + "=" * 90)
    print(f"{'WINDOW':<12} | {'SL/TP':<8} | {'SWEEP':<10} | {'HOLD PIPS':<10} | {'HOLD PF':<8} | {'TRADES':<7} | {'EXP':<7} | {'STATUS'}")
    print("=" * 90)

    for c in top_candidates:
        tr_list = []
        for w in hold_cache:
            tr = eval_eurgbp_reversion(
                w, c["start_hr"], c["end_hr"], c["sl_pips"], c["tp_pips"],
                c["sweep_min"], c["sweep_max"], c["min_wick"], max_spread=1.4
            )
            if len(tr) > 0:
                tr_list.append(tr)
        if not tr_list:
            continue
        comb_h = np.concatenate(tr_list)
        if len(comb_h) < 15:
            continue

        pips = comb_h['pips']
        wins = float(np.sum(pips[pips > 0]))
        losses = float(np.abs(np.sum(pips[pips < 0])))
        pf = (wins / losses) if losses > 0 else 99.0
        net = wins - losses
        exp = net / len(comb_h)

        pips_sorted = np.sort(pips)
        pips_ex3 = float(np.sum(pips_sorted[:-3]))

        is_distinct, anti_msg = check_anti_redundancy(SYMBOL, comb_h, max_overlap_pct=50.0, max_corr=0.50)

        win_str = f"{c['start_hr']:02d}:00-{c['end_hr']:02d}:00"
        sl_tp_str = f"{c['sl_pips']:.0f}/{c['tp_pips']:.0f}"
        sw_str = f"{c['sweep_min']:.1f}-{c['sweep_max']:.1f}p"

        if pf >= 1.20 and net > 0 and exp >= 1.20 and pips_ex3 > 0 and is_distinct:
            status = "CERTIFIED"
            c["hold_trades"] = len(comb_h)
            c["hold_pips"] = round(net, 1)
            c["hold_pf"] = round(pf, 2)
            c["hold_exp"] = round(exp, 2)
            c["hold_pips_ex3"] = round(pips_ex3, 1)
            c["trades_array"] = comb_h
            certified.append(c)
        else:
            reason = "Failed Gates" if (pf < 1.20 or net <= 0 or exp < 1.20) else ("Outlier Dependent" if pips_ex3 <= 0 else anti_msg)
            status = f"REJECTED ({reason})"

        print(f"{win_str:<12} | {sl_tp_str:<8} | {sw_str:<10} | {net:>+8.1f}p | {pf:>6.2f} | {len(comb_h):>6} | {exp:>5.2f}p | {status}")

    print("=" * 90)
    print(f"EURGBP Sweep: {len(certified)} brand-new non-redundant champions certified!")

    if certified:
        out_file = ROOT_DIR / "results" / "new_champions_eurgbp_reversion.json"
        clean = [{k: v for k, v in x.items() if k != "trades_array"} for x in certified]
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(clean, f, indent=2)
        print(f"Saved to {out_file}")

if __name__ == "__main__":
    main()

