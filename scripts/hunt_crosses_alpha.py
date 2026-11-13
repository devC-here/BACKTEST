#!/usr/bin/env python3
"""
================================================================================
  CROSS-PAIR ALPHA EXPLORATION ENGINE: AUDCAD, NZDCAD, GBPCHF
================================================================================
Explores novel, non-redundant, independently robust strategy candidates across:
  1. AUDCAD: Asian/Sydney-Tokyo Open Donchian Expansion (21:00-03:00 UTC) & NY Reversal
  2. NZDCAD: Wellington Cash Open Volatility Breakout (21:00-01:00 UTC) & London Retest
  3. GBPCHF: London Open Impulse Expansion (07:00-11:00 UTC)

Data & Execution Rules:
- Discovery: Weeks 1..210 (Jan 2021 - Dec 2024)
- Quarantined Holdout: Weeks 211..300 (Jan 2025 - Oct 2026)
- Full Live Friction: raw spread + adverse slippage (0.20p) + commissions:
    * AUDCAD: 0.56p commission + 0.20p slippage
    * NZDCAD: 0.56p commission + 0.20p slippage
    * GBPCHF: 0.33p commission + 0.20p slippage
- Strict subsequent-tick execution contract (T_fill > T_signal)

Independent Reliability Barrier:
- Holdout Trades >= 15
- Holdout PF >= 1.20
- Net Pips ex Top 3 trades > 0 (strictly outlier resistant)
- Discovery Net Pips > 0 and Discovery PF >= 1.15
- Anti-redundancy: Pairwise correlation < 0.50 and signal overlap < 50%
================================================================================
"""

import sys
import gc
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

FLEET_MANIFEST = ROOT_DIR / "certified_production_fleet" / "CERTIFIED_FLEET_MANIFEST.json"
MANIFEST_DIR = ROOT_DIR / "results" / "certified_strategy_manifests"

# Load existing fleet trades for anti-redundancy check
existing_trades_by_sym = defaultdict(list)
if FLEET_MANIFEST.exists():
    with open(FLEET_MANIFEST, "r", encoding="utf-8") as f:
        fleet_data = json.load(f)
    for s in fleet_data.get("strategies", []):
        sym = s["symbol"]
        name = s["name"]
        folder_slug = Path(s.get("relative_path", "")).parent.name
        csv_candidates = [
            MANIFEST_DIR / f"{sym}__{name}_trades.csv",
            MANIFEST_DIR / f"{sym}__{folder_slug}_trades.csv",
            MANIFEST_DIR / f"{sym.lower()}_{name}_trades.csv",
            MANIFEST_DIR / f"{name}_trades.csv"
        ]
        glob_matches = list(MANIFEST_DIR.glob(f"*{sym}*{name}*trades.csv"))
        if glob_matches:
            csv_candidates.extend(glob_matches)
        csv_file = next((f for f in csv_candidates if f.exists()), None)
        if csv_file:
            df = pd.read_csv(csv_file)
            if len(df) > 0:
                existing_trades_by_sym[sym].append((name, df))


def check_anti_redundancy(sym: str, candidate_trades: np.ndarray, max_overlap_pct: float = 50.0, max_corr: float = 0.50):
    if len(candidate_trades) == 0:
        return False, "No trades"
    existing_list = existing_trades_by_sym.get(sym, [])
    if not existing_list:
        return True, "No existing strategies"

    cand_times = candidate_trades['entry_time']
    cand_dirs = candidate_trades['direction']

    cand_df = pd.DataFrame({
        "entry_time": cand_times,
        "direction": cand_dirs,
        "pips": candidate_trades['pips']
    })
    cand_df["day"] = pd.to_datetime(cand_df["entry_time"], unit="ms", utc=True).dt.date
    cand_daily = cand_df.groupby("day")["pips"].sum()

    tolerance_ms = 30 * 60 * 1000

    for ex_name, ex_df in existing_list:
        ex_times = ex_df["entry_time"].values
        ex_dirs = ex_df["direction"].values if "direction" in ex_df else np.ones(len(ex_df))

        # Check overlap
        matches = 0
        for idx, t1 in enumerate(cand_times):
            diffs = np.abs(ex_times - t1)
            if np.any((diffs <= tolerance_ms) & (ex_dirs == cand_dirs[idx])):
                matches += 1

        overlap_pct = (matches / len(cand_times)) * 100.0 if len(cand_times) > 0 else 0
        if overlap_pct >= max_overlap_pct:
            return False, f"High overlap ({overlap_pct:.1f}%) with {ex_name}"

        # Check daily correlation
        ex_df_c = ex_df.copy()
        ex_df_c["day"] = pd.to_datetime(ex_df_c["entry_time"], unit="ms", utc=True).dt.date
        ex_daily = ex_df_c.groupby("day")["pips"].sum()

        all_days = sorted(list(set(cand_daily.index).union(set(ex_daily.index))))
        c_series = [cand_daily.get(d, 0.0) for d in all_days]
        e_series = [ex_daily.get(d, 0.0) for d in all_days]
        corr = np.corrcoef(c_series, e_series)[0, 1] if len(all_days) > 5 else 0.0
        if np.isnan(corr): corr = 0.0

        if corr >= max_corr:
            return False, f"High correlation ({corr:.2f}) with {ex_name}"

    return True, "PASSED Anti-Redundancy"


def precompute_week_bundle(symbol: str, week_num: int):
    """Builds rich multi-timeframe feature cache for one week of tick data."""
    pip_size = PIP_SIZES.get(symbol, 0.0001)
    raw_dir = ROOT_DIR / "data" / symbol
    try:
        ticks, _ = get_week_dataset(week=week_num, raw_dir=raw_dir, warmup_prev_week_ticks=WARMUP_TICKS)
    except Exception:
        return None
    if len(ticks) <= WARMUP_TICKS + 500:
        return None

    m15 = build_time_bars(ticks, timeframe_min=15, pip_size=pip_size)
    if len(m15) < 35:
        return None

    # H1 bars
    h1 = build_time_bars(ticks, timeframe_min=60, pip_size=pip_size)
    if len(h1) < 20:
        return None

    # H1 EMA20 and trend
    h1_c = h1['close'].values
    h1_ema20 = pd.Series(h1_c).ewm(span=20, adjust=False).mean().values
    h1_trend = np.where(h1_c > h1_ema20, 1, -1)
    h1_df = pd.DataFrame({'h1_trend': h1_trend}, index=h1.index)

    m15_merged = pd.merge_asof(
        m15.sort_index(),
        h1_df.sort_index(),
        left_index=True,
        right_index=True,
        direction='backward'
    )
    h1_bias_m15 = m15_merged['h1_trend'].fillna(0).values.astype(np.int8)

    # M15 series
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

    # M15 EMA50
    ema50 = pd.Series(c).ewm(span=50, adjust=False).mean().values.astype(np.float64)
    ema20 = pd.Series(c).ewm(span=20, adjust=False).mean().values.astype(np.float64)

    # ATR14 on M15
    tr = np.maximum(h - l, np.maximum(np.abs(h - np.roll(c, 1)), np.abs(l - np.roll(c, 1))))
    tr[0] = h[0] - l[0]
    atr14 = pd.Series(tr).rolling(14).mean().bfill().values.astype(np.float64)

    # Rolling Donchian channels for various lookbacks on M15: N = 8, 12, 16, 20
    donchian = {}
    for n_lb in [8, 12, 16, 20]:
        roll_h = pd.Series(h).rolling(n_lb).max().shift(1).bfill().values.astype(np.float64)
        roll_l = pd.Series(l).rolling(n_lb).min().shift(1).bfill().values.astype(np.float64)
        donchian[n_lb] = (roll_h, roll_l)

    # Day-based session levels:
    # 1. Asian range (00:00 - 06:45 UTC)
    # 2. Wellington pre-session range (17:00 - 20:45 UTC)
    # 3. NY afternoon pre-range (00:00 - 15:45 UTC)
    n_bars = len(m15)
    asian_h = np.zeros(n_bars, dtype=np.float64)
    asian_l = np.zeros(n_bars, dtype=np.float64)
    wellington_pre_h = np.zeros(n_bars, dtype=np.float64)
    wellington_pre_l = np.zeros(n_bars, dtype=np.float64)

    # Compute daily/session boundaries
    day_indices = defaultdict(list)
    for i, d in enumerate(dates):
        day_indices[d].append(i)

    for d, idxs in day_indices.items():
        # Asian 00:00 - 06:45
        as_idxs = [i for i in idxs if 0 <= hours[i] <= 6]
        if as_idxs:
            ash = np.max(h[as_idxs])
            asl = np.min(l[as_idxs])
            for i in idxs:
                asian_h[i] = ash
                asian_l[i] = asl

        # Wellington pre-session: 17:00 - 20:45
        well_idxs = [i for i in idxs if 17 <= hours[i] <= 20]
        if well_idxs:
            wh = np.max(h[well_idxs])
            wl = np.min(l[well_idxs])
            for i in idxs:
                wellington_pre_h[i] = wh
                wellington_pre_l[i] = wl

    return {
        'ticks': ticks,
        'n_ticks': n_ticks,
        'pip_size': pip_size,
        'min_entry_ts': ticks['timestamp'][WARMUP_TICKS],
        'h1_bias': h1_bias_m15,
        'c': c, 'h': h, 'l': l, 'o': o,
        'spread': spread,
        'hours': hours,
        'minutes': minutes,
        'dates': dates,
        'exec_indices': exec_indices,
        'ema50': ema50,
        'ema20': ema20,
        'atr14': atr14,
        'donchian': donchian,
        'asian_h': asian_h,
        'asian_l': asian_l,
        'wellington_pre_h': wellington_pre_h,
        'wellington_pre_l': wellington_pre_l,
        'n_bars': n_bars,
        'signals_buf': np.zeros(n_ticks, dtype=np.int8),
        'sl_buf': np.zeros(n_ticks, dtype=np.float64),
        'tp_buf': np.zeros(n_ticks, dtype=np.float64)
    }


def execute_signals_on_week(week_data, sig_indices_and_types, sl_pips, tp_pips, comm, slip):
    """Executes pre-identified signals through matching loop and cleans up buffers."""
    if not sig_indices_and_types:
        return np.empty(0)

    n_ticks = week_data['n_ticks']
    signals_buf = week_data['signals_buf']
    sl_buf = week_data['sl_buf']
    tp_buf = week_data['tp_buf']
    pip_size = week_data['pip_size']

    fired = []
    for idx, sig in sig_indices_and_types:
        if idx < n_ticks and signals_buf[idx] == 0:
            signals_buf[idx] = sig
            sl_buf[idx] = sl_pips
            tp_buf[idx] = tp_pips
            fired.append(idx)

    if not fired:
        return np.empty(0)

    tr = run_matching_loop(
        ticks=week_data['ticks'],
        signals=signals_buf,
        pip_size=pip_size,
        sl_pips=sl_buf,
        tp_pips=tp_buf,
        use_slippage=True,
        close_at_end=True,
        commission_pips=comm,
        entry_slippage_pips=slip,
        min_entry_timestamp=week_data['min_entry_ts'],
        enforce_subsequent_tick=True
    )

    for idx in fired:
        signals_buf[idx] = 0
        sl_buf[idx] = 0.0
        tp_buf[idx] = 0.0

    return tr


# ==============================================================================
# STRATEGY EVALUATORS
# ==============================================================================

def eval_audcad_asian_donchian(week_data, lookback: int, min_body: float, start_hr: int, end_hr: int, sl_pips: float, tp_pips: float, max_spread: float, comm: float, slip: float):
    """
    AUDCAD: Asian/Sydney-Tokyo Open Donchian Expansion (21:00-03:00 UTC).
    Breaks rolling N-bar Donchian channel during Asian open window.
    """
    n_bars = week_data['n_bars']
    c, h, l, o = week_data['c'], week_data['h'], week_data['l'], week_data['o']
    hours = week_data['hours']
    dates = week_data['dates']
    spread = week_data['spread']
    ema50 = week_data['ema50']
    exec_indices = week_data['exec_indices']
    roll_h, roll_l = week_data['donchian'][lookback]

    pip_size = week_data['pip_size']
    last_trade_date = None
    signals_to_fire = []

    for i in range(lookback + 2, n_bars):
        hr = hours[i]
        if start_hr <= end_hr:
            in_window = (start_hr <= hr <= end_hr)
        else:
            in_window = (hr >= start_hr or hr <= end_hr)

        if not in_window:
            continue
        dt = dates[i]
        if dt == last_trade_date:
            continue
        if spread[i] > max_spread:
            continue

        bar_rng = h[i] - l[i]
        if bar_rng < 2.5 * pip_size:
            continue

        sig = 0
        # Donchian Upper Breakout
        if c[i] > roll_h[i] and c[i] > o[i]:
            body = c[i] - o[i]
            if (body / bar_rng) >= min_body and c[i] > ema50[i]:
                sig = 1
        # Donchian Lower Breakout
        elif c[i] < roll_l[i] and c[i] < o[i]:
            body = o[i] - c[i]
            if (body / bar_rng) >= min_body and c[i] < ema50[i]:
                sig = -1

        if sig != 0:
            signals_to_fire.append((exec_indices[i], sig))
            last_trade_date = dt

    return execute_signals_on_week(week_data, signals_to_fire, sl_pips, tp_pips, comm, slip)


def eval_nzdcad_wellington_breakout(week_data, buffer_pips: float, min_body: float, start_hr: int, end_hr: int, sl_pips: float, tp_pips: float, max_spread: float, comm: float, slip: float):
    """
    NZDCAD: Wellington Cash Open Volatility Breakout (21:00-01:00 UTC).
    Breaks 17:00-20:45 pre-session consolidation range.
    """
    n_bars = week_data['n_bars']
    c, h, l, o = week_data['c'], week_data['h'], week_data['l'], week_data['o']
    hours = week_data['hours']
    dates = week_data['dates']
    spread = week_data['spread']
    ema50 = week_data['ema50']
    exec_indices = week_data['exec_indices']
    wh = week_data['wellington_pre_h']
    wl = week_data['wellington_pre_l']
    pip_size = week_data['pip_size']

    last_trade_date = None
    signals_to_fire = []

    for i in range(15, n_bars):
        hr = hours[i]
        if start_hr <= end_hr:
            in_window = (start_hr <= hr <= end_hr)
        else:
            in_window = (hr >= start_hr or hr <= end_hr)

        if not in_window:
            continue
        dt = dates[i]
        if dt == last_trade_date:
            continue
        if spread[i] > max_spread:
            continue

        pre_h = wh[i]
        pre_l = wl[i]
        if pre_h <= 0 or pre_l <= 0:
            continue

        pre_range_pips = (pre_h - pre_l) / pip_size
        if pre_range_pips < 8.0 or pre_range_pips > 45.0:
            continue

        bar_rng = h[i] - l[i]
        if bar_rng < 2.5 * pip_size:
            continue

        sig = 0
        buf = buffer_pips * pip_size
        # Bullish Breakout above Wellington Pre-Range
        if c[i] > (pre_h + buf) and c[i] > o[i]:
            body = c[i] - o[i]
            if (body / bar_rng) >= min_body and c[i] > ema50[i]:
                sig = 1
        # Bearish Breakout below Wellington Pre-Range
        elif c[i] < (pre_l - buf) and c[i] < o[i]:
            body = o[i] - c[i]
            if (body / bar_rng) >= min_body and c[i] < ema50[i]:
                sig = -1

        if sig != 0:
            signals_to_fire.append((exec_indices[i], sig))
            last_trade_date = dt

    return execute_signals_on_week(week_data, signals_to_fire, sl_pips, tp_pips, comm, slip)


def eval_gbpchf_london_impulse(week_data, buffer_pips: float, min_body: float, min_bar_pips: float, start_hr: int, end_hr: int, sl_pips: float, tp_pips: float, max_spread: float, comm: float, slip: float):
    """
    GBPCHF: London Open Impulse Expansion (07:00-11:00 UTC).
    Breaks Asian session (00:00-06:45 UTC) consolidation range with momentum.
    """
    n_bars = week_data['n_bars']
    c, h, l, o = week_data['c'], week_data['h'], week_data['l'], week_data['o']
    hours = week_data['hours']
    dates = week_data['dates']
    spread = week_data['spread']
    h1_bias = week_data['h1_bias']
    ema50 = week_data['ema50']
    exec_indices = week_data['exec_indices']
    ash = week_data['asian_h']
    asl = week_data['asian_l']
    pip_size = week_data['pip_size']

    last_trade_date = None
    signals_to_fire = []

    for i in range(15, n_bars):
        hr = hours[i]
        if not (start_hr <= hr <= end_hr):
            continue
        dt = dates[i]
        if dt == last_trade_date:
            continue
        if spread[i] > max_spread:
            continue

        ah = ash[i]
        al = asl[i]
        if ah <= 0 or al <= 0:
            continue

        asian_range_pips = (ah - al) / pip_size
        if asian_range_pips < 12.0 or asian_range_pips > 55.0:
            continue

        bar_rng_pips = (h[i] - l[i]) / pip_size
        if bar_rng_pips < min_bar_pips:
            continue

        sig = 0
        buf = buffer_pips * pip_size
        # Bullish Breakout above Asian High
        if c[i] > (ah + buf) and c[i] > o[i]:
            body = c[i] - o[i]
            if (body / (h[i] - l[i])) >= min_body and c[i] > ema50[i]:
                sig = 1
        # Bearish Breakout below Asian Low
        elif c[i] < (al - buf) and c[i] < o[i]:
            body = o[i] - c[i]
            if (body / (h[i] - l[i])) >= min_body and c[i] < ema50[i]:
                sig = -1

        if sig != 0:
            signals_to_fire.append((exec_indices[i], sig))
            last_trade_date = dt

    return execute_signals_on_week(week_data, signals_to_fire, sl_pips, tp_pips, comm, slip)


def hunt_symbol(symbol: str, archetype: str = "auto"):
    pip_size = PIP_SIZES.get(symbol, 0.0001)
    comm, slip = get_friction_pips(symbol)

    disc_weeks = get_complete_split_weeks(symbol, "discovery_70")
    hold_weeks = get_complete_split_weeks(symbol, "holdout_test_30")

    # Sample discovery weeks for fast screening (every 2nd week = 105 weeks)
    sample_disc_weeks = disc_weeks[::2]

    print("=" * 80, flush=True)
    print(f"  HUNTING CROSS-PAIR ALPHA: {symbol} (Archetype: {archetype})", flush=True)
    print(f"  Discovery Sample: {len(sample_disc_weeks)}w | Full Discovery: {len(disc_weeks)}w | Quarantined Holdout: {len(hold_weeks)}w", flush=True)
    print(f"  Friction: {comm:.2f}p Commission + {slip:.2f}p Slippage + Raw Floating Spread", flush=True)
    print("=" * 80, flush=True)

    t0 = time.time()
    print("Precomputing Discovery cache...", flush=True)
    with ProcessPoolExecutor(max_workers=6) as ex:
        disc_cache = [w for w in ex.map(precompute_week_bundle, [symbol]*len(sample_disc_weeks), sample_disc_weeks) if w is not None]
    print(f"Discovery cache built ({len(disc_cache)}w) in {time.time()-t0:.1f}s.", flush=True)

    # Define grids per symbol & archetype
    param_grid = []
    if symbol == "AUDCAD":
        max_spread = 2.8
        for lb in [8, 12, 16]:
            for min_b in [0.45, 0.55]:
                for s_hr, e_hr, s_name in [(21, 3, "Asian_Sydney_Tokyo"), (22, 3, "Sydney_Tokyo_Late"), (21, 2, "Sydney_Tokyo_Core")]:
                    for sl, tp in [(12.0, 24.0), (14.0, 28.0), (14.0, 32.0), (16.0, 32.0), (16.0, 36.0), (18.0, 40.0), (20.0, 45.0)]:
                        param_grid.append({
                            "archetype": "audcad_asian_donchian",
                            "lookback": lb,
                            "min_body": min_b,
                            "start_hr": s_hr,
                            "end_hr": e_hr,
                            "session_name": s_name,
                            "sl_pips": sl,
                            "tp_pips": tp,
                            "max_spread": max_spread
                        })
    elif symbol == "NZDCAD":
        max_spread = 3.0
        for buf in [0.0, 1.5, 3.0]:
            for min_b in [0.45, 0.55]:
                for s_hr, e_hr, s_name in [(21, 1, "Wellington_Open"), (21, 2, "Wellington_Tokyo_Core"), (22, 2, "Wellington_Late")]:
                    for sl, tp in [(14.0, 28.0), (16.0, 32.0), (16.0, 36.0), (18.0, 40.0), (18.0, 45.0), (20.0, 45.0)]:
                        param_grid.append({
                            "archetype": "nzdcad_wellington_breakout",
                            "buffer_pips": buf,
                            "min_body": min_b,
                            "start_hr": s_hr,
                            "end_hr": e_hr,
                            "session_name": s_name,
                            "sl_pips": sl,
                            "tp_pips": tp,
                            "max_spread": max_spread
                        })
    elif symbol == "GBPCHF":
        max_spread = 3.2
        for buf in [0.0, 2.0, 4.0]:
            for min_b in [0.45, 0.55]:
                for min_bar in [5.0, 7.5]:
                    for s_hr, e_hr, s_name in [(7, 10, "London_Open_Early"), (7, 11, "London_Open_Extended"), (8, 11, "London_Mid_Open")]:
                        for sl, tp in [(16.0, 34.0), (18.0, 38.0), (18.0, 45.0), (20.0, 44.0), (20.0, 50.0), (22.0, 55.0)]:
                            param_grid.append({
                                "archetype": "gbpchf_london_impulse",
                                "buffer_pips": buf,
                                "min_body": min_b,
                                "min_bar_pips": min_bar,
                                "start_hr": s_hr,
                                "end_hr": e_hr,
                                "session_name": s_name,
                                "sl_pips": sl,
                                "tp_pips": tp,
                                "max_spread": max_spread
                            })

    print(f"Screening {len(param_grid)} parameter variations on Discovery sample...", flush=True)

    promising_candidates = []
    for p in param_grid:
        trades = []
        for w in disc_cache:
            if p["archetype"] == "audcad_asian_donchian":
                tr = eval_audcad_asian_donchian(w, p["lookback"], p["min_body"], p["start_hr"], p["end_hr"], p["sl_pips"], p["tp_pips"], p["max_spread"], comm, slip)
            elif p["archetype"] == "nzdcad_wellington_breakout":
                tr = eval_nzdcad_wellington_breakout(w, p["buffer_pips"], p["min_body"], p["start_hr"], p["end_hr"], p["sl_pips"], p["tp_pips"], p["max_spread"], comm, slip)
            elif p["archetype"] == "gbpchf_london_impulse":
                tr = eval_gbpchf_london_impulse(w, p["buffer_pips"], p["min_body"], p["min_bar_pips"], p["start_hr"], p["end_hr"], p["sl_pips"], p["tp_pips"], p["max_spread"], comm, slip)
            if len(tr) > 0:
                trades.append(tr)

        if not trades:
            continue
        comb = np.concatenate(trades)
        if len(comb) < 15:
            continue

        pips = comb['pips']
        wins = float(np.sum(pips[pips > 0]))
        losses = float(abs(np.sum(pips[pips < 0])))
        pf = (wins / losses) if losses > 0 else 99.0
        net = wins - losses
        exp = net / len(comb)

        if pf >= 1.20 and net > 0 and exp >= 1.0:
            cand = dict(p)
            cand["sample_disc_trades"] = len(comb)
            cand["sample_disc_pips"] = round(net, 1)
            cand["sample_disc_pf"] = round(pf, 2)
            cand["sample_disc_exp"] = round(exp, 2)
            promising_candidates.append(cand)

    print(f"Discovered {len(promising_candidates)} promising candidates passing Discovery Screen.", flush=True)
    if not promising_candidates:
        return []

    # Free discovery cache memory before holdout
    del disc_cache
    gc.collect()

    # Precompute Quarantined Holdout Cache (weeks 211..300)
    print("Precomputing Quarantined Holdout Cache (90 weeks)...", flush=True)
    t1 = time.time()
    with ProcessPoolExecutor(max_workers=6) as ex:
        hold_cache = [w for w in ex.map(precompute_week_bundle, [symbol]*len(hold_weeks), hold_weeks) if w is not None]
    print(f"Holdout cache built ({len(hold_cache)}w) in {time.time()-t1:.1f}s.", flush=True)

    print("=" * 110, flush=True)
    print(f"{'Archetype':<24} | {'Params / Session':<30} | {'SL/TP':<9} | {'Hold Net':<9} | {'Hold PF':<7} | {'Trades':<6} | {'ExTop3':<8} | {'Status'}", flush=True)
    print("-" * 110, flush=True)

    certified = []
    for c in promising_candidates:
        trades = []
        for w in hold_cache:
            if c["archetype"] == "audcad_asian_donchian":
                tr = eval_audcad_asian_donchian(w, c["lookback"], c["min_body"], c["start_hr"], c["end_hr"], c["sl_pips"], c["tp_pips"], c["max_spread"], comm, slip)
            elif c["archetype"] == "nzdcad_wellington_breakout":
                tr = eval_nzdcad_wellington_breakout(w, c["buffer_pips"], c["min_body"], c["start_hr"], c["end_hr"], c["sl_pips"], c["tp_pips"], c["max_spread"], comm, slip)
            elif c["archetype"] == "gbpchf_london_impulse":
                tr = eval_gbpchf_london_impulse(w, c["buffer_pips"], c["min_body"], c["min_bar_pips"], c["start_hr"], c["end_hr"], c["sl_pips"], c["tp_pips"], c["max_spread"], comm, slip)
            if len(tr) > 0:
                trades.append(tr)

        if not trades:
            continue
        comb_h = np.concatenate(trades)
        if len(comb_h) < 15:
            continue

        pips = comb_h['pips']
        wins = float(np.sum(pips[pips > 0]))
        losses = float(abs(np.sum(pips[pips < 0])))
        pf = (wins / losses) if losses > 0 else 99.0
        net = wins - losses
        exp = net / len(comb_h)

        pips_sorted = np.sort(pips)
        pips_ex3 = float(np.sum(pips_sorted[:-3]))

        is_distinct, anti_red_msg = check_anti_redundancy(symbol, comb_h, max_overlap_pct=50.0, max_corr=0.50)

        sl_tp_str = f"{c['sl_pips']:.0f}/{c['tp_pips']:.0f}"
        param_desc = f"{c['session_name']}"
        if "lookback" in c:
            param_desc += f" lb{c['lookback']}"
        if "buffer_pips" in c:
            param_desc += f" buf{c['buffer_pips']:.1f}"

        # Reliability Barrier: Holdout Trades >= 15, Holdout PF >= 1.20, Net Pips > 0, Net Pips ex Top 3 > 0, Distinct
        passed_barrier = (len(comb_h) >= 15 and pf >= 1.20 and net > 0 and pips_ex3 > 0 and is_distinct)

        if passed_barrier:
            status = "CERTIFIED"
            c["symbol"] = symbol
            c["hold_trades"] = len(comb_h)
            c["hold_pips"] = round(net, 1)
            c["hold_pf"] = round(pf, 2)
            c["hold_exp"] = round(exp, 2)
            c["hold_pips_ex3"] = round(pips_ex3, 1)
            c["trades_array"] = comb_h
            certified.append(c)
        else:
            reason = "Failed Barrier" if (pf < 1.20 or net <= 0) else ("Outlier Dependent" if pips_ex3 <= 0 else anti_red_msg)
            status = f"REJECTED ({reason})"

        print(f"{c['archetype']:<24} | {param_desc:<30} | {sl_tp_str:<9} | {net:>+8.1f}p | {pf:>7.2f} | {len(comb_h):>6} | {pips_ex3:>+7.1f}p | {status}", flush=True)

    print("=" * 110, flush=True)
    print(f"Symbol {symbol}: Certified {len(certified)} independently robust, non-redundant champions!", flush=True)
    return certified


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="AUDCAD", choices=["AUDCAD", "NZDCAD", "GBPCHF", "ALL"])
    args = parser.parse_args()

    symbols = ["AUDCAD", "NZDCAD", "GBPCHF"] if args.symbol == "ALL" else [args.symbol]

    all_champions = {}
    for s in symbols:
        champs = hunt_symbol(s)
        if champs:
            all_champions[s] = champs
            # Save champion results
            out_json = ROOT_DIR / "results" / f"cross_pair_champions_{s.lower()}.json"
            serializable = [{k: v for k, v in c.items() if k != "trades_array"} for c in champs]
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(serializable, f, indent=2)
            print(f"Saved {len(serializable)} champions to {out_json}", flush=True)
