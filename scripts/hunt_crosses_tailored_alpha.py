#!/usr/bin/env python3
"""
================================================================================
  TAILORED CROSS-PAIR ALPHA EXPLORATION ENGINE: AUDCAD, NZDCAD, GBPCHF
================================================================================
Explores the high-probability microstructure regimes tailored to the 84-87%
sweep & fakeout tendencies and session momentum of the target crosses:

1. AUDCAD:
   - NY Afternoon Reversal / Daily Extreme Fade (15:00 - 20:00 UTC)
   - Asian Tokyo Boundary Fade / Rejection Retest (22:00 - 03:00 UTC)
2. NZDCAD:
   - London Morning Retest & Asian Liquidity Fade (07:00 - 10:30 UTC)
   - Wellington Cash Open Volatility Breakout (21:00 - 01:00 UTC)
3. GBPCHF:
   - H1 Keltner Asymmetric Runner (2.2R - 3.0R) (07:00 - 16:00 UTC)
   - London Open Impulse Expansion (07:00 - 10:30 UTC)

Data & Execution Rules:
- Discovery: Weeks 1..210 (Jan 2021 - Dec 2024)
- Quarantined Holdout: Weeks 211..300 (Jan 2025 - Oct 2026)
- Full Live Friction (Commissions + 0.20p slippage + raw spread)
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

# Load existing fleet trades for anti-redundancy checks
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


def precompute_comprehensive_week(symbol: str, week_num: int):
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

    # H1 bars for Keltner Squeeze & Trend
    h1 = build_time_bars(ticks, timeframe_min=60, pip_size=pip_size)
    if len(h1) < 20:
        return None

    h1_c = h1['close'].values
    h1_h = h1['high'].values
    h1_l = h1['low'].values
    c_s = pd.Series(h1_c)
    h1_ema20 = c_s.ewm(span=20, adjust=False).mean().values
    h1_std20 = c_s.rolling(20).std().values
    bb_upper = h1_ema20 + 2.0 * h1_std20
    bb_lower = h1_ema20 - 2.0 * h1_std20

    tr_h1 = np.maximum(h1_h - h1_l, np.maximum(np.abs(h1_h - np.roll(h1_c, 1)), np.abs(h1_l - np.roll(h1_c, 1))))
    tr_h1[0] = h1_h[0] - h1_l[0]
    atr20_h1 = pd.Series(tr_h1).rolling(20).mean().bfill().values

    # Precalculate squeeze for mult 1.5 and 1.8
    keltner_data = {}
    for mult in [1.5, 1.8]:
        kc_upper = h1_ema20 + mult * atr20_h1
        kc_lower = h1_ema20 - mult * atr20_h1
        sqz_on = (bb_lower > kc_lower) & (bb_upper < kc_upper)
        sqz_fired = pd.Series(sqz_on).shift(1).fillna(False).values & (~sqz_on)
        h1_mom = np.where(h1_c > h1_ema20, 1, np.where(h1_c < h1_ema20, -1, 0))
        h1_df = pd.DataFrame({'sqz_fire': sqz_fired, 'h1_mom': h1_mom}, index=h1.index)
        m15_merged = pd.merge_asof(m15.sort_index(), h1_df.sort_index(), left_index=True, right_index=True, direction='backward')
        keltner_data[mult] = {
            'sqz_fire': m15_merged['sqz_fire'].fillna(False).values,
            'h1_mom': m15_merged['h1_mom'].fillna(0).values.astype(np.int8)
        }

    # M15 series
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
    ema50 = pd.Series(c).ewm(span=50, adjust=False).mean().values.astype(np.float64)
    ema20 = pd.Series(c).ewm(span=20, adjust=False).mean().values.astype(np.float64)

    # Intraday tracking:
    # 1. Asian range: 00:00 to 06:45 UTC
    # 2. Daily range up to 15:00 UTC: day_h_15, day_l_15
    n_bars = len(m15)
    asian_h = np.zeros(n_bars, dtype=np.float64)
    asian_l = np.zeros(n_bars, dtype=np.float64)
    day_h_15 = np.zeros(n_bars, dtype=np.float64)
    day_l_15 = np.zeros(n_bars, dtype=np.float64)

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

        # Daily levels before 15:00 UTC
        pre15_idxs = [i for i in idxs if hours[i] < 15]
        if pre15_idxs:
            dh = np.max(h[pre15_idxs])
            dl = np.min(l[pre15_idxs])
            for i in idxs:
                day_h_15[i] = dh
                day_l_15[i] = dl

    return {
        'ticks': ticks,
        'n_ticks': n_ticks,
        'pip_size': pip_size,
        'min_entry_ts': ticks['timestamp'][WARMUP_TICKS],
        'keltner_data': keltner_data,
        'c': c, 'h': h, 'l': l, 'o': o,
        'spread': spread,
        'hours': hours,
        'dates': dates,
        'exec_indices': exec_indices,
        'ema50': ema50,
        'ema20': ema20,
        'asian_h': asian_h,
        'asian_l': asian_l,
        'day_h_15': day_h_15,
        'day_l_15': day_l_15,
        'n_bars': n_bars,
        'signals_buf': np.zeros(n_ticks, dtype=np.int8),
        'sl_buf': np.zeros(n_ticks, dtype=np.float64),
        'tp_buf': np.zeros(n_ticks, dtype=np.float64)
    }


def execute_signals_on_week(week_data, sig_indices_and_types, sl_pips, tp_pips, comm, slip):
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
# SPECIALIZED ARCHETYPES
# ==============================================================================

def eval_ny_afternoon_reversal(week_data, min_wick: float, start_hr: int, end_hr: int, sl_pips: float, tp_pips: float, max_spread: float, comm: float, slip: float):
    """
    AUDCAD: NY Afternoon Reversal / Session Extreme Fade (15:00 - 20:00 UTC).
    Exploits exhaustion of daily moves after London close.
    """
    n_bars = week_data['n_bars']
    c, h, l, o = week_data['c'], week_data['h'], week_data['l'], week_data['o']
    hours = week_data['hours']
    dates = week_data['dates']
    spread = week_data['spread']
    exec_indices = week_data['exec_indices']
    day_h = week_data['day_h_15']
    day_l = week_data['day_l_15']
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

        dh = day_h[i]
        dl = day_l[i]
        if dh <= 0 or dl <= 0:
            continue

        bar_rng = h[i] - l[i]
        if bar_rng < 2.5 * pip_size:
            continue

        upper_wick = h[i] - max(o[i], c[i])
        lower_wick = min(o[i], c[i]) - l[i]

        sig = 0
        # Bearish Reversal from Daily High
        if h[i] >= (dh - 1.5 * pip_size) and c[i] < dh and (upper_wick / bar_rng) >= min_wick and c[i] < o[i]:
            sig = -1
        # Bullish Reversal from Daily Low
        elif l[i] <= (dl + 1.5 * pip_size) and c[i] > dl and (lower_wick / bar_rng) >= min_wick and c[i] > o[i]:
            sig = 1

        if sig != 0:
            signals_to_fire.append((exec_indices[i], sig))
            last_trade_date = dt

    return execute_signals_on_week(week_data, signals_to_fire, sl_pips, tp_pips, comm, slip)


def eval_asian_liquidity_fade(week_data, min_wick: float, start_hr: int, end_hr: int, sl_pips: float, tp_pips: float, max_spread: float, comm: float, slip: float):
    """
    NZDCAD: London Morning Retest & Asian Liquidity Sweep Fade (07:00 - 10:30 UTC).
    Exploits 84% Asian breakout fakeout rate via liquidity rejection.
    """
    n_bars = week_data['n_bars']
    c, h, l, o = week_data['c'], week_data['h'], week_data['l'], week_data['o']
    hours = week_data['hours']
    dates = week_data['dates']
    spread = week_data['spread']
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

        bar_rng = h[i] - l[i]
        if bar_rng < 2.5 * pip_size:
            continue

        upper_wick = h[i] - max(o[i], c[i])
        lower_wick = min(o[i], c[i]) - l[i]

        sig = 0
        # Sweep Asian High and Reject
        if h[i] >= (ah - 1.5 * pip_size) and c[i] < ah and (upper_wick / bar_rng) >= min_wick and c[i] < o[i]:
            sig = -1
        # Sweep Asian Low and Reject
        elif l[i] <= (al + 1.5 * pip_size) and c[i] > al and (lower_wick / bar_rng) >= min_wick and c[i] > o[i]:
            sig = 1

        if sig != 0:
            signals_to_fire.append((exec_indices[i], sig))
            last_trade_date = dt

    return execute_signals_on_week(week_data, signals_to_fire, sl_pips, tp_pips, comm, slip)


def hunt_tailored_for_symbol(symbol: str):
    pip_size = PIP_SIZES.get(symbol, 0.0001)
    comm, slip = get_friction_pips(symbol)

    disc_weeks = get_complete_split_weeks(symbol, "discovery_70")
    hold_weeks = get_complete_split_weeks(symbol, "holdout_test_30")

    # Sample discovery weeks for screening (every 2nd week = 105 weeks)
    sample_disc_weeks = disc_weeks[::2]

    print("=" * 80, flush=True)
    print(f"  TAILORED ALPHA EXPLORATION FOR {symbol}", flush=True)
    print(f"  Discovery Sample: {len(sample_disc_weeks)}w | Quarantined Holdout: {len(hold_weeks)}w", flush=True)
    print(f"  Friction: {comm:.2f}p Commission + {slip:.2f}p Slippage + Raw Floating Spread", flush=True)
    print("=" * 80, flush=True)

    t0 = time.time()
    print("Precomputing Discovery cache...", flush=True)
    with ProcessPoolExecutor(max_workers=6) as ex:
        disc_cache = [w for w in ex.map(precompute_comprehensive_week, [symbol]*len(sample_disc_weeks), sample_disc_weeks) if w is not None]
    print(f"Discovery cache built ({len(disc_cache)}w) in {time.time()-t0:.1f}s.", flush=True)

    param_grid = []
    if symbol == "AUDCAD":
        # NY Afternoon Reversal / Extreme Fade
        for min_w in [0.30, 0.40]:
            for s_hr, e_hr in [(15, 19), (16, 20), (15, 20)]:
                for sl, tp in [(12.0, 24.0), (14.0, 28.0), (14.0, 32.0), (16.0, 32.0), (16.0, 36.0), (18.0, 36.0), (18.0, 42.0)]:
                    for ms in [2.2, 2.6]:
                        param_grid.append({
                            "archetype": "audcad_ny_reversal",
                            "min_wick": min_w,
                            "start_hr": s_hr,
                            "end_hr": e_hr,
                            "sl_pips": sl,
                            "tp_pips": tp,
                            "max_spread": ms,
                            "desc": f"NY_Fade_{s_hr}-{e_hr}_w{int(min_w*100)}"
                        })
    elif symbol == "NZDCAD":
        # London Morning Retest & Asian Liquidity Fade
        for min_w in [0.30, 0.40]:
            for s_hr, e_hr in [(7, 10), (7, 11), (8, 11)]:
                for sl, tp in [(12.0, 24.0), (14.0, 28.0), (14.0, 32.0), (16.0, 32.0), (16.0, 36.0), (18.0, 36.0), (18.0, 42.0)]:
                    for ms in [2.4, 2.8]:
                        param_grid.append({
                            "archetype": "nzdcad_asian_fade",
                            "min_wick": min_w,
                            "start_hr": s_hr,
                            "end_hr": e_hr,
                            "sl_pips": sl,
                            "tp_pips": tp,
                            "max_spread": ms,
                            "desc": f"London_Retest_{s_hr}-{e_hr}_w{int(min_w*100)}"
                        })
    elif symbol == "GBPCHF":
        # H1 Keltner Asymmetric Runner
        for kc in [1.5, 1.8]:
            for s_hr, e_hr in [(7, 16), (7, 12), (8, 16)]:
                for sl, tp in [(16.0, 40.0), (18.0, 45.0), (18.0, 50.0), (20.0, 48.0), (20.0, 55.0), (22.0, 60.0)]:
                    for ms in [3.0, 3.2]:
                        param_grid.append({
                            "archetype": "gbpchf_keltner_runner",
                            "kc_mult": kc,
                            "start_hr": s_hr,
                            "end_hr": e_hr,
                            "sl_pips": sl,
                            "tp_pips": tp,
                            "max_spread": ms,
                            "desc": f"KC{kc:.1f}_{s_hr}-{e_hr}"
                        })

    print(f"Screening {len(param_grid)} parameter variations on Discovery sample...", flush=True)

    promising = []
    for p in param_grid:
        trades = []
        for w in disc_cache:
            if p["archetype"] == "audcad_ny_reversal":
                tr = eval_ny_afternoon_reversal(w, p["min_wick"], p["start_hr"], p["end_hr"], p["sl_pips"], p["tp_pips"], p["max_spread"], comm, slip)
            elif p["archetype"] == "nzdcad_asian_fade":
                tr = eval_asian_liquidity_fade(w, p["min_wick"], p["start_hr"], p["end_hr"], p["sl_pips"], p["tp_pips"], p["max_spread"], comm, slip)
            elif p["archetype"] == "gbpchf_keltner_runner":
                tr = eval_keltner_asymmetric_runner(w, p["kc_mult"], p["start_hr"], p["end_hr"], p["sl_pips"], p["tp_pips"], p["max_spread"], comm, slip)
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
            c_dict = dict(p)
            c_dict["disc_trades"] = len(comb)
            c_dict["disc_pips"] = round(net, 1)
            c_dict["disc_pf"] = round(pf, 2)
            c_dict["disc_exp"] = round(exp, 2)
            promising.append(c_dict)

    print(f"Discovery Screening complete: {len(promising)} promising candidates qualified.", flush=True)
    if not promising:
        return []

    # Free discovery cache
    del disc_cache
    gc.collect()

    print("Precomputing Quarantined Holdout Cache (90 weeks)...", flush=True)
    t1 = time.time()
    with ProcessPoolExecutor(max_workers=6) as ex:
        hold_cache = [w for w in ex.map(precompute_comprehensive_week, [symbol]*len(hold_weeks), hold_weeks) if w is not None]
    print(f"Holdout cache built ({len(hold_cache)}w) in {time.time()-t1:.1f}s.", flush=True)

    print("=" * 115, flush=True)
    print(f"{'Archetype':<24} | {'Setup Desc':<26} | {'SL/TP':<9} | {'Hold Net':<9} | {'Hold PF':<7} | {'Trades':<6} | {'ExTop3':<8} | {'Status'}", flush=True)
    print("-" * 115, flush=True)

    certified = []
    for c in promising:
        trades = []
        for w in hold_cache:
            if c["archetype"] == "audcad_ny_reversal":
                tr = eval_ny_afternoon_reversal(w, c["min_wick"], c["start_hr"], c["end_hr"], c["sl_pips"], c["tp_pips"], c["max_spread"], comm, slip)
            elif c["archetype"] == "nzdcad_asian_fade":
                tr = eval_asian_liquidity_fade(w, c["min_wick"], c["start_hr"], c["end_hr"], c["sl_pips"], c["tp_pips"], c["max_spread"], comm, slip)
            elif c["archetype"] == "gbpchf_keltner_runner":
                tr = eval_keltner_asymmetric_runner(w, c["kc_mult"], c["start_hr"], c["end_hr"], c["sl_pips"], c["tp_pips"], c["max_spread"], comm, slip)
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

        print(f"{c['archetype']:<24} | {c['desc']:<26} | {sl_tp_str:<9} | {net:>+8.1f}p | {pf:>7.2f} | {len(comb_h):>6} | {pips_ex3:>+7.1f}p | {status}", flush=True)

    print("=" * 115, flush=True)
    print(f"Symbol {symbol}: Certified {len(certified)} independently robust, non-redundant champions!", flush=True)
    return certified


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="AUDCAD", choices=["AUDCAD", "NZDCAD", "GBPCHF", "ALL"])
    args = parser.parse_args()

    symbols = ["AUDCAD", "NZDCAD", "GBPCHF"] if args.symbol == "ALL" else [args.symbol]

    for s in symbols:
        champs = hunt_tailored_for_symbol(s)
        if champs:
            out_json = ROOT_DIR / "results" / f"tailored_champions_{s.lower()}.json"
            serializable = [{k: v for k, v in c.items() if k != "trades_array"} for c in champs]
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(serializable, f, indent=2)
            print(f"Saved {len(serializable)} champions to {out_json}", flush=True)

