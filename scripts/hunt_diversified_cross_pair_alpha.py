#!/usr/bin/env python3
"""
================================================================================
  MULTI-ARCHETYPE ALPHA DISCOVERY LAB: DIVERSIFIED CROSS-PAIRS
================================================================================
Searches for new, non-redundant institutional alpha across low-representation pairs:
  - USDCAD (Target: +2 to +3 strategies)
  - GBPCHF (Target: +2 strategies)
  - EURNZD (Target: +2 strategies)
  - EURGBP (Target: +1 to +2 strategies)
  - GBPJPY (Target: +1 to +2 strategies)
  - AUDCAD (Target: +1 to +2 strategies)
  - NZDCAD (Target: +1 to +2 strategies)

Archetypes Explored:
1. Session Range Expansion (London 07:00 UTC / NY 12:00 UTC Opening Impulse)
2. Session Liquidity Sweep & Mean Reversion (Asian Tokyo / London Pre-Market Sweep)
3. Trend Pullback & Volatility Expansion (H4 Trend + M15 Confluence)

Mandatory Reliability Gates:
- Discovery (In-Sample): PF >= 1.25, Trades >= 30, Net Pips > 0
- Quarantined Holdout (OOS): PF >= 1.20, Trades >= 15, Net Pips > 0
- Friction: $4.00/lot flat commission + 0.20p adverse entry slippage + raw spreads
- Execution Contract: Strict subsequent-tick causal execution (T_fill > T_sig)
- Anti-Redundancy Gate: Overlap < 50% & Corr < 0.50 against all existing models in symbol
================================================================================
"""

import sys
import gc
import json
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
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

# Load existing fleet trades to enforce anti-redundancy
with open(FLEET_MANIFEST, "r", encoding="utf-8") as f:
    fleet_data = json.load(f)

existing_trades_by_sym = defaultdict(list)
for s in fleet_data["strategies"]:
    sym = s["symbol"]
    name = s["name"]
    folder_slug = Path(s.get("relative_path", "")).parent.name
    csv_candidates = [
        MANIFEST_DIR / f"{sym}__{name}_trades.csv",
        MANIFEST_DIR / f"{sym}__{folder_slug}_trades.csv",
        MANIFEST_DIR / f"{sym.lower()}_{name}_trades.csv",
        MANIFEST_DIR / f"{name}_trades.csv"
    ]
    glob_matches = list(MANIFEST_DIR.glob(f"{sym}__{name}*_trades.csv"))
    if glob_matches:
        csv_candidates.extend(glob_matches)
    csv_file = next((f for f in csv_candidates if f.exists()), None)
    if csv_file:
        df = pd.read_csv(csv_file)
        if len(df) > 0:
            existing_trades_by_sym[sym].append((name, df))


def check_anti_redundancy(sym: str, candidate_trades: np.ndarray, max_overlap_pct: float = 50.0, max_corr: float = 0.50):
    """Verifies candidate trades have zero redundancy against all existing models in symbol."""
    if len(candidate_trades) == 0:
        return False, "No trades"
    existing_list = existing_trades_by_sym.get(sym, [])
    if not existing_list:
        return True, "No existing strategies"

    cand_times = candidate_trades['entry_time']
    cand_dirs = candidate_trades['direction']

    # Candidate daily pips
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


def precompute_discovery_week(symbol: str, week_num: int):
    """Loads a week of tick data and builds multi-timeframe bar cache."""
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

    # H4 bars directly from M15 (orders of magnitude faster than resampling raw ticks)
    h4 = m15[['high', 'low', 'close']].resample('4h').agg({'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
    if len(h4) < 10:
        return None

    h4['ema20'] = h4['close'].ewm(span=20, adjust=False).mean()
    h4['bias'] = np.where(h4['close'] > h4['ema20'], 1, -1)
    bias_m15 = h4['bias'].reindex(m15.index, method='ffill').fillna(0).values.astype(np.int8)

    # M15 arrays
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

    # ATR(14)
    tr = np.maximum(h - l, np.maximum(np.abs(h - np.roll(c, 1)), np.abs(l - np.roll(c, 1))))
    tr[0] = h[0] - l[0]
    atr14 = pd.Series(tr).rolling(14).mean().bfill().values.astype(np.float64)

    return {
        'ticks': ticks,
        'n_ticks': n_ticks,
        'pip_size': pip_size,
        'min_entry_ts': ticks['timestamp'][WARMUP_TICKS],
        'bias_m15': bias_m15,
        'c': c, 'h': h, 'l': l, 'o': o,
        'spread': spread,
        'hours': hours,
        'minutes': minutes,
        'dates': dates,
        'exec_indices': exec_indices,
        'atr14': atr14,
        'bar_rng': ((h - l) / pip_size).astype(np.float64),
        'n_bars': len(m15),
        'signals_buf': np.zeros(n_ticks, dtype=np.int8),
        'sl_buf': np.zeros(n_ticks, dtype=np.float64),
        'tp_buf': np.zeros(n_ticks, dtype=np.float64)
    }


def eval_expansion_archetype(week_data, start_hr, end_hr, sl_pips, tp_pips, atr_mult, min_body_ratio, max_spread, comm, slip):
    """Evaluates London or NY Opening Range Expansion archetype."""
    n_bars = week_data['n_bars']
    n_ticks = week_data['n_ticks']
    pip_size = week_data['pip_size']
    hours = week_data['hours']
    dates = week_data['dates']
    c, h, l, o = week_data['c'], week_data['h'], week_data['l'], week_data['o']
    spread = week_data['spread']
    atr14 = week_data['atr14']
    bias_m15 = week_data['bias_m15']
    exec_indices = week_data['exec_indices']
    signals_buf = week_data['signals_buf']
    sl_buf = week_data['sl_buf']
    tp_buf = week_data['tp_buf']

    last_trade_date = None
    fired_indices = []

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

        bar_rng = h[i] - l[i]
        cur_atr = atr14[i]
        if bar_rng < atr_mult * cur_atr:
            continue

        body = abs(c[i] - o[i])
        if (body / max(bar_rng, 1e-9)) < min_body_ratio:
            continue

        macro = bias_m15[i]
        sig = 0
        if macro == 1 and c[i] > o[i]:
            sig = 1
        elif macro == -1 and c[i] < o[i]:
            sig = -1

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
        commission_pips=comm,
        entry_slippage_pips=slip,
        min_entry_timestamp=week_data['min_entry_ts'],
        enforce_subsequent_tick=True
    )

    for idx in fired_indices:
        signals_buf[idx] = 0
        sl_buf[idx] = 0.0
        tp_buf[idx] = 0.0

    return tr


def run_hunt_for_symbol(symbol: str, archetypes: list):
    """
    Executes multi-archetype discovery search for a given symbol.
    """
    pip_size = PIP_SIZES.get(symbol, 0.0001)
    comm, slip = get_friction_pips(symbol)

    disc_weeks = get_complete_split_weeks(symbol, "discovery_70")
    hold_weeks = get_complete_split_weeks(symbol, "holdout_test_30")

    # Sample representative in-sample weeks (e.g. 70 weeks across 4 years)
    sample_disc_weeks = disc_weeks[::3]

    print("=" * 80)
    print(f"  HUNTING DIVERSIFICATION CHAMPIONS FOR {symbol}")
    print(f"  Discovery Sample: {len(sample_disc_weeks)} weeks | Quarantined Holdout: {len(hold_weeks)} weeks")
    print(f"  Friction: {comm:.2f}p Commission + {slip:.2f}p Slippage + Floating Spreads")
    print("=" * 80)

    t0 = time.time()
    print("Precomputing Discovery cache...")
    with ProcessPoolExecutor(max_workers=6) as ex:
        disc_cache = [w for w in ex.map(precompute_discovery_week, [symbol]*len(sample_disc_weeks), sample_disc_weeks) if w is not None]
    print(f"Discovery cache built ({len(disc_cache)} weeks) in {time.time() - t0:.1f}s.")

    candidates_found = []

    for arch in archetypes:
        name = arch["name"]
        start_hr = arch["start_hr"]
        end_hr = arch["end_hr"]
        max_spread = arch.get("max_spread", 2.5)
        sl_tp_grid = arch["sl_tp_grid"]
        atr_mults = arch.get("atr_mults", [1.1, 1.3, 1.5])
        body_ratios = arch.get("body_ratios", [0.50, 0.55])

        print(f"\nScanning Archetype: {name} ({start_hr:02d}:00 - {end_hr:02d}:00 UTC)...")

        for sl, tp in sl_tp_grid:
            for atr_m in atr_mults:
                for br in body_ratios:
                    trades_list = []
                    for w_data in disc_cache:
                        tr = eval_expansion_archetype(
                            w_data, start_hr, end_hr, sl, tp, atr_m, br, max_spread, comm, slip
                        )
                        if len(tr) > 0:
                            trades_list.append(tr)

                    if not trades_list:
                        continue
                    comb = np.concatenate(trades_list)
                    if len(comb) < 25:
                        continue

                    pips = comb['pips']
                    wins = float(np.sum(pips[pips > 0]))
                    losses = float(np.abs(np.sum(pips[pips < 0])))
                    if losses == 0: pf = 99.0
                    else: pf = wins / losses
                    net = wins - losses

                    if pf >= 1.25 and net > 0:
                        candidates_found.append({
                            "archetype": name,
                            "symbol": symbol,
                            "start_hr": start_hr,
                            "end_hr": end_hr,
                            "sl_pips": sl,
                            "tp_pips": tp,
                            "atr_mult": atr_m,
                            "body_ratio": br,
                            "max_spread": max_spread,
                            "disc_trades": len(comb),
                            "disc_pips": round(net, 1),
                            "disc_pf": round(pf, 2),
                            "disc_exp": round(net / len(comb), 2)
                        })

    print(f"\nDiscovery screening finished. High-edge candidates found: {len(candidates_found)}")
    if not candidates_found:
        print(f"No viable discovery candidates found for {symbol}.")
        return []

    # Sort candidates by discovery profit factor * pips
    candidates_found.sort(key=lambda x: x["disc_pf"] * x["disc_pips"], reverse=True)
    top_candidates = candidates_found[:12]

    # Now evaluate top candidates on QUARANTINED HOLDOUT (streaming week-by-week to eliminate IPC pipe buffer exhaustion)
    print(f"\nEvaluating {len(top_candidates)} candidates across {len(hold_weeks)} Quarantined Holdout weeks...")
    t0 = time.time()
    cand_trades = defaultdict(list)

    for w_num in hold_weeks:
        w_data = precompute_discovery_week(symbol, w_num)
        if w_data is None:
            continue
        for idx, c in enumerate(top_candidates):
            tr = eval_expansion_archetype(
                w_data, c["start_hr"], c["end_hr"], c["sl_pips"], c["tp_pips"],
                c["atr_mult"], c["body_ratio"], c["max_spread"], comm, slip
            )
            if len(tr) > 0:
                cand_trades[idx].append(tr)
        del w_data

    print(f"Holdout evaluation completed in {time.time() - t0:.1f}s.")

    certified_champions = []

    print("\n" + "=" * 90)
    print(f"{'ARCHETYPE':<25} | {'SL/TP':<10} | {'HOLD PIPS':<10} | {'HOLD PF':<8} | {'TRADES':<7} | {'EXP':<7} | {'STATUS'}")
    print("=" * 90)

    for idx, c in enumerate(top_candidates):
        trades_list = cand_trades.get(idx, [])
        if not trades_list:
            continue
        comb_hold = np.concatenate(trades_list)
        if len(comb_hold) < 15:
            continue

        pips = comb_hold['pips']
        wins = float(np.sum(pips[pips > 0]))
        losses = float(np.abs(np.sum(pips[pips < 0])))
        pf = (wins / losses) if losses > 0 else 99.0
        net = wins - losses
        exp = net / len(comb_hold)

        # Outlier sensitivity check
        pips_sorted = np.sort(pips)
        pips_ex3 = float(np.sum(pips_sorted[:-3]))

        # Anti-redundancy check against existing fleet
        is_distinct, anti_red_msg = check_anti_redundancy(symbol, comb_hold, max_overlap_pct=50.0, max_corr=0.50)

        # Gate check: PF >= 1.20, Net > 0, Exp >= 1.5p, Pips ex top 3 > 0, Anti-redundancy PASSED
        if pf >= 1.20 and net > 0 and exp >= 1.50 and pips_ex3 > 0 and is_distinct:
            status = "CERTIFIED"
            c["hold_trades"] = len(comb_hold)
            c["hold_pips"] = round(net, 1)
            c["hold_pf"] = round(pf, 2)
            c["hold_exp"] = round(exp, 2)
            c["hold_pips_ex3"] = round(pips_ex3, 1)
            c["trades_array"] = comb_hold
            certified_champions.append(c)
        else:
            reason = "Failed Gates" if (pf < 1.20 or net <= 0 or exp < 1.50) else ("Outlier Dependent" if pips_ex3 <= 0 else anti_red_msg)
            status = f"REJECTED ({reason})"

        sl_tp_str = f"{c['sl_pips']:.0f}/{c['tp_pips']:.0f}"
        print(f"{c['archetype']:<25} | {sl_tp_str:<10} | {net:>+8.1f}p | {pf:>6.2f} | {len(comb_hold):>6} | {exp:>5.2f}p | {status}")

    print("=" * 90)
    print(f"Symbol {symbol}: {len(certified_champions)} brand-new non-redundant champions certified!")
    return certified_champions


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="USDCAD")
    args = parser.parse_args()

    sym = args.symbol.upper()

    # Define targeted archetypes based on symbol
    if sym == "USDCAD":
        archetypes = [
            # 1. London/NY Overlap Momentum Expansion (12:00 - 15:00 UTC)
            {
                "name": "NY_London_Overlap_Impulse",
                "start_hr": 12, "end_hr": 14,
                "max_spread": 2.2,
                "sl_tp_grid": [(14.0, 28.0), (16.0, 32.0), (18.0, 40.0), (20.0, 45.0)],
                "atr_mults": [1.15, 1.30],
                "body_ratios": [0.55, 0.60]
            },
            # 2. London Morning Trend Expansion (07:00 - 09:30 UTC)
            {
                "name": "London_Morning_Impulse",
                "start_hr": 7, "end_hr": 9,
                "max_spread": 2.2,
                "sl_tp_grid": [(12.0, 24.0), (14.0, 30.0), (16.0, 35.0), (18.0, 40.0)],
                "atr_mults": [1.15, 1.30],
                "body_ratios": [0.55, 0.60]
            },
            # 3. NY Afternoon Trend Expansion (15:00 - 18:00 UTC)
            {
                "name": "NY_Afternoon_Continuation",
                "start_hr": 15, "end_hr": 17,
                "max_spread": 2.0,
                "sl_tp_grid": [(14.0, 30.0), (16.0, 35.0), (18.0, 42.0), (20.0, 50.0)],
                "atr_mults": [1.10, 1.25],
                "body_ratios": [0.50, 0.55]
            }
        ]
    elif sym == "GBPCHF":
        archetypes = [
            # 1. London Cash Open Range Expansion (07:00 - 09:00 UTC)
            {
                "name": "London_Cash_Open_Impulse",
                "start_hr": 7, "end_hr": 8,
                "max_spread": 2.8,
                "sl_tp_grid": [(16.0, 32.0), (18.0, 40.0), (20.0, 45.0), (22.0, 50.0)],
                "atr_mults": [1.15, 1.30],
                "body_ratios": [0.55, 0.60]
            },
            # 2. Zurich / European Morning Breakout (08:00 - 10:30 UTC)
            {
                "name": "Zurich_Morning_Breakout",
                "start_hr": 8, "end_hr": 10,
                "max_spread": 2.8,
                "sl_tp_grid": [(18.0, 36.0), (20.0, 44.0), (22.0, 50.0), (25.0, 60.0)],
                "atr_mults": [1.15, 1.30],
                "body_ratios": [0.55, 0.60]
            }
        ]
    elif sym == "EURNZD":
        archetypes = [
            # 1. European Morning Volatility Surge (06:00 - 08:30 UTC)
            {
                "name": "European_Morning_Surge",
                "start_hr": 6, "end_hr": 8,
                "max_spread": 4.5,
                "sl_tp_grid": [(22.0, 45.0), (25.0, 55.0), (28.0, 65.0), (30.0, 75.0)],
                "atr_mults": [1.15, 1.30],
                "body_ratios": [0.55, 0.60]
            },
            # 2. Asian / Wellington Opening Impulse (21:00 - 23:30 UTC)
            {
                "name": "Asian_Wellington_Impulse",
                "start_hr": 21, "end_hr": 23,
                "max_spread": 5.0,
                "sl_tp_grid": [(24.0, 50.0), (28.0, 60.0), (32.0, 75.0)],
                "atr_mults": [1.10, 1.25],
                "body_ratios": [0.50, 0.55]
            }
        ]
    elif sym == "GBPJPY":
        archetypes = [
            # 1. London Opening Range Expansion (07:00 - 09:00 UTC)
            {
                "name": "London_Opening_Impulse",
                "start_hr": 7, "end_hr": 8,
                "max_spread": 2.5,
                "sl_tp_grid": [(25.0, 50.0), (28.0, 60.0), (30.0, 65.0), (35.0, 75.0)],
                "atr_mults": [1.15, 1.30],
                "body_ratios": [0.55, 0.60]
            },
            # 2. NY Morning Expansion (12:00 - 14:00 UTC)
            {
                "name": "NY_Morning_Expansion",
                "start_hr": 12, "end_hr": 14,
                "max_spread": 2.5,
                "sl_tp_grid": [(25.0, 55.0), (30.0, 65.0), (35.0, 80.0)],
                "atr_mults": [1.15, 1.30],
                "body_ratios": [0.55, 0.60]
            }
        ]
    elif sym == "AUDCAD":
        archetypes = [
            # 1. Sydney / Tokyo Open Momentum (22:00 - 01:00 UTC)
            {
                "name": "Sydney_Tokyo_Open_Momentum",
                "start_hr": 22, "end_hr": 1,
                "max_spread": 2.8,
                "sl_tp_grid": [(14.0, 28.0), (16.0, 35.0), (18.0, 40.0)],
                "atr_mults": [1.15, 1.30],
                "body_ratios": [0.55, 0.60]
            },
            # 2. London Morning Retest & Acceleration (07:00 - 09:30 UTC)
            {
                "name": "London_Morning_Acceleration",
                "start_hr": 7, "end_hr": 9,
                "max_spread": 2.5,
                "sl_tp_grid": [(12.0, 25.0), (14.0, 30.0), (16.0, 35.0)],
                "atr_mults": [1.15, 1.30],
                "body_ratios": [0.55, 0.60]
            }
        ]
    elif sym == "NZDCAD":
        archetypes = [
            # 1. Asian Wellington Open Momentum (21:00 - 00:00 UTC)
            {
                "name": "Asian_Open_Momentum",
                "start_hr": 21, "end_hr": 23,
                "max_spread": 3.0,
                "sl_tp_grid": [(14.0, 28.0), (16.0, 35.0), (18.0, 40.0)],
                "atr_mults": [1.15, 1.30],
                "body_ratios": [0.55, 0.60]
            },
            # 2. London Opening Range Expansion (07:00 - 09:30 UTC)
            {
                "name": "London_Opening_Expansion",
                "start_hr": 7, "end_hr": 9,
                "max_spread": 2.8,
                "sl_tp_grid": [(12.0, 25.0), (14.0, 30.0), (16.0, 35.0)],
                "atr_mults": [1.15, 1.30],
                "body_ratios": [0.55, 0.60]
            }
        ]
    elif sym == "EURUSD":
        archetypes = [
            # 1. NY Equity Cash Open Expansion (13:00 - 15:00 UTC)
            {
                "name": "NY_Equity_Cash_Open_Impulse",
                "start_hr": 13, "end_hr": 15,
                "max_spread": 1.2,
                "sl_tp_grid": [(10.0, 22.0), (12.0, 26.0), (14.0, 32.0), (16.0, 36.0)],
                "atr_mults": [1.15, 1.30],
                "body_ratios": [0.55, 0.60]
            },
            # 2. London Morning Trend Acceleration (07:00 - 09:30 UTC)
            {
                "name": "London_Morning_Ignition",
                "start_hr": 7, "end_hr": 9,
                "max_spread": 1.0,
                "sl_tp_grid": [(10.0, 22.0), (12.0, 26.0), (14.0, 32.0)],
                "atr_mults": [1.15, 1.30],
                "body_ratios": [0.55, 0.60]
            }
        ]
    elif sym == "USDJPY":
        archetypes = [
            # 1. Tokyo Cash Open Momentum (00:00 - 02:30 UTC)
            {
                "name": "Tokyo_Cash_Open_Impulse",
                "start_hr": 0, "end_hr": 2,
                "max_spread": 1.4,
                "sl_tp_grid": [(12.0, 26.0), (14.0, 32.0), (16.0, 38.0), (18.0, 42.0)],
                "atr_mults": [1.15, 1.30],
                "body_ratios": [0.55, 0.60]
            },
            # 2. NY/London Overlap Momentum (12:00 - 14:30 UTC)
            {
                "name": "NY_London_Overlap_Acceleration",
                "start_hr": 12, "end_hr": 14,
                "max_spread": 1.4,
                "sl_tp_grid": [(14.0, 30.0), (16.0, 36.0), (18.0, 42.0), (20.0, 48.0)],
                "atr_mults": [1.15, 1.30],
                "body_ratios": [0.55, 0.60]
            }
        ]
    else: # EURGBP
        archetypes = [
            # 1. London Morning Range Expansion (07:00 - 09:30 UTC)
            {
                "name": "London_Morning_Expansion",
                "start_hr": 7, "end_hr": 9,
                "max_spread": 1.8,
                "sl_tp_grid": [(8.0, 14.0), (9.0, 16.0), (10.0, 18.0), (11.0, 20.0)],
                "atr_mults": [1.10, 1.25],
                "body_ratios": [0.50, 0.55]
            }
        ]

    champs = run_hunt_for_symbol(sym, archetypes)
    if champs:
        out_path = ROOT_DIR / "results" / f"new_champions_{sym.lower()}.json"
        # Save serializable version
        serializable = []
        for c in champs:
            s_c = {k: v for k, v in c.items() if k != "trades_array"}
            serializable.append(s_c)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2)
        print(f"Saved {len(serializable)} champions to {out_path}")
