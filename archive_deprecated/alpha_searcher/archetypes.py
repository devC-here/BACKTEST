#!/usr/bin/env python3
"""
Alpha Searcher Strategy Archetypes: Multi-Session Sweep & Retest Generators
==========================================================================
Generates diverse candidate grids tailored to symbol characteristics,
incorporating both standard and high-frequency (30-80+ trades/year) parameters.
"""

from typing import Dict, List, Any


def get_symbol_scale(symbol: str) -> float:
    s = symbol.upper()
    if "JPY" in s:
        return 1.0 if "GBP" in s else 0.7
    elif "NZD" in s:
        return 1.4 if "EUR" in s else 0.8
    elif "CHF" in s:
        return 1.1
    elif "GBP" in s:
        return 0.75
    elif "CAD" in s:
        return 0.85
    else:
        # EURUSD
        return 0.7


def generate_candidate_battery(symbol: str, high_frequency_bias: bool = True) -> List[Dict[str, Any]]:
    """
    Generates a comprehensive grid of zero-bias alpha candidates across 5 archetypes:
    1. Asian Sweep Fade (London Open Fade)
    2. London Sweep Fade (NY Open Fade)
    3. NY Session Sweep Fade (Late NY Fade)
    4. H1 Structure Retest (Faster-cycling structure)
    5. H4 Macro Structure Retest (Swing-level retests)
    """
    s = symbol.upper()
    scale = get_symbol_scale(s)
    cands = []

    # 1. Asian Sweep Fade (Asian: 00:00-06:00 UTC, Entry: 06:00-14:00 UTC)
    deltas = [round(d * scale, 1) for d in ([2.0, 3.0, 4.0, 5.0, 6.5] if high_frequency_bias else [4.0, 6.0, 8.0, 10.0])]
    sl_tp_pairs = [
        (round(10.0 * scale, 1), round(20.0 * scale, 1)),
        (round(11.0 * scale, 1), round(22.0 * scale, 1)),
        (round(12.0 * scale, 1), round(24.0 * scale, 1)),
        (round(12.0 * scale, 1), round(28.0 * scale, 1)),
        (round(14.0 * scale, 1), round(28.0 * scale, 1)),
        (round(15.0 * scale, 1), round(30.0 * scale, 1)),
        (round(16.0 * scale, 1), round(35.0 * scale, 1)),
        (round(18.0 * scale, 1), round(40.0 * scale, 1)),
    ]
    ranges = [
        (round(10.0 * scale, 1), round(40.0 * scale, 1)),
        (round(12.0 * scale, 1), round(50.0 * scale, 1)),
        (round(15.0 * scale, 1), round(65.0 * scale, 1)),
    ]
    for d in deltas:
        for sl, tp in sl_tp_pairs:
            for min_r, max_r in ranges:
                cands.append({
                    "family": "ASIAN_SWEEP",
                    "symbol": s,
                    "delta": d,
                    "sl": sl,
                    "tp": tp,
                    "min_rng": min_r,
                    "max_rng": max_r,
                    "name": f"AsianSweep_D{d:.1f}_SL{sl:.0f}_TP{tp:.0f}_R{min_r:.0f}-{max_r:.0f}"
                })

    # 2. London Sweep Fade (Asian/Early London: 00:00-08:00 UTC, Entry: 08:00-16:00 UTC)
    london_deltas = [round(d * scale, 1) for d in [3.0, 4.5, 6.0, 7.5]]
    for d in london_deltas:
        for sl, tp in sl_tp_pairs:
            for min_r, max_r in ranges:
                cands.append({
                    "family": "LONDON_SWEEP",
                    "symbol": s,
                    "delta": d,
                    "sl": sl,
                    "tp": tp,
                    "min_rng": min_r,
                    "max_rng": max_r,
                    "name": f"LondonSweep_D{d:.1f}_SL{sl:.0f}_TP{tp:.0f}_R{min_r:.0f}-{max_r:.0f}"
                })

    # 3. NY Session Sweep Fade (European Range: 07:00-12:00 UTC, Entry: 12:00-17:00 UTC)
    ny_deltas = [round(d * scale, 1) for d in [3.5, 4.5, 5.5, 7.0]]
    for d in ny_deltas:
        for sl, tp in sl_tp_pairs:
            for min_r, max_r in ranges:
                cands.append({
                    "family": "NY_SWEEP",
                    "symbol": s,
                    "delta": d,
                    "sl": sl,
                    "tp": tp,
                    "min_rng": min_r,
                    "max_rng": max_r,
                    "name": f"NYSweep_D{d:.1f}_SL{sl:.0f}_TP{tp:.0f}_R{min_r:.0f}-{max_r:.0f}"
                })

    # 4. H1 Structure Retests (Dynamic Support/Resistance on H1, 3x faster than H4)
    h1_lbs = [8, 12, 16]
    h1_dists = [round(d * scale, 1) for d in [8.0, 10.0, 12.0, 15.0]]
    h1_wicks = [0.25, 0.30, 0.35]
    for lb in h1_lbs:
        for dist in h1_dists:
            for wick in h1_wicks:
                for sl, tp in sl_tp_pairs[:6]:
                    cands.append({
                        "family": "H1_RETEST",
                        "symbol": s,
                        "lb": lb,
                        "dist": dist,
                        "wick": wick,
                        "sl": sl,
                        "tp": tp,
                        "name": f"H1Retest_LB{lb}_D{dist:.0f}_W{wick:.2f}_SL{sl:.0f}_TP{tp:.0f}"
                    })

    # 5. H4 Macro Structure Retest (Swing inflections with single causal shift)
    h4_lbs = [8, 10, 12]
    h4_dists = [round(d * scale, 1) for d in [12.0, 16.0, 20.0]]
    h4_wicks = [0.35, 0.40, 0.45]
    for lb in h4_lbs:
        for dist in h4_dists:
            for wick in h4_wicks:
                for sl, tp in sl_tp_pairs[2:]:
                    cands.append({
                        "family": "H4_RETEST",
                        "symbol": s,
                        "lb": lb,
                        "dist": dist,
                        "wick": wick,
                        "sl": sl,
                        "tp": tp,
                        "name": f"H4Retest_LB{lb}_D{dist:.0f}_W{wick:.2f}_SL{sl:.0f}_TP{tp:.0f}"
                    })

    return cands
