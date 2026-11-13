#!/usr/bin/env python3
"""
================================================================================
  QUARANTINE FRAGILE STRATEGIES & HARDEN PRODUCTION FLEET
================================================================================
Moves the 16 Grade F (fragile / stress-failing / outlier-dependent) strategies
from certified_production_fleet/ to archive_deprecated/pruned_low_edge_candidates/,
leaving 85 pure, stress-resilient institutional strategies across all 10 pairs.
================================================================================
"""

import os
import shutil
import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
FLEET_DIR = ROOT_DIR / "certified_production_fleet"
ARCHIVE_DIR = ROOT_DIR / "archive_deprecated" / "pruned_low_edge_candidates"
ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

# The 16 Grade F fragile strategies identified in the individual stress & reliability audit:
FRAGILE_STRATEGIES = [
    # EURUSD (7 strategies)
    ("EURUSD", "disc38_london_midday_macd_acceleration_elite", "eurusd_champions/disc38_london_midday_macd_acceleration_elite"),
    ("EURUSD", "disc41_thursday_macro_trend_ignition_elite", "eurusd_champions/disc41_thursday_macro_trend_ignition_elite"),
    ("EURUSD", "disc80_double_swing_neckline_retest_elite", "eurusd_champions/disc80_double_swing_neckline_retest_elite"),
    ("EURUSD", "disc82_chaikin_volatility_surge_elite", "eurusd_champions/disc82_chaikin_volatility_surge_elite"),
    ("EURUSD", "disc85_session_open_impulse_elite", "eurusd_champions/disc85_session_open_impulse_elite"),
    ("EURUSD", "strategy_09_eurusd_ny_sweep_fade", "eurusd_ny_sweeps/strategy_09_eurusd_ny_sweep_fade.py"),
    ("EURUSD", "strategy_10_eurusd_ny_sweep_fade", "eurusd_ny_sweeps/strategy_10_eurusd_ny_sweep_fade.py"),
    
    # USDJPY (9 strategies)
    ("USDJPY", "disc100_double_structure_neckline_breakout_elite", "usdjpy_champions/disc100_double_structure_neckline_breakout_elite"),
    ("USDJPY", "disc102_keltner_midline_expansion_elite", "usdjpy_champions/disc102_keltner_midline_expansion_elite"),
    ("USDJPY", "disc68_cmf_momentum_expansion_elite", "usdjpy_champions/disc68_cmf_momentum_expansion_elite"),
    ("USDJPY", "disc80_double_swing_neckline_retest_elite", "usdjpy_champions/disc80_double_swing_neckline_retest_elite"),
    ("USDJPY", "disc82_chaikin_volatility_surge_elite", "usdjpy_champions/disc82_chaikin_volatility_surge_elite"),
    ("USDJPY", "disc84_london_30m_orb_continuation_elite", "usdjpy_champions/disc84_london_30m_orb_continuation_elite"),
    ("USDJPY", "disc85_session_open_impulse_elite", "usdjpy_champions/disc85_session_open_impulse_elite"),
    ("USDJPY", "disc87_pivot_r1_s1_breakout_elite", "usdjpy_champions/disc87_pivot_r1_s1_breakout_elite"),
    ("USDJPY", "disc96_cardwell_rsi50_momentum_bounce_elite", "usdjpy_champions/disc96_cardwell_rsi50_momentum_bounce_elite"),
]


def quarantine():
    print(f"Quarantining {len(FRAGILE_STRATEGIES)} fragile strategies...")
    quarantined_count = 0

    for sym, name, rel_path in FRAGILE_STRATEGIES:
        src = FLEET_DIR / rel_path
        dst = ARCHIVE_DIR / rel_path
        dst.parent.mkdir(parents=True, exist_ok=True)

        if src.exists():
            if src.is_dir():
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.move(str(src), str(dst))
            else:
                shutil.move(str(src), str(dst))
            print(f"  [QUARANTINED] {sym} {name} -> {dst.relative_to(ROOT_DIR)}")
            quarantined_count += 1
        else:
            print(f"  [NOT FOUND] {src}")

    print(f"\nSuccessfully quarantined {quarantined_count} fragile strategies.")


if __name__ == "__main__":
    quarantine()

