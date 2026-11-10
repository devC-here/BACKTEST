#!/usr/bin/env python3
"""
================================================================================
  CONSOLIDATE ECONOMICALLY VIABLE FLEET & ARCHIVE NON-VIABLE MODELS
================================================================================
1. Deploys the complete, verified Economically Viable Fleet into:
     strategies/qualified_strategies/
       ├── all_pairs/   (Universal Cross-Pair Champions)
       ├── eurusd/      (Verified High-Expectancy EURUSD Specialists)
       ├── usdjpy/      (Verified High-Expectancy USDJPY Specialists)
       ├── eurgbp/      (Verified High-Expectancy EURGBP Specialists)
       └── README.md    (Master Economic Scorecard)

2. Archives all unviable, negative PnL, or paper-profit models into:
     strategies/archive_non_viable/
================================================================================
"""

import os
import sys
import shutil
import re
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
QUALIFIED_DIR = ROOT_DIR / "strategies" / "qualified_strategies"
ARCHIVE_DIR = ROOT_DIR / "strategies" / "archive_non_viable"

SOURCE_DIRS = [
    ROOT_DIR / "strategies" / "eurusd_production_fleet",
    ROOT_DIR / "strategies" / "usdjpy_fleet",
    ROOT_DIR / "strategies" / "eurgbp_fleet",
    ROOT_DIR / "strategies" / "new_discoveries",
    ROOT_DIR / "strategies" / "qualified_strategies" / "all_pairs",
    ROOT_DIR / "strategies" / "qualified_strategies" / "eurusd",
    ROOT_DIR / "strategies" / "qualified_strategies" / "usdjpy",
    ROOT_DIR / "strategies" / "qualified_strategies" / "eurgbp",
    ROOT_DIR / "strategies"
]

# The complete economically viable roster
SELECTIONS = {
    "all_pairs": [
        "disc42_macro_structure_h4_m30_elite",
        "disc43_intraday_precision_h1_m5_elite",
        "disc45_continuous_m5_rejection_elite",
        "disc49_intraday_macro_retest_elite",
        "disc51_macro_asymmetric_runner_elite",
        "disc58_h1_keltner_squeeze_explosion_elite",
        "disc65_session_vwap_band_fade_elite",
        "disc74_daily_pivot_pullback_elite"
    ],
    "eurusd": [
        "disc56_triple_screen_m15_impulse_elite",
        "disc53_m15_3bar_trend_pullback_elite",
        "disc65_session_vwap_band_fade_elite",
        "disc86_ema_cross_atr_expansion_elite",
        "disc82_chaikin_volatility_surge_elite",
        "disc62_stoch_rsi_pullback_hook_elite",
        "disc75_weekly_pivot_pullback_elite",
        "disc58_h1_keltner_squeeze_explosion_elite",
        "disc76_vwap_1sd_rebound_elite",
        "disc59_m15_macd_zero_cross_momentum_elite",
        "disc80_double_swing_neckline_retest_elite",
        "disc74_daily_pivot_pullback_elite",
        "disc51_macro_asymmetric_runner_elite",
        "disc85_session_open_impulse_elite",
        "disc79_london_open_15m_expansion_elite",
        "disc49_intraday_macro_retest_elite",
        "disc71_asian_hl_retest_elite",
        "disc52_fib_golden_pocket_pin_defense_elite",
        "disc50_m30_m5_precision_retest_elite",
        "disc60_asian_compression_breakout_elite",
        "disc40_rsi_regime_range_shift_elite",
        "disc45_continuous_m5_rejection_elite",
        "disc42_macro_structure_h4_m30_elite"
    ],
    "usdjpy": [
        "disc56_triple_screen_m15_impulse_elite",
        "disc53_m15_3bar_trend_pullback_elite",
        "disc77_keltner_midline_expansion_elite",
        "disc54_h1_inside_bar_expansion_elite",
        "disc68_cmf_momentum_expansion_elite",
        "disc65_session_vwap_band_fade_elite",
        "disc83_parabolic_sar_flip_elite",
        "disc84_london_30m_orb_continuation_elite",
        "disc58_h1_keltner_squeeze_explosion_elite",
        "disc64_asian_midpoint_rebound_elite",
        "disc75_weekly_pivot_pullback_elite",
        "disc87_pivot_r1_s1_breakout_elite",
        "disc57_london_ignition_range_break_elite",
        "disc59_m15_macd_zero_cross_momentum_elite",
        "disc52_fib_golden_pocket_pin_defense_elite",
        "disc67_london_lunch_expansion_elite",
        "disc76_vwap_1sd_rebound_elite",
        "disc81_bb_squeeze_m5_ignition_elite",
        "disc61_double_structure_neckline_breakout_elite",
        "disc74_daily_pivot_pullback_elite",
        "disc55_m15_200ema_dynamic_bounce_elite",
        "disc46_daily_pivot_wall_elite",
        "disc48_camarilla_floor_elite",
        "disc45_continuous_m5_rejection_elite",
        "disc42_macro_structure_h4_m30_elite"
    ],
    "eurgbp": [
        "disc74_daily_pivot_pullback_elite",
        "disc58_h1_keltner_squeeze_explosion_elite",
        "disc60_asian_compression_breakout_elite",
        "disc51_macro_asymmetric_runner_elite",
        "disc49_intraday_macro_retest_elite",
        "disc48_camarilla_floor_elite",
        "disc46_daily_pivot_wall_elite",
        "disc47_weekly_pivot_wall_elite",
        "disc42_macro_structure_h4_m30_elite",
        "disc45_continuous_m5_rejection_elite"
    ]
}

def find_source(slug: str) -> Path | None:
    for s_dir in SOURCE_DIRS:
        cand = s_dir / slug
        if cand.exists() and (cand / "strategy.py").exists():
            return cand
    return None

def patch_strategy(content: str) -> str:
    root_pattern = re.compile(r'ROOT_DIR\s*=\s*Path\(__file__\)[^\n]+')
    dynamic_root = (
        'ROOT_DIR = Path(__file__).resolve().parent\n'
        'while ROOT_DIR.name and not (ROOT_DIR / "tick_engine").exists():\n'
        '    ROOT_DIR = ROOT_DIR.parent'
    )
    if root_pattern.search(content):
        return root_pattern.sub(dynamic_root, content, count=1)
    return content

def main():
    QUALIFIED_DIR.mkdir(parents=True, exist_ok=True)
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

    total_deployed = 0
    for category, slugs in SELECTIONS.items():
        cat_dir = QUALIFIED_DIR / category
        cat_dir.mkdir(parents=True, exist_ok=True)
        init_file = cat_dir / "__init__.py"
        if not init_file.exists():
            init_file.write_text(f'"""Qualified {category.upper()} specialist fleet."""\n', encoding="utf-8")

        for slug in slugs:
            src = find_source(slug)
            if not src:
                print(f"Warning: could not find source for {slug}")
                continue

            dst = cat_dir / slug
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)

            strat_py = dst / "strategy.py"
            if strat_py.exists():
                code = strat_py.read_text(encoding="utf-8")
                patched = patch_strategy(code)
                strat_py.write_text(patched, encoding="utf-8")

            total_deployed += 1
            print(f"[{category.upper()}] Deployed: {slug}")

    # Now archive standalone unviable roots from strategies/
    unviable_roots = [
        "comb12_bollinger_squeeze_elite",
        "comb27_squeeze_supertrend_hybrid_elite",
        "s10_dual_ema_pullback_elite",
        "s14_inside_bar_m30_elite",
        "s66_bhanushali_44ma_elite",
        "high_frequency_m15_scalper_elite",
        "disc03_ny_open_overlap_elite",
        "disc38_london_midday_macd_acceleration_elite"
    ]
    strat_root = ROOT_DIR / "strategies"
    for r in unviable_roots:
        p = strat_root / r
        if p.exists() and p.is_dir():
            dst = ARCHIVE_DIR / r
            if dst.exists():
                shutil.rmtree(dst)
            shutil.move(str(p), str(dst))
            print(f"[ARCHIVE] Moved non-viable root: {r} -> strategies/archive_non_viable/")

    print("\n" + "=" * 80)
    print(f"Successfully deployed {total_deployed} strategy instances across qualified_strategies/!")
    print(f"Archived non-viable folders to strategies/archive_non_viable/.")
    print("=" * 80)

if __name__ == "__main__":
    main()
