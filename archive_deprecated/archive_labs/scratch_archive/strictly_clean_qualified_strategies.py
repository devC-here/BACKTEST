#!/usr/bin/env python3
"""
Strictly clean strategies/qualified_strategies/ so that ONLY the 11 verified,
economically viable champions remain in their respective sub-fleet folders.
All other legacy, unviable, or unvetted discovery folders are moved to archive_non_viable/.
"""

import shutil
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
QUAL_DIR = ROOT_DIR / "strategies" / "qualified_strategies"
ARCH_DIR = ROOT_DIR / "strategies" / "archive_non_viable"
ARCH_DIR.mkdir(parents=True, exist_ok=True)

WHITELIST = {
    "all_pairs": {
        "disc45_continuous_m5_rejection_elite",
    },
    "eurusd": {
        "disc86_ema_cross_atr_expansion_elite",
        "disc40_rsi_regime_range_shift_elite",
        "disc45_continuous_m5_rejection_elite",
        "disc85_session_open_impulse_elite",
    },
    "usdjpy": {
        "disc90_opening_range_impulse_expansion_elite",
        "disc91_session_donchian_midline_bounce_elite",
        "disc87_pivot_r1_s1_breakout_elite",
        "disc89_ny_london_vwap_ignition_elite",
        "disc92_asian_sr_flip_retest_elite",
        "disc86_ema_cross_atr_expansion_elite",
        "disc85_session_open_impulse_elite",
        "disc45_continuous_m5_rejection_elite",
    },
    "eurgbp": {
        "disc48_camarilla_floor_elite",
        "disc46_daily_pivot_wall_elite",
        "disc45_continuous_m5_rejection_elite",
    },
}


def main():
    print("=" * 80)
    print("  STRICTLY PURGING ALL NON-WHITELISTED FOLDERS FROM QUALIFIED_STRATEGIES")
    print("=" * 80)

    moved_count = 0
    for sub, allowed in WHITELIST.items():
        sub_dir = QUAL_DIR / sub
        if not sub_dir.exists():
            continue
        for child in list(sub_dir.iterdir()):
            if child.is_dir() and not child.name.startswith(('__', '.')):
                if child.name not in allowed:
                    dst = ARCH_DIR / f"{sub}_{child.name}"
                    if dst.exists():
                        shutil.rmtree(dst)
                    shutil.move(str(child), str(dst))
                    print(f"Moved: {sub}/{child.name} -> archive_non_viable/{dst.name}")
                    moved_count += 1

    print(f"\nPurge complete! Moved {moved_count} unviable/redundant folders to archive.")

    print("\n--- REMAINING VERIFIED FOLDERS IN QUALIFIED_STRATEGIES ---")
    for sub in WHITELIST:
        sub_dir = QUAL_DIR / sub
        remaining = [c.name for c in sub_dir.iterdir() if c.is_dir() and not c.name.startswith(('__', '.'))]
        print(f"  {sub}/: {len(remaining)} folders -> {sorted(remaining)}")


if __name__ == "__main__":
    main()

