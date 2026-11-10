#!/usr/bin/env python3
"""
Prune all non-viable, low-return, low-expectancy, and sub-1.25 PF strategies
from strategies/qualified_strategies/ into strategies/archive_non_viable/.
Keeps ONLY the proven, economically justifiable champions.
"""

from __future__ import annotations
import sys
import shutil
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
QUAL_DIR = ROOT_DIR / "strategies" / "qualified_strategies"
ARCH_DIR = ROOT_DIR / "strategies" / "archive_non_viable"
ARCH_DIR.mkdir(parents=True, exist_ok=True)

# List of folders to archive from qualified_strategies
# 1. Sub-1.25 PF or High Drawdown:
#    - disc88_three_drive_ignition_elite (PF 1.14, 225p DD)
#    - disc47_weekly_pivot_wall_elite (PF 1.19, 389p DD)
# 2. Low Annual Return (<100 pips/yr) and low Ret/DD (< 3.0x):
#    - disc41_thursday_macro_trend_ignition_elite (67p/yr, Ret/DD 2.57x)
#    - disc03_ny_open_overlap_elite (81p/yr, Ret/DD 2.44x)
#    - disc38_london_midday_macd_acceleration_elite (83p/yr, Ret/DD 2.51x)
#    - disc43_intraday_precision_h1_m5_elite (87p/yr)
#    - disc42_macro_structure_h4_m30_elite (Only 48 trades in 5 years, <10 tr/yr)

PRUNE_TARGETS = [
    ("eurusd", "disc41_thursday_macro_trend_ignition_elite"),
    ("eurusd", "disc88_three_drive_ignition_elite"),
    ("eurusd", "disc03_ny_open_overlap_elite"),
    ("eurusd", "disc38_london_midday_macd_acceleration_elite"),
    ("eurusd", "disc43_intraday_precision_h1_m5_elite"),
    ("eurusd", "disc42_macro_structure_h4_m30_elite"),

    ("usdjpy", "disc88_three_drive_ignition_elite"),
    ("usdjpy", "disc38_london_midday_macd_acceleration_elite"),
    ("usdjpy", "disc43_intraday_precision_h1_m5_elite"),
    ("usdjpy", "disc42_macro_structure_h4_m30_elite"),
    ("usdjpy", "disc46_daily_pivot_wall_elite"),       # 1 trade on USDJPY
    ("usdjpy", "disc48_camarilla_floor_elite"),        # 5 trades on USDJPY
    ("usdjpy", "disc49_intraday_macro_retest_elite"),   # legacy redundant
    ("usdjpy", "disc50_m30_m5_precision_retest_elite"), # legacy redundant

    ("eurgbp", "disc47_weekly_pivot_wall_elite"),
    ("eurgbp", "disc42_macro_structure_h4_m30_elite"),
    ("eurgbp", "disc43_intraday_precision_h1_m5_elite"),

    ("all_pairs", "disc42_macro_structure_h4_m30_elite"),
    ("all_pairs", "disc43_intraday_precision_h1_m5_elite"),
]


def main():
    print("=" * 80)
    print("  PRUNING QUALIFIED DIRECTORY TO STRICTLY ECONOMICALLY VIABLE FLEET")
    print("=" * 80)

    for subfolder, slug in PRUNE_TARGETS:
        src = QUAL_DIR / subfolder / slug
        if src.exists():
            dst = ARCH_DIR / f"{subfolder}_{slug}"
            if dst.exists():
                shutil.rmtree(dst)
            shutil.move(str(src), str(dst))
            print(f"Archived: {subfolder}/{slug} -> archive_non_viable/{subfolder}_{slug}")

    print("\nFinished moving unviable/marginal strategies to archive.")


if __name__ == "__main__":
    main()
