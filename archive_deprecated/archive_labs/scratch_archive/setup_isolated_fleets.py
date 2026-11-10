#!/usr/bin/env python3
"""
Sets up the clean physical separation between:
  1. strategies/eurusd_production_fleet/ (All 14 frozen, untouched EURUSD models)
  2. strategies/eurgbp_fleet/ (EURGBP verified Premier Trinity and specialists)
"""

import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "strategies"
EURUSD_DIR = ROOT / "eurusd_production_fleet"
EURGBP_DIR = ROOT / "eurgbp_fleet"

EURUSD_DIR.mkdir(parents=True, exist_ok=True)
EURGBP_DIR.mkdir(parents=True, exist_ok=True)


def setup_fleets():
    print("=" * 80)
    print("  PHYSICAL FLEET SEPARATION: EURUSD FROZEN FLEET vs EURGBP FLEET  ")
    print("=" * 80)

    # 1. DISC46
    src_d46 = ROOT / "new_discoveries" / "disc46_eurgbp_daily_pivot_wall_elite"
    dest_d46 = EURGBP_DIR / "disc46_daily_pivot_wall_elite"
    if src_d46.exists():
        if dest_d46.exists():
            shutil.rmtree(dest_d46)
        shutil.copytree(src_d46, dest_d46)
        print("[+] Copied DISC46 into strategies/eurgbp_fleet/disc46_daily_pivot_wall_elite")

    # 2. DISC42 for EURGBP
    src_d42 = ROOT / "new_discoveries" / "disc42_macro_structure_h4_m30_elite"
    dest_d42 = EURGBP_DIR / "disc42_macro_structure_h4_m30_elite"
    if src_d42.exists():
        if dest_d42.exists():
            shutil.rmtree(dest_d42)
        shutil.copytree(src_d42, dest_d42)
        # Update default max_spread to 2.0 in the EURGBP copy
        strat_file = dest_d42 / "strategy.py"
        content = strat_file.read_text(encoding="utf-8")
        content = content.replace("max_spread: float = 0.60", "max_spread: float = 2.0")
        content = content.replace('"Macro_Structure_H4_M30_Elite"', '"EURGBP_Macro_Structure_H4_M30_Elite"')
        strat_file.write_text(content, encoding="utf-8")
        print("[+] Adapted DISC42 into strategies/eurgbp_fleet/disc42_macro_structure_h4_m30_elite (max_spread=2.0)")

    # 3. DISC45 for EURGBP
    src_d45 = ROOT / "new_discoveries" / "disc45_continuous_m5_rejection_elite"
    dest_d45 = EURGBP_DIR / "disc45_continuous_m5_rejection_elite"
    if src_d45.exists():
        if dest_d45.exists():
            shutil.rmtree(dest_d45)
        shutil.copytree(src_d45, dest_d45)
        # Update default max_spread to 2.0 in the EURGBP copy
        strat_file = dest_d45 / "strategy.py"
        content = strat_file.read_text(encoding="utf-8")
        content = content.replace("max_spread: float = 0.55", "max_spread: float = 2.0")
        content = content.replace('"Continuous_M5_Rejection_Elite"', '"EURGBP_Continuous_M5_Rejection_Elite"')
        strat_file.write_text(content, encoding="utf-8")
        print("[+] Adapted DISC45 into strategies/eurgbp_fleet/disc45_continuous_m5_rejection_elite (max_spread=2.0)")

    # 4. DISC43 for EURGBP
    src_d43 = ROOT / "new_discoveries" / "disc43_intraday_precision_h1_m5_elite"
    dest_d43 = EURGBP_DIR / "disc43_intraday_precision_h1_m5_elite"
    if src_d43.exists():
        if dest_d43.exists():
            shutil.rmtree(dest_d43)
        shutil.copytree(src_d43, dest_d43)
        # Update default max_spread to 2.0 in the EURGBP copy
        strat_file = dest_d43 / "strategy.py"
        content = strat_file.read_text(encoding="utf-8")
        content = content.replace("max_spread: float = 0.55", "max_spread: float = 2.0")
        content = content.replace('"Intraday_Precision_H1_M5_Elite"', '"EURGBP_Intraday_Precision_H1_M5_Elite"')
        strat_file.write_text(content, encoding="utf-8")
        print("[+] Adapted DISC43 into strategies/eurgbp_fleet/disc43_intraday_precision_h1_m5_elite (max_spread=2.0)")

    # 5. EURGBP Fleet __init__.py
    init_code = '''"""
EURGBP Specialist Fleet (The Premier Trinity + Structural Suite).
Audited on 157 continuous weeks of EURGBP (~65.89M ticks across 2021, 2022, 2024).

Champions:
  1. DISC46 (Daily Pivot Wall): +604.2p | 62.3% WR | PF 2.54 | Max DD 34.5p | Ret/DD 17.51x
  2. DISC42 (Macro Structure H4/M30): +339.5p | 64.9% WR | PF 2.61 | Max DD 46.7p | Ret/DD 7.27x
  3. DISC45 (Continuous M5 Rejection): +302.8p | 49.2% WR | PF 1.27 | Max DD 61.5p | Ret/DD 4.92x
  4. DISC43 (Intraday Precision H1/M5): +151.9p | 44.2% WR | PF 1.25 | Max DD 54.4p | Ret/DD 2.79x

Combined Premier Trinity: +1,246.5 Net Pips | 53.6% Win Rate | PF 1.72 | Max DD 43.0p | Ret/DD 28.99x
"""
from .disc46_daily_pivot_wall_elite.strategy import EURGBPDailyPivotWallElite
from .disc42_macro_structure_h4_m30_elite.strategy import MacroStructureH4M30Elite
from .disc45_continuous_m5_rejection_elite.strategy import ContinuousM5RejectionElite
from .disc43_intraday_precision_h1_m5_elite.strategy import IntradayPrecisionH1M5Elite

__all__ = [
    "EURGBPDailyPivotWallElite",
    "MacroStructureH4M30Elite",
    "ContinuousM5RejectionElite",
    "IntradayPrecisionH1M5Elite",
]
'''
    (EURGBP_DIR / "__init__.py").write_text(init_code, encoding="utf-8")
    print("[+] Created strategies/eurgbp_fleet/__init__.py")

    print("\n[+] Fleet physical separation complete!")
    print(f"  * EURUSD Production Fleet: {len(list(EURUSD_DIR.iterdir()))} items in strategies/eurusd_production_fleet/")
    print(f"  * EURGBP Production Fleet: {len(list(EURGBP_DIR.iterdir()))} items in strategies/eurgbp_fleet/")


if __name__ == "__main__":
    setup_fleets()
