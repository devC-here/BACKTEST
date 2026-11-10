#!/usr/bin/env python3
"""
================================================================================
  PRUNE FRAGILE STRATEGIES: KEEP ONLY REAL-LIFE VIABLE FLEET
================================================================================
Based on the Top 30% Winners Removal Stress Test and Outlier Sensitivity Audit,
prunes strategies that fail real-life deployment viability:

Archived from EURUSD Production Fleet (7 Models):
  1. high_frequency_m15_scalper_elite (HF_M15): Bleeds -5,046.5p when trimmed; massive spread drag.
  2. s66_bhanushali_44ma_elite (S66_44MA): Negative net return under strict tick execution slippage.
  3. s14_inside_bar_m30_elite (S14_IB): Negative net return under strict tick execution slippage.
  4. s10_dual_ema_pullback_elite (S10_EMA): Poor Ret/DD (1.21x), bleeds -1,073.5p when trimmed.
  5. disc39_stochastic_deep_pullback_elite (DISC39): Low Ret/DD (2.39x), bleeds -648.5p when trimmed.
  6. comb12_bollinger_squeeze_elite (COMB12): Low Ret/DD (2.10x), bleeds -587.6p when trimmed.
  7. comb27_squeeze_supertrend_hybrid_elite (COMB27): Low Ret/DD (1.82x), bleeds -745.6p when trimmed.

Archived from EURGBP Fleet (1 Model):
  8. disc43_intraday_precision_h1_m5_elite (DISC43): Only +151p on EURGBP, weak Ret/DD (2.79x), bleeds -80.1p when trimmed.
     (Note: DISC43 is kept in EURUSD where it has +437.2p and passed the outlier test with positive trimmed profit).

Retained Pristine Live Fleet:
  EURUSD Fleet (7 Elite Models):
    - DISC42_Macro_Structure_H4_M30 (+387.6p | Trimmed +233.8p | PF 3.97 | Ret/DD 19.00x)
    - DISC45_Continuous_M5_Rejection (+999.4p | Trimmed +125.0p | PF 1.57 | Ret/DD 16.41x)
    - DISC40_RSI_Regime_Range_Shift (+1,112.8p | PF 1.31 | Ret/DD 7.72x)
    - DISC43_Intraday_Precision_H1_M5 (+437.2p | Trimmed +20.3p | PF 1.47 | Ret/DD 6.36x)
    - DISC03_NY_Open_Overlap (+405.3p | PF 1.29 | Ret/DD 2.89x)
    - DISC41_Thursday_Macro_Ignition (+336.1p | PF 1.30 | Ret/DD 2.57x)
    - DISC38_London_Midday_MACD (+372.7p | PF 1.28 | Ret/DD 2.52x)

  EURGBP Fleet (5 Elite Models):
    - DISC46_Daily_Pivot_Wall (+604.2p | Trimmed +296.1p | PF 2.54 | Ret/DD 17.51x)
    - DISC48_Camarilla_Floor (+647.2p | Trimmed +178.6p | PF 1.72 | Ret/DD 12.54x)
    - DISC42_Macro_Structure (+339.5p | Trimmed +172.9p | PF 2.61 | Ret/DD 7.27x)
    - DISC45_Continuous_M5 (+302.8p | PF 1.27 | Ret/DD 4.92x)
    - DISC47_Weekly_Pivot_Wall (+473.6p | PF 1.37 | Ret/DD 3.88x)
================================================================================
"""

from __future__ import annotations
import shutil
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
STRAT_DIR = ROOT_DIR / "strategies"
EURUSD_FLEET = STRAT_DIR / "eurusd_production_fleet"
EURGBP_FLEET = STRAT_DIR / "eurgbp_fleet"
ARCHIVE_DIR = STRAT_DIR / "archive_unprofitable"
ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

# 1. Models to prune from EURUSD Fleet
eurusd_prune = [
    "high_frequency_m15_scalper_elite",
    "s66_bhanushali_44ma_elite",
    "s14_inside_bar_m30_elite",
    "s10_dual_ema_pullback_elite",
    "disc39_stochastic_deep_pullback_elite",
    "comb12_bollinger_squeeze_elite",
    "comb27_squeeze_supertrend_hybrid_elite"
]

# 2. Models to prune from EURGBP Fleet
eurgbp_prune = [
    "disc43_intraday_precision_h1_m5_elite"
]

print("=" * 80)
print("             PRUNING FRAGILE & HIGH-DRAG STRATEGIES FROM ACTIVE FLEETS             ".center(80))
print("=" * 80)

for folder_name in eurusd_prune:
    src = EURUSD_FLEET / folder_name
    dest = ARCHIVE_DIR / f"eurusd_{folder_name}"
    if src.exists():
        if dest.exists():
            shutil.rmtree(dest)
        shutil.move(str(src), str(dest))
        print(f"[+] Archived from EURUSD Fleet: {folder_name} -> archive_unprofitable/eurusd_{folder_name}")

for folder_name in eurgbp_prune:
    src = EURGBP_FLEET / folder_name
    dest = ARCHIVE_DIR / f"eurgbp_{folder_name}"
    if src.exists():
        if dest.exists():
            shutil.rmtree(dest)
        shutil.move(str(src), str(dest))
        print(f"[+] Archived from EURGBP Fleet: {folder_name} -> archive_unprofitable/eurgbp_{folder_name}")

# Also check root level strategies/ to keep it clean
root_prune = [
    "comb12_bollinger_squeeze_elite",
    "comb27_squeeze_supertrend_hybrid_elite",
    "high_frequency_m15_scalper_elite",
    "s10_dual_ema_pullback_elite",
    "s14_inside_bar_m30_elite",
    "s66_bhanushali_44ma_elite"
]
for folder_name in root_prune:
    src = STRAT_DIR / folder_name
    if src.exists():
        dest = ARCHIVE_DIR / f"root_{folder_name}"
        if dest.exists():
            shutil.rmtree(dest)
        shutil.move(str(src), str(dest))
        print(f"[+] Cleaned from root strategies/: {folder_name}")

print("\n[+] Pruning complete. Updating package exports...")
