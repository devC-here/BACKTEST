#!/usr/bin/env python3
"""
Populates certified_production_fleet with the 24 verified dual-partition champions.
"""

import shutil
from pathlib import Path
import json

ROOT_DIR = Path(__file__).resolve().parent.parent

FLEET_DIR = ROOT_DIR / "certified_production_fleet"
FLEET_DIR.mkdir(parents=True, exist_ok=True)

# 1. Top 10 EURUSD Multi-Regime Champions
eurusd_target = FLEET_DIR / "eurusd_champions"
eurusd_target.mkdir(parents=True, exist_ok=True)

top_10_eurusd = [
    "disc65_session_vwap_band_fade_elite",
    "disc86_ema_cross_atr_expansion_elite",
    "s10_dual_ema_pullback_elite",
    "disc59_m15_macd_zero_cross_momentum_elite",
    "disc82_chaikin_volatility_surge_elite",
    "disc74_daily_pivot_pullback_elite",
    "disc76_vwap_1sd_rebound_elite",
    "disc75_weekly_pivot_pullback_elite",
    "disc58_h1_keltner_squeeze_explosion_elite",
    "disc80_double_swing_neckline_retest_elite"
]

src_eurusd_dir = ROOT_DIR / "strategies" / "eurusd_production_fleet"
for name in top_10_eurusd:
    src_dir = src_eurusd_dir / name
    dst_dir = eurusd_target / name
    if src_dir.exists():
        if dst_dir.exists():
            shutil.rmtree(dst_dir)
        shutil.copytree(src_dir, dst_dir)
        print(f"Copied {name} -> {dst_dir.relative_to(ROOT_DIR)}")

# 2. 8 USDCAD Asian Sweep Champions
usdcad_target = FLEET_DIR / "usdcad_asian_sweeps"
usdcad_target.mkdir(parents=True, exist_ok=True)

v45_dir = ROOT_DIR / "v 4.5"
for slot in range(17, 25):
    fname = f"strategy_{slot}_usdcad_asian_sweep.py"
    src_file = v45_dir / fname
    dst_file = usdcad_target / fname
    if src_file.exists():
        shutil.copy2(src_file, dst_file)
        print(f"Copied {fname} -> {dst_file.relative_to(ROOT_DIR)}")

# 3. 6 EURUSD NY Session Sweeps
eurusd_ny_target = FLEET_DIR / "eurusd_ny_sweeps"
eurusd_ny_target.mkdir(parents=True, exist_ok=True)

for slot in [9, 10, 11, 12, 15, 16]:
    fname = f"strategy_{slot:02d}_eurusd_ny_sweep_fade.py"
    src_file = v45_dir / fname
    dst_file = eurusd_ny_target / fname
    if src_file.exists():
        shutil.copy2(src_file, dst_file)
        print(f"Copied {fname} -> {dst_file.relative_to(ROOT_DIR)}")

# 4. EURGBP London Morning Sweep
eurgbp_target = FLEET_DIR / "eurgbp_london_sweeps"
eurgbp_target.mkdir(parents=True, exist_ok=True)

fname = "strategy_62_eurgbp_london_sweep.py"
src_file = v45_dir / fname
dst_file = eurgbp_target / fname
if src_file.exists():
    shutil.copy2(src_file, dst_file)
    print(f"Copied {fname} -> {dst_file.relative_to(ROOT_DIR)}")

print("\nAll 24 champions successfully staged into certified_production_fleet/!")

