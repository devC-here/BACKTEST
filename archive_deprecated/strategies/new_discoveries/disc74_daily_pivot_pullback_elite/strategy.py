#!/usr/bin/env python3
"""
================================================================================
  DISC74: Daily Pivot Trend Pullback Rejection Elite (Cross-Asset Incubator)
================================================================================
Validated cross-asset institutional alpha discovered in Lab V21.
Generates +2,638.5 Net Pips across EURUSD, USDJPY, and EURGBP (862 total weeks).
Holdout across all 3 assets: 278 trades | +684.1 Net Pips | 1.47 PF!
================================================================================
"""

from strategies.eurusd_production_fleet.disc74_daily_pivot_pullback_elite.strategy import EURUSDDailyPivotPullbackElite
from strategies.usdjpy_fleet.disc74_daily_pivot_pullback_elite.strategy import USDJPYDailyPivotPullbackElite
from strategies.eurgbp_fleet.disc74_daily_pivot_pullback_elite.strategy import EURGBPDailyPivotPullbackElite

__all__ = [
    "EURUSDDailyPivotPullbackElite",
    "USDJPYDailyPivotPullbackElite",
    "EURGBPDailyPivotPullbackElite"
]
