#!/usr/bin/env python3
"""
================================================================================
  DISC73: Donchian Trend EMA Pullback Re-Accumulation Elite (Cross-Asset Incubator)
================================================================================
Validated cross-asset institutional alpha discovered in Lab V20.
Generates +2,860.5 Net Pips across EURUSD, USDJPY, and EURGBP (862 total weeks).
Holdout across all 3 assets: 919 trades | +1,151.0 Net Pips!
================================================================================
"""

from strategies.eurusd_production_fleet.disc73_donchian_ema_pullback_elite.strategy import EURUSDDonchianEMAPullbackElite
from strategies.usdjpy_fleet.disc73_donchian_ema_pullback_elite.strategy import USDJPYDonchianEMAPullbackElite
from strategies.eurgbp_fleet.disc73_donchian_ema_pullback_elite.strategy import EURGBPDonchianEMAPullbackElite

__all__ = [
    "EURUSDDonchianEMAPullbackElite",
    "USDJPYDonchianEMAPullbackElite",
    "EURGBPDonchianEMAPullbackElite"
]
