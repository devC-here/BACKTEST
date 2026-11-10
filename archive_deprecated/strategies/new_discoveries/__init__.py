"""
NEW DISCOVERIES INCUBATOR ROSTER (VERIFIED ELITE MODELS)
========================================================
Dedicated repository for newly mined quantitative alpha models discovered via
the Data Engineering Suite. Kept strictly separate from legacy models.

Verified Models in this Suite:
  1. DISC38: London & Midday MACD Acceleration Elite (+372.7p, PF 1.28)
  2. DISC39: Stochastic Deep Pullback in Trend Elite (+483.4p, PF 1.22)
  3. DISC40: RSI Bull/Bear Regime Range Shift Elite (+1,112.8p, PF 1.31)
  4. DISC41: Thursday Macro Trend Ignition Elite (+336.1p, PF 1.30)
  5. DISC42: Macro Structure H4 Level + M30 Trigger Elite (+387.6p, PF 3.97, Max DD 20.4p)
  6. DISC43: Intraday Precision H1 Level + M5 Rejection Elite (+437.2p, PF 1.47, Max DD 68.7p)
"""

try:
    from strategies.archive_non_viable.disc38_london_midday_macd_acceleration_elite.strategy import LondonMiddayMACDAccelerationElite
except Exception:
    LondonMiddayMACDAccelerationElite = None
from .disc39_stochastic_deep_pullback_elite.strategy import StochasticDeepPullbackElite
from .disc40_rsi_regime_range_shift_elite.strategy import RSIRegimeRangeShiftElite
from .disc41_thursday_macro_trend_ignition_elite.strategy import ThursdayMacroTrendIgnitionElite
from .disc42_macro_structure_h4_m30_elite.strategy import MacroStructureH4M30Elite
from .disc43_intraday_precision_h1_m5_elite.strategy import IntradayPrecisionH1M5Elite
from .disc45_continuous_m5_rejection_elite.strategy import ContinuousM5RejectionElite

__all__ = [
    "LondonMiddayMACDAccelerationElite",
    "StochasticDeepPullbackElite",
    "RSIRegimeRangeShiftElite",
    "ThursdayMacroTrendIgnitionElite",
    "MacroStructureH4M30Elite",
    "IntradayPrecisionH1M5Elite",
    "ContinuousM5RejectionElite"
]
