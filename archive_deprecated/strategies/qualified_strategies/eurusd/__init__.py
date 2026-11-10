"""
Titan 8 Production Fleet - EURUSD Champions (3 Models)
Strictly least-correlated institutional models verified across 5-year full cycle and 30% holdout.
"""

from .disc105_chaikin_volatility_surge_elite.strategy import EURUSDChaikinVolatilitySurgeElite
from .disc103_double_swing_neckline_retest_elite.strategy import EURUSDDoubleSwingNecklineRetestElite
from .disc45_continuous_m5_rejection_elite.strategy import ContinuousM5RejectionElite

__all__ = [
    "EURUSDChaikinVolatilitySurgeElite",
    "EURUSDDoubleSwingNecklineRetestElite",
    "ContinuousM5RejectionElite",
]
