"""
Titan 8 Production Fleet - USDJPY Champions (3 Models)
Strictly least-correlated institutional models verified across 5-year full cycle and 30% holdout.
"""

from .disc90_opening_range_impulse_expansion_elite.strategy import USDJPYOpeningRangeImpulseExpansionElite
from .disc96_cardwell_rsi50_momentum_bounce_elite.strategy import USDJPYCardwellRSI50MomentumBounceElite
from .disc100_double_structure_neckline_breakout_elite.strategy import USDJPYDoubleStructureNecklineBreakoutElite

__all__ = [
    "USDJPYOpeningRangeImpulseExpansionElite",
    "USDJPYCardwellRSI50MomentumBounceElite",
    "USDJPYDoubleStructureNecklineBreakoutElite",
]
