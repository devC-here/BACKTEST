"""
================================================================================
  EURGBP Specialist Quantitative Fleet (8 Production Champions)
================================================================================
Audited across 5.77 continuous years (300 continuous weeks / 85,813,388 ticks).
Balanced to the Top 8 highest-performing, least-correlated production champions.
Every strategy certified on quarantined 30% holdout with Annual Return >= 15.0%.
================================================================================
"""

from .disc121_session_vwap_fade_elite.strategy import EURGBPSessionVWAPFadeElite
from .disc123_daily_pivot_pullback_elite.strategy import EURGBPDailyPivotPullbackSniperElite
from .disc56_triple_screen_m15_impulse_elite.strategy import EURGBPTripleScreenM15ImpulseElite
from .disc122_fib_golden_pocket_lb36_elite.strategy import EURGBPFibGoldenPocketLB36Elite
from .disc125_fib_golden_pocket_lb48_elite.strategy import EURGBPFibGoldenPocketLB48HighWRElite
from .disc48_camarilla_floor_elite.strategy import EURGBPCamarillaFloorElite
from .disc46_daily_pivot_wall_elite.strategy import EURGBPDailyPivotWallElite
from .disc58_h1_keltner_squeeze_explosion_elite.strategy import EURGBPH1KeltnerSqueezeExplosionElite

__all__ = [
    "EURGBPSessionVWAPFadeElite",
    "EURGBPDailyPivotPullbackSniperElite",
    "EURGBPTripleScreenM15ImpulseElite",
    "EURGBPFibGoldenPocketLB36Elite",
    "EURGBPFibGoldenPocketLB48HighWRElite",
    "EURGBPCamarillaFloorElite",
    "EURGBPDailyPivotWallElite",
    "EURGBPH1KeltnerSqueezeExplosionElite",
]
