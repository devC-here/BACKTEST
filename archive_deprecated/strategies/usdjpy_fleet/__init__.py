"""
USDJPY Specialist Fleet (The 8 Quantitative Champions).
Audited on 300 continuous weeks of USDJPY (192.32M authentic Dukascopy ticks across 2021-2026).

Universal Structure & Precision Models:
  1. DISC49 (Intraday Macro Retest H1/M15): +404.8p | 72.2% WR | PF 4.35 | Max DD 24.1p | Ret/DD 16.80x
  2. DISC50 (M30/M5 Precision Retest):     +309.6p | 72.7% WR | PF 4.41 | Max DD 30.2p | Ret/DD 10.25x
  3. DISC42 (Macro Structure H4/M30):       +265.7p | 74.2% WR | PF 4.28 | Max DD 20.2p | Ret/DD 13.15x
  4. DISC46 (Daily Pivot Wall):             +492.5p | 60.0% WR | PF 2.43 | Max DD 67.0p | Ret/DD 7.35x
  5. DISC45 (Continuous M5 Scalper):        +545.7p | 53.1% WR | PF 1.48 | Max DD 167.3p | Ret/DD 3.26x
  6. DISC47 (Weekly Pivot Wall):            +451.7p | 49.3% WR | PF 1.31 | Max DD 191.0p | Ret/DD 2.36x
  7. DISC48 (Camarilla Floor):              +416.9p | 51.5% WR | PF 1.44 | Max DD 162.7p | Ret/DD 2.56x
  8. DISC43 (Intraday Precision H1/M5):     +257.7p | 48.2% WR | PF 1.48 | Max DD 58.1p | Ret/DD 4.44x

Total 5-Year USDJPY Fleet Net Pips: +3,144.6 Pips
"""

from .disc49_intraday_macro_retest_elite.strategy import USDJPYIntradayMacroRetestElite
from .disc50_m30_m5_precision_retest_elite.strategy import USDJPYM30M5PrecisionRetestElite
from .disc51_macro_asymmetric_runner_elite.strategy import USDJPYMacroAsymmetricRunnerElite
from .disc52_fib_golden_pocket_pin_defense_elite.strategy import USDJPYFibGoldenPocketPinDefenseElite
from .disc53_m15_3bar_trend_pullback_elite.strategy import USDJPYM15ThreeBarTrendPullbackElite
from .disc54_h1_inside_bar_expansion_elite.strategy import USDJPYH1InsideBarExpansionElite
from .disc55_m15_200ema_dynamic_bounce_elite.strategy import USDJPYM15TwoHundredEMABounceElite
from .disc56_triple_screen_m15_impulse_elite.strategy import USDJPYTripleScreenM15ImpulseElite
from .disc57_london_ignition_range_break_elite.strategy import USDJPYLondonIgnitionRangeBreakoutElite
from .disc58_h1_keltner_squeeze_explosion_elite.strategy import USDJPYH1KeltnerSqueezeExplosionElite
from .disc59_m15_macd_zero_cross_momentum_elite.strategy import USDJPYM15MACDZeroCrossElite
from .disc60_asian_compression_breakout_elite.strategy import USDJPYAsianCompressionBreakoutElite
from .disc61_double_structure_neckline_breakout_elite.strategy import USDJPYDoubleStructureNecklineBreakoutElite
from .disc62_stoch_rsi_pullback_hook_elite.strategy import USDJPYStochRSIPullbackHookElite
from .disc64_asian_midpoint_rebound_elite.strategy import USDJPYAsianMidpointReboundElite
from .disc65_session_vwap_band_fade_elite.strategy import USDJPYSessionVWAPBandFadeElite
from .disc66_heikin_ashi_surge_elite.strategy import USDJPYHeikinAshiSurgeElite
from .disc67_london_lunch_expansion_elite.strategy import USDJPYLondonLunchExpansionElite
from .disc68_cmf_momentum_expansion_elite.strategy import USDJPYCMFMomentumExpansionElite
from .disc71_asian_hl_retest_elite.strategy import USDJPYAsianHLRetestElite
from .disc73_donchian_ema_pullback_elite.strategy import USDJPYDonchianEMAPullbackElite
from .disc74_daily_pivot_pullback_elite.strategy import USDJPYDailyPivotPullbackElite
from .disc75_weekly_pivot_pullback_elite.strategy import USDJPYWeeklyPivotPullbackElite
from .disc76_vwap_1sd_rebound_elite.strategy import USDJPYVWAP1SDReboundElite
from .disc77_keltner_midline_expansion_elite.strategy import USDJPYKeltnerMidlineExpansionElite
from .disc78_three_bar_inside_confirmation_elite.strategy import USDJPYThreeBarInsideConfirmationElite
from .disc79_london_open_15m_expansion_elite.strategy import USDJPYLondonOpen15MExpansionElite
from .disc80_double_swing_neckline_retest_elite.strategy import USDJPYDoubleSwingNecklineRetestElite
from .disc81_bb_squeeze_m5_ignition_elite.strategy import USDJPYBBSqueezeIgnitionElite
from .disc82_chaikin_volatility_surge_elite.strategy import USDJPYChaikinVolatilitySurgeElite
from .disc83_parabolic_sar_flip_elite.strategy import USDJPYParabolicSARFlipElite
from .disc84_london_30m_orb_continuation_elite.strategy import USDJPYLondon30MORBContinuationElite
from .disc85_session_open_impulse_elite.strategy import USDJPYSessionOpenImpulseElite
from .disc86_ema_cross_atr_expansion_elite.strategy import USDJPYEMACrossATRExpansionElite
from .disc87_pivot_r1_s1_breakout_elite.strategy import USDJPYPivotR1S1BreakoutElite
from .disc88_three_drive_ignition_elite.strategy import USDJPYThreeDriveIgnitionElite
from strategies.eurusd_production_fleet.disc42_macro_structure_h4_m30_elite.strategy import MacroStructureH4M30Elite
from strategies.eurusd_production_fleet.disc45_continuous_m5_rejection_elite.strategy import ContinuousM5RejectionElite
from strategies.eurusd_production_fleet.disc43_intraday_precision_h1_m5_elite.strategy import IntradayPrecisionH1M5Elite
from strategies.eurgbp_fleet.disc46_daily_pivot_wall_elite.strategy import EURGBPDailyPivotWallElite
from strategies.eurgbp_fleet.disc47_weekly_pivot_wall_elite.strategy import EURGBPWeeklyPivotWallElite
from strategies.eurgbp_fleet.disc48_camarilla_floor_elite.strategy import EURGBPCamarillaFloorElite

__all__ = [
    "USDJPYThreeDriveIgnitionElite",
    "USDJPYPivotR1S1BreakoutElite",
    "USDJPYEMACrossATRExpansionElite",
    "USDJPYSessionOpenImpulseElite",
    "USDJPYLondon30MORBContinuationElite",
    "USDJPYParabolicSARFlipElite",
    "USDJPYChaikinVolatilitySurgeElite",
    "USDJPYBBSqueezeIgnitionElite",
    "USDJPYDoubleSwingNecklineRetestElite",
    "USDJPYLondonOpen15MExpansionElite",
    "USDJPYThreeBarInsideConfirmationElite",
    "USDJPYKeltnerMidlineExpansionElite",
    "USDJPYVWAP1SDReboundElite",
    "USDJPYWeeklyPivotPullbackElite",
    "USDJPYDailyPivotPullbackElite",
    "USDJPYDonchianEMAPullbackElite",
    "USDJPYAsianHLRetestElite",
    "USDJPYCMFMomentumExpansionElite",
    "USDJPYHeikinAshiSurgeElite",
    "USDJPYLondonLunchExpansionElite",
    "USDJPYSessionVWAPBandFadeElite",
    "USDJPYAsianMidpointReboundElite",
    "USDJPYStochRSIPullbackHookElite",
    "USDJPYDoubleStructureNecklineBreakoutElite",
    "USDJPYAsianCompressionBreakoutElite",
    "USDJPYM15MACDZeroCrossElite",
    "USDJPYH1KeltnerSqueezeExplosionElite",
    "USDJPYLondonIgnitionRangeBreakoutElite",
    "USDJPYTripleScreenM15ImpulseElite",
    "USDJPYMacroAsymmetricRunnerElite",
    "USDJPYIntradayMacroRetestElite",
    "USDJPYM30M5PrecisionRetestElite",
    "USDJPYFibGoldenPocketPinDefenseElite",
    "USDJPYM15ThreeBarTrendPullbackElite",
    "USDJPYH1InsideBarExpansionElite",
    "USDJPYM15TwoHundredEMABounceElite",
    "MacroStructureH4M30Elite",
    "ContinuousM5RejectionElite",
    "IntradayPrecisionH1M5Elite",
    "EURGBPDailyPivotWallElite",
    "EURGBPWeeklyPivotWallElite",
    "EURGBPCamarillaFloorElite",
]
