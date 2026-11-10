"""
Pristine, Frozen Production Fleet for EURUSD (14 Elite Models).
Audited on 5 Full Calendar Years (262 Continuous Weeks / ~123.22M Dukascopy Ticks).
DO NOT MODIFY OR MINGLE WITH EXPERIMENTAL CROSS-PAIR CODE.
"""

from .disc42_macro_structure_h4_m30_elite.strategy import MacroStructureH4M30Elite
from .disc45_continuous_m5_rejection_elite.strategy import ContinuousM5RejectionElite
from .disc40_rsi_regime_range_shift_elite.strategy import RSIRegimeRangeShiftElite
from .disc43_intraday_precision_h1_m5_elite.strategy import IntradayPrecisionH1M5Elite
from .s66_bhanushali_44ma_elite.strategy import S66Bhanushali44MAElite
from .high_frequency_m15_scalper_elite.strategy import HighFrequencyM15ScalperElite
from .s10_dual_ema_pullback_elite.strategy import S10DualEMAPullbackElite
from .s14_inside_bar_m30_elite.strategy import S14InsideBarM30Elite
from .disc03_ny_open_overlap_elite.strategy import NewYorkOpenOverlapBreakoutElite
from .disc41_thursday_macro_trend_ignition_elite.strategy import ThursdayMacroTrendIgnitionElite
from .disc38_london_midday_macd_acceleration_elite.strategy import LondonMiddayMACDAccelerationElite
from .disc39_stochastic_deep_pullback_elite.strategy import StochasticDeepPullbackElite
from .comb12_bollinger_squeeze_elite.strategy import BollingerBandwidthExpansionElite
from .comb27_squeeze_supertrend_hybrid_elite.strategy import BollingerSupertrendHybridElite

from .disc49_intraday_macro_retest_elite.strategy import IntradayMacroRetestElite
from .disc50_m30_m5_precision_retest_elite.strategy import M30M5PrecisionRetestElite
from .disc51_macro_asymmetric_runner_elite.strategy import MacroAsymmetricRunnerElite
from .disc52_fib_golden_pocket_pin_defense_elite.strategy import FibGoldenPocketPinDefenseElite
from .disc53_m15_3bar_trend_pullback_elite.strategy import EURUSDM15ThreeBarTrendPullbackElite
from .disc54_h1_inside_bar_expansion_elite.strategy import EURUSDH1InsideBarExpansionElite
from .disc56_triple_screen_m15_impulse_elite.strategy import EURUSDTripleScreenM15ImpulseElite
from .disc57_london_ignition_range_break_elite.strategy import EURUSDLondonIgnitionRangeBreakoutElite
from .disc58_h1_keltner_squeeze_explosion_elite.strategy import EURUSDH1KeltnerSqueezeExplosionElite
from .disc59_m15_macd_zero_cross_momentum_elite.strategy import EURUSDM15MACDZeroCrossElite
from .disc60_asian_compression_breakout_elite.strategy import EURUSDAsianCompressionBreakoutElite
from .disc61_double_structure_neckline_breakout_elite.strategy import EURUSDDoubleStructureNecklineBreakoutElite
from .disc62_stoch_rsi_pullback_hook_elite.strategy import EURUSDStochRSIPullbackHookElite
from .disc64_asian_midpoint_rebound_elite.strategy import EURUSDAsianMidpointReboundElite
from .disc65_session_vwap_band_fade_elite.strategy import EURUSDSessionVWAPBandFadeElite
from .disc71_asian_hl_retest_elite.strategy import EURUSDAsianHLRetestElite
from .disc72_h4_stoch_m5_wick_scalper_elite.strategy import EURUSDH4StochM5WickScalperElite
from .disc73_donchian_ema_pullback_elite.strategy import EURUSDDonchianEMAPullbackElite
from .disc74_daily_pivot_pullback_elite.strategy import EURUSDDailyPivotPullbackElite
from .disc75_weekly_pivot_pullback_elite.strategy import EURUSDWeeklyPivotPullbackElite
from .disc76_vwap_1sd_rebound_elite.strategy import EURUSDVWAP1SDReboundElite
from .disc77_keltner_midline_expansion_elite.strategy import EURUSDKeltnerMidlineExpansionElite
from .disc78_three_bar_inside_confirmation_elite.strategy import EURUSDThreeBarInsideConfirmationElite
from .disc79_london_open_15m_expansion_elite.strategy import EURUSDLondonOpen15MExpansionElite
from .disc80_double_swing_neckline_retest_elite.strategy import EURUSDDoubleSwingNecklineRetestElite
from .disc82_chaikin_volatility_surge_elite.strategy import EURUSDChaikinVolatilitySurgeElite
from .disc83_parabolic_sar_flip_elite.strategy import EURUSDParabolicSARFlipElite
from .disc85_session_open_impulse_elite.strategy import EURUSDSessionOpenImpulseElite
from .disc86_ema_cross_atr_expansion_elite.strategy import EURUSDEMACrossATRExpansionElite
from .disc88_three_drive_ignition_elite.strategy import EURUSDThreeDriveIgnitionElite

__all__ = [
    "EURUSDThreeDriveIgnitionElite",
    "EURUSDEMACrossATRExpansionElite",
    "EURUSDSessionOpenImpulseElite",
    "EURUSDParabolicSARFlipElite",
    "EURUSDChaikinVolatilitySurgeElite",
    "EURUSDDoubleSwingNecklineRetestElite",
    "EURUSDLondonOpen15MExpansionElite",
    "EURUSDThreeBarInsideConfirmationElite",
    "EURUSDKeltnerMidlineExpansionElite",
    "EURUSDVWAP1SDReboundElite",
    "EURUSDWeeklyPivotPullbackElite",
    "EURUSDDailyPivotPullbackElite",
    "EURUSDDonchianEMAPullbackElite",
    "EURUSDH4StochM5WickScalperElite",
    "EURUSDAsianHLRetestElite",
    "EURUSDSessionVWAPBandFadeElite",
    "EURUSDAsianMidpointReboundElite",
    "EURUSDStochRSIPullbackHookElite",
    "EURUSDDoubleStructureNecklineBreakoutElite",
    "EURUSDAsianCompressionBreakoutElite",
    "EURUSDM15MACDZeroCrossElite",
    "EURUSDH1KeltnerSqueezeExplosionElite",
    "EURUSDLondonIgnitionRangeBreakoutElite",
    "EURUSDTripleScreenM15ImpulseElite",
    "EURUSDH1InsideBarExpansionElite",
    "EURUSDM15ThreeBarTrendPullbackElite",
    "FibGoldenPocketPinDefenseElite",
    "MacroAsymmetricRunnerElite",
    "IntradayMacroRetestElite",
    "M30M5PrecisionRetestElite",
    "MacroStructureH4M30Elite",
    "ContinuousM5RejectionElite",
    "RSIRegimeRangeShiftElite",
    "IntradayPrecisionH1M5Elite",
    "S66Bhanushali44MAElite",
    "HighFrequencyM15ScalperElite",
    "S10DualEMAPullbackElite",
    "S14InsideBarM30Elite",
    "NewYorkOpenOverlapBreakoutElite",
    "ThursdayMacroTrendIgnitionElite",
    "LondonMiddayMACDAccelerationElite",
    "StochasticDeepPullbackElite",
    "BollingerBandwidthExpansionElite",
    "BollingerSupertrendHybridElite",
]

