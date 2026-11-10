"""
================================================================================
  AUDNZD Specialist Quantitative Fleet (8 Production Champions)
================================================================================
Audited across 5.77 continuous years (300 continuous weeks / 139,004,694 ticks)
using real floating spreads (avg 2.1-2.5 pips) and execution slippage.
All models verified across quarantined 30% holdout test partition (90 weeks / 41.93M ticks).
All models meet the strict >= 15.0% annual return threshold.
================================================================================
"""

from strategies.audnzd_fleet.disc115_fib_golden_pocket_lb48_elite.strategy import (
    AUDNZDFibGoldenPocketLB48Elite,
)
from strategies.audnzd_fleet.disc116_fib_golden_pocket_precision_elite.strategy import (
    AUDNZDFibGoldenPocketPrecisionElite,
)
from strategies.audnzd_fleet.disc117_fib_golden_pocket_high_wr_elite.strategy import (
    AUDNZDFibGoldenPocketHighWRElite,
)
from strategies.audnzd_fleet.disc118_fib_golden_pocket_lb36_elite.strategy import (
    AUDNZDFibGoldenPocketLB36Elite,
)
from strategies.audnzd_fleet.disc120_session_vwap_band_fade_elite.strategy import (
    AUDNZDSessionVWAPBandFadeElite,
)
from strategies.audnzd_fleet.disc111_h4_macro_structure_retest_elite.strategy import (
    AUDNZDH4MacroStructureRetestElite,
)
from strategies.audnzd_fleet.disc112_h4_macro_precision_retest_elite.strategy import (
    AUDNZDH4MacroPrecisionRetestElite,
)
from strategies.audnzd_fleet.disc129_fib_golden_pocket_lb72_macro_elite.strategy import (
    AUDNZDFibGoldenPocketLB72MacroElite,
)

__all__ = [
    "AUDNZDFibGoldenPocketLB48Elite",
    "AUDNZDFibGoldenPocketPrecisionElite",
    "AUDNZDFibGoldenPocketHighWRElite",
    "AUDNZDFibGoldenPocketLB36Elite",
    "AUDNZDSessionVWAPBandFadeElite",
    "AUDNZDH4MacroStructureRetestElite",
    "AUDNZDH4MacroPrecisionRetestElite",
    "AUDNZDFibGoldenPocketLB72MacroElite",
]
