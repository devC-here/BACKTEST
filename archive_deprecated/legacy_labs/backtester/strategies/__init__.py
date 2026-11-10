"""
Built-in strategy registry for the backtest engine.
"""

from backtester.strategies.dual_ema import DualEMAStrategy
from backtester.strategies.rsi_pullback import RSIPullbackStrategy
from backtester.strategies.breakout import DonchianBreakoutStrategy
from backtester.strategies.supertrend import SupertrendStrategy
from backtester.strategies.ict_double_sweep import ICTDoubleSweepStrategy
from backtester.strategies.london_breakout import LondonBreakoutStrategy
from backtester.strategies.rayner_teo_rsi import RaynerTeoRSIStrategy
from backtester.strategies.asia_forex_mentor import AsiaForexMentorStrategy
from backtester.strategies.booming_halftrend import BoomingHalfTrendStrategy

AVAILABLE_STRATEGIES = {
    "dual_ema": DualEMAStrategy,
    "rsi_pullback": RSIPullbackStrategy,
    "breakout": DonchianBreakoutStrategy,
    "supertrend": SupertrendStrategy,
    "ict_double_sweep": ICTDoubleSweepStrategy,
    "london_breakout": LondonBreakoutStrategy,
    "rayner_teo_rsi": RaynerTeoRSIStrategy,
    "asia_forex_mentor": AsiaForexMentorStrategy,
    "booming_halftrend": BoomingHalfTrendStrategy,
}


def get_strategy(name: str, **kwargs):
    """Factory function to get strategy instance by name."""
    key = name.lower().replace("-", "_").replace(" ", "_")
    if key not in AVAILABLE_STRATEGIES:
        raise ValueError(
            f"Strategy '{name}' not found. Available: {list(AVAILABLE_STRATEGIES.keys())}"
        )
    return AVAILABLE_STRATEGIES[key](**kwargs)
