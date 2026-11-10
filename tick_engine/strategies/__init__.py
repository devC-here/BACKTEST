"""
Strategies registry and factory for tick_engine.
"""

from __future__ import annotations
from typing import Dict, Type, Any, Callable
from .base import BaseStrategy, SIGNAL_HOLD, SIGNAL_LONG, SIGNAL_SHORT, SIGNAL_FORCE_EXIT
from .example_strategy import EMACrossoverStrategy
from .momentum import TickMomentumStrategy
from .top15 import Top15Strategy, TOP_15_FLEET, TOP_15_BY_ID, TOP_15_BY_CODE, build_rates_by_timeframe

STRATEGY_REGISTRY: Dict[str, Any] = {
    "ema_crossover": EMACrossoverStrategy,
    "emacrossover": EMACrossoverStrategy,
    "ema": EMACrossoverStrategy,
    "tick_momentum": TickMomentumStrategy,
    "momentum": TickMomentumStrategy,
}

# Register all 15 institutional models
for s in TOP_15_FLEET:
    sid = s["id"]
    code_key = s["code"].lower()
    
    def _make_factory(strat_id: int):
        return lambda **kwargs: Top15Strategy(strategy_id=strat_id, **kwargs)

    STRATEGY_REGISTRY[code_key] = _make_factory(sid)
    STRATEGY_REGISTRY[f"s{sid:02d}"] = _make_factory(sid)
    STRATEGY_REGISTRY[f"strategy_{sid}"] = _make_factory(sid)

# Named aliases
STRATEGY_REGISTRY["s01_ict"] = _make_factory(1)
STRATEGY_REGISTRY["s03_tg_london"] = _make_factory(3)
STRATEGY_REGISTRY["s06_fib_golden"] = _make_factory(6)
STRATEGY_REGISTRY["s08_smart_money"] = _make_factory(8)
STRATEGY_REGISTRY["s10_dual_ema"] = _make_factory(10)
STRATEGY_REGISTRY["s14_rayner_teo"] = _make_factory(14)
STRATEGY_REGISTRY["s18_macd_momentum"] = _make_factory(18)
STRATEGY_REGISTRY["s23_cpr"] = _make_factory(23)
STRATEGY_REGISTRY["s26_mamba_fx"] = _make_factory(26)
STRATEGY_REGISTRY["s28_chandelier"] = _make_factory(28)
STRATEGY_REGISTRY["s34_heikin_ashi"] = _make_factory(34)
STRATEGY_REGISTRY["s36_berlin_7ema"] = _make_factory(36)
STRATEGY_REGISTRY["s43_halftrend"] = _make_factory(43)
STRATEGY_REGISTRY["s51_booming_bulls"] = _make_factory(51)
STRATEGY_REGISTRY["s54_renko"] = _make_factory(54)


def get_strategy(name: str, **kwargs) -> BaseStrategy:
    """
    Factory function to instantiate registered strategy by name.
    """
    key = name.lower().replace("-", "_").replace(" ", "")
    if key not in STRATEGY_REGISTRY:
        available = ", ".join(sorted(set(STRATEGY_REGISTRY.keys())))
        raise ValueError(f"Unknown strategy '{name}'. Available strategies: {available}")
    
    target = STRATEGY_REGISTRY[key]
    if isinstance(target, type):
        return target(**kwargs)
    elif callable(target):
        return target(**kwargs)
    else:
        raise TypeError(f"Invalid strategy target in registry for key '{key}'")


def list_strategies() -> list[str]:
    """
    Returns unique list of strategy names.
    """
    return sorted(list(STRATEGY_REGISTRY.keys()))


__all__ = [
    "BaseStrategy",
    "SIGNAL_HOLD",
    "SIGNAL_LONG",
    "SIGNAL_SHORT",
    "SIGNAL_FORCE_EXIT",
    "EMACrossoverStrategy",
    "TickMomentumStrategy",
    "Top15Strategy",
    "TOP_15_FLEET",
    "TOP_15_BY_ID",
    "TOP_15_BY_CODE",
    "build_rates_by_timeframe",
    "STRATEGY_REGISTRY",
    "get_strategy",
    "list_strategies",
]
