"""
Titan 8 Production Fleet - EURGBP Champions (2 Models)
Strictly least-correlated institutional models verified across 5-year full cycle and 30% holdout.
"""

from .disc48_camarilla_floor_elite.strategy import EURGBPCamarillaFloorElite
from .disc108_macro_asymmetric_runner_elite.strategy import EURGBPMacroAsymmetricRunnerElite

__all__ = [
    "EURGBPCamarillaFloorElite",
    "EURGBPMacroAsymmetricRunnerElite",
]
