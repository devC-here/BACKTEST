"""
Titan 8 Elite Production Fleet (Least-Correlated Champions)
Contains strictly the best 8 least-correlated trading models across EURUSD, USDJPY, and EURGBP.
Selected via full 5-year covariance optimization and verified on 30% holdout data.
"""

from . import eurusd
from . import usdjpy
from . import eurgbp

__all__ = ["eurusd", "usdjpy", "eurgbp"]
