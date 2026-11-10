"""
Certified Qualified Strategies Repository
Contains strictly verified, robust, and deployable trading models.

Sub-packages:
  - all_pairs: Universal champions that perform well across EURUSD, USDJPY, and EURGBP
  - eurusd: High-conviction specialists for EURUSD
  - usdjpy: High-conviction specialists for USDJPY
  - eurgbp: High-conviction specialists for EURGBP
"""

from . import all_pairs
from . import eurusd
from . import usdjpy
from . import eurgbp

__all__ = ["all_pairs", "eurusd", "usdjpy", "eurgbp"]
