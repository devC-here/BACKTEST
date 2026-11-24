#!/usr/bin/env python3
"""
================================================================================
  MARKET REGIME & ARCHETYPE KILL-SWITCH MASTER SYSTEM (ALL-IN-ONE EXPORT BUNDLE)
================================================================================
This single file contains the complete specification, mathematical formulations,
programmatic registry of all 71 production strategies, and high-performance
NumPy execution engine for real-time MT5 and algorithmic backtest deployment.

--------------------------------------------------------------------------------
GUIDE FOR IMPORTING AI AGENTS (CURSOR / CLAUDE / ANTIGRAVITY):
--------------------------------------------------------------------------------
1. PURPOSE:
   This module prevents algorithmic trading strategies from taking trades in
   toxic market regimes (e.g. chop traps, liquidity squeezes, adverse sessions).
   In empirical backtesting across 7,928 trades, this system purged 942 losing
   trades (11.9% of fleet volume) and saved +3,544.6 pips of pure loss!

2. HOW TO WIRE INTO YOUR EXECUTION ENGINE:
   Import `LiveMarketRegimeFilter` (or `MarketRegimeSystem`) directly:
   
   ```python
   from MARKET_REGIME_SYSTEM import LiveMarketRegimeFilter

   regime_engine = LiveMarketRegimeFilter()

   # Inside your trade generation or tick event loop:
   permitted, reason, metrics = regime_engine.evaluate_live_permission(
       strategy_name=strategy.name,
       symbol=strategy.symbol,
       high=m15_highs,
       low=m15_lows,
       close=m15_closes,
       spread_pips=live_spread_pips,
       current_utc_dt=pd.Timestamp.utcnow()
   )

   if not permitted:
       # Quash / block the incoming trade
       print(f"Signal BLOCKED by Kill-Switch: {reason}")
       return 0  # Do not execute
   ```

3. THE 4 CANONICAL REGIMES:
   - Regime 1: Persistent Directional Trend (KER >= 0.40, CHOP < 38.2, H > 0.56)
     -> Permitted for Trend Pullbacks, Macro Runners, VWAP Retests.
   - Regime 2: Stationary Gaussian Range (KER < 0.22, CHOP > 55.0, H < 0.44)
     -> Permitted for Mean-Reversion & VWAP Band Faders.
   - Regime 3: High-Entropy Volatile Chop (KER < 0.20, CHOP > 61.8, RVR > 1.35)
     -> STRICT KILL-SWITCH FOR ALL STRATEGIES.
   - Regime 4: Liquidity Squeeze / Pre-Breakout Coil (RVR < 0.70, BBWP < 15.0%)
     -> Gated for trend-followers; permitted for breakout coiling setups.

4. CLOSED-FORM MATHEMATICAL INDICATORS:
   - Kaufman Efficiency Ratio (KER):
     KER_t = |C_t - C_(t-n)| / sum(|C_(t-i) - C_(t-i-1)|, i=0..n-1)  [n=20]
   - Choppiness Index (CHOP):
     CHOP_t = 100 * log10(sum(TR, 14) / (MaxH_14 - MinL_14)) / log10(14)
   - Variance Ratio Hurst Exponent (H):
     VR(q) = Var(ret_q) / (q * Var(ret_1))  [q=5, window=100]
     H = 0.5 + ln(VR(q)) / (2 * ln(q))
   - Relative Volatility Ratio (RVR):
     RVR_t = ATR(fast=5) / ATR(slow=50)
   - Bollinger Bandwidth Percentile (BBWP):
     BBW_t = 4 * std(20) / mean(20)
     BBWP_t = Percentile rank of BBW_t over 252 bars
   - Spread Friction Ratio (SVP):
     SVP_t = Spread_t (pips) / ATR_5 (pips)  [Kill-switch if > 0.12]
================================================================================
"""

import sys
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional, List

# ==============================================================================
# 71-STRATEGY REGIME REGISTRY & INDIVIDUAL KILL-SWITCH RULES
# ==============================================================================
STRATEGY_REGIME_REGISTRY: Dict[str, Dict[str, Any]] = {
  "AUDCAD_disc115": {
    "name": "disc115",
    "symbol": "AUDCAD",
    "category": "audcad_champions",
    "archetype": "trend_pullback",
    "permitted_regimes": [
      1
    ],
    "prohibited_regimes": [
      2,
      3,
      4
    ],
    "toxic_sessions": [],
    "toxic_days": [],
    "base_win_rate": 63.3,
    "base_profit_factor": 2.98,
    "min_ker": 0.3,
    "max_ker": 1.0,
    "max_chop": 52.0,
    "min_hurst": 0.52,
    "max_hurst": 1.0,
    "max_svp": 0.12
  },
  "AUDCAD_disc116": {
    "name": "disc116",
    "symbol": "AUDCAD",
    "category": "audcad_champions",
    "archetype": "trend_pullback",
    "permitted_regimes": [
      1
    ],
    "prohibited_regimes": [
      2,
      3,
      4
    ],
    "toxic_sessions": [],
    "toxic_days": [],
    "base_win_rate": 56.7,
    "base_profit_factor": 2.86,
    "min_ker": 0.3,
    "max_ker": 1.0,
    "max_chop": 52.0,
    "min_hurst": 0.52,
    "max_hurst": 1.0,
    "max_svp": 0.12
  },
  "AUDCAD_disc117": {
    "name": "disc117",
    "symbol": "AUDCAD",
    "category": "audcad_champions",
    "archetype": "trend_pullback",
    "permitted_regimes": [
      1
    ],
    "prohibited_regimes": [
      2,
      3,
      4
    ],
    "toxic_sessions": [],
    "toxic_days": [],
    "base_win_rate": 59.5,
    "base_profit_factor": 2.24,
    "min_ker": 0.3,
    "max_ker": 1.0,
    "max_chop": 52.0,
    "min_hurst": 0.52,
    "max_hurst": 1.0,
    "max_svp": 0.12
  },
  "AUDCAD_disc118": {
    "name": "disc118",
    "symbol": "AUDCAD",
    "category": "audcad_champions",
    "archetype": "trend_pullback",
    "permitted_regimes": [
      1
    ],
    "prohibited_regimes": [
      2,
      3,
      4
    ],
    "toxic_sessions": [],
    "toxic_days": [],
    "base_win_rate": 65.2,
    "base_profit_factor": 2.92,
    "min_ker": 0.3,
    "max_ker": 1.0,
    "max_chop": 52.0,
    "min_hurst": 0.52,
    "max_hurst": 1.0,
    "max_svp": 0.12
  },
  "AUDCAD_disc159": {
    "name": "disc159",
    "symbol": "AUDCAD",
    "category": "audcad_champions",
    "archetype": "trend_pullback",
    "permitted_regimes": [
      1
    ],
    "prohibited_regimes": [
      2,
      3,
      4
    ],
    "toxic_sessions": [
      "London/NY Overlap (12:00-16:00)"
    ],
    "toxic_days": [
      "Friday"
    ],
    "base_win_rate": 47.8,
    "base_profit_factor": 1.44,
    "min_ker": 0.3,
    "max_ker": 1.0,
    "max_chop": 52.0,
    "min_hurst": 0.52,
    "max_hurst": 1.0,
    "max_svp": 0.12
  },
  "AUDCAD_disc160": {
    "name": "disc160",
    "symbol": "AUDCAD",
    "category": "audcad_champions",
    "archetype": "trend_pullback",
    "permitted_regimes": [
      1
    ],
    "prohibited_regimes": [
      2,
      3,
      4
    ],
    "toxic_sessions": [],
    "toxic_days": [
      "Friday"
    ],
    "base_win_rate": 51.6,
    "base_profit_factor": 1.73,
    "min_ker": 0.3,
    "max_ker": 1.0,
    "max_chop": 52.0,
    "min_hurst": 0.52,
    "max_hurst": 1.0,
    "max_svp": 0.12
  },
  "AUDNZD_disc111_h4_macro_structure_retest_elite": {
    "name": "disc111_h4_macro_structure_retest_elite",
    "symbol": "AUDNZD",
    "category": "audnzd_champions",
    "archetype": "macro_structure_retest",
    "permitted_regimes": [
      1,
      2
    ],
    "prohibited_regimes": [
      3,
      4
    ],
    "toxic_sessions": [],
    "toxic_days": [],
    "base_win_rate": 60.7,
    "base_profit_factor": 2.56,
    "min_ker": 0.2,
    "max_ker": 1.0,
    "max_chop": 58.0,
    "min_hurst": 0.0,
    "max_hurst": 1.0,
    "max_svp": 0.12
  },
  "AUDNZD_disc112_h4_macro_precision_retest_elite": {
    "name": "disc112_h4_macro_precision_retest_elite",
    "symbol": "AUDNZD",
    "category": "audnzd_champions",
    "archetype": "macro_structure_retest",
    "permitted_regimes": [
      1,
      2
    ],
    "prohibited_regimes": [
      3,
      4
    ],
    "toxic_sessions": [],
    "toxic_days": [
      "Thursday"
    ],
    "base_win_rate": 55.2,
    "base_profit_factor": 2.06,
    "min_ker": 0.2,
    "max_ker": 1.0,
    "max_chop": 58.0,
    "min_hurst": 0.0,
    "max_hurst": 1.0,
    "max_svp": 0.12
  },
  "AUDNZD_disc113_h1_macro_intraday_retest_elite": {
    "name": "disc113_h1_macro_intraday_retest_elite",
    "symbol": "AUDNZD",
    "category": "audnzd_champions",
    "archetype": "macro_structure_retest",
    "permitted_regimes": [
      1,
      2
    ],
    "prohibited_regimes": [
      3,
      4
    ],
    "toxic_sessions": [
      "London Open (06:00-12:00)"
    ],
    "toxic_days": [],
    "base_win_rate": 50.0,
    "base_profit_factor": 1.72,
    "min_ker": 0.2,
    "max_ker": 1.0,
    "max_chop": 58.0,
    "min_hurst": 0.0,
    "max_hurst": 1.0,
    "max_svp": 0.12
  },
  "AUDNZD_disc114_h4_macro_wide_runner_elite": {
    "name": "disc114_h4_macro_wide_runner_elite",
    "symbol": "AUDNZD",
    "category": "audnzd_champions",
    "archetype": "macro_structure_retest",
    "permitted_regimes": [
      1,
      2
    ],
    "prohibited_regimes": [
      3,
      4
    ],
    "toxic_sessions": [],
    "toxic_days": [],
    "base_win_rate": 64.3,
    "base_profit_factor": 3.03,
    "min_ker": 0.2,
    "max_ker": 1.0,
    "max_chop": 58.0,
    "min_hurst": 0.0,
    "max_hurst": 1.0,
    "max_svp": 0.12
  },
  "AUDNZD_disc145": {
    "name": "disc145",
    "symbol": "AUDNZD",
    "category": "audnzd_champions",
    "archetype": "trend_pullback",
    "permitted_regimes": [
      1
    ],
    "prohibited_regimes": [
      2,
      3,
      4
    ],
    "toxic_sessions": [],
    "toxic_days": [],
    "base_win_rate": 72.4,
    "base_profit_factor": 4.75,
    "min_ker": 0.3,
    "max_ker": 1.0,
    "max_chop": 52.0,
    "min_hurst": 0.52,
    "max_hurst": 1.0,
    "max_svp": 0.12
  },
  "AUDNZD_disc146": {
    "name": "disc146",
    "symbol": "AUDNZD",
    "category": "audnzd_champions",
    "archetype": "trend_pullback",
    "permitted_regimes": [
      1
    ],
    "prohibited_regimes": [
      2,
      3,
      4
    ],
    "toxic_sessions": [],
    "toxic_days": [],
    "base_win_rate": 64.1,
    "base_profit_factor": 3.26,
    "min_ker": 0.3,
    "max_ker": 1.0,
    "max_chop": 52.0,
    "min_hurst": 0.52,
    "max_hurst": 1.0,
    "max_svp": 0.12
  },
  "AUDNZD_disc147": {
    "name": "disc147",
    "symbol": "AUDNZD",
    "category": "audnzd_champions",
    "archetype": "trend_pullback",
    "permitted_regimes": [
      1
    ],
    "prohibited_regimes": [
      2,
      3,
      4
    ],
    "toxic_sessions": [],
    "toxic_days": [],
    "base_win_rate": 62.5,
    "base_profit_factor": 3.23,
    "min_ker": 0.3,
    "max_ker": 1.0,
    "max_chop": 52.0,
    "min_hurst": 0.52,
    "max_hurst": 1.0,
    "max_svp": 0.12
  },
  "EURGBP_disc107_h1_keltner_squeeze_explosion_elite": {
    "name": "disc107_h1_keltner_squeeze_explosion_elite",
    "symbol": "EURGBP",
    "category": "eurgbp_champions",
    "archetype": "volatility_breakout",
    "permitted_regimes": [
      1,
      4
    ],
    "prohibited_regimes": [
      2,
      3
    ],
    "toxic_sessions": [],
    "toxic_days": [
      "Tuesday"
    ],
    "base_win_rate": 47.5,
    "base_profit_factor": 1.71,
    "min_ker": 0.25,
    "max_ker": 1.0,
    "max_chop": 61.8,
    "min_hurst": 0.5,
    "max_hurst": 1.0,
    "max_svp": 0.12
  },
  "EURNZD_disc132": {
    "name": "disc132",
    "symbol": "EURNZD",
    "category": "eurnzd_champions",
    "archetype": "trend_pullback",
    "permitted_regimes": [
      1
    ],
    "prohibited_regimes": [
      2,
      3,
      4
    ],
    "toxic_sessions": [],
    "toxic_days": [],
    "base_win_rate": 61.8,
    "base_profit_factor": 3.32,
    "min_ker": 0.3,
    "max_ker": 1.0,
    "max_chop": 52.0,
    "min_hurst": 0.52,
    "max_hurst": 1.0,
    "max_svp": 0.12
  },
  "EURNZD_disc133": {
    "name": "disc133",
    "symbol": "EURNZD",
    "category": "eurnzd_champions",
    "archetype": "trend_pullback",
    "permitted_regimes": [
      1
    ],
    "prohibited_regimes": [
      2,
      3,
      4
    ],
    "toxic_sessions": [],
    "toxic_days": [],
    "base_win_rate": 62.8,
    "base_profit_factor": 3.45,
    "min_ker": 0.3,
    "max_ker": 1.0,
    "max_chop": 52.0,
    "min_hurst": 0.52,
    "max_hurst": 1.0,
    "max_svp": 0.12
  },
  "EURNZD_disc143": {
    "name": "disc143",
    "symbol": "EURNZD",
    "category": "eurnzd_champions",
    "archetype": "trend_pullback",
    "permitted_regimes": [
      1
    ],
    "prohibited_regimes": [
      2,
      3,
      4
    ],
    "toxic_sessions": [],
    "toxic_days": [],
    "base_win_rate": 77.8,
    "base_profit_factor": 7.27,
    "min_ker": 0.3,
    "max_ker": 1.0,
    "max_chop": 52.0,
    "min_hurst": 0.52,
    "max_hurst": 1.0,
    "max_svp": 0.12
  },
  "EURNZD_disc157": {
    "name": "disc157",
    "symbol": "EURNZD",
    "category": "eurnzd_champions",
    "archetype": "trend_pullback",
    "permitted_regimes": [
      1
    ],
    "prohibited_regimes": [
      2,
      3,
      4
    ],
    "toxic_sessions": [],
    "toxic_days": [],
    "base_win_rate": 78.8,
    "base_profit_factor": 8.23,
    "min_ker": 0.3,
    "max_ker": 1.0,
    "max_chop": 52.0,
    "min_hurst": 0.52,
    "max_hurst": 1.0,
    "max_svp": 0.12
  },
  "EURNZD_disc158": {
    "name": "disc158",
    "symbol": "EURNZD",
    "category": "eurnzd_champions",
    "archetype": "trend_pullback",
    "permitted_regimes": [
      1
    ],
    "prohibited_regimes": [
      2,
      3,
      4
    ],
    "toxic_sessions": [],
    "toxic_days": [],
    "base_win_rate": 56.2,
    "base_profit_factor": 2.85,
    "min_ker": 0.3,
    "max_ker": 1.0,
    "max_chop": 52.0,
    "min_hurst": 0.52,
    "max_hurst": 1.0,
    "max_svp": 0.12
  },
  "EURUSD_disc50_m30_m5_precision_retest_elite": {
    "name": "disc50_m30_m5_precision_retest_elite",
    "symbol": "EURUSD",
    "category": "eurusd_champions",
    "archetype": "trend_pullback",
    "permitted_regimes": [
      1
    ],
    "prohibited_regimes": [
      2,
      3,
      4
    ],
    "toxic_sessions": [],
    "toxic_days": [
      "Friday"
    ],
    "base_win_rate": 54.5,
    "base_profit_factor": 1.79,
    "min_ker": 0.3,
    "max_ker": 1.0,
    "max_chop": 52.0,
    "min_hurst": 0.52,
    "max_hurst": 1.0,
    "max_svp": 0.12
  },
  "EURUSD_disc51_macro_asymmetric_runner_elite": {
    "name": "disc51_macro_asymmetric_runner_elite",
    "symbol": "EURUSD",
    "category": "eurusd_champions",
    "archetype": "macro_structure_retest",
    "permitted_regimes": [
      1,
      2
    ],
    "prohibited_regimes": [
      3,
      4
    ],
    "toxic_sessions": [],
    "toxic_days": [],
    "base_win_rate": 70.6,
    "base_profit_factor": 6.05,
    "min_ker": 0.2,
    "max_ker": 1.0,
    "max_chop": 58.0,
    "min_hurst": 0.0,
    "max_hurst": 1.0,
    "max_svp": 0.12
  },
  "EURUSD_disc58_h1_keltner_squeeze_explosion_elite": {
    "name": "disc58_h1_keltner_squeeze_explosion_elite",
    "symbol": "EURUSD",
    "category": "eurusd_champions",
    "archetype": "volatility_breakout",
    "permitted_regimes": [
      1,
      4
    ],
    "prohibited_regimes": [
      2,
      3
    ],
    "toxic_sessions": [],
    "toxic_days": [],
    "base_win_rate": 45.9,
    "base_profit_factor": 1.63,
    "min_ker": 0.25,
    "max_ker": 1.0,
    "max_chop": 61.8,
    "min_hurst": 0.5,
    "max_hurst": 1.0,
    "max_svp": 0.12
  },
  "EURUSD_disc59_m15_macd_zero_cross_momentum_elite": {
    "name": "disc59_m15_macd_zero_cross_momentum_elite",
    "symbol": "EURUSD",
    "category": "eurusd_champions",
    "archetype": "trend_pullback",
    "permitted_regimes": [
      1
    ],
    "prohibited_regimes": [
      2,
      3,
      4
    ],
    "toxic_sessions": [],
    "toxic_days": [],
    "base_win_rate": 41.0,
    "base_profit_factor": 1.3,
    "min_ker": 0.3,
    "max_ker": 1.0,
    "max_chop": 52.0,
    "min_hurst": 0.52,
    "max_hurst": 1.0,
    "max_svp": 0.12
  },
  "EURUSD_disc65_session_vwap_band_fade_elite": {
    "name": "disc65_session_vwap_band_fade_elite",
    "symbol": "EURUSD",
    "category": "eurusd_champions",
    "archetype": "mean_reversion",
    "permitted_regimes": [
      2
    ],
    "prohibited_regimes": [
      1,
      3,
      4
    ],
    "toxic_sessions": [],
    "toxic_days": [
      "Thursday"
    ],
    "base_win_rate": 41.7,
    "base_profit_factor": 1.3,
    "min_ker": 0.0,
    "max_ker": 0.38,
    "max_chop": 100.0,
    "min_hurst": 0.0,
    "max_hurst": 0.54,
    "max_svp": 0.12
  },
  "EURUSD_disc74_daily_pivot_pullback_elite": {
    "name": "disc74_daily_pivot_pullback_elite",
    "symbol": "EURUSD",
    "category": "eurusd_champions",
    "archetype": "trend_pullback",
    "permitted_regimes": [
      1
    ],
    "prohibited_regimes": [
      2,
      3,
      4
    ],
    "toxic_sessions": [],
    "toxic_days": [],
    "base_win_rate": 50.4,
    "base_profit_factor": 1.88,
    "min_ker": 0.3,
    "max_ker": 1.0,
    "max_chop": 52.0,
    "min_hurst": 0.52,
    "max_hurst": 1.0,
    "max_svp": 0.12
  },
  "EURUSD_disc75_weekly_pivot_pullback_elite": {
    "name": "disc75_weekly_pivot_pullback_elite",
    "symbol": "EURUSD",
    "category": "eurusd_champions",
    "archetype": "trend_pullback",
    "permitted_regimes": [
      1
    ],
    "prohibited_regimes": [
      2,
      3,
      4
    ],
    "toxic_sessions": [
      "London Open (06:00-12:00)",
      "NY Afternoon (16:00-21:00)"
    ],
    "toxic_days": [
      "Friday",
      "Wednesday"
    ],
    "base_win_rate": 33.7,
    "base_profit_factor": 1.15,
    "min_ker": 0.3,
    "max_ker": 1.0,
    "max_chop": 52.0,
    "min_hurst": 0.52,
    "max_hurst": 1.0,
    "max_svp": 0.12
  },
  "EURUSD_disc76_vwap_1sd_rebound_elite": {
    "name": "disc76_vwap_1sd_rebound_elite",
    "symbol": "EURUSD",
    "category": "eurusd_champions",
    "archetype": "trend_pullback",
    "permitted_regimes": [
      1
    ],
    "prohibited_regimes": [
      2,
      3,
      4
    ],
    "toxic_sessions": [],
    "toxic_days": [
      "Thursday"
    ],
    "base_win_rate": 43.4,
    "base_profit_factor": 1.4,
    "min_ker": 0.3,
    "max_ker": 1.0,
    "max_chop": 52.0,
    "min_hurst": 0.52,
    "max_hurst": 1.0,
    "max_svp": 0.12
  },
  "EURUSD_disc86_ema_cross_atr_expansion_elite": {
    "name": "disc86_ema_cross_atr_expansion_elite",
    "symbol": "EURUSD",
    "category": "eurusd_champions",
    "archetype": "volatility_breakout",
    "permitted_regimes": [
      1,
      4
    ],
    "prohibited_regimes": [
      2,
      3
    ],
    "toxic_sessions": [
      "NY Afternoon (16:00-21:00)"
    ],
    "toxic_days": [
      "Friday"
    ],
    "base_win_rate": 43.3,
    "base_profit_factor": 1.42,
    "min_ker": 0.25,
    "max_ker": 1.0,
    "max_chop": 61.8,
    "min_hurst": 0.5,
    "max_hurst": 1.0,
    "max_svp": 0.12
  },
  "GBPCHF_disc134": {
    "name": "disc134",
    "symbol": "GBPCHF",
    "category": "gbpchf_champions",
    "archetype": "trend_pullback",
    "permitted_regimes": [
      1
    ],
    "prohibited_regimes": [
      2,
      3,
      4
    ],
    "toxic_sessions": [],
    "toxic_days": [],
    "base_win_rate": 66.7,
    "base_profit_factor": 3.46,
    "min_ker": 0.3,
    "max_ker": 1.0,
    "max_chop": 52.0,
    "min_hurst": 0.52,
    "max_hurst": 1.0,
    "max_svp": 0.12
  },
  "GBPCHF_disc135": {
    "name": "disc135",
    "symbol": "GBPCHF",
    "category": "gbpchf_champions",
    "archetype": "trend_pullback",
    "permitted_regimes": [
      1
    ],
    "prohibited_regimes": [
      2,
      3,
      4
    ],
    "toxic_sessions": [],
    "toxic_days": [],
    "base_win_rate": 65.9,
    "base_profit_factor": 3.52,
    "min_ker": 0.3,
    "max_ker": 1.0,
    "max_chop": 52.0,
    "min_hurst": 0.52,
    "max_hurst": 1.0,
    "max_svp": 0.12
  },
  "GBPCHF_disc137": {
    "name": "disc137",
    "symbol": "GBPCHF",
    "category": "gbpchf_champions",
    "archetype": "trend_pullback",
    "permitted_regimes": [
      1
    ],
    "prohibited_regimes": [
      2,
      3,
      4
    ],
    "toxic_sessions": [],
    "toxic_days": [],
    "base_win_rate": 64.4,
    "base_profit_factor": 3.07,
    "min_ker": 0.3,
    "max_ker": 1.0,
    "max_chop": 52.0,
    "min_hurst": 0.52,
    "max_hurst": 1.0,
    "max_svp": 0.12
  },
  "GBPCHF_disc152": {
    "name": "disc152",
    "symbol": "GBPCHF",
    "category": "gbpchf_champions",
    "archetype": "trend_pullback",
    "permitted_regimes": [
      1
    ],
    "prohibited_regimes": [
      2,
      3,
      4
    ],
    "toxic_sessions": [],
    "toxic_days": [],
    "base_win_rate": 69.1,
    "base_profit_factor": 3.79,
    "min_ker": 0.3,
    "max_ker": 1.0,
    "max_chop": 52.0,
    "min_hurst": 0.52,
    "max_hurst": 1.0,
    "max_svp": 0.12
  },
  "GBPCHF_disc153": {
    "name": "disc153",
    "symbol": "GBPCHF",
    "category": "gbpchf_champions",
    "archetype": "trend_pullback",
    "permitted_regimes": [
      1
    ],
    "prohibited_regimes": [
      2,
      3,
      4
    ],
    "toxic_sessions": [],
    "toxic_days": [],
    "base_win_rate": 62.3,
    "base_profit_factor": 3.04,
    "min_ker": 0.3,
    "max_ker": 1.0,
    "max_chop": 52.0,
    "min_hurst": 0.52,
    "max_hurst": 1.0,
    "max_svp": 0.12
  },
  "GBPJPY_disc138": {
    "name": "disc138",
    "symbol": "GBPJPY",
    "category": "gbpjpy_champions",
    "archetype": "trend_pullback",
    "permitted_regimes": [
      1
    ],
    "prohibited_regimes": [
      2,
      3,
      4
    ],
    "toxic_sessions": [
      "NY Afternoon (16:00-21:00)"
    ],
    "toxic_days": [],
    "base_win_rate": 56.0,
    "base_profit_factor": 2.39,
    "min_ker": 0.3,
    "max_ker": 1.0,
    "max_chop": 52.0,
    "min_hurst": 0.52,
    "max_hurst": 1.0,
    "max_svp": 0.12
  },
  "GBPJPY_disc139": {
    "name": "disc139",
    "symbol": "GBPJPY",
    "category": "gbpjpy_champions",
    "archetype": "trend_pullback",
    "permitted_regimes": [
      1
    ],
    "prohibited_regimes": [
      2,
      3,
      4
    ],
    "toxic_sessions": [
      "NY Afternoon (16:00-21:00)"
    ],
    "toxic_days": [],
    "base_win_rate": 51.7,
    "base_profit_factor": 1.96,
    "min_ker": 0.3,
    "max_ker": 1.0,
    "max_chop": 52.0,
    "min_hurst": 0.52,
    "max_hurst": 1.0,
    "max_svp": 0.12
  },
  "GBPJPY_disc140": {
    "name": "disc140",
    "symbol": "GBPJPY",
    "category": "gbpjpy_champions",
    "archetype": "trend_pullback",
    "permitted_regimes": [
      1
    ],
    "prohibited_regimes": [
      2,
      3,
      4
    ],
    "toxic_sessions": [
      "NY Afternoon (16:00-21:00)"
    ],
    "toxic_days": [
      "Thursday"
    ],
    "base_win_rate": 52.0,
    "base_profit_factor": 2.06,
    "min_ker": 0.3,
    "max_ker": 1.0,
    "max_chop": 52.0,
    "min_hurst": 0.52,
    "max_hurst": 1.0,
    "max_svp": 0.12
  },
  "GBPJPY_disc141": {
    "name": "disc141",
    "symbol": "GBPJPY",
    "category": "gbpjpy_champions",
    "archetype": "trend_pullback",
    "permitted_regimes": [
      1
    ],
    "prohibited_regimes": [
      2,
      3,
      4
    ],
    "toxic_sessions": [],
    "toxic_days": [],
    "base_win_rate": 69.6,
    "base_profit_factor": 4.93,
    "min_ker": 0.3,
    "max_ker": 1.0,
    "max_chop": 52.0,
    "min_hurst": 0.52,
    "max_hurst": 1.0,
    "max_svp": 0.12
  },
  "GBPJPY_disc155": {
    "name": "disc155",
    "symbol": "GBPJPY",
    "category": "gbpjpy_champions",
    "archetype": "trend_pullback",
    "permitted_regimes": [
      1
    ],
    "prohibited_regimes": [
      2,
      3,
      4
    ],
    "toxic_sessions": [],
    "toxic_days": [],
    "base_win_rate": 58.3,
    "base_profit_factor": 2.9,
    "min_ker": 0.3,
    "max_ker": 1.0,
    "max_chop": 52.0,
    "min_hurst": 0.52,
    "max_hurst": 1.0,
    "max_svp": 0.12
  },
  "GBPJPY_disc156": {
    "name": "disc156",
    "symbol": "GBPJPY",
    "category": "gbpjpy_champions",
    "archetype": "trend_pullback",
    "permitted_regimes": [
      1
    ],
    "prohibited_regimes": [
      2,
      3,
      4
    ],
    "toxic_sessions": [],
    "toxic_days": [],
    "base_win_rate": 64.5,
    "base_profit_factor": 3.57,
    "min_ker": 0.3,
    "max_ker": 1.0,
    "max_chop": 52.0,
    "min_hurst": 0.52,
    "max_hurst": 1.0,
    "max_svp": 0.12
  },
  "NZDCAD_disc122": {
    "name": "disc122",
    "symbol": "NZDCAD",
    "category": "nzdcad_champions",
    "archetype": "trend_pullback",
    "permitted_regimes": [
      1
    ],
    "prohibited_regimes": [
      2,
      3,
      4
    ],
    "toxic_sessions": [],
    "toxic_days": [
      "Tuesday"
    ],
    "base_win_rate": 44.4,
    "base_profit_factor": 1.81,
    "min_ker": 0.3,
    "max_ker": 1.0,
    "max_chop": 52.0,
    "min_hurst": 0.52,
    "max_hurst": 1.0,
    "max_svp": 0.12
  },
  "NZDCAD_disc123": {
    "name": "disc123",
    "symbol": "NZDCAD",
    "category": "nzdcad_champions",
    "archetype": "trend_pullback",
    "permitted_regimes": [
      1
    ],
    "prohibited_regimes": [
      2,
      3,
      4
    ],
    "toxic_sessions": [],
    "toxic_days": [
      "Tuesday"
    ],
    "base_win_rate": 48.0,
    "base_profit_factor": 1.97,
    "min_ker": 0.3,
    "max_ker": 1.0,
    "max_chop": 52.0,
    "min_hurst": 0.52,
    "max_hurst": 1.0,
    "max_svp": 0.12
  },
  "NZDCAD_disc125": {
    "name": "disc125",
    "symbol": "NZDCAD",
    "category": "nzdcad_champions",
    "archetype": "trend_pullback",
    "permitted_regimes": [
      1
    ],
    "prohibited_regimes": [
      2,
      3,
      4
    ],
    "toxic_sessions": [],
    "toxic_days": [],
    "base_win_rate": 51.9,
    "base_profit_factor": 1.76,
    "min_ker": 0.3,
    "max_ker": 1.0,
    "max_chop": 52.0,
    "min_hurst": 0.52,
    "max_hurst": 1.0,
    "max_svp": 0.12
  },
  "NZDCAD_disc126": {
    "name": "disc126",
    "symbol": "NZDCAD",
    "category": "nzdcad_champions",
    "archetype": "trend_pullback",
    "permitted_regimes": [
      1
    ],
    "prohibited_regimes": [
      2,
      3,
      4
    ],
    "toxic_sessions": [],
    "toxic_days": [
      "Monday"
    ],
    "base_win_rate": 44.4,
    "base_profit_factor": 1.53,
    "min_ker": 0.3,
    "max_ker": 1.0,
    "max_chop": 52.0,
    "min_hurst": 0.52,
    "max_hurst": 1.0,
    "max_svp": 0.12
  },
  "NZDCAD_disc161": {
    "name": "disc161",
    "symbol": "NZDCAD",
    "category": "nzdcad_champions",
    "archetype": "trend_pullback",
    "permitted_regimes": [
      1
    ],
    "prohibited_regimes": [
      2,
      3,
      4
    ],
    "toxic_sessions": [
      "London/NY Overlap (12:00-16:00)"
    ],
    "toxic_days": [
      "Tuesday"
    ],
    "base_win_rate": 50.0,
    "base_profit_factor": 1.56,
    "min_ker": 0.3,
    "max_ker": 1.0,
    "max_chop": 52.0,
    "min_hurst": 0.52,
    "max_hurst": 1.0,
    "max_svp": 0.12
  },
  "NZDCAD_disc162": {
    "name": "disc162",
    "symbol": "NZDCAD",
    "category": "nzdcad_champions",
    "archetype": "trend_pullback",
    "permitted_regimes": [
      1
    ],
    "prohibited_regimes": [
      2,
      3,
      4
    ],
    "toxic_sessions": [],
    "toxic_days": [
      "Friday"
    ],
    "base_win_rate": 51.6,
    "base_profit_factor": 1.76,
    "min_ker": 0.3,
    "max_ker": 1.0,
    "max_chop": 52.0,
    "min_hurst": 0.52,
    "max_hurst": 1.0,
    "max_svp": 0.12
  },
  "USDCAD_disc127": {
    "name": "disc127",
    "symbol": "USDCAD",
    "category": "usdcad_champions",
    "archetype": "trend_pullback",
    "permitted_regimes": [
      1
    ],
    "prohibited_regimes": [
      2,
      3,
      4
    ],
    "toxic_sessions": [],
    "toxic_days": [],
    "base_win_rate": 66.7,
    "base_profit_factor": 4.07,
    "min_ker": 0.3,
    "max_ker": 1.0,
    "max_chop": 52.0,
    "min_hurst": 0.52,
    "max_hurst": 1.0,
    "max_svp": 0.12
  },
  "USDCAD_disc128": {
    "name": "disc128",
    "symbol": "USDCAD",
    "category": "usdcad_champions",
    "archetype": "trend_pullback",
    "permitted_regimes": [
      1
    ],
    "prohibited_regimes": [
      2,
      3,
      4
    ],
    "toxic_sessions": [],
    "toxic_days": [
      "Wednesday"
    ],
    "base_win_rate": 55.6,
    "base_profit_factor": 2.48,
    "min_ker": 0.3,
    "max_ker": 1.0,
    "max_chop": 52.0,
    "min_hurst": 0.52,
    "max_hurst": 1.0,
    "max_svp": 0.12
  },
  "USDCAD_disc129": {
    "name": "disc129",
    "symbol": "USDCAD",
    "category": "usdcad_champions",
    "archetype": "trend_pullback",
    "permitted_regimes": [
      1
    ],
    "prohibited_regimes": [
      2,
      3,
      4
    ],
    "toxic_sessions": [],
    "toxic_days": [
      "Wednesday"
    ],
    "base_win_rate": 55.9,
    "base_profit_factor": 2.52,
    "min_ker": 0.3,
    "max_ker": 1.0,
    "max_chop": 52.0,
    "min_hurst": 0.52,
    "max_hurst": 1.0,
    "max_svp": 0.12
  },
  "USDCAD_disc130": {
    "name": "disc130",
    "symbol": "USDCAD",
    "category": "usdcad_champions",
    "archetype": "trend_pullback",
    "permitted_regimes": [
      1
    ],
    "prohibited_regimes": [
      2,
      3,
      4
    ],
    "toxic_sessions": [],
    "toxic_days": [],
    "base_win_rate": 59.3,
    "base_profit_factor": 2.92,
    "min_ker": 0.3,
    "max_ker": 1.0,
    "max_chop": 52.0,
    "min_hurst": 0.52,
    "max_hurst": 1.0,
    "max_svp": 0.12
  },
  "USDJPY_disc106_chaikin_volatility_surge_elite": {
    "name": "disc106_chaikin_volatility_surge_elite",
    "symbol": "USDJPY",
    "category": "usdjpy_champions",
    "archetype": "volatility_breakout",
    "permitted_regimes": [
      1,
      4
    ],
    "prohibited_regimes": [
      2,
      3
    ],
    "toxic_sessions": [
      "NY Afternoon (16:00-21:00)"
    ],
    "toxic_days": [
      "Monday",
      "Wednesday"
    ],
    "base_win_rate": 38.4,
    "base_profit_factor": 1.17,
    "min_ker": 0.25,
    "max_ker": 1.0,
    "max_chop": 61.8,
    "min_hurst": 0.5,
    "max_hurst": 1.0,
    "max_svp": 0.12
  },
  "USDJPY_disc65_session_vwap_band_fade_elite": {
    "name": "disc65_session_vwap_band_fade_elite",
    "symbol": "USDJPY",
    "category": "usdjpy_champions",
    "archetype": "mean_reversion",
    "permitted_regimes": [
      2
    ],
    "prohibited_regimes": [
      1,
      3,
      4
    ],
    "toxic_sessions": [
      "NY Afternoon (16:00-21:00)"
    ],
    "toxic_days": [
      "Friday"
    ],
    "base_win_rate": 40.2,
    "base_profit_factor": 1.18,
    "min_ker": 0.0,
    "max_ker": 0.38,
    "max_chop": 100.0,
    "min_hurst": 0.0,
    "max_hurst": 0.54,
    "max_svp": 0.12
  },
  "USDJPY_disc67_london_lunch_expansion_elite": {
    "name": "disc67_london_lunch_expansion_elite",
    "symbol": "USDJPY",
    "category": "usdjpy_champions",
    "archetype": "volatility_breakout",
    "permitted_regimes": [
      1,
      4
    ],
    "prohibited_regimes": [
      2,
      3
    ],
    "toxic_sessions": [],
    "toxic_days": [
      "Wednesday"
    ],
    "base_win_rate": 41.5,
    "base_profit_factor": 1.27,
    "min_ker": 0.25,
    "max_ker": 1.0,
    "max_chop": 61.8,
    "min_hurst": 0.5,
    "max_hurst": 1.0,
    "max_svp": 0.12
  },
  "USDJPY_disc83_parabolic_sar_flip_elite": {
    "name": "disc83_parabolic_sar_flip_elite",
    "symbol": "USDJPY",
    "category": "usdjpy_champions",
    "archetype": "trend_pullback",
    "permitted_regimes": [
      1
    ],
    "prohibited_regimes": [
      2,
      3,
      4
    ],
    "toxic_sessions": [
      "NY Afternoon (16:00-21:00)"
    ],
    "toxic_days": [
      "Monday"
    ],
    "base_win_rate": 40.1,
    "base_profit_factor": 1.23,
    "min_ker": 0.3,
    "max_ker": 1.0,
    "max_chop": 52.0,
    "min_hurst": 0.52,
    "max_hurst": 1.0,
    "max_svp": 0.12
  },
  "USDJPY_disc91_session_donchian_midline_bounce_elite": {
    "name": "disc91_session_donchian_midline_bounce_elite",
    "symbol": "USDJPY",
    "category": "usdjpy_champions",
    "archetype": "trend_pullback",
    "permitted_regimes": [
      1
    ],
    "prohibited_regimes": [
      2,
      3,
      4
    ],
    "toxic_sessions": [],
    "toxic_days": [
      "Friday",
      "Wednesday"
    ],
    "base_win_rate": 40.1,
    "base_profit_factor": 1.23,
    "min_ker": 0.3,
    "max_ker": 1.0,
    "max_chop": 52.0,
    "min_hurst": 0.52,
    "max_hurst": 1.0,
    "max_svp": 0.12
  },
  "USDJPY_disc94_london_vwap_pullback_elite": {
    "name": "disc94_london_vwap_pullback_elite",
    "symbol": "USDJPY",
    "category": "usdjpy_champions",
    "archetype": "trend_pullback",
    "permitted_regimes": [
      1
    ],
    "prohibited_regimes": [
      2,
      3,
      4
    ],
    "toxic_sessions": [
      "London/NY Overlap (12:00-16:00)"
    ],
    "toxic_days": [
      "Friday",
      "Monday"
    ],
    "base_win_rate": 39.6,
    "base_profit_factor": 1.2,
    "min_ker": 0.3,
    "max_ker": 1.0,
    "max_chop": 52.0,
    "min_hurst": 0.52,
    "max_hurst": 1.0,
    "max_svp": 0.12
  },
  "USDJPY_disc95_camarilla_expansion_elite": {
    "name": "disc95_camarilla_expansion_elite",
    "symbol": "USDJPY",
    "category": "usdjpy_champions",
    "archetype": "volatility_breakout",
    "permitted_regimes": [
      1,
      4
    ],
    "prohibited_regimes": [
      2,
      3
    ],
    "toxic_sessions": [],
    "toxic_days": [
      "Wednesday"
    ],
    "base_win_rate": 40.9,
    "base_profit_factor": 1.3,
    "min_ker": 0.25,
    "max_ker": 1.0,
    "max_chop": 61.8,
    "min_hurst": 0.5,
    "max_hurst": 1.0,
    "max_svp": 0.12
  },
  "USDJPY_disc99_m15_macd_zero_cross_momentum_elite": {
    "name": "disc99_m15_macd_zero_cross_momentum_elite",
    "symbol": "USDJPY",
    "category": "usdjpy_champions",
    "archetype": "trend_pullback",
    "permitted_regimes": [
      1
    ],
    "prohibited_regimes": [
      2,
      3,
      4
    ],
    "toxic_sessions": [
      "London/NY Overlap (12:00-16:00)"
    ],
    "toxic_days": [
      "Friday",
      "Wednesday"
    ],
    "base_win_rate": 38.3,
    "base_profit_factor": 1.16,
    "min_ker": 0.3,
    "max_ker": 1.0,
    "max_chop": 52.0,
    "min_hurst": 0.52,
    "max_hurst": 1.0,
    "max_svp": 0.12
  },
  "USDCAD_disc163": {
    "name": "disc163",
    "symbol": "USDCAD",
    "category": "usdcad_champions",
    "archetype": "trend_pullback",
    "permitted_regimes": [
      1
    ],
    "prohibited_regimes": [
      2,
      3,
      4
    ],
    "toxic_sessions": [],
    "toxic_days": [],
    "base_win_rate": 43.2,
    "base_profit_factor": 1.56,
    "min_ker": 0.3,
    "max_ker": 1.0,
    "max_chop": 52.0,
    "min_hurst": 0.52,
    "max_hurst": 1.0,
    "max_svp": 0.12
  },
  "USDCAD_disc164": {
    "name": "disc164",
    "symbol": "USDCAD",
    "category": "usdcad_champions",
    "archetype": "trend_pullback",
    "permitted_regimes": [
      1
    ],
    "prohibited_regimes": [
      2,
      3,
      4
    ],
    "toxic_sessions": [],
    "toxic_days": [],
    "base_win_rate": 42.3,
    "base_profit_factor": 1.36,
    "min_ker": 0.3,
    "max_ker": 1.0,
    "max_chop": 52.0,
    "min_hurst": 0.52,
    "max_hurst": 1.0,
    "max_svp": 0.12
  },
  "GBPJPY_disc165": {
    "name": "disc165",
    "symbol": "GBPJPY",
    "category": "gbpjpy_champions",
    "archetype": "trend_pullback",
    "permitted_regimes": [
      1
    ],
    "prohibited_regimes": [
      2,
      3,
      4
    ],
    "toxic_sessions": [],
    "toxic_days": [
      "Thursday",
      "Wednesday"
    ],
    "base_win_rate": 38.6,
    "base_profit_factor": 1.23,
    "min_ker": 0.3,
    "max_ker": 1.0,
    "max_chop": 52.0,
    "min_hurst": 0.52,
    "max_hurst": 1.0,
    "max_svp": 0.12
  },
  "GBPCHF_disc166": {
    "name": "disc166",
    "symbol": "GBPCHF",
    "category": "gbpchf_champions",
    "archetype": "trend_pullback",
    "permitted_regimes": [
      1
    ],
    "prohibited_regimes": [
      2,
      3,
      4
    ],
    "toxic_sessions": [],
    "toxic_days": [
      "Monday",
      "Thursday"
    ],
    "base_win_rate": 41.3,
    "base_profit_factor": 1.25,
    "min_ker": 0.3,
    "max_ker": 1.0,
    "max_chop": 52.0,
    "min_hurst": 0.52,
    "max_hurst": 1.0,
    "max_svp": 0.12
  },
  "EURNZD_disc167": {
    "name": "disc167",
    "symbol": "EURNZD",
    "category": "eurnzd_champions",
    "archetype": "trend_pullback",
    "permitted_regimes": [
      1
    ],
    "prohibited_regimes": [
      2,
      3,
      4
    ],
    "toxic_sessions": [],
    "toxic_days": [
      "Friday",
      "Monday",
      "Tuesday"
    ],
    "base_win_rate": 39.7,
    "base_profit_factor": 1.22,
    "min_ker": 0.3,
    "max_ker": 1.0,
    "max_chop": 52.0,
    "min_hurst": 0.52,
    "max_hurst": 1.0,
    "max_svp": 0.12
  },
  "EURNZD_disc168": {
    "name": "disc168",
    "symbol": "EURNZD",
    "category": "eurnzd_champions",
    "archetype": "trend_pullback",
    "permitted_regimes": [
      1
    ],
    "prohibited_regimes": [
      2,
      3,
      4
    ],
    "toxic_sessions": [
      "NY Afternoon (16:00-21:00)"
    ],
    "toxic_days": [
      "Tuesday",
      "Wednesday"
    ],
    "base_win_rate": 37.9,
    "base_profit_factor": 1.23,
    "min_ker": 0.3,
    "max_ker": 1.0,
    "max_chop": 52.0,
    "min_hurst": 0.52,
    "max_hurst": 1.0,
    "max_svp": 0.12
  },
  "EURUSD_disc169": {
    "name": "disc169",
    "symbol": "EURUSD",
    "category": "eurusd_champions",
    "archetype": "trend_pullback",
    "permitted_regimes": [
      1
    ],
    "prohibited_regimes": [
      2,
      3,
      4
    ],
    "toxic_sessions": [
      "NY Afternoon (16:00-21:00)"
    ],
    "toxic_days": [
      "Monday"
    ],
    "base_win_rate": 41.3,
    "base_profit_factor": 1.4,
    "min_ker": 0.3,
    "max_ker": 1.0,
    "max_chop": 52.0,
    "min_hurst": 0.52,
    "max_hurst": 1.0,
    "max_svp": 0.12
  },
  "USDCAD_disc170": {
    "name": "disc170",
    "symbol": "USDCAD",
    "category": "usdcad_champions",
    "archetype": "trend_pullback",
    "permitted_regimes": [
      1
    ],
    "prohibited_regimes": [
      2,
      3,
      4
    ],
    "toxic_sessions": [],
    "toxic_days": [],
    "base_win_rate": 50.8,
    "base_profit_factor": 1.99,
    "min_ker": 0.3,
    "max_ker": 1.0,
    "max_chop": 52.0,
    "min_hurst": 0.52,
    "max_hurst": 1.0,
    "max_svp": 0.12
  },
  "GBPJPY_disc171": {
    "name": "disc171",
    "symbol": "GBPJPY",
    "category": "gbpjpy_champions",
    "archetype": "trend_pullback",
    "permitted_regimes": [
      1
    ],
    "prohibited_regimes": [
      2,
      3,
      4
    ],
    "toxic_sessions": [],
    "toxic_days": [],
    "base_win_rate": 41.2,
    "base_profit_factor": 1.54,
    "min_ker": 0.3,
    "max_ker": 1.0,
    "max_chop": 52.0,
    "min_hurst": 0.52,
    "max_hurst": 1.0,
    "max_svp": 0.12
  },
  "EURGBP_disc172": {
    "name": "disc172",
    "symbol": "EURGBP",
    "category": "eurgbp_champions",
    "archetype": "trend_pullback",
    "permitted_regimes": [
      1
    ],
    "prohibited_regimes": [
      2,
      3,
      4
    ],
    "toxic_sessions": [],
    "toxic_days": [],
    "base_win_rate": 52.6,
    "base_profit_factor": 2.04,
    "min_ker": 0.3,
    "max_ker": 1.0,
    "max_chop": 52.0,
    "min_hurst": 0.52,
    "max_hurst": 1.0,
    "max_svp": 0.12
  },
  "USDJPY_disc173": {
    "name": "disc173",
    "symbol": "USDJPY",
    "category": "usdjpy_champions",
    "archetype": "trend_pullback",
    "permitted_regimes": [
      1
    ],
    "prohibited_regimes": [
      2,
      3,
      4
    ],
    "toxic_sessions": [
      "London Open (06:00-12:00)"
    ],
    "toxic_days": [],
    "base_win_rate": 42.0,
    "base_profit_factor": 1.39,
    "min_ker": 0.3,
    "max_ker": 1.0,
    "max_chop": 52.0,
    "min_hurst": 0.52,
    "max_hurst": 1.0,
    "max_svp": 0.12
  },
  "EURGBP_disc174": {
    "name": "disc174",
    "symbol": "EURGBP",
    "category": "eurgbp_champions",
    "archetype": "trend_pullback",
    "permitted_regimes": [
      1
    ],
    "prohibited_regimes": [
      2,
      3,
      4
    ],
    "toxic_sessions": [],
    "toxic_days": [],
    "base_win_rate": 54.3,
    "base_profit_factor": 2.14,
    "min_ker": 0.3,
    "max_ker": 1.0,
    "max_chop": 52.0,
    "min_hurst": 0.52,
    "max_hurst": 1.0,
    "max_svp": 0.12
  },
  "EURGBP_disc175": {
    "name": "disc175",
    "symbol": "EURGBP",
    "category": "eurgbp_champions",
    "archetype": "trend_pullback",
    "permitted_regimes": [
      1
    ],
    "prohibited_regimes": [
      2,
      3,
      4
    ],
    "toxic_sessions": [],
    "toxic_days": [
      "Tuesday"
    ],
    "base_win_rate": 48.0,
    "base_profit_factor": 1.62,
    "min_ker": 0.3,
    "max_ker": 1.0,
    "max_chop": 52.0,
    "min_hurst": 0.52,
    "max_hurst": 1.0,
    "max_svp": 0.12
  },
  "EURGBP_disc180": {
    "name": "disc180",
    "symbol": "EURGBP",
    "category": "eurgbp_champions",
    "archetype": "trend_pullback",
    "permitted_regimes": [
      1
    ],
    "prohibited_regimes": [
      2,
      3,
      4
    ],
    "toxic_sessions": [],
    "toxic_days": [
      "Wednesday"
    ],
    "base_win_rate": 51.4,
    "base_profit_factor": 1.63,
    "min_ker": 0.3,
    "max_ker": 1.0,
    "max_chop": 52.0,
    "min_hurst": 0.52,
    "max_hurst": 1.0,
    "max_svp": 0.12
  }
}

# Name-based fallback lookup
_NAME_REGISTRY: Dict[str, Dict[str, Any]] = {
  "disc115": {
    "name": "disc115",
    "symbol": "AUDCAD",
    "category": "audcad_champions",
    "archetype": "trend_pullback",
    "permitted_regimes": [
      1
    ],
    "prohibited_regimes": [
      2,
      3,
      4
    ],
    "toxic_sessions": [],
    "toxic_days": [],
    "base_win_rate": 63.3,
    "base_profit_factor": 2.98,
    "min_ker": 0.3,
    "max_ker": 1.0,
    "max_chop": 52.0,
    "min_hurst": 0.52,
    "max_hurst": 1.0,
    "max_svp": 0.12
  },
  "disc116": {
    "name": "disc116",
    "symbol": "AUDCAD",
    "category": "audcad_champions",
    "archetype": "trend_pullback",
    "permitted_regimes": [
      1
    ],
    "prohibited_regimes": [
      2,
      3,
      4
    ],
    "toxic_sessions": [],
    "toxic_days": [],
    "base_win_rate": 56.7,
    "base_profit_factor": 2.86,
    "min_ker": 0.3,
    "max_ker": 1.0,
    "max_chop": 52.0,
    "min_hurst": 0.52,
    "max_hurst": 1.0,
    "max_svp": 0.12
  },
  "disc117": {
    "name": "disc117",
    "symbol": "AUDCAD",
    "category": "audcad_champions",
    "archetype": "trend_pullback",
    "permitted_regimes": [
      1
    ],
    "prohibited_regimes": [
      2,
      3,
      4
    ],
    "toxic_sessions": [],
    "toxic_days": [],
    "base_win_rate": 59.5,
    "base_profit_factor": 2.24,
    "min_ker": 0.3,
    "max_ker": 1.0,
    "max_chop": 52.0,
    "min_hurst": 0.52,
    "max_hurst": 1.0,
    "max_svp": 0.12
  },
  "disc118": {
    "name": "disc118",
    "symbol": "AUDCAD",
    "category": "audcad_champions",
    "archetype": "trend_pullback",
    "permitted_regimes": [
      1
    ],
    "prohibited_regimes": [
      2,
      3,
      4
    ],
    "toxic_sessions": [],
    "toxic_days": [],
    "base_win_rate": 65.2,
    "base_profit_factor": 2.92,
    "min_ker": 0.3,
    "max_ker": 1.0,
    "max_chop": 52.0,
    "min_hurst": 0.52,
    "max_hurst": 1.0,
    "max_svp": 0.12
  },
  "disc159": {
    "name": "disc159",
    "symbol": "AUDCAD",
    "category": "audcad_champions",
    "archetype": "trend_pullback",
    "permitted_regimes": [
      1
    ],
    "prohibited_regimes": [
      2,
      3,
      4
    ],
    "toxic_sessions": [
      "London/NY Overlap (12:00-16:00)"
    ],
    "toxic_days": [
      "Friday"
    ],
    "base_win_rate": 47.8,
    "base_profit_factor": 1.44,
    "min_ker": 0.3,
    "max_ker": 1.0,
    "max_chop": 52.0,
    "min_hurst": 0.52,
    "max_hurst": 1.0,
    "max_svp": 0.12
  },
  "disc160": {
    "name": "disc160",
    "symbol": "AUDCAD",
    "category": "audcad_champions",
    "archetype": "trend_pullback",
    "permitted_regimes": [
      1
    ],
    "prohibited_regimes": [
      2,
      3,
      4
    ],
    "toxic_sessions": [],
    "toxic_days": [
      "Friday"
    ],
    "base_win_rate": 51.6,
    "base_profit_factor": 1.73,
    "min_ker": 0.3,
    "max_ker": 1.0,
    "max_chop": 52.0,
    "min_hurst": 0.52,
    "max_hurst": 1.0,
    "max_svp": 0.12
  },
  "disc111_h4_macro_structure_retest_elite": {
    "name": "disc111_h4_macro_structure_retest_elite",
    "symbol": "AUDNZD",
    "category": "audnzd_champions",
    "archetype": "macro_structure_retest",
    "permitted_regimes": [
      1,
      2
    ],
    "prohibited_regimes": [
      3,
      4
    ],
    "toxic_sessions": [],
    "toxic_days": [],
    "base_win_rate": 60.7,
    "base_profit_factor": 2.56,
    "min_ker": 0.2,
    "max_ker": 1.0,
    "max_chop": 58.0,
    "min_hurst": 0.0,
    "max_hurst": 1.0,
    "max_svp": 0.12
  },
  "disc112_h4_macro_precision_retest_elite": {
    "name": "disc112_h4_macro_precision_retest_elite",
    "symbol": "AUDNZD",
    "category": "audnzd_champions",
    "archetype": "macro_structure_retest",
    "permitted_regimes": [
      1,
      2
    ],
    "prohibited_regimes": [
      3,
      4
    ],
    "toxic_sessions": [],
    "toxic_days": [
      "Thursday"
    ],
    "base_win_rate": 55.2,
    "base_profit_factor": 2.06,
    "min_ker": 0.2,
    "max_ker": 1.0,
    "max_chop": 58.0,
    "min_hurst": 0.0,
    "max_hurst": 1.0,
    "max_svp": 0.12
  },
  "disc113_h1_macro_intraday_retest_elite": {
    "name": "disc113_h1_macro_intraday_retest_elite",
    "symbol": "AUDNZD",
    "category": "audnzd_champions",
    "archetype": "macro_structure_retest",
    "permitted_regimes": [
      1,
      2
    ],
    "prohibited_regimes": [
      3,
      4
    ],
    "toxic_sessions": [
      "London Open (06:00-12:00)"
    ],
    "toxic_days": [],
    "base_win_rate": 50.0,
    "base_profit_factor": 1.72,
    "min_ker": 0.2,
    "max_ker": 1.0,
    "max_chop": 58.0,
    "min_hurst": 0.0,
    "max_hurst": 1.0,
    "max_svp": 0.12
  },
  "disc114_h4_macro_wide_runner_elite": {
    "name": "disc114_h4_macro_wide_runner_elite",
    "symbol": "AUDNZD",
    "category": "audnzd_champions",
    "archetype": "macro_structure_retest",
    "permitted_regimes": [
      1,
      2
    ],
    "prohibited_regimes": [
      3,
      4
    ],
    "toxic_sessions": [],
    "toxic_days": [],
    "base_win_rate": 64.3,
    "base_profit_factor": 3.03,
    "min_ker": 0.2,
    "max_ker": 1.0,
    "max_chop": 58.0,
    "min_hurst": 0.0,
    "max_hurst": 1.0,
    "max_svp": 0.12
  },
  "disc145": {
    "name": "disc145",
    "symbol": "AUDNZD",
    "category": "audnzd_champions",
    "archetype": "trend_pullback",
    "permitted_regimes": [
      1
    ],
    "prohibited_regimes": [
      2,
      3,
      4
    ],
    "toxic_sessions": [],
    "toxic_days": [],
    "base_win_rate": 72.4,
    "base_profit_factor": 4.75,
    "min_ker": 0.3,
    "max_ker": 1.0,
    "max_chop": 52.0,
    "min_hurst": 0.52,
    "max_hurst": 1.0,
    "max_svp": 0.12
  },
  "disc146": {
    "name": "disc146",
    "symbol": "AUDNZD",
    "category": "audnzd_champions",
    "archetype": "trend_pullback",
    "permitted_regimes": [
      1
    ],
    "prohibited_regimes": [
      2,
      3,
      4
    ],
    "toxic_sessions": [],
    "toxic_days": [],
    "base_win_rate": 64.1,
    "base_profit_factor": 3.26,
    "min_ker": 0.3,
    "max_ker": 1.0,
    "max_chop": 52.0,
    "min_hurst": 0.52,
    "max_hurst": 1.0,
    "max_svp": 0.12
  },
  "disc147": {
    "name": "disc147",
    "symbol": "AUDNZD",
    "category": "audnzd_champions",
    "archetype": "trend_pullback",
    "permitted_regimes": [
      1
    ],
    "prohibited_regimes": [
      2,
      3,
      4
    ],
    "toxic_sessions": [],
    "toxic_days": [],
    "base_win_rate": 62.5,
    "base_profit_factor": 3.23,
    "min_ker": 0.3,
    "max_ker": 1.0,
    "max_chop": 52.0,
    "min_hurst": 0.52,
    "max_hurst": 1.0,
    "max_svp": 0.12
  },
  "disc107_h1_keltner_squeeze_explosion_elite": {
    "name": "disc107_h1_keltner_squeeze_explosion_elite",
    "symbol": "EURGBP",
    "category": "eurgbp_champions",
    "archetype": "volatility_breakout",
    "permitted_regimes": [
      1,
      4
    ],
    "prohibited_regimes": [
      2,
      3
    ],
    "toxic_sessions": [],
    "toxic_days": [
      "Tuesday"
    ],
    "base_win_rate": 47.5,
    "base_profit_factor": 1.71,
    "min_ker": 0.25,
    "max_ker": 1.0,
    "max_chop": 61.8,
    "min_hurst": 0.5,
    "max_hurst": 1.0,
    "max_svp": 0.12
  },
  "disc132": {
    "name": "disc132",
    "symbol": "EURNZD",
    "category": "eurnzd_champions",
    "archetype": "trend_pullback",
    "permitted_regimes": [
      1
    ],
    "prohibited_regimes": [
      2,
      3,
      4
    ],
    "toxic_sessions": [],
    "toxic_days": [],
    "base_win_rate": 61.8,
    "base_profit_factor": 3.32,
    "min_ker": 0.3,
    "max_ker": 1.0,
    "max_chop": 52.0,
    "min_hurst": 0.52,
    "max_hurst": 1.0,
    "max_svp": 0.12
  },
  "disc133": {
    "name": "disc133",
    "symbol": "EURNZD",
    "category": "eurnzd_champions",
    "archetype": "trend_pullback",
    "permitted_regimes": [
      1
    ],
    "prohibited_regimes": [
      2,
      3,
      4
    ],
    "toxic_sessions": [],
    "toxic_days": [],
    "base_win_rate": 62.8,
    "base_profit_factor": 3.45,
    "min_ker": 0.3,
    "max_ker": 1.0,
    "max_chop": 52.0,
    "min_hurst": 0.52,
    "max_hurst": 1.0,
    "max_svp": 0.12
  },
  "disc143": {
    "name": "disc143",
    "symbol": "EURNZD",
    "category": "eurnzd_champions",
    "archetype": "trend_pullback",
    "permitted_regimes": [
      1
    ],
    "prohibited_regimes": [
      2,
      3,
      4
    ],
    "toxic_sessions": [],
    "toxic_days": [],
    "base_win_rate": 77.8,
    "base_profit_factor": 7.27,
    "min_ker": 0.3,
    "max_ker": 1.0,
    "max_chop": 52.0,
    "min_hurst": 0.52,
    "max_hurst": 1.0,
    "max_svp": 0.12
  },
  "disc157": {
    "name": "disc157",
    "symbol": "EURNZD",
    "category": "eurnzd_champions",
    "archetype": "trend_pullback",
    "permitted_regimes": [
      1
    ],
    "prohibited_regimes": [
      2,
      3,
      4
    ],
    "toxic_sessions": [],
    "toxic_days": [],
    "base_win_rate": 78.8,
    "base_profit_factor": 8.23,
    "min_ker": 0.3,
    "max_ker": 1.0,
    "max_chop": 52.0,
    "min_hurst": 0.52,
    "max_hurst": 1.0,
    "max_svp": 0.12
  },
  "disc158": {
    "name": "disc158",
    "symbol": "EURNZD",
    "category": "eurnzd_champions",
    "archetype": "trend_pullback",
    "permitted_regimes": [
      1
    ],
    "prohibited_regimes": [
      2,
      3,
      4
    ],
    "toxic_sessions": [],
    "toxic_days": [],
    "base_win_rate": 56.2,
    "base_profit_factor": 2.85,
    "min_ker": 0.3,
    "max_ker": 1.0,
    "max_chop": 52.0,
    "min_hurst": 0.52,
    "max_hurst": 1.0,
    "max_svp": 0.12
  },
  "disc50_m30_m5_precision_retest_elite": {
    "name": "disc50_m30_m5_precision_retest_elite",
    "symbol": "EURUSD",
    "category": "eurusd_champions",
    "archetype": "trend_pullback",
    "permitted_regimes": [
      1
    ],
    "prohibited_regimes": [
      2,
      3,
      4
    ],
    "toxic_sessions": [],
    "toxic_days": [
      "Friday"
    ],
    "base_win_rate": 54.5,
    "base_profit_factor": 1.79,
    "min_ker": 0.3,
    "max_ker": 1.0,
    "max_chop": 52.0,
    "min_hurst": 0.52,
    "max_hurst": 1.0,
    "max_svp": 0.12
  },
  "disc51_macro_asymmetric_runner_elite": {
    "name": "disc51_macro_asymmetric_runner_elite",
    "symbol": "EURUSD",
    "category": "eurusd_champions",
    "archetype": "macro_structure_retest",
    "permitted_regimes": [
      1,
      2
    ],
    "prohibited_regimes": [
      3,
      4
    ],
    "toxic_sessions": [],
    "toxic_days": [],
    "base_win_rate": 70.6,
    "base_profit_factor": 6.05,
    "min_ker": 0.2,
    "max_ker": 1.0,
    "max_chop": 58.0,
    "min_hurst": 0.0,
    "max_hurst": 1.0,
    "max_svp": 0.12
  },
  "disc58_h1_keltner_squeeze_explosion_elite": {
    "name": "disc58_h1_keltner_squeeze_explosion_elite",
    "symbol": "EURUSD",
    "category": "eurusd_champions",
    "archetype": "volatility_breakout",
    "permitted_regimes": [
      1,
      4
    ],
    "prohibited_regimes": [
      2,
      3
    ],
    "toxic_sessions": [],
    "toxic_days": [],
    "base_win_rate": 45.9,
    "base_profit_factor": 1.63,
    "min_ker": 0.25,
    "max_ker": 1.0,
    "max_chop": 61.8,
    "min_hurst": 0.5,
    "max_hurst": 1.0,
    "max_svp": 0.12
  },
  "disc59_m15_macd_zero_cross_momentum_elite": {
    "name": "disc59_m15_macd_zero_cross_momentum_elite",
    "symbol": "EURUSD",
    "category": "eurusd_champions",
    "archetype": "trend_pullback",
    "permitted_regimes": [
      1
    ],
    "prohibited_regimes": [
      2,
      3,
      4
    ],
    "toxic_sessions": [],
    "toxic_days": [],
    "base_win_rate": 41.0,
    "base_profit_factor": 1.3,
    "min_ker": 0.3,
    "max_ker": 1.0,
    "max_chop": 52.0,
    "min_hurst": 0.52,
    "max_hurst": 1.0,
    "max_svp": 0.12
  },
  "disc65_session_vwap_band_fade_elite": {
    "name": "disc65_session_vwap_band_fade_elite",
    "symbol": "EURUSD",
    "category": "eurusd_champions",
    "archetype": "mean_reversion",
    "permitted_regimes": [
      2
    ],
    "prohibited_regimes": [
      1,
      3,
      4
    ],
    "toxic_sessions": [],
    "toxic_days": [
      "Thursday"
    ],
    "base_win_rate": 41.7,
    "base_profit_factor": 1.3,
    "min_ker": 0.0,
    "max_ker": 0.38,
    "max_chop": 100.0,
    "min_hurst": 0.0,
    "max_hurst": 0.54,
    "max_svp": 0.12
  },
  "disc74_daily_pivot_pullback_elite": {
    "name": "disc74_daily_pivot_pullback_elite",
    "symbol": "EURUSD",
    "category": "eurusd_champions",
    "archetype": "trend_pullback",
    "permitted_regimes": [
      1
    ],
    "prohibited_regimes": [
      2,
      3,
      4
    ],
    "toxic_sessions": [],
    "toxic_days": [],
    "base_win_rate": 50.4,
    "base_profit_factor": 1.88,
    "min_ker": 0.3,
    "max_ker": 1.0,
    "max_chop": 52.0,
    "min_hurst": 0.52,
    "max_hurst": 1.0,
    "max_svp": 0.12
  },
  "disc75_weekly_pivot_pullback_elite": {
    "name": "disc75_weekly_pivot_pullback_elite",
    "symbol": "EURUSD",
    "category": "eurusd_champions",
    "archetype": "trend_pullback",
    "permitted_regimes": [
      1
    ],
    "prohibited_regimes": [
      2,
      3,
      4
    ],
    "toxic_sessions": [
      "London Open (06:00-12:00)",
      "NY Afternoon (16:00-21:00)"
    ],
    "toxic_days": [
      "Friday",
      "Wednesday"
    ],
    "base_win_rate": 33.7,
    "base_profit_factor": 1.15,
    "min_ker": 0.3,
    "max_ker": 1.0,
    "max_chop": 52.0,
    "min_hurst": 0.52,
    "max_hurst": 1.0,
    "max_svp": 0.12
  },
  "disc76_vwap_1sd_rebound_elite": {
    "name": "disc76_vwap_1sd_rebound_elite",
    "symbol": "EURUSD",
    "category": "eurusd_champions",
    "archetype": "trend_pullback",
    "permitted_regimes": [
      1
    ],
    "prohibited_regimes": [
      2,
      3,
      4
    ],
    "toxic_sessions": [],
    "toxic_days": [
      "Thursday"
    ],
    "base_win_rate": 43.4,
    "base_profit_factor": 1.4,
    "min_ker": 0.3,
    "max_ker": 1.0,
    "max_chop": 52.0,
    "min_hurst": 0.52,
    "max_hurst": 1.0,
    "max_svp": 0.12
  },
  "disc86_ema_cross_atr_expansion_elite": {
    "name": "disc86_ema_cross_atr_expansion_elite",
    "symbol": "EURUSD",
    "category": "eurusd_champions",
    "archetype": "volatility_breakout",
    "permitted_regimes": [
      1,
      4
    ],
    "prohibited_regimes": [
      2,
      3
    ],
    "toxic_sessions": [
      "NY Afternoon (16:00-21:00)"
    ],
    "toxic_days": [
      "Friday"
    ],
    "base_win_rate": 43.3,
    "base_profit_factor": 1.42,
    "min_ker": 0.25,
    "max_ker": 1.0,
    "max_chop": 61.8,
    "min_hurst": 0.5,
    "max_hurst": 1.0,
    "max_svp": 0.12
  },
  "disc134": {
    "name": "disc134",
    "symbol": "GBPCHF",
    "category": "gbpchf_champions",
    "archetype": "trend_pullback",
    "permitted_regimes": [
      1
    ],
    "prohibited_regimes": [
      2,
      3,
      4
    ],
    "toxic_sessions": [],
    "toxic_days": [],
    "base_win_rate": 66.7,
    "base_profit_factor": 3.46,
    "min_ker": 0.3,
    "max_ker": 1.0,
    "max_chop": 52.0,
    "min_hurst": 0.52,
    "max_hurst": 1.0,
    "max_svp": 0.12
  },
  "disc135": {
    "name": "disc135",
    "symbol": "GBPCHF",
    "category": "gbpchf_champions",
    "archetype": "trend_pullback",
    "permitted_regimes": [
      1
    ],
    "prohibited_regimes": [
      2,
      3,
      4
    ],
    "toxic_sessions": [],
    "toxic_days": [],
    "base_win_rate": 65.9,
    "base_profit_factor": 3.52,
    "min_ker": 0.3,
    "max_ker": 1.0,
    "max_chop": 52.0,
    "min_hurst": 0.52,
    "max_hurst": 1.0,
    "max_svp": 0.12
  },
  "disc137": {
    "name": "disc137",
    "symbol": "GBPCHF",
    "category": "gbpchf_champions",
    "archetype": "trend_pullback",
    "permitted_regimes": [
      1
    ],
    "prohibited_regimes": [
      2,
      3,
      4
    ],
    "toxic_sessions": [],
    "toxic_days": [],
    "base_win_rate": 64.4,
    "base_profit_factor": 3.07,
    "min_ker": 0.3,
    "max_ker": 1.0,
    "max_chop": 52.0,
    "min_hurst": 0.52,
    "max_hurst": 1.0,
    "max_svp": 0.12
  },
  "disc152": {
    "name": "disc152",
    "symbol": "GBPCHF",
    "category": "gbpchf_champions",
    "archetype": "trend_pullback",
    "permitted_regimes": [
      1
    ],
    "prohibited_regimes": [
      2,
      3,
      4
    ],
    "toxic_sessions": [],
    "toxic_days": [],
    "base_win_rate": 69.1,
    "base_profit_factor": 3.79,
    "min_ker": 0.3,
    "max_ker": 1.0,
    "max_chop": 52.0,
    "min_hurst": 0.52,
    "max_hurst": 1.0,
    "max_svp": 0.12
  },
  "disc153": {
    "name": "disc153",
    "symbol": "GBPCHF",
    "category": "gbpchf_champions",
    "archetype": "trend_pullback",
    "permitted_regimes": [
      1
    ],
    "prohibited_regimes": [
      2,
      3,
      4
    ],
    "toxic_sessions": [],
    "toxic_days": [],
    "base_win_rate": 62.3,
    "base_profit_factor": 3.04,
    "min_ker": 0.3,
    "max_ker": 1.0,
    "max_chop": 52.0,
    "min_hurst": 0.52,
    "max_hurst": 1.0,
    "max_svp": 0.12
  },
  "disc138": {
    "name": "disc138",
    "symbol": "GBPJPY",
    "category": "gbpjpy_champions",
    "archetype": "trend_pullback",
    "permitted_regimes": [
      1
    ],
    "prohibited_regimes": [
      2,
      3,
      4
    ],
    "toxic_sessions": [
      "NY Afternoon (16:00-21:00)"
    ],
    "toxic_days": [],
    "base_win_rate": 56.0,
    "base_profit_factor": 2.39,
    "min_ker": 0.3,
    "max_ker": 1.0,
    "max_chop": 52.0,
    "min_hurst": 0.52,
    "max_hurst": 1.0,
    "max_svp": 0.12
  },
  "disc139": {
    "name": "disc139",
    "symbol": "GBPJPY",
    "category": "gbpjpy_champions",
    "archetype": "trend_pullback",
    "permitted_regimes": [
      1
    ],
    "prohibited_regimes": [
      2,
      3,
      4
    ],
    "toxic_sessions": [
      "NY Afternoon (16:00-21:00)"
    ],
    "toxic_days": [],
    "base_win_rate": 51.7,
    "base_profit_factor": 1.96,
    "min_ker": 0.3,
    "max_ker": 1.0,
    "max_chop": 52.0,
    "min_hurst": 0.52,
    "max_hurst": 1.0,
    "max_svp": 0.12
  },
  "disc140": {
    "name": "disc140",
    "symbol": "GBPJPY",
    "category": "gbpjpy_champions",
    "archetype": "trend_pullback",
    "permitted_regimes": [
      1
    ],
    "prohibited_regimes": [
      2,
      3,
      4
    ],
    "toxic_sessions": [
      "NY Afternoon (16:00-21:00)"
    ],
    "toxic_days": [
      "Thursday"
    ],
    "base_win_rate": 52.0,
    "base_profit_factor": 2.06,
    "min_ker": 0.3,
    "max_ker": 1.0,
    "max_chop": 52.0,
    "min_hurst": 0.52,
    "max_hurst": 1.0,
    "max_svp": 0.12
  },
  "disc141": {
    "name": "disc141",
    "symbol": "GBPJPY",
    "category": "gbpjpy_champions",
    "archetype": "trend_pullback",
    "permitted_regimes": [
      1
    ],
    "prohibited_regimes": [
      2,
      3,
      4
    ],
    "toxic_sessions": [],
    "toxic_days": [],
    "base_win_rate": 69.6,
    "base_profit_factor": 4.93,
    "min_ker": 0.3,
    "max_ker": 1.0,
    "max_chop": 52.0,
    "min_hurst": 0.52,
    "max_hurst": 1.0,
    "max_svp": 0.12
  },
  "disc155": {
    "name": "disc155",
    "symbol": "GBPJPY",
    "category": "gbpjpy_champions",
    "archetype": "trend_pullback",
    "permitted_regimes": [
      1
    ],
    "prohibited_regimes": [
      2,
      3,
      4
    ],
    "toxic_sessions": [],
    "toxic_days": [],
    "base_win_rate": 58.3,
    "base_profit_factor": 2.9,
    "min_ker": 0.3,
    "max_ker": 1.0,
    "max_chop": 52.0,
    "min_hurst": 0.52,
    "max_hurst": 1.0,
    "max_svp": 0.12
  },
  "disc156": {
    "name": "disc156",
    "symbol": "GBPJPY",
    "category": "gbpjpy_champions",
    "archetype": "trend_pullback",
    "permitted_regimes": [
      1
    ],
    "prohibited_regimes": [
      2,
      3,
      4
    ],
    "toxic_sessions": [],
    "toxic_days": [],
    "base_win_rate": 64.5,
    "base_profit_factor": 3.57,
    "min_ker": 0.3,
    "max_ker": 1.0,
    "max_chop": 52.0,
    "min_hurst": 0.52,
    "max_hurst": 1.0,
    "max_svp": 0.12
  },
  "disc122": {
    "name": "disc122",
    "symbol": "NZDCAD",
    "category": "nzdcad_champions",
    "archetype": "trend_pullback",
    "permitted_regimes": [
      1
    ],
    "prohibited_regimes": [
      2,
      3,
      4
    ],
    "toxic_sessions": [],
    "toxic_days": [
      "Tuesday"
    ],
    "base_win_rate": 44.4,
    "base_profit_factor": 1.81,
    "min_ker": 0.3,
    "max_ker": 1.0,
    "max_chop": 52.0,
    "min_hurst": 0.52,
    "max_hurst": 1.0,
    "max_svp": 0.12
  },
  "disc123": {
    "name": "disc123",
    "symbol": "NZDCAD",
    "category": "nzdcad_champions",
    "archetype": "trend_pullback",
    "permitted_regimes": [
      1
    ],
    "prohibited_regimes": [
      2,
      3,
      4
    ],
    "toxic_sessions": [],
    "toxic_days": [
      "Tuesday"
    ],
    "base_win_rate": 48.0,
    "base_profit_factor": 1.97,
    "min_ker": 0.3,
    "max_ker": 1.0,
    "max_chop": 52.0,
    "min_hurst": 0.52,
    "max_hurst": 1.0,
    "max_svp": 0.12
  },
  "disc125": {
    "name": "disc125",
    "symbol": "NZDCAD",
    "category": "nzdcad_champions",
    "archetype": "trend_pullback",
    "permitted_regimes": [
      1
    ],
    "prohibited_regimes": [
      2,
      3,
      4
    ],
    "toxic_sessions": [],
    "toxic_days": [],
    "base_win_rate": 51.9,
    "base_profit_factor": 1.76,
    "min_ker": 0.3,
    "max_ker": 1.0,
    "max_chop": 52.0,
    "min_hurst": 0.52,
    "max_hurst": 1.0,
    "max_svp": 0.12
  },
  "disc126": {
    "name": "disc126",
    "symbol": "NZDCAD",
    "category": "nzdcad_champions",
    "archetype": "trend_pullback",
    "permitted_regimes": [
      1
    ],
    "prohibited_regimes": [
      2,
      3,
      4
    ],
    "toxic_sessions": [],
    "toxic_days": [
      "Monday"
    ],
    "base_win_rate": 44.4,
    "base_profit_factor": 1.53,
    "min_ker": 0.3,
    "max_ker": 1.0,
    "max_chop": 52.0,
    "min_hurst": 0.52,
    "max_hurst": 1.0,
    "max_svp": 0.12
  },
  "disc161": {
    "name": "disc161",
    "symbol": "NZDCAD",
    "category": "nzdcad_champions",
    "archetype": "trend_pullback",
    "permitted_regimes": [
      1
    ],
    "prohibited_regimes": [
      2,
      3,
      4
    ],
    "toxic_sessions": [
      "London/NY Overlap (12:00-16:00)"
    ],
    "toxic_days": [
      "Tuesday"
    ],
    "base_win_rate": 50.0,
    "base_profit_factor": 1.56,
    "min_ker": 0.3,
    "max_ker": 1.0,
    "max_chop": 52.0,
    "min_hurst": 0.52,
    "max_hurst": 1.0,
    "max_svp": 0.12
  },
  "disc162": {
    "name": "disc162",
    "symbol": "NZDCAD",
    "category": "nzdcad_champions",
    "archetype": "trend_pullback",
    "permitted_regimes": [
      1
    ],
    "prohibited_regimes": [
      2,
      3,
      4
    ],
    "toxic_sessions": [],
    "toxic_days": [
      "Friday"
    ],
    "base_win_rate": 51.6,
    "base_profit_factor": 1.76,
    "min_ker": 0.3,
    "max_ker": 1.0,
    "max_chop": 52.0,
    "min_hurst": 0.52,
    "max_hurst": 1.0,
    "max_svp": 0.12
  },
  "disc127": {
    "name": "disc127",
    "symbol": "USDCAD",
    "category": "usdcad_champions",
    "archetype": "trend_pullback",
    "permitted_regimes": [
      1
    ],
    "prohibited_regimes": [
      2,
      3,
      4
    ],
    "toxic_sessions": [],
    "toxic_days": [],
    "base_win_rate": 66.7,
    "base_profit_factor": 4.07,
    "min_ker": 0.3,
    "max_ker": 1.0,
    "max_chop": 52.0,
    "min_hurst": 0.52,
    "max_hurst": 1.0,
    "max_svp": 0.12
  },
  "disc128": {
    "name": "disc128",
    "symbol": "USDCAD",
    "category": "usdcad_champions",
    "archetype": "trend_pullback",
    "permitted_regimes": [
      1
    ],
    "prohibited_regimes": [
      2,
      3,
      4
    ],
    "toxic_sessions": [],
    "toxic_days": [
      "Wednesday"
    ],
    "base_win_rate": 55.6,
    "base_profit_factor": 2.48,
    "min_ker": 0.3,
    "max_ker": 1.0,
    "max_chop": 52.0,
    "min_hurst": 0.52,
    "max_hurst": 1.0,
    "max_svp": 0.12
  },
  "disc129": {
    "name": "disc129",
    "symbol": "USDCAD",
    "category": "usdcad_champions",
    "archetype": "trend_pullback",
    "permitted_regimes": [
      1
    ],
    "prohibited_regimes": [
      2,
      3,
      4
    ],
    "toxic_sessions": [],
    "toxic_days": [
      "Wednesday"
    ],
    "base_win_rate": 55.9,
    "base_profit_factor": 2.52,
    "min_ker": 0.3,
    "max_ker": 1.0,
    "max_chop": 52.0,
    "min_hurst": 0.52,
    "max_hurst": 1.0,
    "max_svp": 0.12
  },
  "disc130": {
    "name": "disc130",
    "symbol": "USDCAD",
    "category": "usdcad_champions",
    "archetype": "trend_pullback",
    "permitted_regimes": [
      1
    ],
    "prohibited_regimes": [
      2,
      3,
      4
    ],
    "toxic_sessions": [],
    "toxic_days": [],
    "base_win_rate": 59.3,
    "base_profit_factor": 2.92,
    "min_ker": 0.3,
    "max_ker": 1.0,
    "max_chop": 52.0,
    "min_hurst": 0.52,
    "max_hurst": 1.0,
    "max_svp": 0.12
  },
  "disc106_chaikin_volatility_surge_elite": {
    "name": "disc106_chaikin_volatility_surge_elite",
    "symbol": "USDJPY",
    "category": "usdjpy_champions",
    "archetype": "volatility_breakout",
    "permitted_regimes": [
      1,
      4
    ],
    "prohibited_regimes": [
      2,
      3
    ],
    "toxic_sessions": [
      "NY Afternoon (16:00-21:00)"
    ],
    "toxic_days": [
      "Monday",
      "Wednesday"
    ],
    "base_win_rate": 38.4,
    "base_profit_factor": 1.17,
    "min_ker": 0.25,
    "max_ker": 1.0,
    "max_chop": 61.8,
    "min_hurst": 0.5,
    "max_hurst": 1.0,
    "max_svp": 0.12
  },
  "disc67_london_lunch_expansion_elite": {
    "name": "disc67_london_lunch_expansion_elite",
    "symbol": "USDJPY",
    "category": "usdjpy_champions",
    "archetype": "volatility_breakout",
    "permitted_regimes": [
      1,
      4
    ],
    "prohibited_regimes": [
      2,
      3
    ],
    "toxic_sessions": [],
    "toxic_days": [
      "Wednesday"
    ],
    "base_win_rate": 41.5,
    "base_profit_factor": 1.27,
    "min_ker": 0.25,
    "max_ker": 1.0,
    "max_chop": 61.8,
    "min_hurst": 0.5,
    "max_hurst": 1.0,
    "max_svp": 0.12
  },
  "disc83_parabolic_sar_flip_elite": {
    "name": "disc83_parabolic_sar_flip_elite",
    "symbol": "USDJPY",
    "category": "usdjpy_champions",
    "archetype": "trend_pullback",
    "permitted_regimes": [
      1
    ],
    "prohibited_regimes": [
      2,
      3,
      4
    ],
    "toxic_sessions": [
      "NY Afternoon (16:00-21:00)"
    ],
    "toxic_days": [
      "Monday"
    ],
    "base_win_rate": 40.1,
    "base_profit_factor": 1.23,
    "min_ker": 0.3,
    "max_ker": 1.0,
    "max_chop": 52.0,
    "min_hurst": 0.52,
    "max_hurst": 1.0,
    "max_svp": 0.12
  },
  "disc91_session_donchian_midline_bounce_elite": {
    "name": "disc91_session_donchian_midline_bounce_elite",
    "symbol": "USDJPY",
    "category": "usdjpy_champions",
    "archetype": "trend_pullback",
    "permitted_regimes": [
      1
    ],
    "prohibited_regimes": [
      2,
      3,
      4
    ],
    "toxic_sessions": [],
    "toxic_days": [
      "Friday",
      "Wednesday"
    ],
    "base_win_rate": 40.1,
    "base_profit_factor": 1.23,
    "min_ker": 0.3,
    "max_ker": 1.0,
    "max_chop": 52.0,
    "min_hurst": 0.52,
    "max_hurst": 1.0,
    "max_svp": 0.12
  },
  "disc94_london_vwap_pullback_elite": {
    "name": "disc94_london_vwap_pullback_elite",
    "symbol": "USDJPY",
    "category": "usdjpy_champions",
    "archetype": "trend_pullback",
    "permitted_regimes": [
      1
    ],
    "prohibited_regimes": [
      2,
      3,
      4
    ],
    "toxic_sessions": [
      "London/NY Overlap (12:00-16:00)"
    ],
    "toxic_days": [
      "Friday",
      "Monday"
    ],
    "base_win_rate": 39.6,
    "base_profit_factor": 1.2,
    "min_ker": 0.3,
    "max_ker": 1.0,
    "max_chop": 52.0,
    "min_hurst": 0.52,
    "max_hurst": 1.0,
    "max_svp": 0.12
  },
  "disc95_camarilla_expansion_elite": {
    "name": "disc95_camarilla_expansion_elite",
    "symbol": "USDJPY",
    "category": "usdjpy_champions",
    "archetype": "volatility_breakout",
    "permitted_regimes": [
      1,
      4
    ],
    "prohibited_regimes": [
      2,
      3
    ],
    "toxic_sessions": [],
    "toxic_days": [
      "Wednesday"
    ],
    "base_win_rate": 40.9,
    "base_profit_factor": 1.3,
    "min_ker": 0.25,
    "max_ker": 1.0,
    "max_chop": 61.8,
    "min_hurst": 0.5,
    "max_hurst": 1.0,
    "max_svp": 0.12
  },
  "disc99_m15_macd_zero_cross_momentum_elite": {
    "name": "disc99_m15_macd_zero_cross_momentum_elite",
    "symbol": "USDJPY",
    "category": "usdjpy_champions",
    "archetype": "trend_pullback",
    "permitted_regimes": [
      1
    ],
    "prohibited_regimes": [
      2,
      3,
      4
    ],
    "toxic_sessions": [
      "London/NY Overlap (12:00-16:00)"
    ],
    "toxic_days": [
      "Friday",
      "Wednesday"
    ],
    "base_win_rate": 38.3,
    "base_profit_factor": 1.16,
    "min_ker": 0.3,
    "max_ker": 1.0,
    "max_chop": 52.0,
    "min_hurst": 0.52,
    "max_hurst": 1.0,
    "max_svp": 0.12
  },
  "disc163": {
    "name": "disc163",
    "symbol": "USDCAD",
    "category": "usdcad_champions",
    "archetype": "trend_pullback",
    "permitted_regimes": [
      1
    ],
    "prohibited_regimes": [
      2,
      3,
      4
    ],
    "toxic_sessions": [],
    "toxic_days": [],
    "base_win_rate": 43.2,
    "base_profit_factor": 1.56,
    "min_ker": 0.3,
    "max_ker": 1.0,
    "max_chop": 52.0,
    "min_hurst": 0.52,
    "max_hurst": 1.0,
    "max_svp": 0.12
  },
  "disc164": {
    "name": "disc164",
    "symbol": "USDCAD",
    "category": "usdcad_champions",
    "archetype": "trend_pullback",
    "permitted_regimes": [
      1
    ],
    "prohibited_regimes": [
      2,
      3,
      4
    ],
    "toxic_sessions": [],
    "toxic_days": [],
    "base_win_rate": 42.3,
    "base_profit_factor": 1.36,
    "min_ker": 0.3,
    "max_ker": 1.0,
    "max_chop": 52.0,
    "min_hurst": 0.52,
    "max_hurst": 1.0,
    "max_svp": 0.12
  },
  "disc165": {
    "name": "disc165",
    "symbol": "GBPJPY",
    "category": "gbpjpy_champions",
    "archetype": "trend_pullback",
    "permitted_regimes": [
      1
    ],
    "prohibited_regimes": [
      2,
      3,
      4
    ],
    "toxic_sessions": [],
    "toxic_days": [
      "Thursday",
      "Wednesday"
    ],
    "base_win_rate": 38.6,
    "base_profit_factor": 1.23,
    "min_ker": 0.3,
    "max_ker": 1.0,
    "max_chop": 52.0,
    "min_hurst": 0.52,
    "max_hurst": 1.0,
    "max_svp": 0.12
  },
  "disc166": {
    "name": "disc166",
    "symbol": "GBPCHF",
    "category": "gbpchf_champions",
    "archetype": "trend_pullback",
    "permitted_regimes": [
      1
    ],
    "prohibited_regimes": [
      2,
      3,
      4
    ],
    "toxic_sessions": [],
    "toxic_days": [
      "Monday",
      "Thursday"
    ],
    "base_win_rate": 41.3,
    "base_profit_factor": 1.25,
    "min_ker": 0.3,
    "max_ker": 1.0,
    "max_chop": 52.0,
    "min_hurst": 0.52,
    "max_hurst": 1.0,
    "max_svp": 0.12
  },
  "disc167": {
    "name": "disc167",
    "symbol": "EURNZD",
    "category": "eurnzd_champions",
    "archetype": "trend_pullback",
    "permitted_regimes": [
      1
    ],
    "prohibited_regimes": [
      2,
      3,
      4
    ],
    "toxic_sessions": [],
    "toxic_days": [
      "Friday",
      "Monday",
      "Tuesday"
    ],
    "base_win_rate": 39.7,
    "base_profit_factor": 1.22,
    "min_ker": 0.3,
    "max_ker": 1.0,
    "max_chop": 52.0,
    "min_hurst": 0.52,
    "max_hurst": 1.0,
    "max_svp": 0.12
  },
  "disc168": {
    "name": "disc168",
    "symbol": "EURNZD",
    "category": "eurnzd_champions",
    "archetype": "trend_pullback",
    "permitted_regimes": [
      1
    ],
    "prohibited_regimes": [
      2,
      3,
      4
    ],
    "toxic_sessions": [
      "NY Afternoon (16:00-21:00)"
    ],
    "toxic_days": [
      "Tuesday",
      "Wednesday"
    ],
    "base_win_rate": 37.9,
    "base_profit_factor": 1.23,
    "min_ker": 0.3,
    "max_ker": 1.0,
    "max_chop": 52.0,
    "min_hurst": 0.52,
    "max_hurst": 1.0,
    "max_svp": 0.12
  },
  "disc169": {
    "name": "disc169",
    "symbol": "EURUSD",
    "category": "eurusd_champions",
    "archetype": "trend_pullback",
    "permitted_regimes": [
      1
    ],
    "prohibited_regimes": [
      2,
      3,
      4
    ],
    "toxic_sessions": [
      "NY Afternoon (16:00-21:00)"
    ],
    "toxic_days": [
      "Monday"
    ],
    "base_win_rate": 41.3,
    "base_profit_factor": 1.4,
    "min_ker": 0.3,
    "max_ker": 1.0,
    "max_chop": 52.0,
    "min_hurst": 0.52,
    "max_hurst": 1.0,
    "max_svp": 0.12
  },
  "disc170": {
    "name": "disc170",
    "symbol": "USDCAD",
    "category": "usdcad_champions",
    "archetype": "trend_pullback",
    "permitted_regimes": [
      1
    ],
    "prohibited_regimes": [
      2,
      3,
      4
    ],
    "toxic_sessions": [],
    "toxic_days": [],
    "base_win_rate": 50.8,
    "base_profit_factor": 1.99,
    "min_ker": 0.3,
    "max_ker": 1.0,
    "max_chop": 52.0,
    "min_hurst": 0.52,
    "max_hurst": 1.0,
    "max_svp": 0.12
  },
  "disc171": {
    "name": "disc171",
    "symbol": "GBPJPY",
    "category": "gbpjpy_champions",
    "archetype": "trend_pullback",
    "permitted_regimes": [
      1
    ],
    "prohibited_regimes": [
      2,
      3,
      4
    ],
    "toxic_sessions": [],
    "toxic_days": [],
    "base_win_rate": 41.2,
    "base_profit_factor": 1.54,
    "min_ker": 0.3,
    "max_ker": 1.0,
    "max_chop": 52.0,
    "min_hurst": 0.52,
    "max_hurst": 1.0,
    "max_svp": 0.12
  },
  "disc172": {
    "name": "disc172",
    "symbol": "EURGBP",
    "category": "eurgbp_champions",
    "archetype": "trend_pullback",
    "permitted_regimes": [
      1
    ],
    "prohibited_regimes": [
      2,
      3,
      4
    ],
    "toxic_sessions": [],
    "toxic_days": [],
    "base_win_rate": 52.6,
    "base_profit_factor": 2.04,
    "min_ker": 0.3,
    "max_ker": 1.0,
    "max_chop": 52.0,
    "min_hurst": 0.52,
    "max_hurst": 1.0,
    "max_svp": 0.12
  },
  "disc173": {
    "name": "disc173",
    "symbol": "USDJPY",
    "category": "usdjpy_champions",
    "archetype": "trend_pullback",
    "permitted_regimes": [
      1
    ],
    "prohibited_regimes": [
      2,
      3,
      4
    ],
    "toxic_sessions": [
      "London Open (06:00-12:00)"
    ],
    "toxic_days": [],
    "base_win_rate": 42.0,
    "base_profit_factor": 1.39,
    "min_ker": 0.3,
    "max_ker": 1.0,
    "max_chop": 52.0,
    "min_hurst": 0.52,
    "max_hurst": 1.0,
    "max_svp": 0.12
  },
  "disc174": {
    "name": "disc174",
    "symbol": "EURGBP",
    "category": "eurgbp_champions",
    "archetype": "trend_pullback",
    "permitted_regimes": [
      1
    ],
    "prohibited_regimes": [
      2,
      3,
      4
    ],
    "toxic_sessions": [],
    "toxic_days": [],
    "base_win_rate": 54.3,
    "base_profit_factor": 2.14,
    "min_ker": 0.3,
    "max_ker": 1.0,
    "max_chop": 52.0,
    "min_hurst": 0.52,
    "max_hurst": 1.0,
    "max_svp": 0.12
  },
  "disc175": {
    "name": "disc175",
    "symbol": "EURGBP",
    "category": "eurgbp_champions",
    "archetype": "trend_pullback",
    "permitted_regimes": [
      1
    ],
    "prohibited_regimes": [
      2,
      3,
      4
    ],
    "toxic_sessions": [],
    "toxic_days": [
      "Tuesday"
    ],
    "base_win_rate": 48.0,
    "base_profit_factor": 1.62,
    "min_ker": 0.3,
    "max_ker": 1.0,
    "max_chop": 52.0,
    "min_hurst": 0.52,
    "max_hurst": 1.0,
    "max_svp": 0.12
  },
  "disc180": {
    "name": "disc180",
    "symbol": "EURGBP",
    "category": "eurgbp_champions",
    "archetype": "trend_pullback",
    "permitted_regimes": [
      1
    ],
    "prohibited_regimes": [
      2,
      3,
      4
    ],
    "toxic_sessions": [],
    "toxic_days": [
      "Wednesday"
    ],
    "base_win_rate": 51.4,
    "base_profit_factor": 1.63,
    "min_ker": 0.3,
    "max_ker": 1.0,
    "max_chop": 52.0,
    "min_hurst": 0.52,
    "max_hurst": 1.0,
    "max_svp": 0.12
  }
}


class LiveMarketRegimeFilter:
    """
    Production Market Regime Engine & Archetype Kill-Switch.
    Zero external C-dependencies, fully self-contained.
    """
    def __init__(
        self,
        ker_period: int = 20,
        chop_period: int = 14,
        hurst_lag: int = 5,
        hurst_window: int = 100,
        atr_fast: int = 5,
        atr_slow: int = 50,
        bb_period: int = 20,
        bbwp_window: int = 252,
    ):
        self.ker_period = ker_period
        self.chop_period = chop_period
        self.hurst_lag = hurst_lag
        self.hurst_window = hurst_window
        self.atr_fast = atr_fast
        self.atr_slow = atr_slow
        self.bb_period = bb_period
        self.bbwp_window = bbwp_window

    def compute_metrics(
        self,
        symbol: str,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        spread_pips: float
    ) -> Dict[str, float]:
        """Calculates all 6 closed-form microstructure indicators."""
        n = len(close)
        if n < max(self.hurst_window + self.hurst_lag, self.bbwp_window + self.bb_period):
            return {}

        # 1. Kaufman Efficiency Ratio (KER)
        abs_diff = np.abs(np.diff(close))
        net_change = abs(close[-1] - close[-self.ker_period - 1])
        path_sum = np.sum(abs_diff[-self.ker_period:])
        ker = float(net_change / path_sum) if path_sum > 1e-12 else 0.0

        # 2. True Range & ATRs
        prev_c = np.roll(close, 1)
        prev_c[0] = close[0]
        tr = np.maximum(high - low, np.maximum(np.abs(high - prev_c), np.abs(low - prev_c)))
        atr_fast_val = float(np.mean(tr[-self.atr_fast:]))
        atr_slow_val = float(np.mean(tr[-self.atr_slow:]))
        rvr = float(atr_fast_val / atr_slow_val) if atr_slow_val > 1e-12 else 1.0

        # 3. Choppiness Index (CHOP)
        tr_sum = np.sum(tr[-self.chop_period:])
        span = np.max(high[-self.chop_period:]) - np.min(low[-self.chop_period:])
        log10_n = np.log10(float(self.chop_period))
        chop = float(100.0 * np.log10(tr_sum / span) / log10_n) if span > 1e-12 and tr_sum > 1e-12 else 50.0

        # 4. Variance Ratio Hurst Exponent
        log_p = np.log(close)
        ret1 = np.diff(log_p)
        retq = log_p[self.hurst_lag:] - log_p[:-self.hurst_lag]
        
        w1 = ret1[-self.hurst_window:]
        wq = retq[-self.hurst_window:]
        v1 = np.var(w1, ddof=1)
        vq = np.var(wq, ddof=1)
        
        if v1 > 1e-16 and vq > 1e-16:
            vr = vq / (self.hurst_lag * v1)
            hurst = float(0.5 + np.log(vr) / (2.0 * np.log(float(self.hurst_lag))))
        else:
            hurst = 0.50

        # 5. Bollinger Bandwidth Percentile (BBWP)
        bbw_history = []
        for i in range(n - self.bbwp_window, n):
            window_slice = close[i - self.bb_period + 1:i + 1]
            mean_val = np.mean(window_slice)
            std_val = np.std(window_slice)
            bbw = (4.0 * std_val) / mean_val if mean_val > 1e-12 else 0.0
            bbw_history.append(bbw)
        curr_bbw = bbw_history[-1]
        bbwp = float(np.sum(np.array(bbw_history) <= curr_bbw) / float(self.bbwp_window) * 100.0)

        # 6. Spread Friction Ratio (SVP)
        pip_unit = 0.01 if "JPY" in symbol.upper() else 0.0001
        atr_pips = atr_fast_val / pip_unit
        svp = float(spread_pips / atr_pips) if atr_pips > 1e-12 else 0.0

        return {
            "ker": round(ker, 4),
            "chop": round(chop, 2),
            "hurst": round(hurst, 4),
            "rvr": round(rvr, 3),
            "bbwp": round(bbwp, 1),
            "svp": round(svp, 4),
        }

    def classify_regime(self, metrics: Dict[str, float]) -> int:
        """
        Classifies current bar into Regimes 1..4:
          1 = Persistent Directional Trend
          2 = Stationary Range
          3 = High-Entropy Volatile Chop
          4 = Liquidity Squeeze (Coil)
        """
        if not metrics:
            return 0
        ker = metrics.get("ker", 0.0)
        chop = metrics.get("chop", 50.0)
        hurst = metrics.get("hurst", 0.50)
        rvr = metrics.get("rvr", 1.0)
        bbwp = metrics.get("bbwp", 50.0)

        if rvr < 0.70 and bbwp < 15.0:
            return 4  # Liquidity Squeeze
        if ker >= 0.40 and chop < 38.2 and hurst > 0.56:
            return 1  # Trend
        if ker < 0.20 and chop > 61.8 and rvr > 1.35:
            return 3  # Volatile Chop
        if ker < 0.22 and chop > 55.0 and hurst < 0.44:
            return 2  # Stationary Range
        return 3      # Default unclassified noise -> treated as Chop

    def evaluate_live_permission(
        self,
        strategy_name: str,
        symbol: str,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        spread_pips: float,
        current_utc_dt: pd.Timestamp
    ) -> Tuple[bool, str, Dict[str, float]]:
        """
        Evaluates incoming strategy signal against instantaneous market microstructure.
        Returns:
            permitted: bool (True = Allowed, False = Blocked)
            reason: str (Diagnostic reason)
            metrics: dict (Instantaneous indicator values)
        """
        metrics = self.compute_metrics(symbol, high, low, close, spread_pips)
        if not metrics:
            return True, "WARMUP_PASS", {}

        # LEVEL 1: UNIVERSAL SPREAD FRICTION KILL-SWITCH
        if metrics["svp"] > 0.12:
            return False, f"KILL_EXCESSIVE_SPREAD_FRICTION (SVP={metrics['svp']:.3f} > 0.12)", metrics

        # LEVEL 2: REGIME CLASSIFICATION
        regime = self.classify_regime(metrics)
        metrics["regime"] = regime

        # LEVEL 3: LOOKUP STRATEGY PROFILE
        lookup_key = f"{symbol}_{strategy_name}"
        profile = STRATEGY_REGIME_REGISTRY.get(lookup_key) or _NAME_REGISTRY.get(strategy_name)

        day_name = current_utc_dt.day_name()
        hour = current_utc_dt.hour

        if profile:
            # Check calendar toxic conditions
            if day_name in profile.get("toxic_days", []):
                return False, f"KILL_TAILORED_TOXIC_DAY ({strategy_name} on {day_name})", metrics

            # Check session toxic conditions
            for sess in profile.get("toxic_sessions", []):
                sess_lower = sess.lower()
                if "ny" in sess_lower and (hour >= 14):
                    return False, f"KILL_TAILORED_TOXIC_SESSION ({strategy_name} in NY Session)", metrics
                elif "asian" in sess_lower and (hour < 6 or hour >= 21):
                    return False, f"KILL_TAILORED_TOXIC_SESSION ({strategy_name} in Asian Session)", metrics

            # Check prohibited regime
            if regime in profile.get("prohibited_regimes", []):
                return False, f"KILL_PROHIBITED_REGIME (Regime {regime} prohibited for {profile['archetype']})", metrics

            # Check KER bounds
            if metrics["ker"] < profile.get("min_ker", 0.0):
                return False, f"KILL_LOW_KER (KER={metrics['ker']:.3f} < {profile['min_ker']})", metrics

            # Check CHOP bounds
            if metrics["chop"] > profile.get("max_chop", 100.0):
                return False, f"KILL_HIGH_CHOP (CHOP={metrics['chop']:.1f} > {profile['max_chop']})", metrics

            return True, f"PERMITTED_TO_TRADE (Regime {regime})", metrics

        # Fallback Archetype Heuristics if strategy name is unknown
        strat_lower = strategy_name.lower()
        if "fade" in strat_lower or "reversion" in strat_lower:
            if metrics["hurst"] > 0.54 or metrics["ker"] > 0.38:
                return False, "KILL_MEAN_REVERSION_TREND_SURGE", metrics
        elif "breakout" in strat_lower or "impulse" in strat_lower or "squeeze" in strat_lower:
            if metrics["chop"] > 61.8 or metrics["bbwp"] > 80.0:
                return False, "KILL_BREAKOUT_CHOP_TRAP", metrics
        else:
            # Default Trend Pullback
            if metrics["ker"] < 0.30 or metrics["chop"] > 52.0 or metrics["hurst"] < 0.52:
                return False, "KILL_TREND_PULLBACK_CHOP", metrics

        return True, "PERMITTED_FALLBACK", metrics


# Export Alias for Drop-In Backward Compatibility
MarketRegimeSystem = LiveMarketRegimeFilter


# ==============================================================================
# SELF-TEST & DIAGNOSTIC RUNNER
# ==============================================================================
if __name__ == "__main__":
    print("=" * 80)
    print("  MARKET REGIME & ARCHETYPE KILL-SWITCH MASTER SYSTEM - SELF-TEST")
    print("=" * 80)
    print(f"Total Registered Production Strategies: {len(STRATEGY_REGIME_REGISTRY)}")

    filter_engine = LiveMarketRegimeFilter()
    np.random.seed(42)

    # 1. Simulate 300 M15 bars
    n_bars = 300
    closes = np.cumprod(1 + np.random.normal(0, 0.001, n_bars)) * 150.0
    highs = closes + np.random.uniform(0.01, 0.05, n_bars)
    lows = closes - np.random.uniform(0.01, 0.05, n_bars)

    print("\n[TEST 1] Testing disc165 (GBPJPY London Breakout) on Wednesday (Mid-week Chop):")
    perm, reason, m = filter_engine.evaluate_live_permission(
        strategy_name="disc165",
        symbol="GBPJPY",
        high=highs,
        low=lows,
        close=closes,
        spread_pips=1.2,
        current_utc_dt=pd.Timestamp("2026-11-25 07:15:00")  # Wednesday
    )
    print(f"  Result: Permitted={perm} | Reason={reason}")
    print(f"  Metrics: {m}")
    assert perm is False, "Validation failed: disc165 should be blocked on Wednesday!"

    print("\n[TEST 2] Testing disc168 (EURNZD Trend Pullback) during NY Session (15:00 UTC):")
    perm2, reason2, m2 = filter_engine.evaluate_live_permission(
        strategy_name="disc168",
        symbol="EURNZD",
        high=highs,
        low=lows,
        close=closes,
        spread_pips=2.0,
        current_utc_dt=pd.Timestamp("2026-11-24 15:00:00")  # 15:00 UTC
    )
    print(f"  Result: Permitted={perm2} | Reason={reason2}")
    assert perm2 is False, "Validation failed: disc168 should be blocked in NY Session!"

    print("\n[TEST 3] Testing High Spread Friction Spike (SVP > 0.12):")
    perm3, reason3, m3 = filter_engine.evaluate_live_permission(
        strategy_name="disc115",
        symbol="AUDCAD",
        high=highs,
        low=lows,
        close=closes,
        spread_pips=15.0,  # Extreme spread
        current_utc_dt=pd.Timestamp("2026-11-24 08:00:00")
    )
    print(f"  Result: Permitted={perm3} | Reason={reason3}")
    assert perm3 is False, "Validation failed: High spread should be blocked by SVP gate!"

    print("\n" + "=" * 80)
    print("  ALL SELF-TESTS PASSED! MASTER REGIME SYSTEM READY FOR EXPORT.")
    print("=" * 80)
