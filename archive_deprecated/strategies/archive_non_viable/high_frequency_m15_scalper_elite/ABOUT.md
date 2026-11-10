# High-Frequency M15 Scalper (All-Weather Elite Edition)

## Strategy Overview
- **Asset Class**: Foreign Exchange (Pure EUR/USD)
- **Timeframe**: 15-Minute (M15)
- **Strategy Class**: High Trade Volume / Multiple Trades Per Day Intraday Scalper
- **Execution Engine**: Millisecond Tick Lookup with Authentic Dukascopy Floating Spread
- **Trading Window**: Peak Institutional Hours [07, 08, 09, 11, 12, 13, 14, 15, 17] UTC
- **Target Risk-to-Reward**: 2.20 : 1

## Core Logic & All-Weather Refinements
1. **Multi-Timeframe Macro Regime Gating**:
   - Long setups require confirmation above H1 50-EMA macro trend.
   - Short setups require confirmation below H1 50-EMA macro trend.
   - Eliminates counter-trend traps during aggressive macro expansions.
2. **Dynamic ATR Stop Loss Sizing**:
   - Replaces fragile fixed micro-stops with dynamic breathing room: $\text{SL} \in [0.8 \times \text{ATR}, 1.8 \times \text{ATR}]$.
   - Protects against institutional liquidity sweepouts in high-volatility regimes.
3. **Dual Structural Triggers**:
   - **Pattern A (4-Hour Liquidity Sweeps)**: Fades institutional false breakouts beyond rolling 4-hour highs/lows.
   - **Pattern B (Fair Value Gap Mitigation)**: Capitalizes on high-velocity displacement imbalances ($\ge 3.5$ pips).

---

## Dual-Regime Performance Scorecard

### 1. Out-of-Sample Unseen Live Sim (Q4 2021: 14 Weeks / 3,452,917 Ticks)
- **Total Trades**: 79 trades (~5.6 trades/week / ~1.13 trades/day)
- **Win Rate**: 38.0%
- **Profit Factor**: **1.34**
- **Realized Risk-to-Reward**: **2.21 : 1**
- **Net Pips Delivered**: **+155.9 pips**
- **Max Portfolio Drawdown**: **104.2 pips**

### 2. In-Sample Benchmark (2025-2026: 53 Weeks / 18,026,436 Ticks)
- **Total Trades**: 340 trades (~6.4 trades/week / ~1.28 trades/day)
- **Win Rate**: 35.6%
- **Profit Factor**: **1.21**
- **Realized Risk-to-Reward**: **2.14 : 1**
- **Net Pips Delivered**: **+420.6 pips**
- **Max Portfolio Drawdown**: **134.1 pips**
