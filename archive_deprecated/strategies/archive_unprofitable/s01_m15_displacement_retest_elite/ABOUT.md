# S01: M15 Displacement & Structure Retest Scalper (Elite Edition)

## Strategy Overview
- **Asset Class**: Foreign Exchange (Pure EUR/USD)
- **Timeframe**: 15-Minute (M15)
- **Execution Engine**: Millisecond Tick Lookup with Authentic Dukascopy Floating Spread
- **Trading Window**: 07:00 – 16:59 UTC (European & US active liquidity hours)
- **Target Risk-to-Reward**: 2.80 : 1

## Core Logic & Mechanics
1. **Displacement Identification**: Scans M15 candles for clean institutional expansion bodies exceeding 4.0 pips that breach prior candle structural extremes.
2. **Structure Retest Mitigation**: Requires immediate pullback retesting the broken structural level without breaching the displacement origin.
3. **Macro Trend Alignment**: Trades only in alignment with the M15 50-EMA vs 200-EMA macro slope.
4. **Volatility Gating**: Rejects trades during dead volatility compression where ATR(14) is below 5.0 pips.
5. **Structural Risk**: Dynamic stop loss placed 1.5 pips beyond the displacement origin bar (5.0 to 14.0 pips risk window).

## Verified Performance Scorecard (53 Weeks / 18,026,436 Ticks)
- **Annual Trade Count**: 67 trades (~0.25 trades/day / ~1.3 trades/week)
- **Win Rate**: 35.8%
- **Profit Factor**: 1.48
- **Realized Risk-to-Reward**: 2.65 : 1
- **Net Pips Delivered**: +217.0 pips
- **Max Portfolio Drawdown**: 60.8 pips
