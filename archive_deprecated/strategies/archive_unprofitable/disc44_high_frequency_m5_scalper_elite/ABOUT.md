# DISC44: High-Frequency M5 Scalper (All-Weather Elite Edition)

## Strategy Overview
- **Asset Class**: Foreign Exchange (Pure EUR/USD)
- **Timeframe**: 5-Minute (M5)
- **Strategy Class**: Pure M5 Intraday Microstructure Scalper
- **Execution Engine**: Millisecond Tick Lookup with Authentic Dukascopy Floating Spread (0.4p mean) & Slippage (0.2p)
- **Trading Window**: Peak Institutional Hours [07, 08, 09, 10, 12, 13, 14, 15, 16] UTC
- **Target Risk-to-Reward**: 2.20 : 1

## Core Logic & Spread-Friction Calibrations
1. **Calibrated Fair Value Gap (FVG) Mitigation**:
   - Requires minimum displacement gap $\ge 3.5$ pips (filtering out sub-2p noise where spread consumes 15% of trade equity).
   - Confirms mitigation bounce with directional close.
2. **Displacement 50% Equilibrium Retest**:
   - Institutional impulse candle (range $\ge 7.0$ pips, body $\ge 70\%$).
   - Price retraces within 3 bars to the 50% midpoint and rejects in the direction of the H1 50-EMA macro trend.
3. **M5 Narrow-Range Compression Box Breakout (NR4)**:
   - 4 consecutive M5 bars with range $\le 3.2$ pips.
   - Breakout candle breaking the 4-bar box extreme aligned with macro trend.
4. **Dynamic ATR Stop Loss**:
   - Replaces fragile fixed micro-stops with breathing room $\text{SL} \in [0.9 \times \text{ATR}, 1.8 \times \text{ATR}]$ (7.0p to 16.0p).

---

## 52-Week Tick Audit Scorecard (52 Weeks / ~26M Ticks)
- **Total Trades**: **440 trades** (~8.5 trades/week / ~1.7 trades/day)
- **Win Rate**: **34.1%** (Breakeven at 2.2R is 31.25%)
- **Profit Factor**: **1.10**
- **Net Cumulative Pips**: **+309.5 pips**
- **Max Portfolio Drawdown**: **270.1 pips**
