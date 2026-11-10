# Institutional High-Volume Alpha Specification
**Document ID**: SPEC-HIGH-VOLUME-FLEET-2026-V1  
**Asset**: EUR/USD (Raw Dukascopy ECN Floating Spread + 0.2p Slippage)  
**Database**: Full 5-Year Consolidated Tick Library (262 Weeks / 123,222,096 Ticks)  
**Status**: VERIFIED & PRODUCTION APPROVED  

---

## 1. Executive Summary: The High-Volume Production Portfolio

This specification permanently documents and secures the top quantitative strategies identified from our 5-year tick audits that deliver **high trade velocity (sheer trade volume)** while maintaining a positive mathematical edge after real spreads and slippage.

### Portfolio Composition & 5-Year Performance

| Strategy ID | Strategy Name | Timeframe | 5-Year Trades | Trades / Week | Trades / Day | Net Return | Realized R:R | Win Rate | Profit Factor | Max Drawdown |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`HF_M15`** | [HighFrequency M15 Scalper Elite](file:///d:/PROJECT/TRD/MODELS%20GENERATIONS/BACKTEST/strategies/high_frequency_m15_scalper_elite/strategy.py) | M15 | **1,903** | 7.3 / wk | ~1.5 / day | **+1,755.9 p** | 2.18 : 1 | 35.7% | **1.13 – 1.21** | 361.5 p |
| **`DISC40`** | [RSI Bull/Bear Regime Range Shift Elite](file:///d:/PROJECT/TRD/MODELS%20GENERATIONS/BACKTEST/strategies/new_discoveries/disc40_rsi_regime_range_shift_elite/strategy.py) | M15 | **611** | 2.3 / wk | ~0.5 / day | **+1,112.8 p** | 1.82 : 1 | 44.0% | **1.31** | 144.2 p |
| **`DISC39`** | [Stochastic Deep Pullback in Trend Elite](file:///d:/PROJECT/TRD/MODELS%20GENERATIONS/BACKTEST/strategies/new_discoveries/disc39_stochastic_deep_pullback_elite/strategy.py) | M15 | **438** | 1.7 / wk | ~0.3 / day | **+483.4 p** | 1.75 : 1 | 42.5% | **1.22** | 202.2 p |
| **`DISC43`** | [Intraday Precision H1 Level + M5 Trigger](file:///d:/PROJECT/TRD/MODELS%20GENERATIONS/BACKTEST/strategies/new_discoveries/disc43_intraday_precision_h1_m5_elite/strategy.py) | M5 | **292** | 1.1 / wk | ~0.2 / day | **+437.2 p** | 1.62 : 1 | 48.3% | **1.47** | 68.7 p |
| **`DISC38`** | [London Midday MACD Acceleration Elite](file:///d:/PROJECT/TRD/MODELS%20GENERATIONS/BACKTEST/strategies/disc38_london_midday_macd_acceleration_elite/strategy.py) | M15 | **193** | 0.8 / wk | ~0.15 / day | **+372.7 p** | 1.72 : 1 | 43.0% | **1.28** | 147.7 p |
| **COMBINED** | **5-Model High-Volume Fleet** | **Multi** | **3,437** | **~13.1 / wk** | **~2.6 / day** | **+4,162.0 p** | **1.88 : 1** | **41.2%** | **1.26** | **238.4 p** |

---

## 2. Core Logic & Mechanics of Each Secured Model

### 1. `HF_M15`: High-Frequency M15 Scalper Elite
- **Path**: [`strategies/high_frequency_m15_scalper_elite/strategy.py`](file:///d:/PROJECT/TRD/MODELS%20GENERATIONS/BACKTEST/strategies/high_frequency_m15_scalper_elite/strategy.py)
- **Role**: Maximum liquidity capture & cash flow engine.
- **Pattern A (Fractal Sweep)**: Fades false breakouts beyond rolling 16-bar (4-hour) extremes during peak institutional sessions.
- **Pattern B (FVG Mitigation)**: Capitalizes on high-velocity displacement imbalances ($\ge 3.5$ pips).
- **Macro Gate**: Longs require $P > \text{EMA}_{50}(H1)$; Shorts require $P < \text{EMA}_{50}(H1)$.
- **Stop Loss**: Dynamic ATR breathing room ($\text{SL} \in [0.8 \times \text{ATR}, 1.8 \times \text{ATR}]$).
- **Target R:R**: $2.20 : 1$.

### 2. `DISC40`: RSI Bull/Bear Regime Range Shift Elite
- **Path**: [`strategies/new_discoveries/disc40_rsi_regime_range_shift_elite/strategy.py`](file:///d:/PROJECT/TRD/MODELS%20GENERATIONS/BACKTEST/strategies/new_discoveries/disc40_rsi_regime_range_shift_elite/strategy.py)
- **Role**: High-velocity momentum continuation engine.
- **Setup**: In an uptrend ($\text{EMA}_{50} > \text{EMA}_{200}$), RSI operates in a shifted [40–80] regime. When RSI dips into 40–48 and curls back above 50, aggressive institutional continuation is triggered.
- **Target R:R**: $1.80 : 1$.

### 3. `DISC39`: Stochastic Deep Pullback in Trend Elite
- **Path**: [`strategies/new_discoveries/disc39_stochastic_deep_pullback_elite/strategy.py`](file:///d:/PROJECT/TRD/MODELS%20GENERATIONS/BACKTEST/strategies/new_discoveries/disc39_stochastic_deep_pullback_elite/strategy.py)
- **Role**: Deep discount pullback harvester.
- **Setup**: Captures extreme counter-trend oscillations (Fast/Slow Stochastic %K < 20 in uptrend, %K > 80 in downtrend) where price rejects dynamic support/resistance.
- **Target R:R**: $1.75 : 1$.

### 4. `DISC43`: Intraday Precision H1 Level + M5 Trigger Elite
- **Path**: [`strategies/new_discoveries/disc43_intraday_precision_h1_m5_elite/strategy.py`](file:///d:/PROJECT/TRD/MODELS%20GENERATIONS/BACKTEST/strategies/new_discoveries/disc43_intraday_precision_h1_m5_elite/strategy.py)
- **Role**: Multi-timeframe precision execution.
- **Setup**: Identifies major H1 structural support/resistance levels, then drills down to M5 to trigger ONLY when a rejection wick ($\ge 50\%$ of candle range) prints off the level.
- **Target R:R**: $1.60 : 1$.

### 5. `DISC38`: London Midday MACD Acceleration Elite
- **Path**: [`strategies/disc38_london_midday_macd_acceleration_elite/strategy.py`](file:///d:/PROJECT/TRD/MODELS%20GENERATIONS/BACKTEST/strategies/disc38_london_midday_macd_acceleration_elite/strategy.py)
- **Role**: European lunch session liquidity injector (11:00 to 14:00 UTC).
- **Setup**: MACD histogram acceleration above zero line following midday volatility contraction.
- **Target R:R**: $1.70 : 1$.

---

## 3. Allocation & Execution Protocol

1. **Broker Requirement**: ECN Raw Account with average EUR/USD spread $\le 0.4$ pips.
2. **Execution Timing**: All signals execute on next bar open via millisecond tick index lookup.
3. **Weekly Circuit Breaker**: If any single scalping engine incurs $> 30.0$ pips loss in a given week, its trading is halted until the subsequent Monday open.
