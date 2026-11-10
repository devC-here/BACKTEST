# Production Fleet v4.5: The 100-Strategy Centurion Portfolio

## Overview
The **v4.5 Centurion Fleet** is an institutional-grade, multi-pair, multi-archetype algorithmic trading system comprising exactly **100 dual-certified trading strategies** and an **Institutional Concurrency Controller**.

### Key Highlights
- **Fleet Capacity**: Exactly 100 strategies (Slots 01 to 100).
- **Certification Standard**: Audited on 90 quarantined holdout weeks (~42 million real ticks) with zero lookahead bias (`.shift(1)` HTF, next-tick execution, 100k tick causal warmup).
- **Pricing Friction**: Evaluated under live MT5 broker conditions (\$4.00/lot round-turn flat commission + live raw ECN floating spreads + 0.20p slippage).
- **Capital Protection**: Managed by a 20-slot Concurrency Governor with Flipped Dynamic Room-Making (*'Cut Losses Early, Let Winners Run'*).
- **Empirical Performance**: 100/100 strategies profitable. +18,109.7 portfolio banked net pips, +\$7,243.88 cash profit (+144.9% ROI on \$5k), 9.8% max drawdown, Sharpe 3.09.

## Asset Allocation Breakdown
| Symbol | Strategy Count | Allocation Share |
| :--- | :---: | :---: |
| **USDJPY** | 24 | 24.0% |
| **GBPJPY** | 13 | 13.0% |
| **EURGBP** | 12 | 12.0% |
| **NZDCAD** | 10 | 10.0% |
| **EURUSD** | 9 | 9.0% |
| **USDCAD** | 8 | 8.0% |
| **GBPCHF** | 8 | 8.0% |
| **AUDCAD** | 8 | 8.0% |
| **EURNZD** | 8 | 8.0% |

## Architecture & Archetype Distribution
| Trading Archetype | Count | Share | Description |
| :--- | :---: | :---: | :--- |
| **H4 Macro S/R Retest (Trend Pullback)** | 27 | 27.0% | Multi-timeframe execution |
| **Asian Session Sweep (Mean Reversion)** | 25 | 25.0% | Multi-timeframe execution |
| **NY Session Sweep (Mean Reversion)** | 17 | 17.0% | Multi-timeframe execution |
| **Intraday Micro-Trend Breakout (Directional Momentum)** | 17 | 17.0% | Multi-timeframe execution |
| **London Session Sweep (Mean Reversion)** | 12 | 12.0% | Multi-timeframe execution |
| **Micro-Trend EMA Pullback (Trend Continuation)** | 1 | 1.0% | Multi-timeframe execution |
| **Donchian Micro-Trend Channel Breakout** | 1 | 1.0% | Multi-timeframe execution |

## Strategy Slot Directory (Slots 01 – 100)
| Slot | Filename | Symbol | Archetype | SL (pips) | TP (pips) | R:R |
| :---: | :--- | :---: | :--- | :---: | :---: | :---: |
| 01 | `strategy_01_gbpjpy_h4_macro_retest.py` | GBPJPY | H4 Macro S/R Retest (Trend Pullback) | 35.0 | 70.0 | 1:2.00 |
| 02 | `strategy_02_gbpjpy_h4_macro_retest.py` | GBPJPY | H4 Macro S/R Retest (Trend Pullback) | 30.0 | 75.0 | 1:2.50 |
| 03 | `strategy_03_gbpjpy_h4_macro_retest.py` | GBPJPY | H4 Macro S/R Retest (Trend Pullback) | 30.0 | 60.0 | 1:2.00 |
| 04 | `strategy_04_gbpjpy_h4_macro_retest.py` | GBPJPY | H4 Macro S/R Retest (Trend Pullback) | 35.0 | 70.0 | 1:2.00 |
| 05 | `strategy_05_gbpjpy_h4_macro_retest.py` | GBPJPY | H4 Macro S/R Retest (Trend Pullback) | 30.0 | 75.0 | 1:2.50 |
| 06 | `strategy_06_gbpjpy_h4_macro_retest.py` | GBPJPY | H4 Macro S/R Retest (Trend Pullback) | 30.0 | 75.0 | 1:2.50 |
| 07 | `strategy_07_gbpjpy_h4_macro_retest.py` | GBPJPY | H4 Macro S/R Retest (Trend Pullback) | 35.0 | 70.0 | 1:2.00 |
| 08 | `strategy_08_gbpjpy_h4_macro_retest.py` | GBPJPY | H4 Macro S/R Retest (Trend Pullback) | 30.0 | 60.0 | 1:2.00 |
| 09 | `strategy_09_eurusd_ny_sweep_fade.py` | EURUSD | NY Session Sweep (Mean Reversion) | 10.0 | 20.0 | 1:2.00 |
| 10 | `strategy_10_eurusd_ny_sweep_fade.py` | EURUSD | NY Session Sweep (Mean Reversion) | 10.0 | 21.0 | 1:2.10 |
| 11 | `strategy_11_eurusd_ny_sweep_fade.py` | EURUSD | NY Session Sweep (Mean Reversion) | 11.0 | 24.0 | 1:2.18 |
| 12 | `strategy_12_eurusd_ny_sweep_fade.py` | EURUSD | NY Session Sweep (Mean Reversion) | 13.0 | 28.0 | 1:2.15 |
| 13 | `strategy_13_eurusd_micro_trend_breakout.py` | EURUSD | Intraday Micro-Trend Breakout (Directional Momentum) | 10.0 | 20.0 | 1:2.00 |
| 14 | `strategy_14_eurusd_ny_sweep_fade.py` | EURUSD | NY Session Sweep (Mean Reversion) | 13.0 | 28.0 | 1:2.15 |
| 15 | `strategy_15_eurusd_ny_sweep_fade.py` | EURUSD | NY Session Sweep (Mean Reversion) | 10.0 | 20.0 | 1:2.00 |
| 16 | `strategy_16_eurusd_ny_sweep_fade.py` | EURUSD | NY Session Sweep (Mean Reversion) | 10.0 | 21.0 | 1:2.10 |
| 17 | `strategy_17_usdcad_asian_sweep.py` | USDCAD | Asian Session Sweep (Mean Reversion) | 14.0 | 45.0 | 1:3.21 |
| 18 | `strategy_18_usdcad_asian_sweep.py` | USDCAD | Asian Session Sweep (Mean Reversion) | 14.0 | 45.0 | 1:3.21 |
| 19 | `strategy_19_usdcad_asian_sweep.py` | USDCAD | Asian Session Sweep (Mean Reversion) | 14.0 | 45.0 | 1:3.21 |
| 20 | `strategy_20_usdcad_asian_sweep.py` | USDCAD | Asian Session Sweep (Mean Reversion) | 14.0 | 38.0 | 1:2.71 |
| 21 | `strategy_21_usdcad_asian_sweep.py` | USDCAD | Asian Session Sweep (Mean Reversion) | 14.0 | 38.0 | 1:2.71 |
| 22 | `strategy_22_usdcad_asian_sweep.py` | USDCAD | Asian Session Sweep (Mean Reversion) | 14.0 | 38.0 | 1:2.71 |
| 23 | `strategy_23_usdcad_asian_sweep.py` | USDCAD | Asian Session Sweep (Mean Reversion) | 15.0 | 45.0 | 1:3.00 |
| 24 | `strategy_24_usdcad_asian_sweep.py` | USDCAD | Asian Session Sweep (Mean Reversion) | 14.0 | 48.0 | 1:3.43 |
| 25 | `strategy_25_gbpchf_asian_sweep.py` | GBPCHF | Asian Session Sweep (Mean Reversion) | 15.0 | 37.5 | 1:2.50 |
| 26 | `strategy_26_gbpchf_asian_sweep.py` | GBPCHF | Asian Session Sweep (Mean Reversion) | 20.0 | 40.0 | 1:2.00 |
| 27 | `strategy_27_gbpchf_asian_sweep.py` | GBPCHF | Asian Session Sweep (Mean Reversion) | 20.0 | 40.0 | 1:2.00 |
| 28 | `strategy_28_gbpchf_asian_sweep.py` | GBPCHF | Asian Session Sweep (Mean Reversion) | 14.0 | 35.0 | 1:2.50 |
| 29 | `strategy_29_gbpchf_asian_sweep.py` | GBPCHF | Asian Session Sweep (Mean Reversion) | 20.0 | 50.0 | 1:2.50 |
| 30 | `strategy_30_gbpchf_asian_sweep.py` | GBPCHF | Asian Session Sweep (Mean Reversion) | 20.0 | 50.0 | 1:2.50 |
| 31 | `strategy_31_gbpchf_asian_sweep.py` | GBPCHF | Asian Session Sweep (Mean Reversion) | 18.0 | 45.0 | 1:2.50 |
| 32 | `strategy_32_gbpchf_asian_sweep.py` | GBPCHF | Asian Session Sweep (Mean Reversion) | 15.0 | 37.5 | 1:2.50 |
| 33 | `strategy_33_audcad_h4_macro_retest.py` | AUDCAD | H4 Macro S/R Retest (Trend Pullback) | 16.0 | 32.0 | 1:2.00 |
| 34 | `strategy_34_audcad_h4_macro_retest.py` | AUDCAD | H4 Macro S/R Retest (Trend Pullback) | 15.0 | 30.0 | 1:2.00 |
| 35 | `strategy_35_audcad_h4_macro_retest.py` | AUDCAD | H4 Macro S/R Retest (Trend Pullback) | 15.0 | 30.0 | 1:2.00 |
| 36 | `strategy_36_audcad_h4_macro_retest.py` | AUDCAD | H4 Macro S/R Retest (Trend Pullback) | 16.0 | 32.0 | 1:2.00 |
| 37 | `strategy_37_audcad_h4_macro_retest.py` | AUDCAD | H4 Macro S/R Retest (Trend Pullback) | 16.0 | 32.0 | 1:2.00 |
| 38 | `strategy_38_audcad_h4_macro_retest.py` | AUDCAD | H4 Macro S/R Retest (Trend Pullback) | 15.0 | 30.0 | 1:2.00 |
| 39 | `strategy_39_audcad_h4_macro_retest.py` | AUDCAD | H4 Macro S/R Retest (Trend Pullback) | 14.0 | 28.0 | 1:2.00 |
| 40 | `strategy_40_audcad_h4_macro_retest.py` | AUDCAD | H4 Macro S/R Retest (Trend Pullback) | 15.0 | 30.0 | 1:2.00 |
| 41 | `strategy_41_usdjpy_asian_sweep.py` | USDJPY | Asian Session Sweep (Mean Reversion) | 10.0 | 25.0 | 1:2.50 |
| 42 | `strategy_42_usdjpy_asian_sweep.py` | USDJPY | Asian Session Sweep (Mean Reversion) | 10.0 | 25.0 | 1:2.50 |
| 43 | `strategy_43_usdjpy_asian_sweep.py` | USDJPY | Asian Session Sweep (Mean Reversion) | 12.0 | 30.0 | 1:2.50 |
| 44 | `strategy_44_usdjpy_micro_trend_breakout.py` | USDJPY | Intraday Micro-Trend Breakout (Directional Momentum) | 10.0 | 20.0 | 1:2.00 |
| 45 | `strategy_45_usdjpy_asian_sweep.py` | USDJPY | Asian Session Sweep (Mean Reversion) | 11.0 | 27.5 | 1:2.50 |
| 46 | `strategy_46_usdjpy_asian_sweep.py` | USDJPY | Asian Session Sweep (Mean Reversion) | 12.0 | 30.0 | 1:2.50 |
| 47 | `strategy_47_usdjpy_asian_sweep.py` | USDJPY | Asian Session Sweep (Mean Reversion) | 11.0 | 22.0 | 1:2.00 |
| 48 | `strategy_48_usdjpy_asian_sweep.py` | USDJPY | Asian Session Sweep (Mean Reversion) | 18.0 | 45.0 | 1:2.50 |
| 49 | `strategy_49_eurnzd_h4_macro_retest.py` | EURNZD | H4 Macro S/R Retest (Trend Pullback) | 30.0 | 65.0 | 1:2.17 |
| 50 | `strategy_50_eurnzd_h4_macro_retest.py` | EURNZD | H4 Macro S/R Retest (Trend Pullback) | 30.0 | 85.0 | 1:2.83 |
| 51 | `strategy_51_eurnzd_h4_macro_retest.py` | EURNZD | H4 Macro S/R Retest (Trend Pullback) | 30.0 | 60.0 | 1:2.00 |
| 52 | `strategy_52_eurnzd_h4_macro_retest.py` | EURNZD | H4 Macro S/R Retest (Trend Pullback) | 24.0 | 85.0 | 1:3.54 |
| 53 | `strategy_53_eurnzd_h4_macro_retest.py` | EURNZD | H4 Macro S/R Retest (Trend Pullback) | 30.0 | 80.0 | 1:2.67 |
| 54 | `strategy_54_eurnzd_h4_macro_retest.py` | EURNZD | H4 Macro S/R Retest (Trend Pullback) | 30.0 | 60.0 | 1:2.00 |
| 55 | `strategy_55_eurnzd_h4_macro_retest.py` | EURNZD | H4 Macro S/R Retest (Trend Pullback) | 24.0 | 80.0 | 1:3.33 |
| 56 | `strategy_56_eurnzd_h4_macro_retest.py` | EURNZD | H4 Macro S/R Retest (Trend Pullback) | 24.0 | 85.0 | 1:3.54 |
| 57 | `strategy_57_gbpjpy_h4_macro_retest.py` | GBPJPY | H4 Macro S/R Retest (Trend Pullback) | 30.0 | 70.0 | 1:2.33 |
| 58 | `strategy_58_gbpjpy_h4_macro_retest.py` | GBPJPY | H4 Macro S/R Retest (Trend Pullback) | 32.0 | 75.0 | 1:2.34 |
| 59 | `strategy_59_gbpjpy_h4_macro_retest.py` | GBPJPY | H4 Macro S/R Retest (Trend Pullback) | 30.0 | 65.0 | 1:2.17 |
| 60 | `strategy_60_nzdcad_ny_sweep.py` | NZDCAD | NY Session Sweep (Mean Reversion) | 26.0 | 72.0 | 1:2.77 |
| 61 | `strategy_61_nzdcad_ny_sweep.py` | NZDCAD | NY Session Sweep (Mean Reversion) | 26.0 | 80.0 | 1:3.08 |
| 62 | `strategy_62_eurgbp_london_sweep.py` | EURGBP | London Session Sweep (Mean Reversion) | 11.0 | 22.0 | 1:2.00 |
| 63 | `strategy_63_usdjpy_asian_sweep.py` | USDJPY | Asian Session Sweep (Mean Reversion) | 10.0 | 25.0 | 1:2.50 |
| 64 | `strategy_64_usdjpy_asian_sweep.py` | USDJPY | Asian Session Sweep (Mean Reversion) | 12.0 | 30.0 | 1:2.50 |
| 65 | `strategy_65_eurgbp_london_sweep.py` | EURGBP | London Session Sweep (Mean Reversion) | 12.0 | 24.0 | 1:2.00 |
| 66 | `strategy_66_eurgbp_london_sweep.py` | EURGBP | London Session Sweep (Mean Reversion) | 12.0 | 22.0 | 1:1.83 |
| 67 | `strategy_67_eurgbp_london_sweep.py` | EURGBP | London Session Sweep (Mean Reversion) | 11.0 | 22.0 | 1:2.00 |
| 68 | `strategy_68_eurgbp_london_sweep.py` | EURGBP | London Session Sweep (Mean Reversion) | 12.0 | 24.0 | 1:2.00 |
| 69 | `strategy_69_eurgbp_london_sweep.py` | EURGBP | London Session Sweep (Mean Reversion) | 12.0 | 24.0 | 1:2.00 |
| 70 | `strategy_70_eurgbp_london_sweep.py` | EURGBP | London Session Sweep (Mean Reversion) | 12.0 | 22.0 | 1:1.83 |
| 71 | `strategy_71_eurgbp_london_sweep.py` | EURGBP | London Session Sweep (Mean Reversion) | 12.0 | 24.0 | 1:2.00 |
| 72 | `strategy_72_eurgbp_london_sweep.py` | EURGBP | London Session Sweep (Mean Reversion) | 12.0 | 24.0 | 1:2.00 |
| 73 | `strategy_73_nzdcad_ny_sweep.py` | NZDCAD | NY Session Sweep (Mean Reversion) | 26.0 | 80.0 | 1:3.08 |
| 74 | `strategy_74_nzdcad_ny_sweep.py` | NZDCAD | NY Session Sweep (Mean Reversion) | 26.0 | 70.0 | 1:2.69 |
| 75 | `strategy_75_nzdcad_ny_sweep.py` | NZDCAD | NY Session Sweep (Mean Reversion) | 26.0 | 80.0 | 1:3.08 |
| 76 | `strategy_76_nzdcad_ny_sweep.py` | NZDCAD | NY Session Sweep (Mean Reversion) | 26.0 | 75.0 | 1:2.88 |
| 77 | `strategy_77_nzdcad_ny_sweep.py` | NZDCAD | NY Session Sweep (Mean Reversion) | 26.0 | 80.0 | 1:3.08 |
| 78 | `strategy_78_nzdcad_ny_sweep.py` | NZDCAD | NY Session Sweep (Mean Reversion) | 26.0 | 75.0 | 1:2.88 |
| 79 | `strategy_79_nzdcad_ny_sweep.py` | NZDCAD | NY Session Sweep (Mean Reversion) | 26.0 | 70.0 | 1:2.69 |
| 80 | `strategy_80_nzdcad_ny_sweep.py` | NZDCAD | NY Session Sweep (Mean Reversion) | 26.0 | 85.0 | 1:3.27 |
| 81 | `strategy_81_usdjpy_micro_trend_pullback.py` | USDJPY | Micro-Trend EMA Pullback (Trend Continuation) | 14.0 | 28.0 | 1:2.00 |
| 82 | `strategy_82_usdjpy_micro_trend_breakout.py` | USDJPY | Intraday Micro-Trend Breakout (Directional Momentum) | 10.0 | 20.0 | 1:2.00 |
| 83 | `strategy_83_usdjpy_micro_trend_breakout.py` | USDJPY | Intraday Micro-Trend Breakout (Directional Momentum) | 14.0 | 28.0 | 1:2.00 |
| 84 | `strategy_84_usdjpy_micro_trend_breakout.py` | USDJPY | Intraday Micro-Trend Breakout (Directional Momentum) | 10.0 | 18.0 | 1:1.80 |
| 85 | `strategy_85_usdjpy_micro_trend_breakout.py` | USDJPY | Intraday Micro-Trend Breakout (Directional Momentum) | 10.0 | 18.0 | 1:1.80 |
| 86 | `strategy_86_usdjpy_micro_trend_breakout.py` | USDJPY | Intraday Micro-Trend Breakout (Directional Momentum) | 10.0 | 20.0 | 1:2.00 |
| 87 | `strategy_87_eurgbp_london_sweep.py` | EURGBP | London Session Sweep (Mean Reversion) | 14.0 | 28.0 | 1:2.00 |
| 88 | `strategy_88_eurusd_micro_trend_breakout.py` | EURUSD | Intraday Micro-Trend Breakout (Directional Momentum) | 10.0 | 20.0 | 1:2.00 |
| 89 | `strategy_89_eurgbp_london_sweep.py` | EURGBP | London Session Sweep (Mean Reversion) | 15.0 | 30.0 | 1:2.00 |
| 90 | `strategy_90_eurgbp_london_sweep.py` | EURGBP | London Session Sweep (Mean Reversion) | 12.0 | 24.0 | 1:2.00 |
| 91 | `strategy_91_usdjpy_donchian_micro_trend.py` | USDJPY | Donchian Micro-Trend Channel Breakout | 14.0 | 28.0 | 1:2.00 |
| 92 | `strategy_92_usdjpy_micro_trend_breakout.py` | USDJPY | Intraday Micro-Trend Breakout (Directional Momentum) | 14.0 | 28.0 | 1:2.00 |
| 93 | `strategy_93_gbpjpy_micro_trend_breakout.py` | GBPJPY | Intraday Micro-Trend Breakout (Directional Momentum) | 20.0 | 42.0 | 1:2.10 |
| 94 | `strategy_94_usdjpy_micro_trend_breakout.py` | USDJPY | Intraday Micro-Trend Breakout (Directional Momentum) | 14.0 | 28.0 | 1:2.00 |
| 95 | `strategy_95_usdjpy_micro_trend_breakout.py` | USDJPY | Intraday Micro-Trend Breakout (Directional Momentum) | 12.0 | 24.0 | 1:2.00 |
| 96 | `strategy_96_gbpjpy_micro_trend_breakout.py` | GBPJPY | Intraday Micro-Trend Breakout (Directional Momentum) | 18.0 | 36.0 | 1:2.00 |
| 97 | `strategy_97_usdjpy_micro_trend_breakout.py` | USDJPY | Intraday Micro-Trend Breakout (Directional Momentum) | 12.0 | 24.0 | 1:2.00 |
| 98 | `strategy_98_usdjpy_micro_trend_breakout.py` | USDJPY | Intraday Micro-Trend Breakout (Directional Momentum) | 12.0 | 24.0 | 1:2.00 |
| 99 | `strategy_99_usdjpy_micro_trend_breakout.py` | USDJPY | Intraday Micro-Trend Breakout (Directional Momentum) | 9.0 | 18.0 | 1:2.00 |
| 100 | `strategy_100_usdjpy_micro_trend_breakout.py` | USDJPY | Intraday Micro-Trend Breakout (Directional Momentum) | 11.0 | 22.0 | 1:2.00 |

## Risk & Execution Governance
- **Trade Controller**: `trade_controller.py`
- **Max Concurrent Trades**: 20
- **Timeout Threshold**: 4 hours for stagnant trades
- **Max Holding Period**: 24 hours absolute hard stop
- **Room-Making Rules**:
  1. Free slots (< 20) -> Direct acceptance.
  2. Slot full (20/20) + incoming signal -> Prune timed-out losing trades (Tier 1) or deepest drawdown losing trades (Tier 2).
  3. Active winning trades (> 0 pips) are strictly shielded from premature closure.
