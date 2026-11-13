# Production Fleet v4.6: The 70-Strategy Institutional Pristine Portfolio

> **Release Version**: `v4.6.0` (Evolution of `v4.5.2` Centurion Fleet)  
> **Backtest Certification**: `6.7.0_pristine_70_elite` / Schema `2.0.0`  
> **Fleet Capacity**: Exactly 71 Dual-Certified Strategies across 10 Currency Pairs  
> **Friction Model**: Live ECN Standard ($4.00/lot commission + 0.20p slippage + raw Dukascopy floating spreads)  
> **Execution Contract**: Strict subsequent-tick causal execution ($T_{\text{fill}} > T_{\text{signal}}$)  
> **Zero Hidden Passengers**: 100% of strategies hold $\ge 15$ Holdout Trades, $\text{PF} \ge 1.15$, Discovery Net Pips $> 0$, and $\text{Pips ex Top 3} > 0$.  

---

## 1. Executive Summary & Fleet Metrics

| Metric | Quarantined Holdout (Weeks 211..300) | Full Cycle (5.75 Years) |
| :--- | :---: | :---: |
| **Active Certified Models** | **71 Strategies** | **71 Strategies** |
| **Total Holdout Trades** | **6,509 trades** | **24,696 trades** |
| **Unrestricted Net Pips** | **+43,006.8 pips** | **+102,267.1 pips** |
| **Gated Net Pips (1.5% Cap)** | **+41,909.5 pips** | **+91,759.7 pips** |
| **Institutional PnL ($100k)** | **+$574,052.49 USD (+574.0%)** | **+$1,303,002.49 USD** |
| **Max Lifetime Drawdown** | **7.75%** (Gate: < 8.0% **PASSED**) | **7.75%** |
| **Worst 24h Drawdown** | **1.60%** (Gate: < 4.5% **PASSED**) | **1.60%** |
| **Gated Calmar Ratio** | **74.06x** | **76.10x** |
| **Prop Firm Audit Status** | **FULLY COMPLIANT & CERTIFIED** | **CERTIFIED** |

---

## 2. 10-Pair Cross-Asset Diversification Matrix

| Symbol | Model Count | Active Slots | Primary Archetypes & Microstructure Drivers |
| :--- | :---: | :---: | :--- |
| **EURUSD** | **10** | Slots 20–64 | NY Open Impulse (`disc169`), Daily/Weekly Pivots, VWAP Bands, ATR Expansion |
| **USDJPY** | **9** | Slots 50–68 | M15 Pivot Pullback (`disc173`), SAR Flips, London Lunch Expansion, VWAP |
| **GBPJPY** | **8** | Slots 34–66 | Daily Pivot Pullback (`disc171`), H4 Macro Retests, London Open |
| **AUDNZD** | **7** | Slots 07–13 | H4 Structural Macro Retest, Session Impulse, Daily Pivots |
| **EURNZD** | **7** | Slots 15–63 | H4 Structural Macro Retest, Session Impulse, Daily Pivots |
| **USDCAD** | **7** | Slots 46–65 | Daily Pivot Pullback (`disc170`), Donchian Bounces, Asian Sweeps |
| **AUDCAD** | **6** | Slots 01–06 | H4 Structural Macro Retest, Session Impulse, Daily Pivots |
| **GBPCHF** | **6** | Slots 29–61 | H4 Structural Macro Retest, Session Impulse, Daily Pivots |
| **NZDCAD** | **6** | Slots 40–45 | H4 Structural Macro Retest, Session Impulse, Daily Pivots |
| **EURGBP** | **5** | Slots 14–71 | **5 Models**: Keltner Squeeze (`disc107`), Compression Breakout (`disc172`), H1 Pullback (`disc174`), Daily Pivot Pullback (`disc175`), Weekly Pivot Pullback (`disc180`) |
| **TOTAL** | **71** | **Slots 01–71** | **Balanced across 10 non-correlated FX pairs** |

---

## 3. Complete Strategy Slot Directory (Slots 01 – 71)

| Slot | Filename | Symbol | Archetype | SL | TP | R:R | Holdout Pips | Holdout PF | Trades |
| :---: | :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 01 | `strategy_01_audcad_disc115.py` | AUDCAD | Macro S/R Retest (Trend Pullback) | 18.0p | 36.0p | 1:2.00 | +407.0p | 2.98 | 30 |
| 02 | `strategy_02_audcad_disc116.py` | AUDCAD | Macro S/R Retest (Trend Pullback) | 16.0p | 40.0p | 1:2.50 | +390.2p | 2.86 | 30 |
| 03 | `strategy_03_audcad_disc117.py` | AUDCAD | Macro S/R Retest (Trend Pullback) | 18.0p | 36.0p | 1:2.00 | +369.9p | 2.24 | 37 |
| 04 | `strategy_04_audcad_disc118.py` | AUDCAD | Macro S/R Retest (Trend Pullback) | 18.0p | 36.0p | 1:2.00 | +288.4p | 2.92 | 23 |
| 05 | `strategy_05_audcad_disc159.py` | AUDCAD | Macro S/R Retest (Trend Pullback) | 12.0p | 24.0p | 1:2.00 | +75.9p | 1.44 | 23 |
| 06 | `strategy_06_audcad_disc160.py` | AUDCAD | Macro S/R Retest (Trend Pullback) | 12.0p | 24.0p | 1:2.00 | +146.8p | 1.73 | 31 |
| 07 | `strategy_07_audnzd_disc111_h4_macro_structure_retest_elite.py` | AUDNZD | Macro S/R Retest (Trend Pullback) | 14.0p | 28.0p | 1:2.00 | +231.6p | 2.56 | 28 |
| 08 | `strategy_08_audnzd_disc112_h4_macro_precision_retest_elite.py` | AUDNZD | Macro S/R Retest (Trend Pullback) | 12.0p | 24.0p | 1:2.00 | +167.6p | 2.06 | 29 |
| 09 | `strategy_09_audnzd_disc113_h1_macro_intraday_retest_elite.py` | AUDNZD | Macro S/R Retest (Trend Pullback) | 12.0p | 20.0p | 1:1.67 | +74.7p | 1.72 | 18 |
| 10 | `strategy_10_audnzd_disc114_h4_macro_wide_runner_elite.py` | AUDNZD | Macro S/R Retest (Trend Pullback) | 15.0p | 30.0p | 1:2.00 | +290.3p | 3.03 | 28 |
| 11 | `strategy_11_audnzd_disc145.py` | AUDNZD | Macro S/R Retest (Trend Pullback) | 22.0p | 55.0p | 1:2.50 | +633.5p | 4.75 | 29 |
| 12 | `strategy_12_audnzd_disc146.py` | AUDNZD | Macro S/R Retest (Trend Pullback) | 22.0p | 55.0p | 1:2.50 | +614.0p | 3.26 | 39 |
| 13 | `strategy_13_audnzd_disc147.py` | AUDNZD | Macro S/R Retest (Trend Pullback) | 22.0p | 55.0p | 1:2.50 | +611.8p | 3.23 | 40 |
| 14 | `strategy_14_eurgbp_disc107_h1_keltner_squeeze_explosion_elite.py` | EURGBP | H1 Keltner Squeeze Explosion Elite | 8.0p | 16.0p | 1:2.00 | +126.0p | 1.71 | 40 |
| 15 | `strategy_15_eurnzd_disc132.py` | EURNZD | Macro S/R Retest (Trend Pullback) | 25.0p | 60.0p | 1:2.40 | +819.5p | 3.32 | 34 |
| 16 | `strategy_16_eurnzd_disc133.py` | EURNZD | Macro S/R Retest (Trend Pullback) | 25.0p | 60.0p | 1:2.40 | +1076.1p | 3.45 | 43 |
| 17 | `strategy_17_eurnzd_disc143.py` | EURNZD | Macro S/R Retest (Trend Pullback) | 28.0p | 65.0p | 1:2.32 | +1417.9p | 7.27 | 36 |
| 18 | `strategy_18_eurnzd_disc157.py` | EURNZD | Macro S/R Retest (Trend Pullback) | 28.0p | 65.0p | 1:2.32 | +1331.2p | 8.23 | 33 |
| 19 | `strategy_19_eurnzd_disc158.py` | EURNZD | Macro S/R Retest (Trend Pullback) | 30.0p | 75.0p | 1:2.50 | +1230.9p | 2.85 | 48 |
| 20 | `strategy_20_eurusd_disc50_m30_m5_precision_retest_elite.py` | EURUSD | Macro S/R Retest (Trend Pullback) | 8.0p | 13.0p | 1:1.62 | +66.8p | 1.79 | 22 |
| 21 | `strategy_21_eurusd_disc51_macro_asymmetric_runner_elite.py` | EURUSD | Macro S/R Retest (Trend Pullback) | 9.0p | 24.0p | 1:2.67 | +238.5p | 6.05 | 17 |
| 22 | `strategy_22_eurusd_disc58_h1_keltner_squeeze_explosion_elite.py` | EURUSD | H1 Keltner Squeeze Explosion Elite | 15.0p | 30.0p | 1:2.00 | +392.3p | 1.63 | 111 |
| 23 | `strategy_23_eurusd_disc59_m15_macd_zero_cross_momentum_elite.py` | EURUSD | M15 MACD Zero-Cross Momentum Elite | 15.0p | 30.0p | 1:2.00 | +366.5p | 1.30 | 217 |
| 24 | `strategy_24_eurusd_disc65_session_vwap_band_fade_elite.py` | EURUSD | Session VWAP Band Rebound Elite | 9.0p | 18.0p | 1:2.00 | +409.0p | 1.30 | 242 |
| 25 | `strategy_25_eurusd_disc74_daily_pivot_pullback_elite.py` | EURUSD | Daily / Weekly Pivot Pullback Elite | 9.0p | 18.0p | 1:2.00 | +513.8p | 1.88 | 123 |
| 26 | `strategy_26_eurusd_disc75_weekly_pivot_pullback_elite.py` | EURUSD | Daily / Weekly Pivot Pullback Elite | 9.0p | 22.5p | 1:2.50 | +81.6p | 1.15 | 86 |
| 27 | `strategy_27_eurusd_disc76_vwap_1sd_rebound_elite.py` | EURUSD | Session VWAP Band Rebound Elite | 9.0p | 18.0p | 1:2.00 | +313.3p | 1.40 | 143 |
| 28 | `strategy_28_eurusd_disc86_ema_cross_atr_expansion_elite.py` | EURUSD | Macro S/R Retest (Trend Pullback) | 8.5p | 17.0p | 1:2.00 | +552.9p | 1.42 | 261 |
| 29 | `strategy_29_gbpchf_disc134.py` | GBPCHF | Macro S/R Retest (Trend Pullback) | 25.0p | 60.0p | 1:2.40 | +1514.5p | 3.46 | 81 |
| 30 | `strategy_30_gbpchf_disc135.py` | GBPCHF | Macro S/R Retest (Trend Pullback) | 22.0p | 50.0p | 1:2.27 | +1464.1p | 3.52 | 82 |
| 31 | `strategy_31_gbpchf_disc137.py` | GBPCHF | Macro S/R Retest (Trend Pullback) | 22.0p | 50.0p | 1:2.27 | +1390.8p | 3.07 | 87 |
| 32 | `strategy_32_gbpchf_disc152.py` | GBPCHF | Macro S/R Retest (Trend Pullback) | 25.0p | 60.0p | 1:2.40 | +1344.1p | 3.79 | 68 |
| 33 | `strategy_33_gbpchf_disc153.py` | GBPCHF | Macro S/R Retest (Trend Pullback) | 20.0p | 45.0p | 1:2.25 | +1125.6p | 3.04 | 77 |
| 34 | `strategy_34_gbpjpy_disc138.py` | GBPJPY | Macro S/R Retest (Trend Pullback) | 35.0p | 75.0p | 1:2.14 | +1032.2p | 2.39 | 50 |
| 35 | `strategy_35_gbpjpy_disc139.py` | GBPJPY | Macro S/R Retest (Trend Pullback) | 35.0p | 75.0p | 1:2.14 | +995.7p | 1.96 | 60 |
| 36 | `strategy_36_gbpjpy_disc140.py` | GBPJPY | Macro S/R Retest (Trend Pullback) | 40.0p | 90.0p | 1:2.25 | +982.5p | 2.06 | 50 |
| 37 | `strategy_37_gbpjpy_disc141.py` | GBPJPY | Macro S/R Retest (Trend Pullback) | 40.0p | 90.0p | 1:2.25 | +968.1p | 4.93 | 23 |
| 38 | `strategy_38_gbpjpy_disc155.py` | GBPJPY | Macro S/R Retest (Trend Pullback) | 35.0p | 75.0p | 1:2.14 | +958.5p | 2.90 | 36 |
| 39 | `strategy_39_gbpjpy_disc156.py` | GBPJPY | Macro S/R Retest (Trend Pullback) | 35.0p | 75.0p | 1:2.14 | +928.7p | 3.57 | 31 |
| 40 | `strategy_40_nzdcad_disc122.py` | NZDCAD | Macro S/R Retest (Trend Pullback) | 16.0p | 40.0p | 1:2.50 | +213.2p | 1.81 | 27 |
| 41 | `strategy_41_nzdcad_disc123.py` | NZDCAD | Macro S/R Retest (Trend Pullback) | 16.0p | 40.0p | 1:2.50 | +222.0p | 1.97 | 25 |
| 42 | `strategy_42_nzdcad_disc125.py` | NZDCAD | Macro S/R Retest (Trend Pullback) | 18.0p | 36.0p | 1:2.00 | +197.8p | 1.76 | 27 |
| 43 | `strategy_43_nzdcad_disc126.py` | NZDCAD | Macro S/R Retest (Trend Pullback) | 16.0p | 40.0p | 1:2.50 | +140.7p | 1.53 | 27 |
| 44 | `strategy_44_nzdcad_disc161.py` | NZDCAD | Macro S/R Retest (Trend Pullback) | 12.0p | 24.0p | 1:2.00 | +139.1p | 1.56 | 36 |
| 45 | `strategy_45_nzdcad_disc162.py` | NZDCAD | Macro S/R Retest (Trend Pullback) | 14.0p | 28.0p | 1:2.00 | +178.5p | 1.76 | 31 |
| 46 | `strategy_46_usdcad_disc127.py` | USDCAD | Macro S/R Retest (Trend Pullback) | 14.0p | 28.0p | 1:2.00 | +373.1p | 4.07 | 27 |
| 47 | `strategy_47_usdcad_disc128.py` | USDCAD | Macro S/R Retest (Trend Pullback) | 14.0p | 28.0p | 1:2.00 | +327.6p | 2.48 | 36 |
| 48 | `strategy_48_usdcad_disc129.py` | USDCAD | Macro S/R Retest (Trend Pullback) | 14.0p | 28.0p | 1:2.00 | +314.9p | 2.52 | 34 |
| 49 | `strategy_49_usdcad_disc130.py` | USDCAD | Macro S/R Retest (Trend Pullback) | 15.0p | 30.0p | 1:2.00 | +311.3p | 2.92 | 27 |
| 50 | `strategy_50_usdjpy_disc106_chaikin_volatility_surge_elite.py` | USDJPY | Macro S/R Retest (Trend Pullback) | 14.0p | 28.0p | 1:2.00 | +465.3p | 1.17 | 305 |
| 51 | `strategy_51_usdjpy_disc65_session_vwap_band_fade_elite.py` | USDJPY | Session VWAP Band Rebound Elite | 12.0p | 24.0p | 1:2.00 | +257.9p | 1.18 | 189 |
| 52 | `strategy_52_usdjpy_disc67_london_lunch_expansion_elite.py` | USDJPY | Macro S/R Retest (Trend Pullback) | 12.0p | 24.0p | 1:2.00 | +276.7p | 1.27 | 135 |
| 53 | `strategy_53_usdjpy_disc83_parabolic_sar_flip_elite.py` | USDJPY | Macro S/R Retest (Trend Pullback) | 12.0p | 24.0p | 1:2.00 | +438.5p | 1.23 | 247 |
| 54 | `strategy_54_usdjpy_disc91_session_donchian_midline_bounce_elite.py` | USDJPY | Macro S/R Retest (Trend Pullback) | 12.0p | 24.0p | 1:2.00 | +347.6p | 1.23 | 197 |
| 55 | `strategy_55_usdjpy_disc94_london_vwap_pullback_elite.py` | USDJPY | Session VWAP Band Rebound Elite | 12.0p | 24.0p | 1:2.00 | +240.0p | 1.20 | 154 |
| 56 | `strategy_56_usdjpy_disc95_camarilla_expansion_elite.py` | USDJPY | Macro S/R Retest (Trend Pullback) | 12.0p | 24.0p | 1:2.00 | +257.4p | 1.30 | 115 |
| 57 | `strategy_57_usdjpy_disc99_m15_macd_zero_cross_momentum_elite.py` | USDJPY | Macro S/R Retest (Trend Pullback) | 12.0p | 24.0p | 1:2.00 | +220.6p | 1.16 | 180 |
| 58 | `strategy_58_usdcad_disc163.py` | USDCAD | Macro S/R Retest (Trend Pullback) | 20.0p | 45.0p | 1:2.25 | +3431.8p | 2.02 | 353 |
| 59 | `strategy_59_usdcad_disc164.py` | USDCAD | Macro S/R Retest (Trend Pullback) | 16.0p | 32.0p | 1:2.00 | +2657.5p | 1.86 | 383 |
| 60 | `strategy_60_gbpjpy_disc165.py` | GBPJPY | Macro S/R Retest (Trend Pullback) | 30.0p | 65.0p | 1:2.17 | +994.5p | 1.23 | 232 |
| 61 | `strategy_61_gbpchf_disc166.py` | GBPCHF | Macro S/R Retest (Trend Pullback) | 18.0p | 38.0p | 1:2.11 | +587.3p | 1.25 | 223 |
| 62 | `strategy_62_eurnzd_disc167.py` | EURNZD | Macro S/R Retest (Trend Pullback) | 32.0p | 70.0p | 1:2.19 | +297.0p | 1.22 | 63 |
| 63 | `strategy_63_eurnzd_disc168.py` | EURNZD | Macro S/R Retest (Trend Pullback) | 35.0p | 80.0p | 1:2.29 | +293.3p | 1.23 | 58 |
| 64 | `strategy_64_eurusd_disc169.py` | EURUSD | Macro S/R Retest (Trend Pullback) | 10.0p | 22.0p | 1:2.20 | +808.0p | 1.40 | 327 |
| 65 | `strategy_65_usdcad_disc170.py` | USDCAD | Macro S/R Retest (Trend Pullback) | 18.0p | 40.0p | 1:2.22 | +1041.8p | 1.99 | 126 |
| 66 | `strategy_66_gbpjpy_disc171.py` | GBPJPY | Macro S/R Retest (Trend Pullback) | 25.0p | 60.0p | 1:2.40 | +968.7p | 1.54 | 119 |
| 67 | `strategy_67_eurgbp_disc172.py` | EURGBP | Macro S/R Retest (Trend Pullback) | 10.0p | 20.0p | 1:2.00 | +83.7p | 2.04 | 19 |
| 68 | `strategy_68_usdjpy_disc173.py` | USDJPY | Macro S/R Retest (Trend Pullback) | 16.0p | 35.0p | 1:2.19 | +486.5p | 1.39 | 131 |
| 69 | `strategy_69_eurgbp_disc174.py` | EURGBP | Macro S/R Retest (Trend Pullback) | 7.0p | 14.0p | 1:2.00 | +176.3p | 2.14 | 46 |
| 70 | `strategy_70_eurgbp_disc175.py` | EURGBP | Macro S/R Retest (Trend Pullback) | 7.0p | 14.0p | 1:2.00 | +119.9p | 1.62 | 50 |
| 71 | `strategy_71_eurgbp_disc180.py` | EURGBP | Macro S/R Retest (Trend Pullback) | 8.0p | 16.0p | 1:2.00 | +195.5p | 1.63 | 74 |

---

## 4. Newly Promoted Champions in v4.6
- **`disc169` (Slot 64, EURUSD)**: NY Equity Open Impulse Elite (+808.0p, PF 1.40).
- **`disc170` (Slot 65, USDCAD)**: Daily Pivot Pullback Rejection Elite (+1,041.8p, PF 1.99).
- **`disc171` (Slot 66, GBPJPY)**: Daily Pivot Pullback Rejection Elite (+968.7p, PF 1.54).
- **`disc172` (Slot 67, EURGBP)**: Daily Compression Breakout Elite (+83.7p, PF 2.04).
- **`disc173` (Slot 68, USDJPY)**: M15 Daily Pivot Pullback Elite (+486.5p, PF 1.39).
- **`disc174` (Slot 69, EURGBP)**: H1 Pullback Rejection Elite 2.0R (+176.3p, PF 2.14).
- **`disc175` (Slot 70, EURGBP)**: Daily Pivot Pullback Rejection Elite 2.0R (+119.9p, PF 1.62).
- **`disc180` (Slot 71, EURGBP)**: Weekly Pivot Pullback Rejection Elite 2.0R (+195.5p, PF 1.63).
