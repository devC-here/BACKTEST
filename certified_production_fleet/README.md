# Certified Institutional Production Fleet (72 Strategies)

> **Verification Standard**: Institutional Continuous Chronological Split (In-Sample: Weeks 1..210, Quarantined Holdout: Weeks 211..300).  
> **Pricing Friction**: Mandatory \$4.00/lot commission + 0.20p adverse entry slippage + raw floating Dukascopy spreads.  
> **Execution Contract**: Strict subsequent-tick causal execution ($T_{\text{fill}} > T_{\text{signal}}$). Zero same-tick fills permitted.  
> **Warmup**: 100,000-tick rolling prior-week causal warmup with `min_entry_timestamp` gate.  
> **Data Quality Gate**: Minimum 80-hour trading span, 15,000 minimum ticks, zero inverted quotes, zero non-positive prices.

---

## 1. Portfolio Performance Summary

| Metric | Quarantined Holdout (Weeks 211..300) | In-Sample Discovery (Weeks 1..210) | Combined 5-Year Full Cycle (Weeks 1..300) |
|:---|:---:|:---:|:---:|
| **Net PnL (Pips)** | **+11,723.3 pips** | **+31,037.4 pips** | **+42,760.7 pips** |
| **Total Executed Trades** | **9,726 trades** | **18,452 trades** | **28,178 trades** |
| **Portfolio Profit Factor** | **1.26** | **1.29** | **1.28** |
| **Average Expectancy** | **+1.21 pips/trade** | **+1.68 pips/trade** | **+1.52 pips/trade** |
| **Active Currency Pairs** | **6 Pairs** (EURUSD, USDJPY, USDCAD, AUDNZD, EURGBP, GBPCHF) | **6 Pairs** | **6 Pairs** |
| **Certified Strategies** | **72 Strategies** | **72 Strategies** | **72 Strategies** |

---

## 2. Currency Pair Distribution & Holdout Performance

| Symbol | Certified Strategies | Quarantined Holdout Net Pips | Holdout Trades | Holdout Expectancy | Primary Alpha Drivers |
|:---|:---:|:---:|:---:|:---:|:---|
| **EURUSD** | **29** | **+5,575.1 pips** | 4,210 | +1.32 pips | Session VWAP Bands, Daily/Weekly Pivots, Macro Retests, NY Sweeps, ATR Expansions |
| **USDJPY** | **24** | **+4,422.3 pips** | 3,745 | +1.18 pips | Parabolic SAR Flips, Chaikin Volatility Surges, London Lunch Expansions, ORB Continuations |
| **AUDNZD** | **4** | **+764.2 pips** | 103 | +7.42 pips | H4 Macro Structure & Precision Retests, Wide Runners |
| **USDCAD** | **8** | **+633.4 pips** | 712 | +0.89 pips | Asian Range Liquidity Sweeps (Slots 17..24) |
| **EURGBP** | **4** | **+203.9 pips** | 106 | +1.92 pips | Camarilla Floors, H4/M30 Macro Structure Retests, H1 Keltner Squeeze Explosions |
| **GBPCHF** | **3** | **+124.4 pips** | 262 | +0.47 pips | Asian Session Sweeps (Slots 26, 27, 29) |
| **TOTAL** | **72** | **+11,723.3 pips** | **9,726** | **+1.21 pips** | **Multi-Pair Institutional Edge Engine** |

---

## 3. Fleet Directory Layout

```
certified_production_fleet/
├── CERTIFIED_FLEET_MANIFEST.json        # Canonical JSON manifest with SHA-256 hashes & trade metrics
├── README.md                            # Comprehensive fleet documentation and performance audit
├── eurusd_champions/                    # 23 Multi-Regime EURUSD Champions
│   ├── disc65_session_vwap_band_fade_elite/
│   ├── disc86_ema_cross_atr_expansion_elite/
│   ├── s10_dual_ema_pullback_elite/
│   ├── disc59_m15_macd_zero_cross_momentum_elite/
│   ├── disc82_chaikin_volatility_surge_elite/
│   ├── disc74_daily_pivot_pullback_elite/
│   ├── disc76_vwap_1sd_rebound_elite/
│   ├── disc75_weekly_pivot_pullback_elite/
│   ├── disc58_h1_keltner_squeeze_explosion_elite/
│   ├── disc80_double_swing_neckline_retest_elite/
│   ├── disc51_macro_asymmetric_runner_elite/
│   ├── disc49_intraday_macro_retest_elite/
│   ├── disc42_macro_structure_h4_m30_elite/
│   ├── disc79_london_open_15m_expansion_elite/
│   ├── disc77_keltner_midline_expansion_elite/
│   ├── disc85_session_open_impulse_elite/
│   ├── disc71_asian_hl_retest_elite/
│   ├── disc50_m30_m5_precision_retest_elite/
│   ├── disc38_london_midday_macd_acceleration_elite/
│   ├── disc41_thursday_macro_trend_ignition_elite/
│   ├── disc03_ny_open_overlap_elite/
│   ├── disc78_three_bar_inside_confirmation_elite/
│   └── high_frequency_m15_scalper_elite/
├── eurusd_ny_sweeps/                    # 6 EURUSD New York Session Sweeps
│   ├── strategy_09_eurusd_ny_sweep_fade.py
│   ├── strategy_10_eurusd_ny_sweep_fade.py
│   ├── strategy_11_eurusd_ny_sweep_fade.py
│   ├── strategy_12_eurusd_ny_sweep_fade.py
│   ├── strategy_15_eurusd_ny_sweep_fade.py
│   └── strategy_16_eurusd_ny_sweep_fade.py
├── usdjpy_champions/                    # 24 USDJPY Champions
│   ├── disc83_parabolic_sar_flip_elite/
│   ├── disc106_chaikin_volatility_surge_elite/
│   ├── disc77_keltner_midline_expansion_elite/
│   ├── disc65_session_vwap_band_fade_elite/
│   ├── disc87_pivot_r1_s1_breakout_elite/
│   ├── disc90_opening_range_impulse_expansion_elite/
│   ├── disc84_london_30m_orb_continuation_elite/
│   ├── disc68_cmf_momentum_expansion_elite/
│   ├── disc82_chaikin_volatility_surge_elite/
│   ├── disc91_session_donchian_midline_bounce_elite/
│   ├── disc67_london_lunch_expansion_elite/
│   ├── disc95_camarilla_expansion_elite/
│   ├── disc94_london_vwap_pullback_elite/
│   ├── disc99_m15_macd_zero_cross_momentum_elite/
│   ├── disc85_session_open_impulse_elite/
│   ├── disc102_keltner_midline_expansion_elite/
│   ├── disc89_ny_london_vwap_ignition_elite/
│   ├── disc100_double_structure_neckline_breakout_elite/
│   ├── disc50_m30_m5_precision_retest_elite/
│   ├── disc51_macro_asymmetric_runner_elite/
│   ├── disc49_intraday_macro_retest_elite/
│   ├── disc71_asian_hl_retest_elite/
│   ├── disc80_double_swing_neckline_retest_elite/
│   └── disc96_cardwell_rsi50_momentum_bounce_elite/
├── usdcad_asian_sweeps/                 # 8 USDCAD Asian Session Sweeps
│   ├── strategy_17_usdcad_asian_sweep.py ... strategy_24_usdcad_asian_sweep.py
├── audnzd_champions/                    # 4 AUDNZD H4 Macro Champions
│   ├── disc111_h4_macro_structure_retest_elite/
│   ├── disc112_h4_macro_precision_retest_elite/
│   ├── disc113_h1_macro_intraday_retest_elite/
│   └── disc114_h4_macro_wide_runner_elite/
├── eurgbp_champions/                    # 4 EURGBP Champions
│   ├── disc42_macro_structure_h4_m30_elite/
│   ├── disc48_camarilla_floor_elite/
│   ├── disc107_h1_keltner_squeeze_explosion_elite/
│   └── disc108_macro_asymmetric_runner_elite/
└── gbpchf_champions/                    # 3 GBPCHF Asian Session Sweeps
    ├── strategy_26_gbpchf_asian_sweep/
    ├── strategy_27_gbpchf_asian_sweep/
    └── strategy_29_gbpchf_asian_sweep/
```

---

## 4. How to Verify Any Strategy Individually

Verify any strategy individually against the institutional platform using:

```powershell
# Verify single strategy on continuous holdout (Weeks 211..300)
.venv\Scripts\python.exe scripts/certify_strategy.py --strategy "certified_production_fleet/eurusd_champions/disc65_session_vwap_band_fade_elite/strategy.py" --mode holdout

# Verify single strategy on in-sample discovery (Weeks 1..210)
.venv\Scripts\python.exe scripts/certify_strategy.py --strategy "certified_production_fleet/usdjpy_champions/disc83_parabolic_sar_flip_elite/strategy.py" --mode discovery

# Recompile the master certified manifest
.venv\Scripts\python.exe scripts/generate_certified_fleet_manifest.py
```
