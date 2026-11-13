# 🌳 Project Directory Tree & Turn-by-Turn Changelog
*Automated Living Architecture & Version Control Tracker for BACKTEST Platform*

> [!NOTE]
> This document is automatically maintained by [`scripts/git_checkpoint.py`](file:///d:/PROJECT/TRD/MODELS%20GENERATIONS/BACKTEST/scripts/git_checkpoint.py).
> Every major prompt, optimization, and strategy discovery is committed to Git and logged here with an updated visual directory tree.

---

## 📁 1. Active Project Directory Architecture
```text
BACKTEST/
├── archive_deprecated/ (12 items)
│   ├── alpha_searcher/ (4 items)
│   │   ├── __init__.py
│   │   ├── archetypes.py
│   │   ├── config.py
│   │   └── searcher.py
│   ├── archive_labs/ (26 items)
│   │   ├── collapsed_bfill_fleet/ (71 items)
│   │   ├── legacy_reserve_pre_warmup/ (128 items)
│   │   ├── scratch_archive/ (154 items)
│   │   ├── audit_daily_trade_velocity.py
│   │   ├── audit_time_to_10_percent.py
│   │   ├── discovery_lab_phase5.py
│   │   ├── discovery_lab_phase6.py
│   │   ├── discovery_lab_phase7.py
│   │   ├── discovery_lab_phase8.py
│   │   ├── discovery_lab_phase9.py
│   │   ├── high_winrate_institutional_lab.py
│   │   ├── high_winrate_laboratory.py
│   │   ├── indicator_strategy_lab.py
│   │   ├── multi_indicator_experiment_lab.py
│   │   ├── multi_indicator_experiment_lab_v2.py
│   │   ├── multi_indicator_experiment_lab_v3.py
│   │   ├── multi_indicator_experiment_lab_v4.py
│   │   ├── optimize_all_strategies.py
│   │   ├── optimize_strategy.py
│   │   ├── randomized_live_simulation.py
│   │   ├── run_backtest.py
│   │   ├── test_5_star_prop_fleet.py
│   │   ├── test_elite_fleet_daily_metrics.py
│   │   ├── test_indicator_prop_fleet.py
│   │   ├── test_prop_firm_fleet.py
│   │   └── web_models_institutional_lab.py
│   ├── archive_versions/ (5 items)
│   │   ├── v 4.1/ (2 items)
│   │   ├── v 4.2/ (81 items)
│   │   ├── v 4.3/ (81 items)
│   │   ├── v 4.3_reserve/ (0 items)
│   │   └── v 4.4/ (81 items)
│   ├── discovery_labs/ (71 items)
│   │   ├── [71 files: alpha_discovery_lab_audnzd_cross_specialists.py, alpha_discovery_lab_audnzd_expansion.py, alpha_discovery_lab_audnzd_mean_reversion.py, alpha_discovery_lab_audnzd_peer_fleet.py, alpha_discovery_lab_audnzd_v1.py, ...]
│   ├── legacy_labs/ (8 items)
│   │   ├── backtester/ (9 items)
│   │   ├── edge_lab/ (14 items)
│   │   ├── eurusd_1y_data/ (1 items)
│   │   ├── h1_base_scanners/ (71 items)
│   │   ├── production_models/ (3 items)
│   │   ├── renko_lab/ (13 items)
│   │   ├── supertrend_lab/ (9 items)
│   │   └── top_15_strategies/ (28 items)
│   ├── pruned_low_edge_candidates/ (10 items)
│   │   ├── disc108_macro_asymmetric_runner_elite/ (2 items)
│   │   ├── disc77_keltner_midline_expansion_elite/ (2 items)
│   │   ├── disc78_three_bar_inside_confirmation_elite/ (2 items)
│   │   ├── eurgbp_london_sweeps/ (1 items)
│   │   ├── eurusd_champions/ (5 items)
│   │   ├── eurusd_ny_sweeps/ (2 items)
│   │   ├── gbpchf_champions/ (1 items)
│   │   ├── high_frequency_m15_scalper_elite/ (2 items)
│   │   ├── s10_dual_ema_pullback_elite/ (2 items)
│   │   └── usdjpy_champions/ (9 items)
│   ├── pruned_low_edge_undersampled/ (12 items)
│   │   ├── disc03_ny_open_overlap_elite/ (1 items)
│   │   ├── disc148/ (1 items)
│   │   ├── disc149/ (1 items)
│   │   ├── disc150/ (1 items)
│   │   ├── disc42_macro_structure_h4_m30_elite/ (2 items)
│   │   ├── disc48_camarilla_floor_elite/ (1 items)
│   │   ├── disc49_intraday_macro_retest_elite/ (2 items)
│   │   ├── disc50_m30_m5_precision_retest_elite/ (2 items)
│   │   ├── disc71_asian_hl_retest_elite/ (1 items)
│   │   ├── disc79_london_open_15m_expansion_elite/ (2 items)
│   │   ├── disc89_ny_london_vwap_ignition_elite/ (1 items)
│   │   └── disc90_opening_range_impulse_expansion_elite/ (1 items)
│   ├── pruned_redundant_duplicates/ (9 items)
│   │   ├── AUDCAD/ (2 items)
│   │   ├── EURNZD/ (2 items)
│   │   ├── EURUSD/ (1 items)
│   │   ├── GBPCHF/ (2 items)
│   │   ├── GBPJPY/ (1 items)
│   │   ├── NZDCAD/ (2 items)
│   │   ├── USDCAD/ (2 items)
│   │   ├── USDJPY/ (1 items)
│   │   └── PRUNED_REDUNDANT_AUDIT_LOG.json
│   ├── redundant_parameter_siblings/ (3 items)
│   │   ├── eurusd_ny_sweeps/ (4 items)
│   │   ├── gbpchf_champions/ (2 items)
│   │   └── usdcad_asian_sweeps/ (6 items)
│   ├── research_scratch/ (17 items)
│   │   ├── audit_regimes_sessions_dollars.py
│   │   ├── inspect_champions_diversity.py
│   │   ├── inspect_cross_screen.py
│   │   ├── print_champions_summary.py
│   │   ├── print_fleet_table.py
│   │   ├── promote_clean_champions.py
│   │   ├── screen_and_promote_to_72.py
│   │   ├── screen_candidate_alpha_pool.py
│   │   ├── search_clean_macro_champions.py
│   │   ├── search_eurnzd_more.py
│   │   ├── select_and_verify_19_champions.py
│   │   ├── select_uncorrelated_champions.py
│   │   ├── test_cross_pairs_macro.py
│   │   ├── test_gbpjpy_candidate.py
│   │   ├── test_selection_diversity.py
│   │   ├── test_usdcad_macro.py
│   │   └── test_v45_candidates.py
│   ├── strategies/ (11 items)
│   │   ├── archive_non_viable/ (75 items)
│   │   ├── archive_unprofitable/ (21 items)
│   │   ├── audnzd_fleet/ (12 items)
│   │   ├── certified_29_champions/ (6 items)
│   │   ├── eurgbp_fleet/ (23 items)
│   │   ├── eurusd_production_fleet/ (45 items)
│   │   ├── new_discoveries/ (37 items)
│   │   ├── qualified_strategies/ (5 items)
│   │   ├── usdcad_fleet/ (7 items)
│   │   ├── usdjpy_fleet/ (41 items)
│   │   └── PROVEN_PROFITABLE_STRATEGIES_DIRECTORY.md
│   └── v 4.5/ (104 items)
│       ├── [104 files: FLEET_MANIFEST.json, README.md, run_portfolio_simulation.py, strategy_01_gbpjpy_h4_macro_retest.py, strategy_02_gbpjpy_h4_macro_retest.py, ...]
├── certified_production_fleet/ (13 items)
│   ├── audcad_champions/ (7 items)
│   │   ├── disc115/ (1 items)
│   │   ├── disc116/ (1 items)
│   │   ├── disc117/ (1 items)
│   │   ├── disc118/ (1 items)
│   │   ├── disc159/ (1 items)
│   │   ├── disc160/ (1 items)
│   │   └── __init__.py
│   ├── audnzd_champions/ (8 items)
│   │   ├── disc111_h4_macro_structure_retest_elite/ (1 items)
│   │   ├── disc112_h4_macro_precision_retest_elite/ (1 items)
│   │   ├── disc113_h1_macro_intraday_retest_elite/ (1 items)
│   │   ├── disc114_h4_macro_wide_runner_elite/ (1 items)
│   │   ├── disc145/ (1 items)
│   │   ├── disc146/ (1 items)
│   │   ├── disc147/ (1 items)
│   │   └── __init__.py
│   ├── eurgbp_champions/ (6 items)
│   │   ├── disc107_h1_keltner_squeeze_explosion_elite/ (2 items)
│   │   ├── disc172_eurgbp_inside_day_breakout_elite/ (1 items)
│   │   ├── disc174_eurgbp_h1_pullback_elite/ (1 items)
│   │   ├── disc175_eurgbp_daily_pivot_pullback_elite/ (1 items)
│   │   ├── disc180_eurgbp_weekly_pivot_pullback_elite/ (1 items)
│   │   └── __init__.py
│   ├── eurnzd_champions/ (8 items)
│   │   ├── disc132/ (1 items)
│   │   ├── disc133/ (1 items)
│   │   ├── disc143/ (1 items)
│   │   ├── disc157/ (1 items)
│   │   ├── disc158/ (1 items)
│   │   ├── disc167_eurnzd_frankfurt_london_keltner_elite/ (2 items)
│   │   ├── disc168_eurnzd_ny_overlap_keltner_runner_elite/ (2 items)
│   │   └── __init__.py
│   ├── eurusd_champions/ (10 items)
│   │   ├── disc169_eurusd_ny_equity_open_impulse_elite/ (1 items)
│   │   ├── disc50_m30_m5_precision_retest_elite/ (2 items)
│   │   ├── disc51_macro_asymmetric_runner_elite/ (2 items)
│   │   ├── disc58_h1_keltner_squeeze_explosion_elite/ (1 items)
│   │   ├── disc59_m15_macd_zero_cross_momentum_elite/ (1 items)
│   │   ├── disc65_session_vwap_band_fade_elite/ (1 items)
│   │   ├── disc74_daily_pivot_pullback_elite/ (2 items)
│   │   ├── disc75_weekly_pivot_pullback_elite/ (2 items)
│   │   ├── disc76_vwap_1sd_rebound_elite/ (2 items)
│   │   └── disc86_ema_cross_atr_expansion_elite/ (2 items)
│   ├── eurusd_ny_sweeps/ (0 items)
│   ├── gbpchf_champions/ (7 items)
│   │   ├── disc134/ (1 items)
│   │   ├── disc135/ (1 items)
│   │   ├── disc137/ (1 items)
│   │   ├── disc152/ (1 items)
│   │   ├── disc153/ (1 items)
│   │   ├── disc166_gbpchf_h1_keltner_squeeze_elite/ (2 items)
│   │   └── __init__.py
│   ├── gbpjpy_champions/ (9 items)
│   │   ├── disc138/ (1 items)
│   │   ├── disc139/ (1 items)
│   │   ├── disc140/ (1 items)
│   │   ├── disc141/ (1 items)
│   │   ├── disc155/ (1 items)
│   │   ├── disc156/ (1 items)
│   │   ├── disc165_gbpjpy_london_open_impulse_elite/ (2 items)
│   │   ├── disc171_gbpjpy_daily_pivot_pullback_elite/ (1 items)
│   │   └── __init__.py
│   ├── nzdcad_champions/ (7 items)
│   │   ├── disc122/ (1 items)
│   │   ├── disc123/ (1 items)
│   │   ├── disc125/ (1 items)
│   │   ├── disc126/ (1 items)
│   │   ├── disc161/ (1 items)
│   │   ├── disc162/ (1 items)
│   │   └── __init__.py
│   ├── usdcad_champions/ (8 items)
│   │   ├── disc127/ (1 items)
│   │   ├── disc128/ (1 items)
│   │   ├── disc129/ (1 items)
│   │   ├── disc130/ (1 items)
│   │   ├── disc163_usdcad_ny_london_overlap_elite/ (2 items)
│   │   ├── disc164_usdcad_ny_london_overlap_runner_elite/ (2 items)
│   │   ├── disc170_usdcad_daily_pivot_pullback_elite/ (1 items)
│   │   └── __init__.py
│   ├── usdjpy_champions/ (9 items)
│   │   ├── disc106_chaikin_volatility_surge_elite/ (2 items)
│   │   ├── disc173_usdjpy_daily_pivot_pullback_elite/ (1 items)
│   │   ├── disc65_session_vwap_band_fade_elite/ (1 items)
│   │   ├── disc67_london_lunch_expansion_elite/ (1 items)
│   │   ├── disc83_parabolic_sar_flip_elite/ (2 items)
│   │   ├── disc91_session_donchian_midline_bounce_elite/ (1 items)
│   │   ├── disc94_london_vwap_pullback_elite/ (2 items)
│   │   ├── disc95_camarilla_expansion_elite/ (2 items)
│   │   └── disc99_m15_macd_zero_cross_momentum_elite/ (2 items)
│   ├── CERTIFIED_FLEET_MANIFEST.json
│   └── README.md
├── data_engineering_suite/ (29 items)
│   ├── [29 files: __init__.py, analytics.py, audit_10_symbols_total.py, bars.py, batch_slice_unified_masters.py, ...]
├── data_splits/ (4 items)
│   ├── clean_weeks_registry.json
│   ├── generate_chronological_splits.py
│   ├── master_70_30_splits.json
│   └── master_chronological_splits.json
├── docs/ (14 items)
│   ├── ARGON4_MCP_BRIDGE.md
│   ├── AUDIT_AND_BIAS_CORRECTION_REPORT.md
│   ├── AZURE_VM_CREDENTIALS.md
│   ├── DATASET_70_30_SPLIT_STANDARD.md
│   ├── FLEET_REDUNDANCY_PRUNING_AND_72_STRATEGY_PORTFOLIO_AUDIT.md
│   ├── INDIVIDUAL_STRATEGY_CERTIFICATION_AND_STRESS_AUDIT.md
│   ├── INSTITUTIONAL_STRATEGY_VERIFICATION_PLATFORM_REPORT.md
│   ├── MARKET_REGIMES_SESSIONS_AND_DOLLAR_VIABILITY_AUDIT.md
│   ├── PROJECT_TREE_AND_CHANGELOG.md
│   ├── REMOTE_VM_COMPUTATION_HARNESS_GUIDE.md
│   ├── SOP_3Y_TRAIN_2Y_TEST.md
│   ├── SOP_NEW_SYMBOL_ALPHA_DISCOVERY.md
│   ├── SOP_STRATEGY_TESTING_AND_DEPLOYMENT_QUALIFICATION.md
│   └── STRATEGY_CATALOG_80_CHAMPIONS.xlsx
├── results/ (36 items)
│   ├── archived_random_split_evaluations/ (289 items)
│   │   ├── [289 files: 5_champion_prop_trades_5y.json, 5y_discovered_fleet_audit.json, 5y_discovered_fleet_summary.csv, 5yr_unseen_prop_fleet_audit.json, 5yr_unseen_prop_fleet_summary.csv, ...]
│   ├── certified_strategy_manifests/ (390 items)
│   │   ├── [390 files: AUDCAD__disc115_run_manifest.json, AUDCAD__disc115_trades.csv, AUDCAD__disc116_run_manifest.json, AUDCAD__disc116_trades.csv, AUDCAD__disc117_run_manifest.json, ...]
│   ├── continuous_walkforward_audits/ (8 items)
│   │   ├── certified_29_screen.json
│   │   ├── certified_survivors_fleet_summary.json
│   │   ├── cross_pairs_screen.json
│   │   ├── eurgbp_fleet_screen.json
│   │   ├── eurusd_older_strategies_screen.json
│   │   ├── usdcad_fleet_screen.json
│   │   ├── usdjpy_fleet_screen.json
│   │   └── v45_continuous_holdout_audit.json
│   ├── legacy_archive/ (28 items)
│   │   ├── [28 files: ALL_70_STRATEGIES_PERFORMANCE_MATRIX.xlsx, annual_benchmark_monthly.csv, annual_benchmark_monthly_weekly_close.csv, annual_benchmark_trades.csv, annual_benchmark_trades_weekly_close.csv, ...]
│   ├── optimization/ (9 items)
│   │   ├── hf_m15_optimization_summary.json
│   │   ├── hf_m15_optuna_trials.csv
│   │   ├── s10_optimization_summary.json
│   │   ├── s10_optuna_trials.csv
│   │   ├── s14_optimization_summary.json
│   │   ├── s14_optuna_trials.csv
│   │   ├── s66_constrained_summary.json
│   │   ├── s66_optimization_summary.json
│   │   └── s66_optuna_trials.csv
│   ├── alpha_pool_screening_results.json
│   ├── clean_macro_champions_found.json
│   ├── collectible_market_moves_audit.json
│   ├── eurgbp_champions_found.json
│   ├── eurgbp_champions_overlap_analysis.json
│   ├── fleet_integrity_audit.json
│   ├── individual_strategy_independent_audit.json
│   ├── individual_strategy_reliability_audit.json
│   ├── individual_strategy_stress_and_reliability_report.json
│   ├── keltner_champions_eurnzd.json
│   ├── keltner_champions_gbpchf.json
│   ├── mfe_diagnostic_results.json
│   ├── mfe_efficiency_and_harnessing_audit.json
│   ├── new_champions_crosses_and_eurgbp.json
│   ├── new_champions_eurgbp_fix.json
│   ├── new_champions_eurgbp_h1_pullback.json
│   ├── new_champions_eurgbp_inside_day.json
│   ├── new_champions_eurgbp_pivot_pullback.json
│   ├── new_champions_eurgbp_pivot_slope.json
│   ├── new_champions_eurgbp_tailored.json
│   ├── new_champions_eurusd.json
│   ├── new_champions_gbpjpy.json
│   ├── new_champions_novel_crosses.json
│   ├── new_champions_pivot_crosses.json
│   ├── new_champions_usdcad.json
│   ├── new_champions_usdjpy_nzdcad_pivots.json
│   ├── pair_microstructure_behavior_profiles.json
│   ├── portfolio_simulation_report.json
│   ├── selected_19_champions.json
│   ├── strategy_dollar_viability_report.json
│   └── underrepresented_champions_found.json
├── scripts/ (129 items)
│   ├── [129 files: analyze_fleet_clusters.py, audit_78_fleet_independent_rigor.py, audit_broker_friction_stress.py, audit_continuous_holdout_v45.py, audit_fleet_integrity.py, ...]
├── sop/ (4 items)
│   ├── BACKTESTING_METHODOLOGY_BIAS_AND_SOLUTIONS_MANUAL.md
│   ├── run_symbol_sop_audit.py
│   ├── STANDARDIZED_OPERATING_PROCEDURE_DUAL_VALIDATION.md
│   └── verify_zero_bias_harness.py
├── tick_engine/ (9 items)
│   ├── core/ (7 items)
│   │   ├── __init__.py
│   │   ├── block_sampler.py
│   │   ├── config.py
│   │   ├── dataloader.py
│   │   ├── engine.py
│   │   ├── metrics.py
│   │   └── session_time.py
│   ├── quality/ (1 items)
│   │   └── validator.py
│   ├── reporting/ (1 items)
│   │   └── run_manifest.py
│   ├── strategies/ (6 items)
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── example_strategy.py
│   │   ├── macro_retest_helper.py
│   │   ├── momentum.py
│   │   └── top15.py
│   ├── stress/ (1 items)
│   │   └── stress_suite.py
│   ├── tests/ (3 items)
│   │   ├── test_engine.py
│   │   ├── test_engine_rules.py
│   │   └── test_institutional_platform.py
│   ├── README.md
│   ├── requirements.txt
│   └── run_backtest.py
├── tools/ (5 items)
│   ├── dispatch_to_vm.bat
│   ├── eurusd_volatility_gui.py
│   ├── generate_direct_script_urls.py
│   ├── open_remote_vscode.bat
│   └── start_volatility_gui.bat
├── v 4.6/ (75 items)
│   ├── [75 files: FLEET_MANIFEST.json, README.md, run_portfolio_simulation.py, strategy_01_audcad_disc115.py, strategy_02_audcad_disc116.py, ...]
├── v4.6/ (75 items)
│   ├── [75 files: FLEET_MANIFEST.json, README.md, run_portfolio_simulation.py, strategy_01_audcad_disc115.py, strategy_02_audcad_disc116.py, ...]
├── DIRECT_SCRIPT_URLS.json
├── DIRECT_SCRIPT_URLS.md
├── mcp_config.json
├── README.md
├── requirements.txt
└── trade_controller.py
```

---

## 🛡️ 2. Certified Production Fleet Overview (71 Active Models)
- **Total Certified Strategies**: **71** (100% Dual-Certified & Quarantined Holdout Validated)
- **Currency Pair Distribution**: **AUDCAD**: 6, **AUDNZD**: 7, **EURGBP**: 5, **EURNZD**: 7, **EURUSD**: 10, **GBPCHF**: 6, **GBPJPY**: 8, **NZDCAD**: 6, **USDCAD**: 7, **USDJPY**: 9
- **Aggregate Holdout Profit (W211..300)**: **+43,006.8 Net Pips**
- **Friction Model**: Realistic ECN Broker Model ($4.00/lot commission + 0.20p adverse entry slippage + raw floating spread)
- **Execution Engine**: Subsequent-tick causal execution ($T_{\text{fill}} > T_{\text{signal}}$) + 100k prior-week warmup

| Slot | Symbol | Strategy Name | Category | Holdout Pips | Holdout PF | Win Rate | Max DD (pips) |
| :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: |
| 01 | `AUDCAD` | `disc115` | audcad_champions | **+407.0** | **2.98** | 63.3% | 37.5 |
| 02 | `AUDCAD` | `disc116` | audcad_champions | **+390.2** | **2.86** | 56.7% | 35.2 |
| 03 | `AUDCAD` | `disc117` | audcad_champions | **+369.9** | **2.24** | 59.5% | 52.7 |
| 04 | `AUDCAD` | `disc118` | audcad_champions | **+288.4** | **2.92** | 65.2% | 37.5 |
| 05 | `AUDCAD` | `disc159` | audcad_champions | **+75.9** | **1.44** | 47.8% | 41.8 |
| 06 | `AUDCAD` | `disc160` | audcad_champions | **+146.8** | **1.73** | 51.6% | 63.4 |
| 07 | `AUDNZD` | `disc111_h4_macro_structure_retest_elite` | audnzd_champions | **+231.6** | **2.56** | 60.7% | 58.9 |
| 08 | `AUDNZD` | `disc112_h4_macro_precision_retest_elite` | audnzd_champions | **+167.6** | **2.06** | 55.2% | 54.0 |
| 09 | `AUDNZD` | `disc113_h1_macro_intraday_retest_elite` | audnzd_champions | **+74.7** | **1.72** | 50.0% | 26.5 |
| 10 | `AUDNZD` | `disc114_h4_macro_wide_runner_elite` | audnzd_champions | **+290.3** | **3.03** | 64.3% | 63.0 |
| 11 | `AUDNZD` | `disc145` | audnzd_champions | **+633.5** | **4.75** | 72.4% | 34.6 |
| 12 | `AUDNZD` | `disc146` | audnzd_champions | **+614.0** | **3.26** | 64.1% | 50.6 |
| 13 | `AUDNZD` | `disc147` | audnzd_champions | **+611.8** | **3.23** | 62.5% | 50.6 |
| 14 | `EURGBP` | `disc107_h1_keltner_squeeze_explosion_elite` | eurgbp_champions | **+126.0** | **1.71** | 47.5% | 62.7 |
| 15 | `EURNZD` | `disc132` | eurnzd_champions | **+819.5** | **3.32** | 61.8% | 103.4 |
| 16 | `EURNZD` | `disc133` | eurnzd_champions | **+1076.1** | **3.45** | 62.8% | 77.9 |
| 17 | `EURNZD` | `disc143` | eurnzd_champions | **+1417.9** | **7.27** | 77.8% | 42.1 |
| 18 | `EURNZD` | `disc157` | eurnzd_champions | **+1331.2** | **8.23** | 78.8% | 68.9 |
| 19 | `EURNZD` | `disc158` | eurnzd_champions | **+1230.9** | **2.85** | 56.2% | 154.2 |
| 20 | `EURUSD` | `disc50_m30_m5_precision_retest_elite` | eurusd_champions | **+66.8** | **1.79** | 54.5% | 34.3 |
| 21 | `EURUSD` | `disc51_macro_asymmetric_runner_elite` | eurusd_champions | **+238.5** | **6.05** | 70.6% | 18.8 |
| 22 | `EURUSD` | `disc58_h1_keltner_squeeze_explosion_elite` | eurusd_champions | **+392.3** | **1.63** | 45.9% | 71.0 |
| 23 | `EURUSD` | `disc59_m15_macd_zero_cross_momentum_elite` | eurusd_champions | **+366.5** | **1.30** | 41.0% | 88.6 |
| 24 | `EURUSD` | `disc65_session_vwap_band_fade_elite` | eurusd_champions | **+409.0** | **1.30** | 41.7% | 151.0 |
| 25 | `EURUSD` | `disc74_daily_pivot_pullback_elite` | eurusd_champions | **+513.8** | **1.88** | 50.4% | 67.1 |
| 26 | `EURUSD` | `disc75_weekly_pivot_pullback_elite` | eurusd_champions | **+81.6** | **1.15** | 33.7% | 175.5 |
| 27 | `EURUSD` | `disc76_vwap_1sd_rebound_elite` | eurusd_champions | **+313.3** | **1.40** | 43.4% | 95.9 |
| 28 | `EURUSD` | `disc86_ema_cross_atr_expansion_elite` | eurusd_champions | **+552.9** | **1.42** | 43.3% | 118.5 |
| 29 | `GBPCHF` | `disc134` | gbpchf_champions | **+1514.5** | **3.46** | 66.7% | 77.1 |
| 30 | `GBPCHF` | `disc135` | gbpchf_champions | **+1464.1** | **3.52** | 65.9% | 86.1 |
| 31 | `GBPCHF` | `disc137` | gbpchf_champions | **+1390.8** | **3.07** | 64.4% | 101.8 |
| 32 | `GBPCHF` | `disc152` | gbpchf_champions | **+1344.1** | **3.79** | 69.1% | 101.5 |
| 33 | `GBPCHF` | `disc153` | gbpchf_champions | **+1125.6** | **3.04** | 62.3% | 102.6 |
| 34 | `GBPJPY` | `disc138` | gbpjpy_champions | **+1032.2** | **2.39** | 56.0% | 183.4 |
| 35 | `GBPJPY` | `disc139` | gbpjpy_champions | **+995.7** | **1.96** | 51.7% | 210.9 |
| 36 | `GBPJPY` | `disc140` | gbpjpy_champions | **+982.5** | **2.06** | 52.0% | 204.4 |
| 37 | `GBPJPY` | `disc141` | gbpjpy_champions | **+968.1** | **4.93** | 69.6% | 122.2 |
| 38 | `GBPJPY` | `disc155` | gbpjpy_champions | **+958.5** | **2.90** | 58.3% | 139.5 |
| 39 | `GBPJPY` | `disc156` | gbpjpy_champions | **+928.7** | **3.57** | 64.5% | 107.3 |
| 40 | `NZDCAD` | `disc122` | nzdcad_champions | **+213.2** | **1.81** | 44.4% | 71.0 |
| 41 | `NZDCAD` | `disc123` | nzdcad_champions | **+222.0** | **1.97** | 48.0% | 71.0 |
| 42 | `NZDCAD` | `disc125` | nzdcad_champions | **+197.8** | **1.76** | 51.9% | 58.3 |
| 43 | `NZDCAD` | `disc126` | nzdcad_champions | **+140.7** | **1.53** | 44.4% | 71.0 |
| 44 | `NZDCAD` | `disc161` | nzdcad_champions | **+139.1** | **1.56** | 50.0% | 50.5 |
| 45 | `NZDCAD` | `disc162` | nzdcad_champions | **+178.5** | **1.76** | 51.6% | 59.5 |
| 46 | `USDCAD` | `disc127` | usdcad_champions | **+373.1** | **4.07** | 66.7% | 72.8 |
| 47 | `USDCAD` | `disc128` | usdcad_champions | **+327.6** | **2.48** | 55.6% | 73.1 |
| 48 | `USDCAD` | `disc129` | usdcad_champions | **+314.9** | **2.52** | 55.9% | 58.2 |
| 49 | `USDCAD` | `disc130` | usdcad_champions | **+311.3** | **2.92** | 59.3% | 78.4 |
| 50 | `USDJPY` | `disc106_chaikin_volatility_surge_elite` | usdjpy_champions | **+465.3** | **1.17** | 38.4% | 208.3 |
| 51 | `USDJPY` | `disc65_session_vwap_band_fade_elite` | usdjpy_champions | **+257.9** | **1.18** | 40.2% | 170.4 |
| 52 | `USDJPY` | `disc67_london_lunch_expansion_elite` | usdjpy_champions | **+276.7** | **1.27** | 41.5% | 120.9 |
| 53 | `USDJPY` | `disc83_parabolic_sar_flip_elite` | usdjpy_champions | **+438.5** | **1.23** | 40.1% | 127.2 |
| 54 | `USDJPY` | `disc91_session_donchian_midline_bounce_elite` | usdjpy_champions | **+347.6** | **1.23** | 40.1% | 131.3 |
| 55 | `USDJPY` | `disc94_london_vwap_pullback_elite` | usdjpy_champions | **+240.0** | **1.20** | 39.6% | 171.1 |
| 56 | `USDJPY` | `disc95_camarilla_expansion_elite` | usdjpy_champions | **+257.4** | **1.30** | 40.9% | 89.2 |
| 57 | `USDJPY` | `disc99_m15_macd_zero_cross_momentum_elite` | usdjpy_champions | **+220.6** | **1.16** | 38.3% | 147.4 |
| 58 | `USDCAD` | `disc163` | usdcad_champions | **+3431.8** | **2.02** | 49.6% | 147.8 |
| 59 | `USDCAD` | `disc164` | usdcad_champions | **+2657.5** | **1.86** | 49.3% | 78.0 |
| 60 | `GBPJPY` | `disc165` | gbpjpy_champions | **+994.5** | **1.23** | 37.5% | 364.0 |
| 61 | `GBPCHF` | `disc166` | gbpchf_champions | **+587.3** | **1.25** | 41.3% | 179.9 |
| 62 | `EURNZD` | `disc167` | eurnzd_champions | **+297.0** | **1.22** | 39.7% | 441.3 |
| 63 | `EURNZD` | `disc168` | eurnzd_champions | **+293.3** | **1.23** | 37.9% | 264.2 |
| 64 | `EURUSD` | `disc169` | eurusd_champions | **+808.0** | **1.40** | 41.3% | 118.6 |
| 65 | `USDCAD` | `disc170` | usdcad_champions | **+1041.8** | **1.99** | 50.8% | 109.2 |
| 66 | `GBPJPY` | `disc171` | gbpjpy_champions | **+968.7** | **1.54** | 41.2% | 250.8 |
| 67 | `EURGBP` | `disc172` | eurgbp_champions | **+83.7** | **2.04** | 52.6% | 31.0 |
| 68 | `USDJPY` | `disc173` | usdjpy_champions | **+486.5** | **1.39** | 42.0% | 196.5 |
| 69 | `EURGBP` | `disc174` | eurgbp_champions | **+176.3** | **2.14** | 54.3% | 29.3 |
| 70 | `EURGBP` | `disc175` | eurgbp_champions | **+119.9** | **1.62** | 48.0% | 50.4 |
| 71 | `EURGBP` | `disc180` | eurgbp_champions | **+195.5** | **1.63** | 51.4% | 58.6 |


---

## 📜 Turn-by-Turn Prompt & Milestone Changelog


### Turn Checkpoint: 2026-11-14 00:28:10 | Commit: `30d4ef6`
- **Version**: `v4.6.1`
- **User Prompt / Objective**:
  > Production Fleet v4.6.1 Baseline: 71 Certified Strategies, EURGBP Weekly Pivot Pullback (disc180), and Automated Architecture Tree
- **Actions Executed & Technical Changes**:
  - Integrated and dual-certified Strategy Slot 71 (disc180 EURGBP Weekly Pivot Pullback: +195.5 holdout pips, PF 1.63).
  - Quarantined 16 fragile/stress-failing models to protect portfolio downside.
  - Standardized all 71 self-contained models into v4.6/ and synchronized manifests.
  - Implemented automated visual tree generator and turn-by-turn changelog (docs/PROJECT_TREE_AND_CHANGELOG.md).
  - Verified subsequent-tick causal execution, 100k-tick warmup, and 3-component ECN friction.
- **Fleet State**: 71 Certified Strategies | Holdout PnL: **+43,006.8 pips**
- **Pairs**: **AUDCAD**: 6, **AUDNZD**: 7, **EURGBP**: 5, **EURNZD**: 7, **EURUSD**: 10, **GBPCHF**: 6, **GBPJPY**: 8, **NZDCAD**: 6, **USDCAD**: 7, **USDJPY**: 9

---

### Turn Checkpoint: 2026-11-14 00:28:02 | Commit: `Pending`
- **Version**: `v4.6.1`
- **User Prompt / Objective**:
  > Production Fleet v4.6.1 Baseline: 71 Certified Strategies, EURGBP Weekly Pivot Pullback (disc180), and Automated Architecture Tree
- **Actions Executed & Technical Changes**:
  - Integrated and dual-certified Strategy Slot 71 (disc180 EURGBP Weekly Pivot Pullback: +195.5 holdout pips, PF 1.63).
  - Quarantined 16 fragile/stress-failing models to protect portfolio downside.
  - Standardized all 71 self-contained models into v4.6/ and synchronized manifests.
  - Implemented automated visual tree generator and turn-by-turn changelog (docs/PROJECT_TREE_AND_CHANGELOG.md).
  - Verified subsequent-tick causal execution, 100k-tick warmup, and 3-component ECN friction.
- **Fleet State**: 71 Certified Strategies | Holdout PnL: **+43,006.8 pips**
- **Pairs**: **AUDCAD**: 6, **AUDNZD**: 7, **EURGBP**: 5, **EURNZD**: 7, **EURUSD**: 10, **GBPCHF**: 6, **GBPJPY**: 8, **NZDCAD**: 6, **USDCAD**: 7, **USDJPY**: 9

---

### Turn Checkpoint: 2026-11-14 00:27:22 | Commit: `Pending`
- **Version**: `v4.6.1`
- **User Prompt / Objective**:
  > Establish automated directory tree tracking and version control
- **Actions Executed & Technical Changes**:
  - Created scripts/git_checkpoint.py to automate visual tree generation and prompt logging.
  - Baseline 71-strategy production fleet across 10 currency pairs.
- **Fleet State**: 71 Certified Strategies | Holdout PnL: **+43,006.8 pips**
- **Pairs**: **AUDCAD**: 6, **AUDNZD**: 7, **EURGBP**: 5, **EURNZD**: 7, **EURUSD**: 10, **GBPCHF**: 6, **GBPJPY**: 8, **NZDCAD**: 6, **USDCAD**: 7, **USDJPY**: 9

---

### Turn Checkpoint: 2026-11-14 00:24:41 | Commit: `Pending`
- **Version**: `v4.6.1`
- **User Prompt / Objective**:
  > Establish automated directory tree tracking and version control
- **Actions Executed & Technical Changes**:
  - Created scripts/git_checkpoint.py to automate visual tree generation and prompt logging.
  - Baseline 71-strategy production fleet across 10 currency pairs.
- **Fleet State**: 71 Certified Strategies | Holdout PnL: **+0.0 pips**
- **Pairs**: **AUDCAD**: 6, **AUDNZD**: 7, **EURGBP**: 5, **EURNZD**: 7, **EURUSD**: 10, **GBPCHF**: 6, **GBPJPY**: 8, **NZDCAD**: 6, **USDCAD**: 7, **USDJPY**: 9

---
