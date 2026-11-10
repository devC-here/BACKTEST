# Standard Operating Procedure (SOP): Systematic Alpha Discovery & Production Fleet Certification for New Symbols

**Document ID:** `SOP-TRD-DISC-2026-004`  
**Effective Date:** October 2026  
**Applies To:** All New Currency Pairs, Equity Indices, Commodities, and Exotic Cross Datasets  
**Harness Tool:** [`run_symbol_alpha_discovery.py`](file:///d:/PROJECT/TRD/MODELS%20GENERATIONS/BACKTEST/run_symbol_alpha_discovery.py) / [`data_engineering_suite/master_symbol_alpha_discovery_suite.py`](file:///d:/PROJECT/TRD/MODELS%20GENERATIONS/BACKTEST/data_engineering_suite/master_symbol_alpha_discovery_suite.py)

---

## 1. Executive Mandate: The "Standard Battery First" Protocol

When new tick data arrives for an unstudied symbol (e.g. `AUDCAD`, `GBPJPY`, `USATECHIDXUSD`, `XAUUSD`), **engineers must NOT immediately write bespoke discovery scripts from scratch**.

Over multi-year testing across EURUSD, USDJPY, EURGBP, and AUDNZD (1,162 weeks / 540M+ ticks), we have already formulated, validated, and holdout-certified **102 elite production champions** and built **52 comprehensive Discovery Labs (`v1` through `v45` + cross-specialists)** that derive alpha from universal market mechanics.

### The 3-Tier Standardized Workflow:

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                   NEW SYMBOL STANDARDIZED ALPHA DISCOVERY PIPELINE                               │
├──────────────────────────────────────────────────────────────────────────────────────────────────┤
│  [TIER 1] FAST SCREENING BATTERY (50+ Core Variants / ~60s runtime)                              │
│       │   Command: run_symbol_alpha_discovery.py --symbol <SYMBOL>                               │
│       │   Evaluates 8 universal architecture families on 70% In-Sample Discovery partition.      │
│       │   ├── If >= 8 Champions Qualified ──> Run 30% Holdout Forward-Test & Deploy Fleet        │
│       │   └── If < 8 Champions Qualified  ──> Proceed to Tier 2 Deep Spectrum Lab Battery        │
│       ▼                                                                                          │
│  [TIER 2] DEEP SPECTRUM DISCOVERY LAB BATTERY (52 Historical Discovery Labs / 160+ Setups)        │
│       │   Command: run_symbol_discovery_labs.py --symbol <SYMBOL> --batch all                    │
│       │   Systematically executes all 52 specialized labs (v1 to v45 + Specialists).             │
│       │   ├── If >= 8 Champions Qualified ──> Run 30% Holdout Forward-Test & Deploy Fleet        │
│       │   └── If < 8 Champions Qualified  ──> Proceed to Tier 3 Exhaustion Protocol              │
│       ▼                                                                                          │
│  [TIER 3] EXHAUSTION PROTOCOL & BESPOKE DISCOVERY                                                │
│           Only triggered when BOTH Tier 1 and Tier 2 fail to produce 8 qualified champions.      │
│           Quant engineers are authorized to develop bespoke scripts (e.g. Lab v46+) targeting    │
│           unique asset-specific idiosyncrasies (commodity gaps, index seasonality, etc.).        │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Complete Catalog of All 52 Discovery Labs (v1 to v45 + Specialists)

All historical discovery labs have been standardized to dynamically accept any symbol via `run_symbol_discovery_labs.py --symbol <SYMBOL> --lab <ID>` or `--batch <BATCH>`:

| Lab ID | Batch | Title & Target Setups | Script Path |
|:---|:---:|:---|:---|
| **v1** | `v1-v10` | London open breakout, Asian sweep fade, Momentum pulse, EMA trend ride | [`discovery_labs/alpha_discovery_lab_v1.py`](file:///d:/PROJECT/TRD/MODELS%20GENERATIONS/BACKTEST/discovery_labs/alpha_discovery_lab_v1.py) |
| **v2** | `v1-v10` | Advanced Systematic Architectures: Donchian trend pullbacks, H4 macro retests | [`discovery_labs/alpha_discovery_lab_v2.py`](file:///d:/PROJECT/TRD/MODELS%20GENERATIONS/BACKTEST/discovery_labs/alpha_discovery_lab_v2.py) |
| **v3** | `v1-v10` | Structural Level Mining: Tokyo fix breakout, London morning continuation | [`discovery_labs/alpha_discovery_lab_v3.py`](file:///d:/PROJECT/TRD/MODELS%20GENERATIONS/BACKTEST/discovery_labs/alpha_discovery_lab_v3.py) |
| **v4** | `v1-v10` | Lower-Timeframe Precision: Central Pivot Range (CPR) bounce, Daily pivot wall | [`discovery_labs/alpha_discovery_lab_v4.py`](file:///d:/PROJECT/TRD/MODELS%20GENERATIONS/BACKTEST/discovery_labs/alpha_discovery_lab_v4.py) |
| **v5** | `v1-v10` | Pivot Extensions & Fibonacci Pullbacks: Golden pocket pin defense, NR7 expansion | [`discovery_labs/alpha_discovery_lab_v5.py`](file:///d:/PROJECT/TRD/MODELS%20GENERATIONS/BACKTEST/discovery_labs/alpha_discovery_lab_v5.py) |
| **v6** | `v1-v10` | Fibonacci 61.8% pin defense, Tokyo pivot wall, Narrow range 7 expansion | [`discovery_labs/alpha_discovery_lab_v6.py`](file:///d:/PROJECT/TRD/MODELS%20GENERATIONS/BACKTEST/discovery_labs/alpha_discovery_lab_v6.py) |
| **v7** | `v1-v10` | Institutional Compression: Double structure neckline breakout, NY open overlap | [`discovery_labs/alpha_discovery_lab_v7.py`](file:///d:/PROJECT/TRD/MODELS%20GENERATIONS/BACKTEST/discovery_labs/alpha_discovery_lab_v7.py) |
| **v8** | `v1-v10` | Asian high/low retests, London fix mean reversion, H1 inside bar expansion | [`discovery_labs/alpha_discovery_lab_v8.py`](file:///d:/PROJECT/TRD/MODELS%20GENERATIONS/BACKTEST/discovery_labs/alpha_discovery_lab_v8.py) |
| **v9** | `v1-v10` | Range Expansions: ATR daily stretch exhaustion fade, Asian liquidity sweep spring | [`discovery_labs/alpha_discovery_lab_v9.py`](file:///d:/PROJECT/TRD/MODELS%20GENERATIONS/BACKTEST/discovery_labs/alpha_discovery_lab_v9.py) |
| **v10** | `v1-v10` | Order Blocks: RSI regime range shift, Stochastic deep pullbacks, Session VWAP value | [`discovery_labs/alpha_discovery_lab_v10.py`](file:///d:/PROJECT/TRD/MODELS%20GENERATIONS/BACKTEST/discovery_labs/alpha_discovery_lab_v10.py) |
| **v11** | `v11-v20` | TTM Keltner squeeze, Tokyo range sweep, Thursday trend ignition | [`discovery_labs/alpha_discovery_lab_v11.py`](file:///d:/PROJECT/TRD/MODELS%20GENERATIONS/BACKTEST/discovery_labs/alpha_discovery_lab_v11.py) |
| **v12** | `v11-v20` | M15 MACD zero-cross, Asian compression breakout, RSI 50 bounce | [`discovery_labs/alpha_discovery_lab_v12.py`](file:///d:/PROJECT/TRD/MODELS%20GENERATIONS/BACKTEST/discovery_labs/alpha_discovery_lab_v12.py) |
| **v13** | `v11-v20` | M15 50-EMA pin runner, US morning overlap, Double structure breakout | [`discovery_labs/alpha_discovery_lab_v13.py`](file:///d:/PROJECT/TRD/MODELS%20GENERATIONS/BACKTEST/discovery_labs/alpha_discovery_lab_v13.py) |
| **v14** | `v11-v20` | Asian range reversal fade, Donchian retest, London afternoon drive | [`discovery_labs/alpha_discovery_lab_v14.py`](file:///d:/PROJECT/TRD/MODELS%20GENERATIONS/BACKTEST/discovery_labs/alpha_discovery_lab_v14.py) |
| **v15** | `v11-v20` | Stoch-RSI pullback hook, Tokyo fix trend runner, London midday MACD | [`discovery_labs/alpha_discovery_lab_v15.py`](file:///d:/PROJECT/TRD/MODELS%20GENERATIONS/BACKTEST/discovery_labs/alpha_discovery_lab_v15.py) |
| **v16** | `v11-v20` | Asian midpoint rebound, SuperTrend M15 squeeze, Tokyo 20-EMA pullback | [`discovery_labs/alpha_discovery_lab_v16.py`](file:///d:/PROJECT/TRD/MODELS%20GENERATIONS/BACKTEST/discovery_labs/alpha_discovery_lab_v16.py) |
| **v17** | `v11-v20` | London lunch expansion, Session VWAP 2.0-SD band fade, NY continuation | [`discovery_labs/alpha_discovery_lab_v17.py`](file:///d:/PROJECT/TRD/MODELS%20GENERATIONS/BACKTEST/discovery_labs/alpha_discovery_lab_v17.py) |
| **v18** | `v11-v20` | Heikin-Ashi impulse surge, CMF volume expansion, M15 200-EMA bounce | [`discovery_labs/alpha_discovery_lab_v18.py`](file:///d:/PROJECT/TRD/MODELS%20GENERATIONS/BACKTEST/discovery_labs/alpha_discovery_lab_v18.py) |
| **v19** | `v11-v20` | Asian box retest, MTF micro-scalper, Liquidity sweep spring | [`discovery_labs/alpha_discovery_lab_v19.py`](file:///d:/PROJECT/TRD/MODELS%20GENERATIONS/BACKTEST/discovery_labs/alpha_discovery_lab_v19.py) |
| **v20** | `v11-v20` | Delta absorption scalper, London fix continuation, H4 wick rejection | [`discovery_labs/alpha_discovery_lab_v20.py`](file:///d:/PROJECT/TRD/MODELS%20GENERATIONS/BACKTEST/discovery_labs/alpha_discovery_lab_v20.py) |
| **v21** | `v21-v30` | Daily VWAP bands 2.0-SD, Session tight coil, Double top/bottom neckline | [`discovery_labs/alpha_discovery_lab_v21.py`](file:///d:/PROJECT/TRD/MODELS%20GENERATIONS/BACKTEST/discovery_labs/alpha_discovery_lab_v21.py) |
| **v22** | `v21-v30` | Weekly pivot pullback (DISC75), VWAP 1-SD discount (DISC76), Keltner midline (DISC77) | [`discovery_labs/alpha_discovery_lab_v22.py`](file:///d:/PROJECT/TRD/MODELS%20GENERATIONS/BACKTEST/discovery_labs/alpha_discovery_lab_v22.py) |
| **v23** | `v21-v30` | Three-bar inside breakout (DISC78), London 15m ORB (DISC79), ADX DI cross | [`discovery_labs/alpha_discovery_lab_v23.py`](file:///d:/PROJECT/TRD/MODELS%20GENERATIONS/BACKTEST/discovery_labs/alpha_discovery_lab_v23.py) |
| **v24** | `v21-v30` | Double swing neckline retest (DISC80), London fix reversion, SAR acceleration | [`discovery_labs/alpha_discovery_lab_v24.py`](file:///d:/PROJECT/TRD/MODELS%20GENERATIONS/BACKTEST/discovery_labs/alpha_discovery_lab_v24.py) |
| **v25** | `v21-v30` | Bollinger squeeze ignition (DISC81), Double inside bar, London fix rebalance | [`discovery_labs/alpha_discovery_lab_v25.py`](file:///d:/PROJECT/TRD/MODELS%20GENERATIONS/BACKTEST/discovery_labs/alpha_discovery_lab_v25.py) |
| **v26** | `v21-v30` | Chaikin surge (DISC82), Parabolic SAR flip (DISC83), London 30m ORB (DISC84), EMA-ATR expansion (DISC86) | [`discovery_labs/alpha_discovery_lab_v26.py`](file:///d:/PROJECT/TRD/MODELS%20GENERATIONS/BACKTEST/discovery_labs/alpha_discovery_lab_v26.py) |
| **v27** | `v21-v30` | Session open impulse (DISC85), Pivot R1/S1 breakout (DISC87), 3-drive ignition (DISC88) | [`discovery_labs/alpha_discovery_lab_v27.py`](file:///d:/PROJECT/TRD/MODELS%20GENERATIONS/BACKTEST/discovery_labs/alpha_discovery_lab_v27.py) |
| **v28** | `v21-v30` | Calibrated SuperTrend squeeze, ATR expansion drives, Fixed pip targets | [`discovery_labs/alpha_discovery_lab_v28.py`](file:///d:/PROJECT/TRD/MODELS%20GENERATIONS/BACKTEST/discovery_labs/alpha_discovery_lab_v28.py) |
| **v29** | `v21-v30` | Daily Pivot R2/S2 extreme boundary fade, Reversion oscillators | [`discovery_labs/alpha_discovery_lab_v29.py`](file:///d:/PROJECT/TRD/MODELS%20GENERATIONS/BACKTEST/discovery_labs/alpha_discovery_lab_v29.py) |
| **v30** | `v21-v30` | Asian liquidity sweep spring, False breakout re-entry, Range boundaries | [`discovery_labs/alpha_discovery_lab_v30.py`](file:///d:/PROJECT/TRD/MODELS%20GENERATIONS/BACKTEST/discovery_labs/alpha_discovery_lab_v30.py) |
| **v31** | `v31-v45` | Fix overlap springs, Session transition liquidity traps, Volume absorption | [`discovery_labs/alpha_discovery_lab_v31.py`](file:///d:/PROJECT/TRD/MODELS%20GENERATIONS/BACKTEST/discovery_labs/alpha_discovery_lab_v31.py) |
| **v32** | `v31-v45` | H1-M15 20/50 EMA value bounce, Dynamic pullback accelerators | [`discovery_labs/alpha_discovery_lab_v32.py`](file:///d:/PROJECT/TRD/MODELS%20GENERATIONS/BACKTEST/discovery_labs/alpha_discovery_lab_v32.py) |
| **v33** | `v31-v45` | Bollinger 2.5-SD extreme fade, London lunch pre-NY pullback, Tokyo continuation | [`discovery_labs/alpha_discovery_lab_v33.py`](file:///d:/PROJECT/TRD/MODELS%20GENERATIONS/BACKTEST/discovery_labs/alpha_discovery_lab_v33.py) |
| **v34** | `v31-v45` | Prior day high/low liquidity sweep, False breakout springs | [`discovery_labs/alpha_discovery_lab_v34.py`](file:///d:/PROJECT/TRD/MODELS%20GENERATIONS/BACKTEST/discovery_labs/alpha_discovery_lab_v34.py) |
| **v35** | `v31-v45` | Pivot R1/S1 breakout retests, Structural level flips, Trend continuation | [`discovery_labs/alpha_discovery_lab_v35.py`](file:///d:/PROJECT/TRD/MODELS%20GENERATIONS/BACKTEST/discovery_labs/alpha_discovery_lab_v35.py) |
| **v36** | `v31-v45` | Camarilla H4/L4 expansion drives, Camarilla L3/H3 boundary fades | [`discovery_labs/alpha_discovery_lab_v36.py`](file:///d:/PROJECT/TRD/MODELS%20GENERATIONS/BACKTEST/discovery_labs/alpha_discovery_lab_v36.py) |
| **v37** | `v31-v45` | Cardwell RSI 50 regime support in bull trend, RSI 50 resistance in bear trend | [`discovery_labs/alpha_discovery_lab_v37.py`](file:///d:/PROJECT/TRD/MODELS%20GENERATIONS/BACKTEST/discovery_labs/alpha_discovery_lab_v37.py) |
| **v38** | `v31-v45` | Spread resilience tuning, SL/TP parameter optimization | [`discovery_labs/alpha_optimization_lab_v38.py`](file:///d:/PROJECT/TRD/MODELS%20GENERATIONS/BACKTEST/discovery_labs/alpha_optimization_lab_v38.py) |
| **v39** | `v31-v45` | RSI 50 bounce at Camarilla pivot levels, High-probability confluence | [`discovery_labs/alpha_discovery_lab_v39.py`](file:///d:/PROJECT/TRD/MODELS%20GENERATIONS/BACKTEST/discovery_labs/alpha_discovery_lab_v39.py) |
| **v40** | `v31-v45` | Calibrated Camarilla, Asian compression, Double structure breakouts | [`discovery_labs/alpha_discovery_lab_v40.py`](file:///d:/PROJECT/TRD/MODELS%20GENERATIONS/BACKTEST/discovery_labs/alpha_discovery_lab_v40.py) |
| **v41** | `v31-v45` | 2.0R to 2.5R asymmetric reward multipliers for 15% annual yield | [`discovery_labs/alpha_optimization_lab_v41.py`](file:///d:/PROJECT/TRD/MODELS%20GENERATIONS/BACKTEST/discovery_labs/alpha_optimization_lab_v41.py) |
| **v42** | `v31-v45` | M15 MACD zero-cross, Tokyo expansion ATR, Cardwell RSI 50, Camarilla expansion | [`discovery_labs/alpha_discovery_lab_v42.py`](file:///d:/PROJECT/TRD/MODELS%20GENERATIONS/BACKTEST/discovery_labs/alpha_discovery_lab_v42.py) |
| **v43** | `v31-v45` | Refined oscillator overbought/oversold, Directional squeeze timing | [`discovery_labs/alpha_discovery_lab_v43.py`](file:///d:/PROJECT/TRD/MODELS%20GENERATIONS/BACKTEST/discovery_labs/alpha_discovery_lab_v43.py) |
| **v44** | `v31-v45` | Underwater duration pruning, Multi-year consistency calibration | [`discovery_labs/alpha_discovery_lab_v44.py`](file:///d:/PROJECT/TRD/MODELS%20GENERATIONS/BACKTEST/discovery_labs/alpha_discovery_lab_v44.py) |
| **v45** | `v31-v45` | Cross-currency volatility calibration, Asymmetric runner optimization | [`discovery_labs/alpha_discovery_lab_v45.py`](file:///d:/PROJECT/TRD/MODELS%20GENERATIONS/BACKTEST/discovery_labs/alpha_discovery_lab_v45.py) |
| **audnzd_mean_reversion** | `specialists` | Sydney sweep fade (Turtle Soup), Multi-day Fib Golden Pocket (LB36-72), Macro H4 Donchian | [`discovery_labs/alpha_discovery_lab_audnzd_mean_reversion.py`](file:///d:/PROJECT/TRD/MODELS%20GENERATIONS/BACKTEST/discovery_labs/alpha_discovery_lab_audnzd_mean_reversion.py) |
| **audnzd_expansion** | `specialists` | Sydney 30m ORB, Dual EMA pullbacks, Keltner midline expansions | [`discovery_labs/alpha_discovery_lab_audnzd_expansion.py`](file:///d:/PROJECT/TRD/MODELS%20GENERATIONS/BACKTEST/discovery_labs/alpha_discovery_lab_audnzd_expansion.py) |
| **audnzd_cross_specialists** | `specialists` | Pacific session microstructure, Tokyo liquidity anchors, Spread resistance | [`discovery_labs/alpha_discovery_lab_audnzd_cross_specialists.py`](file:///d:/PROJECT/TRD/MODELS%20GENERATIONS/BACKTEST/discovery_labs/alpha_discovery_lab_audnzd_cross_specialists.py) |
| **audnzd_peer_fleet** | `specialists` | Testing 44 EURUSD and 42 USDJPY champions directly on AUDNZD data | [`discovery_labs/alpha_discovery_lab_audnzd_peer_fleet.py`](file:///d:/PROJECT/TRD/MODELS%20GENERATIONS/BACKTEST/discovery_labs/alpha_discovery_lab_audnzd_peer_fleet.py) |
| **audnzd_v2** | `specialists` | Pacific Open breakout, Tokyo overlap drive, Asian range mean reversion | [`discovery_labs/alpha_discovery_lab_audnzd_v2.py`](file:///d:/PROJECT/TRD/MODELS%20GENERATIONS/BACKTEST/discovery_labs/alpha_discovery_lab_audnzd_v2.py) |
| **audnzd_v3** | `specialists` | H4 macro structure wall retests with wide asymmetric runners (DISC111, DISC112) | [`discovery_labs/alpha_discovery_lab_audnzd_v3.py`](file:///d:/PROJECT/TRD/MODELS%20GENERATIONS/BACKTEST/discovery_labs/alpha_discovery_lab_audnzd_v3.py) |
| **eurgbp_expansion** | `specialists` | Session VWAP 2.0-SD fade (DISC121), Fib Golden Pocket LB36/48 (DISC122, DISC125), Daily Pivot sniper (DISC123) | [`discovery_labs/alpha_discovery_lab_eurgbp_expansion.py`](file:///d:/PROJECT/TRD/MODELS%20GENERATIONS/BACKTEST/discovery_labs/alpha_discovery_lab_eurgbp_expansion.py) |

---

## 2. The 8 Proven Architecture Families in the Master Battery

The Master Battery encapsulates the 8 core quantitative edges that have proven profitable across majors and cross-currency pairs:

| Family # | Architecture Family | Core Market Mechanism | Proven Benchmarks |
|:---:|:---|:---|:---|
| **FAM 1** | **Session VWAP Band Fades** | Fading statistical price overextension at $\ge 2.0\text{--}2.5\sigma$ from intraday anchored VWAP with 2.0R mean-reversion targets back to baseline. | `DISC65`, `DISC120`, `DISC121` |
| **FAM 2** | **Fibonacci Golden Pocket Retracements** | Tracking multi-day rolling swings (36h, 48h, 72h) and entering institutional 61.8%–78.6% pullbacks in H4 trend direction with asymmetric 2.0R to 2.5R targets. | `DISC115`, `DISC116`, `DISC117`, `DISC118`, `DISC122`, `DISC125`, `DISC129` |
| **FAM 3** | **Institutional Daily & Camarilla Pivot Walls** | Defending floor trader reference pivots (P, S1, R1) or Camarilla equation levels (H3/L3) when price tests the wall and prints a structural rejection wick ($\ge 35\%$). | `DISC46`, `DISC48`, `DISC74`, `DISC123` |
| **FAM 4** | **Macro Structural Donchian Retests** | Multi-timeframe H4/H1 Donchian channel breakout level retests with M15 rejection wicks ($\ge 40\%$) and asymmetric 2.0R to 2.5R runner targets. | `DISC42`, `DISC49`, `DISC51`, `DISC111`, `DISC112` |
| **FAM 5** | **Alexander Elder Triple Screen Impulses** | 3-tier cascade: Screen 1 (H4 MACD histogram slope) + Screen 2 (H1 20-EMA wave) + Screen 3 (M15 Stochastic extreme sweep $\le 30 / \ge 70$ + engulfing trigger). | `DISC56`, `DISC53` |
| **FAM 6** | **Volatility Compression Squeeze Explosions** | John Carter TTM Squeeze on H1 (Bollinger Bands compressing inside Keltner Channel) followed by an M15 directional breakout continuation. | `DISC58`, `DISC77`, `DISC81` |
| **FAM 7** | **Session Opening Range Breakouts (ORB)** | London Cash Open (07:00 UTC) 15m/30m expansion candle drives in H4 20-EMA trend direction, and Asian session compression range breakouts. | `DISC60`, `DISC71`, `DISC79`, `DISC84` |
| **FAM 8** | **Momentum Surge & Volatility Accelerators** | Chaikin Volatility expansion ($\ge +30\%$ ROC) or Parabolic SAR reversal flips paired with momentum expansion candles closing in H4 trend direction. | `DISC82`, `DISC83`, `DISC86`, `DISC88` |

---

## 3. Step-by-Step Execution Protocol for Any New Symbol

### Step 1: Place Tick Data in the Standard Directory
Ensure tick parquet files are saved under `data/<SYMBOL>/` with standard weekly files:
```
data/
└── <SYMBOL>/
    ├── <SYMBOL>_2021_W01.parquet
    ├── <SYMBOL>_2021_W02.parquet
    └── ... (continuous weekly files)
```
*(If the symbol arrives as a single 5-year master parquet, unpack it into weekly chunks using `data_engineering_suite/` utilities).*

### Step 2: Run the Master Discovery Pipeline
Execute the automated discovery suite from the project root:
```bash
.venv\Scripts\python.exe run_symbol_alpha_discovery.py --symbol <SYMBOL>
```
*Options:*
- `--hurdle-return 15.0`: Minimum annualized net return percentage at 1% account risk (default: `15.0`).
- `--hurdle-pf 1.25`: Minimum Profit Factor threshold (default: `1.25`).
- `--max-workers 12`: Number of parallel CPU workers for multiprocessing.

### Step 3: Automated Pipeline Execution Stages
The pipeline will automatically execute all 5 phases:
1. **Dynamic Symbol Calibration**: Detects symbol type (standard FX, JPY pair, metal, or index) and assigns appropriate pip sizing, spread filters, and SL/TP scalers.
2. **Weekly Bar Precomputation**: Builds M15, H1, and H4 bars once per week per worker for lightning multi-process evaluation.
3. **In-Sample Screening (70% Partition)**: Evaluates all 50+ strategy variants across the 210 in-sample weeks.
4. **Hurdle Qualification**: Filters models meeting $AnnRet \ge 15.0\%$, $PF \ge 1.25$, $N \ge 100$, and $Ret/DD \ge 3.0x$.
5. **Out-of-Sample Forward Test (30% Partition)**: Evaluates top survivors on the 90 unseen holdout weeks (`Seed 42`) and confirms holdout $PF \ge 1.20$.
6. **Top 8 Fleet Assembly**: Selects the top 8 least-correlated models (limiting max 2 models per family to ensure diversity).

---

## 4. Pipeline Outputs & Scorecards

Upon completion, the pipeline automatically writes standardized artifacts to `results/`:

1. **`results/{symbol}_master_discovery_scorecard.csv`**: Full scorecard ranking every strategy permutation across trades, net pips, profit factor, annual return, max drawdown, and certification status.
2. **`results/{symbol}_master_fleet_recommendation.json`**: Top 8 certified champions formatted with all risk parameters, SL/TP levels, and metrics.
3. **Console Institutional Scorecard**:
```
===================================================================================================================================
  <SYMBOL> TOP 8 CERTIFIED PRODUCTION CHAMPIONS SCORECARD
===================================================================================================================================
Rank | Strategy Name                      | Family          | Trades  | Net Pips  | Ann Ret%  | Full PF | Hold PF | Win%   | Max DD  | Ret/DD 
-----------------------------------------------------------------------------------------------------------------------------------
1    | FibPocket_LB48_SL8.0_TP16.0        | FIB_POCKET      | 301     | +1023.8p  | +22.2%    | 1.88    | 2.02    | 51.2%  | 66.2p   | 15.47x 
2    | VWAP_B2.0_SL7.0_TP14.0             | VWAP_FADE       | 696     | +1285.5p  | +31.8%    | 1.45    | 1.50    | 46.1%  | 160.5p  | 8.01x  
3    | DailyPivot_SL7.5_TP15.0            | PIVOT_WALL      | 721     | +1206.0p  | +27.9%    | 1.39    | 1.50    | 44.5%  | 111.6p  | 10.81x 
4    | MacroDonchian_LB10_SL12.0_TP24.0   | MACRO_STRUCT    | 189     | +1238.9p  | +17.9%    | 2.27    | 2.21    | 55.0%  | 95.6p   | 12.96x 
...
-----------------------------------------------------------------------------------------------------------------------------------
```

---

## 5. The Exhaustion Protocol

At the conclusion of the pipeline, the system outputs an official assessment:

### Scenario A: Suite Sufficient ($\ge 8$ Champions Identified)
```
>> STATUS: SUITE SUFFICIENT (Found 8+ Certified Champions meeting all hurdles).
>> ACTION: Deploy the 8 champions directly into strategies/<symbol>_fleet/. NO NEW BESPOKE SCRIPTS REQUIRED.
```
- **Action**: Create strategy files in `strategies/{symbol}_fleet/`, generate `README.md`, update `PROVEN_PROFITABLE_STRATEGIES_DIRECTORY.md`, and deploy to production.

### Scenario B: Battery Exhausted ($< 8$ Champions Identified)
```
>> STATUS: BATTERY EXHAUSTED (Only K Champions passed out of 8 required).
>> ACTION: TRIGGER BESPOKE MICROSTRUCTURAL DISCOVERY. Create specialized scripts for <SYMBOL>.
```
- **Action**: Only when Scenario B occurs is the quant engineer authorized to formulate new hypotheses, design custom indicators, and write bespoke discovery scripts targeting the asset's specific idiosyncratic behavior (e.g. index gap fills, commodity seasonality, or exotic currency peg mechanics).

