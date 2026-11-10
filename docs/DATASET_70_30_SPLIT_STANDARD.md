# Institutional Standard: 70/30 Randomized Week Partitioning

**Policy Effective Date:** October 2026  
**Deterministic Seed:** `42` (Reproducible, zero-overlap)  
**Central Registry:** [`data_splits/master_70_30_splits.json`](file:///d:/PROJECT/TRD/MODELS%20GENERATIONS/BACKTEST/data_splits/master_70_30_splits.json)  

---

## 1. The Core Quantitative Rule

To completely eliminate lookahead bias, data snooping, and curve-fitting across all multi-asset quantitative research:

1. **70% Discovery Partition (`discovery_70`)**:
   - **Allowed Use**: Exploratory data analysis (EDA), statistical regime classification, indicator discovery, parameter optimization, machine learning training, and candidate validation.
   - **Rule**: All new alphas MUST be developed and calibrated **exclusively** on this 70% partition.
2. **30% Holdout Test Partition (`holdout_test_30`)**:
   - **Allowed Use**: **STRICTLY QUARANTINED FOR FINAL FORWARD VALIDATION**.
   - **Strict Prohibition**: **NEVER** used for scanning, parameter sweeps, feature selection, or edge discovery.
   - **Purpose**: Acts as an uncompromised "True Unseen Forward Market" to verify out-of-sample edge before strategies are deemed production-ready.

---

## 2. Partition Summary Across All Asset Datasets

| Asset Symbol | Total Weeks | Total Ticks | 70% Discovery Weeks | Discovery Ticks | 30% Holdout Test Weeks | Holdout Ticks | Split Manifest File |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **EURUSD** | **262** | 123,223,663 | **183** (69.8%) | 84,904,240 | **79** (30.2%) | 38,319,423 | [split_70_30.json](file:///d:/PROJECT/TRD/MODELS%20GENERATIONS/BACKTEST/data/EURUSD/split_70_30.json) |
| **EURGBP** | **300** | 85,809,973 | **210** (70.0%) | 58,916,398 | **90** (30.0%) | 26,893,575 | [split_70_30.json](file:///d:/PROJECT/TRD/MODELS%20GENERATIONS/BACKTEST/data/EURGBP/split_70_30.json) |
| **USDJPY** | **300** | 192,319,465 | **210** (70.0%) | 136,837,818 | **90** (30.0%) | 55,481,647 | [split_70_30.json](file:///d:/PROJECT/TRD/MODELS%20GENERATIONS/BACKTEST/data/USDJPY/split_70_30.json) |
| **AUDNZD** | **53** | 17,664,818 | **37** (69.8%) | 11,746,514 | **16** (30.2%) | 5,918,304 | [split_70_30.json](file:///d:/PROJECT/TRD/MODELS%20GENERATIONS/BACKTEST/data/AUDNZD/split_70_30.json) |
| **Fleet Total**| **915** | **419,017,919** | **640** (69.9%) | **292,404,970** | **275** (30.1%) | **126,612,949** | [master_70_30_splits.json](file:///d:/PROJECT/TRD/MODELS%20GENERATIONS/BACKTEST/data_splits/master_70_30_splits.json) |

---

## 3. How to Use in Python Code

The data loader in [`tick_engine/core/dataloader.py`](file:///d:/PROJECT/TRD/MODELS%20GENERATIONS/BACKTEST/tick_engine/core/dataloader.py) natively enforces this split:

### A. For Alpha Discovery & Calibration (The 70% Set)
```python
from tick_engine.core.dataloader import get_discovery_weeks, get_week_dataset

# Load strictly the 70% discovery weeks (e.g. 110 weeks for EURGBP, 183 for EURUSD)
discovery_weeks = get_discovery_weeks("EURGBP")

for w in discovery_weeks:
    ticks, meta = get_week_dataset(w, raw_dir="EURGBP")
    # Run strategy research, feature engineering, and calibration here...
```

### B. For Final Out-of-Sample Forward Verification (The 30% Set)
```python
from tick_engine.core.dataloader import get_holdout_test_weeks, get_week_dataset

# Load strictly the 30% quarantined holdout weeks (e.g. 47 weeks for EURGBP, 79 for EURUSD)
test_weeks = get_holdout_test_weeks("EURGBP")

for w in test_weeks:
    ticks, meta = get_week_dataset(w, raw_dir="EURGBP")
    # Final uncompromised test: Never tune parameters on these weeks!
```

---

## 4. Metadata Structure in Manifests

Every item in `manifest_weekly_sets.json` has been updated with its designated split role:
```json
{
  "week": 1,
  "filename": "week_01_2021-01-04_to_2021-01-08.csv",
  "tick_count": 561375,
  "split": "discovery_70",
  "split_category": "discovery",
  "split_role": "Analysis, alpha discovery, and calibration"
}
```
Holdout test weeks are tagged:
```json
{
  "week": 4,
  "filename": "week_04_2021-01-25_to_2021-01-29.csv",
  "tick_count": 519280,
  "split": "holdout_test_30",
  "split_category": "holdout_test",
  "split_role": "Quarantined out-of-sample test only"
}
```
