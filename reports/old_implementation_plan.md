# Implementation Plan: Survival Analysis of Air Raid Alerts in Ukraine

Technical implementation plan for a pet project analyzing and predicting air raid alert durations using survival analysis (Kaplan-Meier, Cox Proportional Hazards, and XGBoost Survival).

## User Review Required

> [!IMPORTANT]
> - **Oblast Level Filtering:** We will filter to keep only `level == 'oblast'` alerts (130,055 records). This naturally avoids 143k localized tactical/artillery alert anomalies (such as a 604-day alert in Lypetska hromada) and focuses prediction on strategic drone/missile threats.
> - **Deduplication:** We must run a global deduplication (`drop_duplicates()`) because 41.66% of the raw dataset consists of duplicate rows (resulting from block-level duplication in the crawler pipeline).
> - **Luhansk and Crimea:** Crimea has 0 records and Luhansk has only 2 records, aligning with the README caveats about permanent alert status. No manual adjustments are needed since their presence is negligible.

## Open Questions

- No major blocking questions. The roadmap starts with Kaplan-Meier and progresses to Cox and XGBoost comparison.

## Proposed Changes

We will build a clean modular structure in Python.

### Core Structure

---

#### [NEW] [requirements.txt](file:///Users/alex/Documents/programming/Air%20Alerts/requirements.txt)
Define project dependencies.
- `pandas`
- `numpy`
- `matplotlib`
- `lifelines` (for Kaplan-Meier and Cox Proportional Hazards)
- `xgboost` (for survival training)
- `scikit-learn`

---

#### [NEW] [data_loader.py](file:///Users/alex/Documents/programming/Air%20Alerts/src/data_loader.py)
Download and pre-process `official_data_en.csv` from Vadimkin GitHub repo.
- Clean duplicate rows globally using `drop_duplicates()`.
- Filter out records with zero or negative durations.
- Filter for `level == 'oblast'` to select regional strategic alerts.
- Convert `started_at` and `finished_at` to datetime.
- Calculate `duration_minutes`.
- Create target column `event` (1 if finished, 0 if active/censored).
- Save cleaned data locally under `data/` cache.

---

#### [NEW] [baseline_km.py](file:///Users/alex/Documents/programming/Air%20Alerts/src/baseline_km.py)
Kaplan-Meier baseline.
- Compute survival curves \(S(t)\) globally and per key region (Kyiv, Kharkiv, Lviv, Odesa).
- Save baseline survival curves plot.

---

#### [NEW] [model_cox.py](file:///Users/alex/Documents/programming/Air%20Alerts/src/model_cox.py)
Cox Proportional Hazards model.
- Encode categorical features (region) using one-hot encoding.
- Build temporal features: `hour_of_day`, `day_of_week`, `month`.
- Train Cox PH model on train split.
- Check proportional hazards assumption.

---

#### [NEW] [model_xgb.py](file:///Users/alex/Documents/programming/Air%20Alerts/src/model_xgb.py)
XGBoost Survival model.
- Format targets for XGBoost survival (`y_lower` and `y_upper` bounds for AFT model).
- Train XGBoost using `survival:aft` or `survival:cox` objective.
- Tune hyperparameters briefly.

---

#### [NEW] [evaluate.py](file:///Users/alex/Documents/programming/Air%20Alerts/src/evaluate.py)
Model comparison and evaluation.
- Compute Concordance Index (C-index) on test split for Cox and XGBoost.
- Plot sample survival curves predicted by all models for specific scenarios.
- Generate comparative report.

## Verification Plan

### Automated Tests
- Validate data pipeline by verifying non-negative durations.
- Print C-index for Cox and XGBoost models to verify mathematical correctness (> 0.5).

### Manual Verification
- Visual inspection of generated survival curve plots.
