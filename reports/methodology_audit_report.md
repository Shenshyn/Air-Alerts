# Methodology Audit Report: Air Alerts Survival Analysis

This document outlines a thorough methodological audit of the machine learning pipeline located in `src/`. The audit focuses on statistical correctness, proper application of survival models, potential data leakages, feature engineering practices, and evaluation metrics.

## 1. Data Leakage and Pipeline Correctness

### Categorical Encoding Leakage
- **Issue**: In `model_cox.py`, `model_xgb.py`, and `evaluate.py`, one-hot encoding for the spatial feature (`oblast`) is applied globally to the entire dataset using `pd.get_dummies()` **before** the chronological train/test split.
- **Impact**: This is a procedural data leakage. The training set is given feature columns for all categories present in the entire dataset, including regions that might theoretically only appear in the future test set.
- **Recommendation**: Fit a categorical encoder (e.g., `sklearn.preprocessing.OneHotEncoder(handle_unknown='ignore')`) strictly on the training set after the split, and apply it to the test set.

### Pipeline Duplication (Risk of Training-Serving Skew)
- **Issue**: Complex feature engineering logic (sweep-line concurrency algorithm, cyclic temporal encoding, and spatial encoding) is copy-pasted verbatim across `model_cox.py`, `model_xgb.py`, and `evaluate.py`.
- **Impact**: Violates the DRY (Don't Repeat Yourself) principle. If a bug is fixed or a feature is tweaked in one file, it easily leads to training-serving skew if not identically updated everywhere, invalidating test evaluations.
- **Recommendation**: Centralize feature engineering into a single transformer class or function within `data_loader.py` or a dedicated `features.py` module.

## 2. Proper Application of Survival Models

### Misuse of Survival Analysis (Zero Censoring)
- **Issue**: `data_loader.py` intentionally filters out any rows missing a `finished_at` timestamp.
- **Impact**: This renders the dataset 100% fully observed (0% right-censored). The primary advantage of survival analysis—its ability to learn from ongoing, unfinished events—is completely lost.
- **Recommendation**: Retain active alerts as right-censored observations (e.g., `event = 0`), defining their duration as the time from `started_at` to the extraction timestamp. This allows the models to learn from currently active alerts.

### XGBoost Target Formatting
- **Observation**: XGBoost's `survival:cox` objective requires targets to be positive for observed events and negative for censored events.
- **Status**: Since the `data_loader.py` guarantees all durations are strictly positive (`duration > 0`) and fully observed, directly passing `duration_minutes` as `y_train` is mechanically correct for the current data state.

### Absence of Validation Split and Early Stopping
- **Issue**: The XGBoost model (`model_xgb.py`) trains for a fixed `n_estimators=200` without a validation set or early stopping.
- **Impact**: Tree-based survival models are highly prone to overfitting the hazard function. Evaluating directly on the test set without a validation split introduces a high risk of suboptimal generalization.
- **Recommendation**: Introduce a validation split (e.g., chronologically before the test split) to monitor validation hazard and apply early stopping.

## 3. Feature Engineering Analysis

### Concurrency Feature Validity
- **Observation**: The sweep-line algorithm calculates the `concurrency` (number of active alerts) exactly at the `started_at` timestamp of each new alert.
- **Status**: This is statistically sound and avoids forward-looking data leakage (lookahead bias). It only utilizes start and end times that have already occurred at or prior to `started_at`.
- **Note for Production**: The current global batch implementation requires all historical data simultaneously. In a live environment, this must be refactored into a stateful counter.

### Suboptimal Cyclic Temporal Encoding
- **Issue**: Temporal features are extracted using `dt.hour` directly (e.g., `np.sin(2 * np.pi * dt.hour / 24.0)`).
- **Impact**: `dt.hour` returns an integer, truncating the minutes. Consequently, an alert starting at 14:01 and one starting at 14:59 are identically encoded as `14.0`, creating an artificial step-function that degrades the continuous nature of time.
- **Recommendation**: Incorporate minutes (and seconds) into the calculation: `dt.hour + dt.minute / 60.0 + dt.second / 3600.0`.

## 4. Evaluation Metrics Adequacy

### C-Index vs. Absolute Error
- **Issue**: The evaluation script (`evaluate.py`) relies exclusively on the Concordance Index (C-index).
- **Impact**: C-index measures relative ranking (i.e., whether the model correctly predicts that Alert A lasts longer than Alert B). Because the dataset contains zero right-censored observations, evaluating absolute prediction error is entirely possible and significantly more relevant for end-users (who need to know *when* an alert will end, not just its relative rank).
- **Recommendation**: While C-index is standard for heavily censored survival data, this uncensored dataset should be evaluated with absolute metrics like Mean Absolute Error (MAE) or Root Mean Squared Error (RMSE) of the expected survival time. Additionally, proper survival scoring rules like the Integrated Brier Score (IBS) should be used to evaluate the calibration of the predicted survival curves over time.
