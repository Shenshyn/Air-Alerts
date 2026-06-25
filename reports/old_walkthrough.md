# Air Raid Alert Duration Survival Analysis: Walkthrough

Summary of the completed pet project analyzing and predicting air raid alert durations using survival analysis in Python.

## 🚀 Accomplished Tasks

- **Git & Env Setup**: Initialized Git repository, set up remote to `https://github.com/Shenshyn/Air-Alerts.git`, and created `requirements.txt`.
- **Data Engineering (`data_loader.py`)**: 
  - Loaded raw official alerts dataset.
  - Eliminated **41.66%** duplicate rows (block duplication in crawler).
  - Scoped to `level == 'oblast'` to avoid localized tactical/artillery alert anomalies (e.g. 604-day alert in Lypetska hromada).
  - Calculated alert durations in minutes and cached cleaned records (65,134 unique oblast alerts).
- **Kaplan-Meier Baseline (`baseline_km.py`)**: Fitted global and regional baseline curves. Saved baseline curves to `plots/baseline_km.png`.
- **Cox Proportional Hazards Model (`model_cox.py`)**:
  - Implemented an $O(N \log N)$ sweep-line algorithm to calculate real-time alert concurrency (number of other active alerts at starting instant).
  - Built cyclic encoding (sine/cosine) for hours, day of week, and month.
  - Split data chronologically (Train before Jan 1, 2026; Test on/after Jan 1, 2026).
  - Saved model to `models/cox_model.pkl`.
- **XGBoost Survival Model (`model_xgb.py`)**: Trained XGBoost using `survival:cox` objective with the same features. Saved model to `models/xgb_model.json`.
- **Model Evaluation (`evaluate.py`)**: Generated comparison statistics and plots.
- **Git Push**: Pushed all codebase and outputs to the remote GitHub main branch.

---

## 📊 Model Comparison Results

Evaluation on the chronological test split (268 alerts on or after Jan 1, 2026):

| Model | Test Concordance Index (C-index) | Key Strengths / Caveats |
| :--- | :---: | :--- |
| **Cox Proportional Hazards** | **0.5759** | Best performance. Simpler linear structure generalizes better to distribution shifts. |
| **XGBoost Survival** | **0.5407** | Non-linear model. Suffers slightly from the small test set size/overfitting. |
| **Regional Kaplan-Meier Baseline** | **0.5000** | Grouped median estimator. Performs equivalent to random guessing due to test split structure. |

### Critical Technical Insights

1. **Why is the KM Baseline exactly 0.5000?**
   - After December 2025, the Ukrainian air defense registry transitioned from oblast-wide sirens to raion/hromada-level sirens.
   - The only exception is **Kyiv City**, which remained as a single unified oblast-level alert zone.
   - Consequently, our test set (all oblast-level alerts in 2026) contains **only Kyiv City alerts**.
   - The Kaplan-Meier baseline predicts a constant value (Kyiv City's training median: 39.97 minutes) for all test alerts, resulting in a C-index of exactly 0.5000 (complete ties).
   - Cox PH and XGBoost achieve predictive power by leveraging time-of-day and concurrency, which vary across Kyiv City alerts.

2. **Why Cox PH outperformed XGBoost:**
   - With a sparse, single-region test set (268 Kyiv City alerts), the complex non-linear combinations learned by XGBoost overfit the historical multi-region training set. The simpler linear structure of the Cox Proportional Hazards model generalized more robustly.

---

## 📈 Visualizations

### Model Performance Comparison
![Model Comparison](model_comparison.png)

### Cox Survival Probability Scenarios
![Cox Predictions](cox_predictions.png)
- **High Threat Intensity (Kharkivska, 03:00, 10 concurrent alerts - Red)**: Has a high hazard rate (survival probability drops sharply), predicting the alert will terminate quickly.
- **Low Threat Intensity (Odeska, 12:00, 0 concurrent alerts - Green)**: Has a low hazard rate, predicting a prolonged duration.
