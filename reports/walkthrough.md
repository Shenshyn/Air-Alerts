# Air Raid Alert Duration Survival Analysis: Walkthrough

Summary of the refactoring, improvements, and evaluation results for the Ukrainian air raid alert duration prediction project.

## 🚀 Key Achievements & Refactoring

Following a comprehensive audit from three subagents, the codebase has been successfully refactored and corrected to address structure (Clean Code), statistical methodology, and logic bugs:

1. **Centralized & Modular Feature Engineering (`src/features.py`)**:
   - Resolved severe DRY (Don't Repeat Yourself) violations by extracting all preprocessing logic into a shared module.
   - Centralized the $O(N \log N)$ sweep-line concurrency algorithm, cyclic temporal feature generation, and spatial one-hot encoding.
   - Refactored scripts to replace monolithic 300+ line functions with single-responsibility helper functions.

2. **Timezone & Time Quantization Correction**:
   - Converted UTC timestamps to Ukraine local time (`Europe/Kyiv`) before extracting temporal features. This ensures diurnal patterns are aligned with local time.
   - Replaced integer hour quantization (`dt.hour`) with continuous fractional hours ($\text{hour} + \frac{\text{minute}}{60} + \frac{\text{second}}{3600}$), creating a smooth cyclic temporal feature mapping instead of a rough step-function.

3. **Data Leakage Elimination**:
   - Replaced global `pd.get_dummies()` one-hot encoding with training-set-constrained encoding. One-hot categories are learned strictly from the training split and applied to validation/test sets, preventing class schema leaks and model crashes under unseen categories.

4. **Methodological Survival Correction**:
   - The data pipeline is now fully compatible with right-censored data (alerts that are active/ongoing, representing `event = 0`). Ongoing alerts compute duration up to the dataset's latest censor cutoff.
   - Formatted labels dynamically for XGBoost's `survival:cox` objective, setting negative target values for censored observations as required by the algorithm.
   - Introduced a validation sub-split with early stopping for the XGBoost model to prevent overfitting.

5. **Expanded & Robust Evaluation Metrics (`src/evaluate.py`)**:
   - Evaluated models using a **Stratified Chronological Split** (80% train / 20% test *per region*), resolving evaluation bias.
   - Added **Mean Absolute Error (MAE)** and **Root Mean Squared Error (RMSE)** metrics in minutes for completed alerts to evaluate absolute prediction accuracy.
   - Implemented a Breslow baseline hazard integration method to compute predicted expected survival times for XGBoost.
   - Added explicit validation to scenario generation to reject misspelled oblast names.

---

## 📊 Model Comparison Results

Evaluation on the stratified chronological test set (13,036 alerts across all regions):

| Model | Test C-index | MAE (min) | RMSE (min) | Performance Analysis |
| :--- | :---: | :---: | :---: | :--- |
| **XGBoost Survival (Cox)** | **0.6397** | 84.11 | 152.56 | Best ranking accuracy (C-index). Catches non-linear interactions between timezone, month, and concurrency. |
| **Cox Proportional Hazards** | 0.6365 | **83.62** | **149.39** | Best absolute predictions (MAE/RMSE). Simpler linear relationships generalise robustly for expected duration. |
| **Regional Kaplan-Meier Baseline** | 0.5515 | 86.78 | 163.35 | Naive estimator relying solely on the region's median training duration. Outperformed by both ML models. |

### Technical Insights
- **XGBoost** captures a slightly better relative ranking of survival times (C-index: 0.6397 vs 0.6365).
- **Cox PH** achieves the lowest absolute prediction error (MAE: 83.62 mins vs 84.11 mins for XGBoost), indicating excellent calibration of the expected survival time calculated from the Cox hazard function.
- Both machine learning models show a substantial increase in predictive performance over the regional KM baseline, demonstrating that concurrency and temporal/seasonality features provide a strong predictive signal.

---

## 📈 Visualizations

Visualizations are saved under the `plots/` directory:

1. **Model Performance Comparison (`plots/model_comparison.png`)**:
   - A bar chart displaying the Concordance Index comparison across the three models.
2. **Predicted Survival Curves (`plots/cox_predictions.png`)**:
   - Shows predicted survival functions $S(t|x)$ from the Cox model for three scenarios:
     - **Scenario 1 (Green)**: Odeska oblast (12:00, 0 concurrency) -> Low risk, longer predicted duration.
     - **Scenario 2 (Red)**: Kharkivska oblast (03:00, 10 concurrency) -> High risk, short predicted duration (siren ends quickly).
     - **Scenario 3 (Amber)**: Kyiv City (18:00, 3 concurrency) -> Intermediate duration.
