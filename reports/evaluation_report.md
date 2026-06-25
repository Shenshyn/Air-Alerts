# Air Raid Alert Duration Forecasting: Model Evaluation Report

This report presents the performance evaluation of three survival models designed to forecast the duration of air raid alerts in Ukraine:
1. **Regional Kaplan-Meier Baseline**: An empirical estimator mapping regional median survival times from the training set to the test set.
2. **XGBoost Survival**: A gradient-boosted decision tree model trained with a Cox objective (`survival:cox`).
3. **Cox Proportional Hazards**: A semi-parametric regression model fitted with L2 regularization (`penalizer=0.1`).

The evaluation was conducted on the chronological test split containing all air raid alerts started on or after **January 1, 2026** (totaling 268 records).

---

## 1. Performance Comparison (Concordance Index)

The **Concordance Index (C-index)** measures the rank correlation between predicted risk (hazard) and actual survival time (alert duration). A C-index of 0.5 indicates performance equivalent to random guessing, while a C-index of 1.0 indicates perfect ordering.

| Model | Test Concordance Index (C-index) | Description |
| :--- | :---: | :--- |
| **Cox Proportional Hazards** | **0.5759** | Semi-parametric model using cyclic temporal, concurrency, and spatial features. |
| **XGBoost Survival** | **0.5407** | Non-linear tree model using the same feature set. |
| **Regional Kaplan-Meier Baseline** | **0.5000** | Grouped median estimator based on oblast. |

### Performance Insights:
- **Cox Proportional Hazards** achieves the highest C-index of **0.5759**, showing that the linear combination of coefficients successfully separates different hazard levels.
- **XGBoost Survival** achieves a C-index of **0.5407**. While it captures non-linear relationships, on this specific test set, it is slightly outperformed by the Cox PH model, potentially due to the small size of the test split or overfitting on complex interaction patterns.
- **Regional Kaplan-Meier Baseline** yields a C-index of exactly **0.5000** (equivalent to random guessing). 

---

## 2. Why is the Kaplan-Meier Baseline exactly 0.5000?

A C-index of 0.5000 indicates that the model has no predictive capacity to rank the alerts by duration. The reason for this lies in the distribution of the test set:
- **Test Set Filter**: The test set is defined by `started_at >= '2026-01-01 00:00:00+00:00'`.
- **Region Homogeneity**: An inspection of the test set reveals that it contains **only Kyiv City alerts** (268 out of 268 records).
- **Constant Predictions**: Because the Regional Kaplan-Meier baseline calculates predictions by mapping each oblast to its median survival time computed during training, and the test split consists entirely of Kyiv City alerts, **every single alert in the test set receives the exact same predicted value** (Kyiv City's training median: `39.97` minutes).
- **Ties in C-index**: With no variation in predicted values, the concordance calculation encounters complete ties for all pairs, resulting in a concordance index of exactly **0.5000**.

In contrast, the Cox PH and XGBoost models utilize temporal features (e.g. time of day cyclic variables) and concurrency, which vary across the 268 alerts in Kyiv City, allowing them to achieve a non-trivial predictive C-index.

---

## 3. Visualizations

Two visualization plots have been generated and saved to the project:

### Model Performance Comparison
![Model Comparison](/Users/alex/.gemini/antigravity/brain/b98655bc-b24c-4c35-8d6c-a1b6056ca941/model_comparison.png)

This plot shows the C-indices of the three models on the 2026 test set. The red dashed line indicates the random guessing baseline (0.5000).

### Cox Survival Curves Scenario Analysis
![Cox Prediction Curves](/Users/alex/.gemini/antigravity/brain/b98655bc-b24c-4c35-8d6c-a1b6056ca941/cox_predictions.png)

This plot displays the predicted survival probabilities (the probability that the alert is still continuing) over time for 3 different scenarios:
- **Scenario 1 (Green)**: Odeska oblast at 12:00 (noon) with 0 concurrency.
- **Scenario 2 (Red)**: Kharkivska oblast at 03:00 (night) with 10 concurrency.
- **Scenario 3 (Orange)**: Kyiv City at 18:00 (evening) with 3 concurrency.

#### Interpretation of Scenario Curves:
- **Scenario 1 (Odeska, 12:00, 0 concurrency)** shows a much higher survival probability over the first 1-2 hours compared to the others, meaning alerts in this scenario tend to last longer.
- **Scenario 2 (Kharkivska, 03:00, 10 concurrency)** drops much more sharply. A high concurrency of alerts across Ukraine combined with night-time hours dramatically increases the hazard rate, indicating that individual alerts are predicted to terminate much more quickly under these conditions.
- **Scenario 3 (Kyiv City, 18:00, 3 concurrency)** lies intermediate between the low-hazard and high-hazard profiles.
