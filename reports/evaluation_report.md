# Model Evaluation and Comparison Report

This report presents the comparative evaluation of baseline and machine learning models trained on preprocessed oblast-level air raid alerts using a stratified chronological split (80% train / 20% test per region).

## Performance Metrics Table

| Model | Test Concordance Index (C-index) | MAE (minutes) | RMSE (minutes) |
| :--- | :---: | :---: | :---: |
| **Regional Kaplan-Meier Baseline** | 0.5515 | 86.78 | 163.35 |
| **Cox Proportional Hazards** | 0.6365 | 83.62 | 149.39 |
| **XGBoost Survival (Cox)** | 0.6397 | 84.11 | 152.56 |

> [!NOTE]
> - **C-index**: Measures the model's ability to correctly order alert durations (higher is better).
> - **MAE & RMSE**: Measures the absolute difference in minutes between the predicted expected survival time and the actual alert duration on observed events (lower is better).
