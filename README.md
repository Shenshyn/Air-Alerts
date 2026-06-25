<a href="README-ukr.md">
    <img alt="Ukrainian Version" src="https://img.shields.io/badge/%D0%A3%D0%BA%D1%80%D0%B0%D1%97%D0%BD%D1%81%D1%8C%D0%BA%D0%B8%D0%B9_%D0%B2%D0%B0%D1%80%D1%96%D0%B0%D0%BD%D1%82-FFD700?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCA5MDAgNjAwIj48cmVjdCB3aWR0aD0iOTAwIiBoZWlnaHQ9IjMwMCIgZmlsbD0iIzAwNTdiZCIvPjxyZWN0IHk9IjMwMCIgd2lkdGg9IjkwMCIgaGVpZ2h0PSIzMDAiIGZpbGw9IiNmZmM3MDAiLz48L3N2Zz4=">
  </a>

# Air Alerts Survival Analysis

> [!NOTE]
> This project was developed as part of the selection process for the **KSE AI Agentic Summer School: Stage 2**.

This project performs survival analysis on Ukrainian air alert data to predict alert durations and understand the risk factors (like region, time of day, and concurrency). 
It employs Baseline Kaplan-Meier Estimators, Cox Proportional Hazards Models, and XGBoost Survival Models to estimate the duration of air alerts.

## Overview

The primary dataset contains historical air alert records for various regions in Ukraine. The goal of this project is to model the duration of these alerts, treating the end of an alert as the "event" in a survival analysis context. 

## Key Challenges and Solutions

During the development and analysis, we encountered several critical data challenges that required specific handling:

### 1. Massive Data Duplication
**Problem**: An audit of the raw data (`official_data_en.csv`) revealed that approximately 41.66% of the rows were exact duplicates. This was caused by block-level crawler repeats during the initial data scraping process.
**Solution**: Applied strict global deduplication (`df.drop_duplicates()`) as the first step in the data pipeline to ensure model training was not biased by duplicate records.

### 2. Methodology Shift and Evaluation Bias
**Problem**: In late 2025, the data collection methodology transitioned to recording alerts at the raion/hromada (district/community) level. Consequently, the 2026 data became heavily biased toward Kyiv City (which remained an oblast-level zone). A simple chronological split for testing (e.g., test set = 2026 data) resulted in a test set composed almost entirely of Kyiv City, destroying our ability to evaluate predictive power across the whole country.
**Solution**: We shifted from a naive chronological split to a **Stratified Chronological Split**. We performed an 80/20 chronological split *within each region independently*. This guarantees that the test set contains recent data representing the whole country proportionally. We also filtered the data to keep only `level == 'oblast'` strategic alerts, dropping raion/hromada level alerts to avoid localized noise from frontline artillery.

### 3. Collinearity and Singularity Errors in Cox Model
**Problem**: When fitting the Cox Proportional Hazards model, we encountered a `LinAlgError: Matrix is singular` error.
**Solution**: This was traced to extreme class imbalance: `Luhanska oblast` had only 2 recorded events in the dataset. Including regions with almost zero events creates a singular matrix in the Cox model's partial likelihood estimation. We filtered out `Luhanska oblast` and `Crimea` (regions without regular alert data) to stabilize the model. We also added a small L2 penalizer to the Cox model to prevent further collinearity issues.

### 4. "Naive" Censoring Artifacts
**Problem**: Some active alerts at the time of data export were naively censored by setting their `ended_at` to the export date (e.g., June 2026), creating artificially massive durations.
**Solution**: We strictly calculated durations from `started_at` and `finished_at`. Active/ongoing alerts without a finished time are properly modeled as right-censored observations (`event = 0`) with their duration computed up to the dataset's latest censor cutoff.

## Methodology

### Feature Engineering
- **Concurrency**: The number of other active oblast-level alerts at the exact instant the current alert started. This was implemented efficiently in $O(N \log N)$ time using a sweep-line algorithm.
- **Temporal Features**: Convert UTC timestamps to Ukraine local time (`Europe/Kyiv`). Continuous fractional hours are calculated ($\text{hour} + \frac{\text{minute}}{60} + \frac{\text{second}}{3600}$) rather than raw integer hours to ensure smooth cyclic encoding (sine/cosine) and avoid artificial step-functions.
- **Spatial Features**: One-hot encoded oblasts. To prevent data leakage, categories are learned strictly from the training split and mapped onto validation and test sets (representing unseen categories as all-zeros).

### Models
1. **Kaplan-Meier Baseline**: Provides a non-parametric estimation of survival probabilities per region.
2. **Cox Proportional Hazards**: A semi-parametric model to evaluate the impact of our engineered features (hazard ratios).
3. **XGBoost Survival**: A gradient-boosted tree approach optimized with the `survival:cox` objective function to capture non-linear interactions. It utilizes early stopping on a validation set and formats censored targets with negative durations as required by XGBoost.

## Model Performance

The models were evaluated on the stratified chronological test set (13,036 alerts across all regions) using the Concordance Index (C-index) for ranking and MAE/RMSE (in minutes) for absolute expected duration:

| Model | Test C-index | MAE (min) | RMSE (min) |
| :--- | :---: | :---: | :---: |
| **XGBoost Survival (Cox)** | **0.6397** | 84.11 | 152.56 |
| **Cox Proportional Hazards** | 0.6365 | **83.62** | **149.39** |
| **Regional Kaplan-Meier Baseline** | 0.5515 | 86.78 | 163.35 |

*Note: A C-index of 0.5 is random guessing, and 1.0 is perfect ranking. Given the highly stochastic nature of air alerts, a C-index of ~0.64 indicates a strong predictive signal over the baseline.*

## Project Structure
- `src/data_loader.py`: Cleans data, removes duplicates, filters oblasts, calculates durations, handles right-censorship, and caches processed data.
- `src/features.py`: Centralized feature engineering including concurrency, continuous temporal encoding, spatial encoding, and stratified splitting.
- `src/model_cox.py`: Trains the Cox PH model and extracts feature importance.
- `src/model_xgb.py`: Trains the XGBoost survival model with validation early stopping.
- `src/evaluate.py`: Compares models using a stratified chronological split and generates performance plots and expected survival metrics.
- `plots/`: Contains generated visualizations (survival curves, model comparisons).

## Running the Code

1. Ensure you have the required libraries installed: `pandas`, `numpy`, `lifelines`, `xgboost`, `matplotlib`, `scipy`.
2. Process the data: `python src/data_loader.py`
3. Run the evaluation pipeline (which also triggers model training):
   - First train Cox: `python src/model_cox.py`
   - Next train XGBoost: `python src/model_xgb.py`
   - Finally evaluate: `python src/evaluate.py`
