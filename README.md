<a href="README-ukr.md">
    <img alt="Ukrainian Version" src="https://img.shields.io/badge/%D0%A3%D0%BA%D1%80%D0%B0%D1%97%D0%BD%D1%81%D1%8C%D0%BA%D0%B8%D0%B9_%D0%B2%D0%B0%D1%80%D1%96%D0%B0%D0%BD%D1%82-FFD700?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCA5MDAgNjAwIj48cmVjdCB3aWR0aD0iOTAwIiBoZWlnaHQ9IjMwMCIgZmlsbD0iIzAwNTdiZCIvPjxyZWN0IHk9IjMwMCIgd2lkdGg9IjkwMCIgaGVpZ2h0PSIzMDAiIGZpbGw9IiNmZmM3MDAiLz48L3N2Zz4=">
  </a>

# Air Alerts Survival Analysis

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
**Solution**: We strictly calculated durations from `started_at` and `finished_at` and dropped null durations. Any unrealistic durations were handled appropriately, maintaining the integrity of our survival modelling.

## Methodology

### Feature Engineering
- **Concurrency**: The number of other active oblast-level alerts at the exact instant the current alert started. This was implemented efficiently in $O(N \log N)$ time using a sweep-line algorithm.
- **Temporal Features**: Cyclic encoding (sine and cosine transformations) of the hour, day of the week, and month to capture seasonality and time-of-day effects.
- **Spatial Features**: One-hot encoded oblasts to capture regional risk baselines.

### Models
1. **Kaplan-Meier Baseline**: Provides a non-parametric estimation of survival probabilities per region.
2. **Cox Proportional Hazards**: A semi-parametric model to evaluate the impact of our engineered features (hazard ratios).
3. **XGBoost Survival**: A gradient-boosted tree approach optimized with the `survival:cox` objective function to capture non-linear interactions.

## Model Performance

The models were evaluated using the Concordance Index (C-index), which measures the model's ability to correctly rank survival times.

| Model | Test C-index |
| --- | --- |
| **XGBoost (survival:cox)** | ~ 0.6524 |
| **Cox Proportional Hazards** | ~ 0.6382 |
| **Kaplan-Meier (Regional Baseline)** | ~ 0.62 |

*Note: A C-index of 0.5 is random guessing, and 1.0 is perfect ranking. Given the highly stochastic nature of air alerts, a C-index of ~0.65 indicates a strong predictive signal over the baseline.*

## Project Structure
- `src/data_loader.py`: Cleans data, removes duplicates, filters oblasts, calculates durations, and caches processed data.
- `src/baseline_km.py`: Fits Kaplan-Meier survival curves for key regions.
- `src/model_cox.py`: Trains the Cox PH model and extracts feature importance.
- `src/model_xgb.py`: Trains the XGBoost survival model.
- `src/evaluate.py`: Compares models using a stratified chronological split and generates performance plots.
- `plots/`: Contains generated visualizations (survival curves, model comparisons).

## Running the Code

1. Ensure you have the required libraries installed: `pandas`, `numpy`, `lifelines`, `xgboost`, `matplotlib`, `seaborn`.
2. Process the data: `python3 src/data_loader.py`
3. Run the evaluation pipeline: `python3 src/evaluate.py`
