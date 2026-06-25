# Air Alerts Logic Audit Report

Based on a thorough review of the Python scripts in `src/`, several logical errors, hidden bugs, and unhandled edge cases were identified across the data processing and survival modeling pipelines.

## 1. Selection Bias / Discarding Right-Censored Data (`data_loader.py`)
**Severity:** Critical
**Location:** `data_loader.py`, lines 48-49 & 64
```python
valid_time_mask = df['started_at'].notna() & df['finished_at'].notna()
df = df[valid_time_mask].copy()
...
df['event'] = 1
```
**Description:** The script drops rows with a missing `finished_at` timestamp. In the context of active alert monitoring, a missing end time typically indicates an **ongoing** alert. By dropping these rows and hardcoding `event = 1`, the pipeline completely ignores **right-censored** data. This defeats the primary advantage of using Survival Analysis (CoxPH, XGBoost Survival) and introduces severe selection bias. Recent alerts that are particularly long are more likely to be ongoing and get dropped, meaning the models will systematically underestimate the duration of alerts.
**Expected Behavior:** Ongoing alerts should be kept, their duration computed up to the dataset extraction time, and marked as censored (`event = 0`).

## 2. Timezone Mishandling in Temporal Features (`model_cox.py`, `model_xgb.py`, `evaluate.py`)
**Severity:** High
**Location:** Cyclic feature generation
```python
df['sin_hour'] = np.sin(2 * np.pi * df['started_at'].dt.hour / 24.0)
```
**Description:** The `started_at` column is correctly parsed as UTC (`utc=True`). However, the scripts extract `dt.hour` directly from the UTC datetime. Air raids exhibit strong local diurnal patterns (e.g., concentrated at 02:00-04:00 local time). Using UTC shifts the hour feature by 2 to 3 hours from local time. Furthermore, because Ukraine observes Daylight Saving Time (DST), the offset changes between winter and summer, causing the same local hour to map to different UTC hours. 
**Expected Behavior:** The datetime should be converted to local time (e.g., `dt.tz_convert('Europe/Kyiv')`) before extracting `hour` and `dayofweek`.

## 3. Data Leakage and Pipeline Brittleness via `get_dummies`
**Severity:** High
**Location:** `model_cox.py` & `model_xgb.py`, executed *before* train/test split
```python
oblast_dummies = pd.get_dummies(df['oblast'], drop_first=True).astype(int)
...
train_mask = df['started_at'] < split_date
```
**Description:** `pd.get_dummies` is applied to the entire dataset *before* the chronological split. If an oblast appears only in the test set, it will still generate a column in the training set (populated with zeros). This leaks feature schema information from the test set into the train set. Additionally, if these scripts were used in production on live data, `pd.get_dummies` would fail to reproduce the exact columns if some regions don't have active alerts at that moment, leading to a mismatched matrix dimension crash during `.predict()`.
**Expected Behavior:** A robust encoder (like `sklearn.preprocessing.OneHotEncoder`) should be fitted strictly on the training split, and applied to the test split (handling unknown categories).

## 4. XGBoost Objective Implicitly Relies on Dropped Censored Data (`model_xgb.py`)
**Severity:** Medium (Silent failure risk)
**Location:** `model_xgb.py`, lines 114-115
```python
y_train = train_df['duration_minutes']
```
**Description:** The XGBoost `survival:cox` objective requires the target array `y` to use negative values to represent right-censored observations (e.g., `y = -duration` if `event == 0`). Currently, `y_train` strictly uses positive durations. This technically "works" right now because `data_loader.py` mistakenly drops all censored data (Bug #1). If `data_loader.py` is ever fixed to properly include ongoing alerts (`event=0`), the XGBoost model will silently treat those censored events as observed (completed) alerts, causing mathematically flawed training.
**Expected Behavior:** `y_train` should dynamically assign negative durations for rows where `event == 0`.

## 5. Harsh Quantization in Cyclic Temporal Encoding
**Severity:** Low / Modeling Flaw
**Location:** Cyclic feature generation
```python
df['sin_hour'] = np.sin(2 * np.pi * df['started_at'].dt.hour / 24.0)
```
**Description:** The `dt.hour` property returns an integer from 0 to 23. This means an alert starting at 10:01 and an alert starting at 10:59 both receive the exact same trigonometric values for `sin_hour` and `cos_hour`. This quantizes a continuous variable into 24 distinct steps, degrading the model's ability to smoothly model temporal proximity.
**Expected Behavior:** The fractional hour should be used (e.g., `time_in_hours = dt.hour + dt.minute / 60.0`).

## 6. Silently Masking Missing / Typo Scenarios (`evaluate.py`)
**Severity:** Low / Debugging Hindrance
**Location:** `evaluate.py`, lines 253, 260, 267
```python
if 'Odeska oblast' in feature_columns:
    scenarios_df.loc[0, 'Odeska oblast'] = 1.0
```
**Description:** Because `drop_first=True` was used for one-hot encoding, one oblast is silently dropped and acts as the "baseline" category (represented by all zeros). The `evaluate.py` scenario setup checks if an oblast name is in `feature_columns` before setting its column to `1.0`. If a developer makes a typo in the scenario oblast name, the `if` condition silently evaluates to false, leaving all columns as `0.0`. This means the script will silently output predictions for the baseline category instead of throwing an error or warning about the missing column.
