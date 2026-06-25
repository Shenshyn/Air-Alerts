import os
from pathlib import Path
import pandas as pd
import numpy as np
import xgboost as xgb
from lifelines.utils import concordance_index

def main():
    # 1. Define paths relative to this script
    project_root = Path(__file__).resolve().parent.parent
    data_path = project_root / "data" / "processed_alerts.csv"
    models_dir = project_root / "models"
    model_path = models_dir / "xgb_model.json"

    print(f"Loading processed data from: {data_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"Processed alerts file not found at: {data_path}")

    # Load data
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df):,} records.")

    # Convert start/end times to datetime and sort
    df['started_at'] = pd.to_datetime(df['started_at'], utc=True)
    df['finished_at'] = pd.to_datetime(df['finished_at'], utc=True)
    df = df.sort_values(by='started_at').reset_index(drop=True)

    # 2. Build features
    # Ensure duration_minutes is present
    if "duration_minutes" not in df.columns:
        if "duration" in df.columns:
            df["duration_minutes"] = df["duration"]
        else:
            raise KeyError("Neither 'duration_minutes' nor 'duration' column found in dataset.")

    # Concurrency using O(N log N) sweep-line (event sorting) algorithm
    print("Calculating alert concurrency using sweep-line algorithm...")
    start_times = df['started_at'].tolist()
    end_times = df['finished_at'].tolist()

    events = []
    for i in range(len(df)):
        events.append((start_times[i], 1, i))
        events.append((end_times[i], -1, i))

    # Sort events: primary key time ascending, secondary key type ascending (-1 before 1)
    events.sort(key=lambda x: (x[0], x[1]))

    concurrency = [0] * len(df)
    active_count = 0
    n_events = len(events)
    i = 0
    while i < n_events:
        current_time = events[i][0]
        ends = 0
        starts = []
        while i < n_events and events[i][0] == current_time:
            ev_type = events[i][1]
            idx = events[i][2]
            if ev_type == -1:
                ends += 1
            else:
                starts.append(idx)
            i += 1
        
        active_count -= ends
        if starts:
            active_count += len(starts)
            for idx in starts:
                # Number of other active alerts is the total active minus 1 (the current alert itself)
                concurrency[idx] = active_count - 1

    df['concurrency'] = concurrency
    print(f"Concurrency calculation complete. Max concurrency: {max(concurrency)}")

    # Temporal features (cyclic encoding)
    print("Generating cyclic temporal features...")
    df['sin_hour'] = np.sin(2 * np.pi * df['started_at'].dt.hour / 24.0)
    df['cos_hour'] = np.cos(2 * np.pi * df['started_at'].dt.hour / 24.0)
    df['sin_dayofweek'] = np.sin(2 * np.pi * df['started_at'].dt.dayofweek / 7.0)
    df['cos_dayofweek'] = np.cos(2 * np.pi * df['started_at'].dt.dayofweek / 7.0)
    df['sin_month'] = np.sin(2 * np.pi * df['started_at'].dt.month / 12.0)
    df['cos_month'] = np.cos(2 * np.pi * df['started_at'].dt.month / 12.0)

    # Spatial features (one-hot encode oblast column, drop first to prevent collinearity)
    print("Generating spatial features (one-hot encoding)...")
    oblast_dummies = pd.get_dummies(df['oblast'], drop_first=True).astype(int)
    df = pd.concat([df, oblast_dummies], axis=1)

    # Define feature columns (exactly the same as model_cox.py covariates)
    feature_columns = [
        'concurrency',
        'sin_hour',
        'cos_hour',
        'sin_dayofweek',
        'cos_dayofweek',
        'sin_month',
        'cos_month'
    ] + list(oblast_dummies.columns)

    model_columns = ['duration_minutes', 'event'] + feature_columns

    # 3. Chronological train/test split
    split_date = pd.to_datetime('2026-01-01 00:00:00+00:00', utc=True)
    train_mask = df['started_at'] < split_date
    train_df = df[train_mask][model_columns].copy()
    test_df = df[~train_mask][model_columns].copy()

    print(f"Train split size: {len(train_df):,} alerts")
    print(f"Test split size: {len(test_df):,} alerts")

    # 4. Format targets for XGBoost Cox Survival:
    # Since all event observations are finished, positive values indicate observed survival times.
    X_train = train_df[feature_columns]
    y_train = train_df['duration_minutes']
    
    X_test = test_df[feature_columns]
    y_test = test_df['duration_minutes']

    # 5. Train XGBoost model using survival:cox objective
    print("Training XGBoost model with survival:cox objective...")
    model = xgb.XGBRegressor(
        objective='survival:cox',
        max_depth=4,
        learning_rate=0.05,
        n_estimators=200,
        random_state=42
    )
    
    model.fit(X_train, y_train)

    # 6. Evaluate performance
    # Predict hazard scores for the test split
    print("Predicting hazard scores for the test split...")
    preds = model.predict(X_test)

    # Calculate Concordance Index (C-index)
    # Note: higher hazard score means shorter duration (higher risk of alert ending).
    # lifelines.utils.concordance_index expects that higher predicted score indicates longer duration.
    # Therefore, we pass -preds (negative of predicted hazard score) to handle the sign correctly.
    test_c_index = concordance_index(
        test_df['duration_minutes'],
        -preds,
        event_observed=test_df['event']
    )

    print("\n" + "="*50)
    print("XGBOOST SURVIVAL MODEL EVALUATION")
    print("="*50)
    print(f"Test Concordance Index (C-index): {test_c_index:.4f}")
    print("="*50)

    # 7. Save the trained XGBoost model
    print(f"Creating directory: {models_dir}")
    models_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving model to: {model_path}")
    model.save_model(str(model_path))
    print("Model saved successfully!")

if __name__ == '__main__':
    main()
