import os
from pathlib import Path
import pandas as pd
import numpy as np
import xgboost as xgb
from lifelines.utils import concordance_index

from features import (
    calculate_concurrency,
    extract_temporal_features,
    encode_spatial_features,
    split_data_stratified
)

# Define paths relative to the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "processed_alerts.csv"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models"
DEFAULT_MODEL_PATH = DEFAULT_MODEL_DIR / "xgb_model.json"

def load_and_prepare_data(data_path: Path = DEFAULT_DATA_PATH) -> tuple:
    """
    Loads processed data, extracts global features, performs train/validation/test 
    splits chronologically and per-region, and encodes spatial features leakage-free.
    """
    print(f"Loading processed data from: {data_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"Processed alerts file not found at: {data_path}")
        
    df = pd.read_csv(data_path)
    df['started_at'] = pd.to_datetime(df['started_at'], utc=True)
    df['finished_at'] = pd.to_datetime(df['finished_at'], utc=True)
    
    # Sort for concurrency calculation
    df = df.sort_values(by='started_at').reset_index(drop=True)
    
    # 1. Global features (safe from lookahead bias)
    print("Calculating concurrency and temporal features...")
    df['concurrency'] = calculate_concurrency(df)
    df = extract_temporal_features(df)
    
    # 2. Chronological Split (80% train / 20% test per region)
    print("Splitting train and test sets (80/20 per region)...")
    train_raw, test_raw = split_data_stratified(df, train_ratio=0.8)
    
    # 3. Further split train into train/validation (90% sub-train / 10% validation)
    print("Splitting train into sub-train and validation sets (90/10 per region)...")
    train_sub_raw, val_raw = split_data_stratified(train_raw, train_ratio=0.9)
    
    # 4. Spatial encoding (leakage-free based on train set only)
    print("Encoding spatial features...")
    train_df, train_oblasts, spatial_cols = encode_spatial_features(train_sub_raw, is_train=True)
    val_df, _, _ = encode_spatial_features(val_raw, train_oblasts=train_oblasts, is_train=False)
    test_df, _, _ = encode_spatial_features(test_raw, train_oblasts=train_oblasts, is_train=False)
    
    feature_cols = [
        'concurrency',
        'sin_hour', 'cos_hour',
        'sin_dayofweek', 'cos_dayofweek',
        'sin_month', 'cos_month'
    ] + spatial_cols
    
    return train_df, val_df, test_df, feature_cols, train_oblasts

def format_targets(df: pd.DataFrame) -> np.ndarray:
    """
    Formats the target array for XGBoost survival:cox objective.
    For survival:cox:
    - Positive values represent observed survival times (event == 1).
    - Negative values represent right-censored survival times (event == 0).
    """
    # np.where multiplies by -1.0 if event is 0 (censored), and 1.0 if event is 1 (observed)
    return df['duration'].values * np.where(df['event'].values == 1, 1.0, -1.0)

def train_xgb_model(
    train_df: pd.DataFrame, 
    val_df: pd.DataFrame, 
    feature_cols: list
) -> xgb.XGBRegressor:
    """
    Trains an XGBoost Survival Regressor with early stopping to prevent overfitting.
    """
    X_train = train_df[feature_cols]
    y_train = format_targets(train_df)
    
    X_val = val_df[feature_cols]
    y_val = format_targets(val_df)
    
    print("Training XGBoost Regressor with survival:cox objective...")
    model = xgb.XGBRegressor(
        objective='survival:cox',
        max_depth=4,
        learning_rate=0.05,
        n_estimators=500,  # Halt earlier via early stopping
        early_stopping_rounds=15,
        random_state=42
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=50
    )
    
    return model

def evaluate_model(model: xgb.XGBRegressor, test_df: pd.DataFrame, feature_cols: list) -> None:
    """
    Evaluates the XGBoost model on the test split using C-index.
    """
    X_test = test_df[feature_cols]
    
    # Predict hazard scores (higher hazard score means higher risk of alert ending, i.e., shorter duration)
    preds = model.predict(X_test)
    
    # Calculate Concordance Index (C-index)
    # Pass -preds (negative of predicted hazard score) so higher values represent longer durations.
    test_c_index = concordance_index(
        test_df['duration'],
        -preds,
        event_observed=test_df['event']
    )
    
    print("\n" + "="*50)
    print("XGBOOST SURVIVAL MODEL EVALUATION")
    print("="*50)
    print(f"Test Concordance Index (C-index): {test_c_index:.4f}")
    print("="*50)

def save_model(model: xgb.XGBRegressor, train_oblasts: list, feature_cols: list, model_path: Path = DEFAULT_MODEL_PATH) -> None:
    """
    Saves the XGBoost model and its associated metadata.
    """
    model_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving XGBoost model to: {model_path}")
    
    # Save the booster model JSON file
    model.save_model(str(model_path))
    
    # Save metadata (oblasts list & feature columns) as a sibling pickle file
    meta_path = model_path.with_suffix('.meta.pkl')
    print(f"Saving model metadata to: {meta_path}")
    
    metadata = {
        'train_oblasts': train_oblasts,
        'feature_cols': feature_cols
    }
    
    import pickle
    with open(meta_path, 'wb') as f:
        pickle.dump(metadata, f)
        
    print("XGBoost model and metadata saved successfully!")

def main() -> None:
    print("--- Air Alerts XGBoost Survival Model Trainer ---")
    try:
        train_df, val_df, test_df, feature_cols, train_oblasts = load_and_prepare_data()
        
        print(f"Train sub-split size: {len(train_df):,} alerts")
        print(f"Val split size:       {len(val_df):,} alerts")
        print(f"Test split size:      {len(test_df):,} alerts")
        
        model = train_xgb_model(train_df, val_df, feature_cols)
        evaluate_model(model, test_df, feature_cols)
        save_model(model, train_oblasts, feature_cols, DEFAULT_MODEL_PATH)
        
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
