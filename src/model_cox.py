import os
import pickle
from pathlib import Path
import pandas as pd
from lifelines import CoxPHFitter

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
DEFAULT_MODEL_PATH = DEFAULT_MODEL_DIR / "cox_model.pkl"

def load_and_prepare_data(data_path: Path = DEFAULT_DATA_PATH) -> tuple:
    """
    Loads preprocessed alerts, calculates global concurrency and temporal features,
    splits them chronologically, and encodes spatial features leakage-free.
    """
    print(f"Loading processed data from: {data_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"Processed alerts file not found at: {data_path}")
        
    df = pd.read_csv(data_path)
    df['started_at'] = pd.to_datetime(df['started_at'], utc=True)
    df['finished_at'] = pd.to_datetime(df['finished_at'], utc=True)
    
    # Ensure sorted order for concurrency
    df = df.sort_values(by='started_at').reset_index(drop=True)
    
    # 1. Global features (safe from lookahead bias)
    print("Calculating concurrency and temporal features...")
    df['concurrency'] = calculate_concurrency(df)
    df = extract_temporal_features(df)
    
    # 2. Stratified Chronological Split (80% train / 20% test per region)
    print("Performing stratified chronological split (80/20 per region)...")
    train_raw, test_raw = split_data_stratified(df, train_ratio=0.8)
    
    # 3. Spatial encoding (leakage-free)
    print("Encoding spatial features...")
    train_df, train_oblasts, spatial_cols = encode_spatial_features(train_raw, is_train=True)
    test_df, _, _ = encode_spatial_features(test_raw, train_oblasts=train_oblasts, is_train=False)
    
    feature_cols = [
        'concurrency',
        'sin_hour', 'cos_hour',
        'sin_dayofweek', 'cos_dayofweek',
        'sin_month', 'cos_month'
    ] + spatial_cols
    
    return train_df, test_df, feature_cols, train_oblasts

def train_cox_model(train_df: pd.DataFrame, feature_cols: list) -> CoxPHFitter:
    """
    Trains a Cox Proportional Hazards model with L2 regularization.
    """
    print("Fitting Cox Proportional Hazards model (penalizer=0.1)...")
    cph = CoxPHFitter(penalizer=0.1)
    
    # Keep only target and features
    model_cols = ['duration', 'event'] + feature_cols
    train_data = train_df[model_cols].copy()
    
    cph.fit(train_data, duration_col='duration', event_col='event')
    return cph

def evaluate_model(cph: CoxPHFitter, test_df: pd.DataFrame, feature_cols: list) -> None:
    """
    Evaluates the model using C-index and prints top coefficients.
    """
    model_cols = ['duration', 'event'] + feature_cols
    test_data = test_df[model_cols].copy()
    
    test_c_index = cph.score(test_data, scoring_method='concordance_index')
    
    print("\n" + "="*50)
    print("COX PH MODEL EVALUATION")
    print("="*50)
    print(f"Test Concordance Index (C-index): {test_c_index:.4f}")
    print("="*50)
    
    # Print top 5 coefficients by absolute value
    summary_df = cph.summary.copy()
    summary_df['abs_coef'] = summary_df['coef'].abs()
    top_5 = summary_df.sort_values(by='abs_coef', ascending=False).head(5)
    
    print("\nTop 5 Coefficients (by absolute value):")
    print(f"{'Covariate':<30} | {'Coefficient':<12} | {'Hazard Ratio':<12} | {'P-value':<10}")
    print("-" * 75)
    for cov, row in top_5.iterrows():
        print(f"{cov:<30} | {row['coef']:12.4f} | {row['exp(coef)']:12.4f} | {row['p']:.4e}")
    print("="*50)

def save_model(cph: CoxPHFitter, train_oblasts: list, feature_cols: list, model_path: Path = DEFAULT_MODEL_PATH) -> None:
    """
    Saves the trained model and metadata for inference/evaluation.
    """
    model_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving Cox model to: {model_path}")
    
    model_package = {
        'model': cph,
        'train_oblasts': train_oblasts,
        'feature_cols': feature_cols
    }
    
    with open(model_path, 'wb') as f:
        pickle.dump(model_package, f)
    print("Model and metadata saved successfully!")

def main() -> None:
    print("--- Air Alerts Cox Model Trainer ---")
    try:
        train_df, test_df, feature_cols, train_oblasts = load_and_prepare_data()
        
        print(f"Train set size: {len(train_df):,} alerts")
        print(f"Test set size:  {len(test_df):,} alerts")
        
        cph = train_cox_model(train_df, feature_cols)
        evaluate_model(cph, test_df, feature_cols)
        save_model(cph, train_oblasts, feature_cols, DEFAULT_MODEL_PATH)
        
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
