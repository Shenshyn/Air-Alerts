import os
import pickle
from pathlib import Path
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.utils import concordance_index

def main():
    # 1. Paths relative to this script
    project_root = Path(__file__).resolve().parent.parent
    data_path = project_root / "data" / "processed_alerts.csv"
    cox_model_path = project_root / "models" / "cox_model.pkl"
    xgb_model_path = project_root / "models" / "xgb_model.json"
    plots_dir = project_root / "plots"
    
    print(f"Project root identified as: {project_root}")
    print(f"Loading processed data from: {data_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"Processed data file not found at {data_path}")
        
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df):,} records.")
    
    # Preprocess dates and sort (identical to model training)
    df['started_at'] = pd.to_datetime(df['started_at'], utc=True)
    df['finished_at'] = pd.to_datetime(df['finished_at'], utc=True)
    df = df.sort_values(by='started_at').reset_index(drop=True)
    
    # Ensure duration_minutes is present
    if "duration_minutes" not in df.columns:
        if "duration" in df.columns:
            df["duration_minutes"] = df["duration"]
        else:
            raise KeyError("Neither 'duration_minutes' nor 'duration' column found in dataset.")
            
    # Calculate alert concurrency using sweep-line algorithm (identical to training scripts)
    print("Calculating alert concurrency using sweep-line algorithm...")
    start_times = df['started_at'].tolist()
    end_times = df['finished_at'].tolist()
    events_sweep = []
    for i in range(len(df)):
        events_sweep.append((start_times[i], 1, i))
        events_sweep.append((end_times[i], -1, i))
    events_sweep.sort(key=lambda x: (x[0], x[1]))
    
    concurrency = [0] * len(df)
    active_count = 0
    n_events = len(events_sweep)
    i = 0
    while i < n_events:
        current_time = events_sweep[i][0]
        ends = 0
        starts = []
        while i < n_events and events_sweep[i][0] == current_time:
            ev_type = events_sweep[i][1]
            idx = events_sweep[i][2]
            if ev_type == -1:
                ends += 1
            else:
                starts.append(idx)
            i += 1
        active_count -= ends
        if starts:
            active_count += len(starts)
            for idx in starts:
                concurrency[idx] = active_count - 1
    df['concurrency'] = concurrency
    
    # Generate cyclic temporal features
    print("Generating cyclic temporal features...")
    df['sin_hour'] = np.sin(2 * np.pi * df['started_at'].dt.hour / 24.0)
    df['cos_hour'] = np.cos(2 * np.pi * df['started_at'].dt.hour / 24.0)
    df['sin_dayofweek'] = np.sin(2 * np.pi * df['started_at'].dt.dayofweek / 7.0)
    df['cos_dayofweek'] = np.cos(2 * np.pi * df['started_at'].dt.dayofweek / 7.0)
    df['sin_month'] = np.sin(2 * np.pi * df['started_at'].dt.month / 12.0)
    df['cos_month'] = np.cos(2 * np.pi * df['started_at'].dt.month / 12.0)
    
    # Generate spatial features (one-hot encoding oblast, drop first)
    print("Generating spatial features (one-hot encoding)...")
    oblast_dummies = pd.get_dummies(df['oblast'], drop_first=True).astype(int)
    df = pd.concat([df, oblast_dummies], axis=1)
    
    # Define features and model columns
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
    
    # Chronological train/test split
    split_date = pd.to_datetime('2026-01-01 00:00:00+00:00', utc=True)
    train_mask = df['started_at'] < split_date
    test_mask = ~train_mask
    
    train_df = df[train_mask].copy()
    test_df = df[test_mask].copy()
    
    print(f"Train set size: {len(train_df):,} records")
    print(f"Test set size: {len(test_df):,} records")
    
    if len(test_df) == 0:
        raise ValueError("Test split is empty. Verify that there is data after '2026-01-01'.")
        
    # Prepare model inputs for evaluation
    test_model_df = test_df[model_columns].copy()
    X_test = test_model_df[feature_columns]
    y_test = test_model_df['duration_minutes']
    event_test = test_model_df['event']
    
    # ==========================================
    # 1. Regional Kaplan-Meier Baseline
    # ==========================================
    print("Evaluating Regional Kaplan-Meier baseline...")
    # Fit global KM to use as a fallback
    global_km = KaplanMeierFitter()
    global_km.fit(train_df['duration_minutes'], event_observed=train_df['event'])
    global_median = global_km.median_survival_time_
    
    # Fit regional KMs
    region_medians = {}
    for region, group in train_df.groupby('oblast'):
        kmf = KaplanMeierFitter()
        kmf.fit(group['duration_minutes'], event_observed=group['event'])
        median = kmf.median_survival_time_
        if pd.isna(median) or np.isinf(median):
            median = global_median
        region_medians[region] = median
        
    # Map predictions to test set
    test_df['km_pred'] = test_df['oblast'].map(region_medians).fillna(global_median)
    
    c_index_km = concordance_index(
        test_df['duration_minutes'],
        test_df['km_pred'],
        event_observed=test_df['event']
    )
    
    # ==========================================
    # 2. Cox Proportional Hazards Model
    # ==========================================
    print("Evaluating Cox Proportional Hazards model...")
    if not cox_model_path.exists():
        raise FileNotFoundError(f"Cox model pickle not found at {cox_model_path}")
    with open(cox_model_path, 'rb') as f:
        cph = pickle.load(f)
        
    # Score using lifelines' built-in score method (which computes concordance index)
    c_index_cox = cph.score(test_model_df, scoring_method='concordance_index')
    
    # ==========================================
    # 3. XGBoost Survival Model
    # ==========================================
    print("Evaluating XGBoost Survival model...")
    if not xgb_model_path.exists():
        raise FileNotFoundError(f"XGBoost model json not found at {xgb_model_path}")
    model_xgb = xgb.XGBRegressor()
    model_xgb.load_model(str(xgb_model_path))
    
    preds_xgb = model_xgb.predict(X_test)
    # Higher predicted hazard means shorter duration, so negate it for C-index
    c_index_xgb = concordance_index(
        y_test,
        -preds_xgb,
        event_observed=event_test
    )
    
    # ==========================================
    # 4. Print Comparison Table
    # ==========================================
    print("\n" + "="*60)
    print("MODEL COMPARISON RESULTS")
    print("="*60)
    print("| Model | Test Concordance Index (C-index) |")
    print("| :--- | :---: |")
    print(f"| Regional Kaplan-Meier Baseline | {c_index_km:.4f} |")
    print(f"| XGBoost Survival | {c_index_xgb:.4f} |")
    print(f"| Cox Proportional Hazards | {c_index_cox:.4f} |")
    print("="*60)
    
    # ==========================================
    # 5. Plot 1: C-index Comparison Bar Chart
    # ==========================================
    print("Generating C-index comparison bar chart...")
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(8, 6), dpi=300)
    plt.rcParams['font.sans-serif'] = 'Arial'
    plt.rcParams['font.family'] = 'sans-serif'
    
    models = ['Regional KM Baseline', 'XGBoost Survival', 'Cox PH']
    c_indices = [c_index_km, c_index_xgb, c_index_cox]
    colors = ['#8E9AA6', '#00838F', '#007ACC'] # Neutral grey-blue, Teal, Premium blue
    
    bars = plt.bar(models, c_indices, color=colors, width=0.55, edgecolor='#CCCCCC', linewidth=0.7)
    
    # Add baseline (random guessing) line
    plt.axhline(y=0.5, color='#E11D48', linestyle='--', linewidth=1.2, label='Random Guessing (0.5000)')
    
    # Styling details
    plt.title("Air Raid Alert Duration Prediction\nModel Comparison on Test Split (2026)", fontsize=13, fontweight='bold', pad=15)
    plt.ylabel("Concordance Index (C-index)", fontsize=11, labelpad=8)
    plt.ylim(0.4, 0.62)
    
    plt.grid(axis='y', linestyle=':', alpha=0.5, color='#CCCCCC')
    
    # Spines
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#888888')
    ax.spines['bottom'].set_color('#888888')
    
    # Annotate bar values
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2.0,
            height + 0.003,
            f"{height:.4f}",
            ha='center',
            va='bottom',
            fontsize=10,
            fontweight='bold'
        )
        
    plt.legend(loc='lower left', frameon=True, facecolor='#F8F9FA', edgecolor='#EAEAEA')
    plt.tight_layout()
    plot_comparison_path = plots_dir / "model_comparison.png"
    plt.savefig(plot_comparison_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Bar chart saved to: {plot_comparison_path}")
    
    # ==========================================
    # 6. Plot 2: Cox Survival Curves for Scenarios
    # ==========================================
    print("Generating Cox survival curves for 3 scenarios...")
    # Create scenario df
    scenarios_df = pd.DataFrame(0.0, index=[0, 1, 2], columns=feature_columns)
    
    # Scenario 1: Odeska oblast at 12:00 with 0 concurrency
    scenarios_df.loc[0, 'concurrency'] = 0.0
    scenarios_df.loc[0, 'sin_hour'] = np.sin(2 * np.pi * 12.0 / 24.0)
    scenarios_df.loc[0, 'cos_hour'] = np.cos(2 * np.pi * 12.0 / 24.0)
    if 'Odeska oblast' in feature_columns:
        scenarios_df.loc[0, 'Odeska oblast'] = 1.0
        
    # Scenario 2: Kharkivska oblast at 03:00 with 10 concurrency
    scenarios_df.loc[1, 'concurrency'] = 10.0
    scenarios_df.loc[1, 'sin_hour'] = np.sin(2 * np.pi * 3.0 / 24.0)
    scenarios_df.loc[1, 'cos_hour'] = np.cos(2 * np.pi * 3.0 / 24.0)
    if 'Kharkivska oblast' in feature_columns:
        scenarios_df.loc[1, 'Kharkivska oblast'] = 1.0
        
    # Scenario 3: Kyiv City at 18:00 with 3 concurrency
    scenarios_df.loc[2, 'concurrency'] = 3.0
    scenarios_df.loc[2, 'sin_hour'] = np.sin(2 * np.pi * 18.0 / 24.0)
    scenarios_df.loc[2, 'cos_hour'] = np.cos(2 * np.pi * 18.0 / 24.0)
    if 'Kyiv City' in feature_columns:
        scenarios_df.loc[2, 'Kyiv City'] = 1.0
        
    # Set neutral values for other features (Wednesday in June)
    for idx in [0, 1, 2]:
        scenarios_df.loc[idx, 'sin_dayofweek'] = np.sin(2 * np.pi * 2.0 / 7.0)
        scenarios_df.loc[idx, 'cos_dayofweek'] = np.cos(2 * np.pi * 2.0 / 7.0)
        scenarios_df.loc[idx, 'sin_month'] = np.sin(2 * np.pi * 6.0 / 12.0)
        scenarios_df.loc[idx, 'cos_month'] = np.cos(2 * np.pi * 6.0 / 12.0)
        
    # Predict survival function
    surv_curves = cph.predict_survival_function(scenarios_df)
    
    # Prepend time 0.0 with 1.0 probability
    surv_curves.loc[0.0] = [1.0, 1.0, 1.0]
    surv_curves = surv_curves.sort_index()
    
    plt.figure(figsize=(10, 6), dpi=300)
    
    # Define scenario descriptions and beautiful colors
    labels = [
        "Scenario 1: Odeska oblast (12:00, 0 concurrency)",
        "Scenario 2: Kharkivska oblast (03:00, 10 concurrency)",
        "Scenario 3: Kyiv City (18:00, 3 concurrency)"
    ]
    colors_scenarios = ['#2E7D32', '#E11D48', '#D97706'] # Forest Green, Crimson Rose, Amber Orange
    
    for i in range(3):
        plt.plot(
            surv_curves.index,
            surv_curves[i],
            color=colors_scenarios[i],
            linewidth=2.5,
            label=labels[i]
        )
        
    # Premium styling details
    plt.title("Predicted Cox Survival Curves by Scenario\nVisualizing Feature Effects on Alert Duration", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Duration (minutes)", fontsize=12, labelpad=10)
    plt.ylabel("Survival Probability (Alert Continuing)", fontsize=12, labelpad=10)
    
    plt.xlim(0, 240)
    plt.ylim(0, 1.05)
    
    plt.grid(True, linestyle=":", alpha=0.5, color="#CCCCCC")
    
    # Remove top and right spines
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#888888')
    ax.spines['bottom'].set_color('#888888')
    
    plt.legend(frameon=True, facecolor="#F8F9FA", edgecolor="#EAEAEA", fontsize=10, loc="upper right")
    plt.tight_layout()
    
    plot_predictions_path = plots_dir / "cox_predictions.png"
    plt.savefig(plot_predictions_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Survival curves plot saved to: {plot_predictions_path}")

if __name__ == '__main__':
    main()
