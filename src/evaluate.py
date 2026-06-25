import os
import pickle
from pathlib import Path
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from lifelines import KaplanMeierFitter
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
DEFAULT_COX_MODEL_PATH = PROJECT_ROOT / "models" / "cox_model.pkl"
DEFAULT_XGB_MODEL_PATH = PROJECT_ROOT / "models" / "xgb_model.json"
DEFAULT_PLOTS_DIR = PROJECT_ROOT / "plots"

def load_models_and_metadata() -> tuple:
    """
    Loads trained Cox and XGBoost models and their respective training metadata.
    """
    print(f"Loading Cox model package from: {DEFAULT_COX_MODEL_PATH}")
    if not DEFAULT_COX_MODEL_PATH.exists():
        raise FileNotFoundError(f"Cox model not found at: {DEFAULT_COX_MODEL_PATH}")
    with open(DEFAULT_COX_MODEL_PATH, 'rb') as f:
        cox_package = pickle.load(f)
        
    print(f"Loading XGBoost model from: {DEFAULT_XGB_MODEL_PATH}")
    if not DEFAULT_XGB_MODEL_PATH.exists():
        raise FileNotFoundError(f"XGBoost model not found at: {DEFAULT_XGB_MODEL_PATH}")
    model_xgb = xgb.XGBRegressor()
    model_xgb.load_model(str(DEFAULT_XGB_MODEL_PATH))
    
    xgb_meta_path = DEFAULT_XGB_MODEL_PATH.with_suffix('.meta.pkl')
    print(f"Loading XGBoost metadata from: {xgb_meta_path}")
    if not xgb_meta_path.exists():
        raise FileNotFoundError(f"XGBoost metadata not found at: {xgb_meta_path}")
    with open(xgb_meta_path, 'rb') as f:
        xgb_metadata = pickle.load(f)
        
    return cox_package, model_xgb, xgb_metadata

def load_and_prepare_test_data(train_oblasts: list, data_path: Path = DEFAULT_DATA_PATH) -> tuple:
    """
    Loads raw preprocessed data, computes global features, splits into train/test,
    and prepares the test set features using the training oblasts mapping to avoid leakage.
    """
    print(f"Loading processed data from: {data_path}")
    df = pd.read_csv(data_path)
    df['started_at'] = pd.to_datetime(df['started_at'], utc=True)
    df['finished_at'] = pd.to_datetime(df['finished_at'], utc=True)
    
    # Sort for concurrency calculation
    df = df.sort_values(by='started_at').reset_index(drop=True)
    
    # 1. Global features
    df['concurrency'] = calculate_concurrency(df)
    df = extract_temporal_features(df)
    
    # 2. Stratified chronological split (80/20 per region)
    train_raw, test_raw = split_data_stratified(df, train_ratio=0.8)
    
    # 3. Spatial encoding (using train_oblasts to align columns, avoiding leakage)
    train_df, _, spatial_cols = encode_spatial_features(train_raw, train_oblasts=train_oblasts, is_train=False)
    test_df, _, _ = encode_spatial_features(test_raw, train_oblasts=train_oblasts, is_train=False)
    
    feature_cols = [
        'concurrency',
        'sin_hour', 'cos_hour',
        'sin_dayofweek', 'cos_dayofweek',
        'sin_month', 'cos_month'
    ] + spatial_cols
    
    return train_df, test_df, feature_cols

def compute_xgb_expected_survival(
    xgb_model: xgb.XGBRegressor, 
    train_df: pd.DataFrame, 
    test_df: pd.DataFrame, 
    feature_cols: list, 
    max_time: float = 240.0
) -> np.ndarray:
    """
    Computes the expected survival time (in minutes) for the test set using a 
    Breslow estimator of the baseline cumulative hazard based on the training set.
    """
    X_train = train_df[feature_cols].copy()
    y_train = train_df['duration'].values
    events_train = train_df['event'].values
    
    X_test = test_df[feature_cols].copy()
    
    # Get training log-hazard predictions
    eta_train = xgb_model.predict(X_train)
    exp_eta_train = np.exp(eta_train)
    
    # Sort training data by duration
    sort_idx = np.argsort(y_train)
    y_train_sorted = y_train[sort_idx]
    events_train_sorted = events_train[sort_idx]
    exp_eta_train_sorted = exp_eta_train[sort_idx]
    
    # Calculate Breslow baseline hazard contribution at each unique event time
    # risk_sums[i] is the sum of exp(eta) for all instances j where duration_j >= duration_i
    risk_sums = np.cumsum(exp_eta_train_sorted[::-1])[::-1]
    
    baseline_hazard = np.zeros(len(y_train_sorted))
    for i in range(len(y_train_sorted)):
        if events_train_sorted[i] == 1 and risk_sums[i] > 0:
            baseline_hazard[i] = 1.0 / risk_sums[i]
            
    baseline_cum_hazard = np.cumsum(baseline_hazard)
    
    # Set up time grid for integration (0 to max_time in minutes)
    grid_times = np.linspace(0, max_time, int(max_time) + 1)
    
    # Map cumulative baseline hazard to the grid times using a step function
    cum_hazard_func = interp1d(
        np.concatenate([[0.0], y_train_sorted]),
        np.concatenate([[0.0], baseline_cum_hazard]),
        kind='previous',
        bounds_error=False,
        fill_value=(0.0, baseline_cum_hazard[-1])
    )
    grid_baseline_cum_hazard = cum_hazard_func(grid_times)
    
    # Get test log-hazard predictions
    eta_test = xgb_model.predict(X_test)
    exp_eta_test = np.exp(eta_test).reshape(-1, 1) # shape: (num_test, 1)
    grid_baseline_cum_hazard = grid_baseline_cum_hazard.reshape(1, -1) # shape: (1, num_grid)
    
    # Compute test survival curves: S(t|x) = exp( - H_0(t) * exp(eta) )
    # shape: (num_test, num_grid)
    surv_curves = np.exp(- grid_baseline_cum_hazard * exp_eta_test)
    
    # Integrate survival probability over the grid using the trapezoidal rule
    expected_survival = np.trapezoid(surv_curves, grid_times, axis=1)
    return expected_survival

def set_scenario_oblast(scenarios_df: pd.DataFrame, idx: int, oblast_name: str, train_oblasts: list, feature_cols: list) -> None:
    """
    Sets the one-hot columns in the scenarios DataFrame for the specified oblast,
    handling baseline dropped categories and raising an error on typos.
    """
    col_name = f"oblast_{oblast_name}"
    
    if col_name in feature_cols:
        scenarios_df.loc[idx, col_name] = 1.0
        print(f"Scenario {idx+1}: Set active oblast to '{oblast_name}'.")
    elif oblast_name == train_oblasts[0]:
        # Baseline oblast (represented by all zeros)
        print(f"Scenario {idx+1}: Oblast '{oblast_name}' is the baseline dropped category (encoded as all-zeros).")
    else:
        raise ValueError(f"Oblast '{oblast_name}' is not recognized. Please check spelling. Available: {train_oblasts}")

def generate_evaluation_plots(cph, test_df: pd.DataFrame, train_oblasts: list, feature_cols: list) -> None:
    """
    Generates predicted survival curves for key scenarios using the Cox model.
    """
    print("Generating scenario survival curves...")
    scenarios_df = pd.DataFrame(0.0, index=[0, 1, 2], columns=feature_cols)
    
    # Scenario 1: Odeska oblast at 12:00 with 0 concurrency
    scenarios_df.loc[0, 'concurrency'] = 0.0
    scenarios_df.loc[0, 'sin_hour'] = np.sin(2 * np.pi * 12.0 / 24.0)
    scenarios_df.loc[0, 'cos_hour'] = np.cos(2 * np.pi * 12.0 / 24.0)
    set_scenario_oblast(scenarios_df, 0, 'Odeska oblast', train_oblasts, feature_cols)
    
    # Scenario 2: Kharkivska oblast at 03:00 with 10 concurrency
    scenarios_df.loc[1, 'concurrency'] = 10.0
    scenarios_df.loc[1, 'sin_hour'] = np.sin(2 * np.pi * 3.0 / 24.0)
    scenarios_df.loc[1, 'cos_hour'] = np.cos(2 * np.pi * 3.0 / 24.0)
    set_scenario_oblast(scenarios_df, 1, 'Kharkivska oblast', train_oblasts, feature_cols)
    
    # Scenario 3: Kyiv City at 18:00 with 3 concurrency
    scenarios_df.loc[2, 'concurrency'] = 3.0
    scenarios_df.loc[2, 'sin_hour'] = np.sin(2 * np.pi * 18.0 / 24.0)
    scenarios_df.loc[2, 'cos_hour'] = np.cos(2 * np.pi * 18.0 / 24.0)
    set_scenario_oblast(scenarios_df, 2, 'Kyiv City', train_oblasts, feature_cols)
    
    # Set neutral day/month parameters (Wednesday in June)
    for idx in [0, 1, 2]:
        scenarios_df.loc[idx, 'sin_dayofweek'] = np.sin(2 * np.pi * 2.0 / 7.0)
        scenarios_df.loc[idx, 'cos_dayofweek'] = np.cos(2 * np.pi * 2.0 / 7.0)
        scenarios_df.loc[idx, 'sin_month'] = np.sin(2 * np.pi * 6.0 / 12.0)
        scenarios_df.loc[idx, 'cos_month'] = np.cos(2 * np.pi * 6.0 / 12.0)
        
    surv_curves = cph.predict_survival_function(scenarios_df)
    
    # Prepend time 0.0 with 1.0 probability
    surv_curves.loc[0.0] = [1.0, 1.0, 1.0]
    surv_curves = surv_curves.sort_index()
    
    plt.figure(figsize=(10, 6), dpi=300)
    labels = [
        "Scenario 1: Odeska oblast (12:00, 0 concurrency)",
        "Scenario 2: Kharkivska oblast (03:00, 10 concurrency)",
        "Scenario 3: Kyiv City (18:00, 3 concurrency)"
    ]
    colors_scenarios = ['#2E7D32', '#E11D48', '#D97706'] # Green, Red, Amber
    
    for i in range(3):
        plt.plot(
            surv_curves.index,
            surv_curves[i],
            color=colors_scenarios[i],
            linewidth=2.5,
            label=labels[i]
        )
        
    plt.title("Predicted Cox Survival Curves by Scenario\nVisualizing Feature Effects on Alert Duration", fontsize=13, fontweight='bold', pad=15)
    plt.xlabel("Duration (minutes)", fontsize=11, labelpad=8)
    plt.ylabel("Survival Probability (Alert Continuing)", fontsize=11, labelpad=8)
    plt.xlim(0, 240)
    plt.ylim(0, 1.05)
    plt.grid(True, linestyle=":", alpha=0.5, color="#CCCCCC")
    
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#888888')
    ax.spines['bottom'].set_color('#888888')
    
    plt.legend(frameon=True, facecolor="#F8F9FA", edgecolor="#EAEAEA", fontsize=9, loc="upper right")
    plt.tight_layout()
    
    DEFAULT_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_predictions_path = DEFAULT_PLOTS_DIR / "cox_predictions.png"
    plt.savefig(plot_predictions_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Survival curves plot saved to: {plot_predictions_path}")

def generate_comparison_bar_chart(c_indices: list, models: list) -> None:
    """
    Generates a premium styled bar chart comparing the C-indices of the models.
    """
    plt.figure(figsize=(8, 6), dpi=300)
    plt.rcParams['font.sans-serif'] = 'Arial'
    plt.rcParams['font.family'] = 'sans-serif'
    
    colors = ['#8E9AA6', '#00838F', '#007ACC'] # Neutral grey-blue, Teal, Premium blue
    bars = plt.bar(models, c_indices, color=colors, width=0.55, edgecolor='#CCCCCC', linewidth=0.7)
    
    plt.axhline(y=0.5, color='#E11D48', linestyle='--', linewidth=1.2, label='Random Guessing (0.5000)')
    
    plt.title("Air Raid Alert Duration Prediction\nModel Comparison on Test Split (Stratified)", fontsize=13, fontweight='bold', pad=15)
    plt.ylabel("Concordance Index (C-index)", fontsize=11, labelpad=8)
    plt.ylim(0.4, 0.68)
    plt.grid(axis='y', linestyle=':', alpha=0.5, color='#CCCCCC')
    
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#888888')
    ax.spines['bottom'].set_color('#888888')
    
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
    
    plot_comparison_path = DEFAULT_PLOTS_DIR / "model_comparison.png"
    plt.savefig(plot_comparison_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Bar chart saved to: {plot_comparison_path}")

def main() -> None:
    print("--- Air Alerts Model Evaluator ---")
    try:
        # 1. Load models and metadata
        cox_package, model_xgb, xgb_metadata = load_models_and_metadata()
        cph = cox_package['model']
        train_oblasts = cox_package['train_oblasts']
        feature_cols = cox_package['feature_cols']
        
        # 2. Load test set and apply mapping
        train_df, test_df, _ = load_and_prepare_test_data(train_oblasts)
        
        print(f"Loaded test split size: {len(test_df):,} alerts")
        if len(test_df) == 0:
            raise ValueError("Test split is empty. Verify dataset timestamps.")
            
        # ==========================================
        # 1. Kaplan-Meier Baseline Evaluation
        # ==========================================
        print("\nEvaluating Regional Kaplan-Meier baseline...")
        global_km = KaplanMeierFitter()
        global_km.fit(train_df['duration'], event_observed=train_df['event'])
        global_median = global_km.median_survival_time_
        
        region_medians = {}
        for region, group in train_df.groupby('oblast'):
            kmf = KaplanMeierFitter()
            kmf.fit(group['duration'], event_observed=group['event'])
            median = kmf.median_survival_time_
            if pd.isna(median) or np.isinf(median):
                median = global_median
            region_medians[region] = median
            
        test_df['km_pred'] = test_df['oblast'].map(region_medians).fillna(global_median)
        
        c_index_km = concordance_index(
            test_df['duration'],
            test_df['km_pred'],
            event_observed=test_df['event']
        )
        
        # ==========================================
        # 2. Cox PH Model Evaluation
        # ==========================================
        print("Evaluating Cox Proportional Hazards model...")
        test_model_df = test_df[['duration', 'event'] + feature_cols].copy()
        c_index_cox = cph.score(test_model_df, scoring_method='concordance_index')
        
        # Calculate Cox expected survival time for MAE/RMSE
        cox_expected_survival = cph.predict_expectation(test_model_df)
        
        # ==========================================
        # 3. XGBoost Model Evaluation
        # ==========================================
        print("Evaluating XGBoost Survival model...")
        X_test = test_df[feature_cols].copy()
        preds_xgb_hazard = model_xgb.predict(X_test)
        
        c_index_xgb = concordance_index(
            test_df['duration'],
            -preds_xgb_hazard,
            event_observed=test_df['event']
        )
        
        # Calculate XGBoost expected survival time for MAE/RMSE using Breslow baseline estimator
        xgb_expected_survival = compute_xgb_expected_survival(model_xgb, train_df, test_df, feature_cols)
        
        # ==========================================
        # 4. Calculate Absolute Metrics (MAE & RMSE)
        # ==========================================
        # Since currently all test alerts are observed (event == 1), absolute metrics are fully valid.
        # If there were censored alerts, we would evaluate MAE/RMSE strictly on event == 1.
        obs_mask = test_df['event'] == 1
        num_obs = obs_mask.sum()
        
        if num_obs > 0:
            durations_obs = test_df.loc[obs_mask, 'duration'].values
            
            # KM
            km_preds_obs = test_df.loc[obs_mask, 'km_pred'].values
            mae_km = np.mean(np.abs(durations_obs - km_preds_obs))
            rmse_km = np.sqrt(np.mean((durations_obs - km_preds_obs) ** 2))
            
            # Cox
            cox_preds_obs = cox_expected_survival.loc[obs_mask].values
            mae_cox = np.mean(np.abs(durations_obs - cox_preds_obs))
            rmse_cox = np.sqrt(np.mean((durations_obs - cox_preds_obs) ** 2))
            
            # XGBoost
            xgb_preds_obs = xgb_expected_survival[obs_mask]
            mae_xgb = np.mean(np.abs(durations_obs - xgb_preds_obs))
            rmse_xgb = np.sqrt(np.mean((durations_obs - xgb_preds_obs) ** 2))
        else:
            mae_km = rmse_km = mae_cox = rmse_cox = mae_xgb = rmse_xgb = np.nan
            
        # ==========================================
        # 5. Print Metrics and Generate Plots
        # ==========================================
        print("\n" + "="*80)
        print("MODEL PERFORMANCE COMPARISON (STRATIFIED TEST SPLIT)")
        print("="*80)
        print(f"{'Model':<35} | {'C-index':<8} | {'MAE (min)':<10} | {'RMSE (min)':<10}")
        print("-" * 80)
        print(f"{'Regional Kaplan-Meier Baseline':<35} | {c_index_km:<8.4f} | {mae_km:<10.2f} | {rmse_km:<10.2f}")
        print(f"{'Cox Proportional Hazards':<35} | {c_index_cox:<8.4f} | {mae_cox:<10.2f} | {rmse_cox:<10.2f}")
        print(f"{'XGBoost Survival (Cox)':<35} | {c_index_xgb:<8.4f} | {mae_xgb:<10.2f} | {rmse_xgb:<10.2f}")
        print("="*80)
        
        # Save results to a markdown report
        report_path = PROJECT_ROOT / "reports" / "evaluation_report.md"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Writing comparison report to: {report_path}")
        with open(report_path, 'w') as f:
            f.write("# Model Evaluation and Comparison Report\n\n")
            f.write("This report presents the comparative evaluation of baseline and machine learning models trained on preprocessed oblast-level air raid alerts using a stratified chronological split (80% train / 20% test per region).\n\n")
            f.write("## Performance Metrics Table\n\n")
            f.write("| Model | Test Concordance Index (C-index) | MAE (minutes) | RMSE (minutes) |\n")
            f.write("| :--- | :---: | :---: | :---: |\n")
            f.write(f"| **Regional Kaplan-Meier Baseline** | {c_index_km:.4f} | {mae_km:.2f} | {rmse_km:.2f} |\n")
            f.write(f"| **Cox Proportional Hazards** | {c_index_cox:.4f} | {mae_cox:.2f} | {rmse_cox:.2f} |\n")
            f.write(f"| **XGBoost Survival (Cox)** | {c_index_xgb:.4f} | {mae_xgb:.2f} | {rmse_xgb:.2f} |\n\n")
            f.write("> [!NOTE]\n")
            f.write("> - **C-index**: Measures the model's ability to correctly order alert durations (higher is better).\n")
            f.write("> - **MAE & RMSE**: Measures the absolute difference in minutes between the predicted expected survival time and the actual alert duration on observed events (lower is better).\n")
            
        # Generate figures
        generate_comparison_bar_chart([c_index_km, c_index_xgb, c_index_cox], ['Regional KM Baseline', 'XGBoost Survival', 'Cox PH'])
        generate_evaluation_plots(cph, test_df, train_oblasts, feature_cols)
        
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
