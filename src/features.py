import numpy as np
import pandas as pd

def calculate_concurrency(df: pd.DataFrame) -> pd.Series:
    """
    Calculates the number of other active oblast-level alerts at the exact instant 
    the current alert started using an O(N log N) sweep-line (event sorting) algorithm.
    """
    df = df.copy()
    
    # In case there are ongoing/censored alerts with null finished_at, 
    # we temporarily fill finished_at with the max finished_at in the dataset to calculate concurrency.
    max_finished = df['finished_at'].max()
    if pd.isna(max_finished):
        max_finished = df['started_at'].max()
        
    finished_times_filled = df['finished_at'].fillna(max_finished)
    
    start_times = df['started_at'].tolist()
    end_times = finished_times_filled.tolist()
    
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
                
    return pd.Series(concurrency, index=df.index)

def extract_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts timestamps to local Ukraine time (Europe/Kyiv) and generates 
    continuous, cyclic temporal features for hour, day of the week, and month.
    """
    df = df.copy()
    
    # 1. Convert to local Kyiv timezone
    local_starts = df['started_at'].dt.tz_convert('Europe/Kyiv')
    
    # 2. Continuous time representations (incorporating hours, minutes, and seconds)
    # This prevents harsh quantization step-functions
    hour_continuous = (
        local_starts.dt.hour + 
        local_starts.dt.minute / 60.0 + 
        local_starts.dt.second / 3600.0
    )
    
    dayofweek_continuous = local_starts.dt.dayofweek + hour_continuous / 24.0
    
    # Approximate month continuous: month index (0-11) + progress within month
    # Using local_starts.dt.days_in_month to scale correctly
    days_in_month = local_starts.dt.days_in_month
    month_continuous = (local_starts.dt.month - 1) + (local_starts.dt.day - 1) / days_in_month
    
    # 3. Generate cyclic features
    df['sin_hour'] = np.sin(2 * np.pi * hour_continuous / 24.0)
    df['cos_hour'] = np.cos(2 * np.pi * hour_continuous / 24.0)
    
    df['sin_dayofweek'] = np.sin(2 * np.pi * dayofweek_continuous / 7.0)
    df['cos_dayofweek'] = np.cos(2 * np.pi * dayofweek_continuous / 7.0)
    
    df['sin_month'] = np.sin(2 * np.pi * month_continuous / 12.0)
    df['cos_month'] = np.cos(2 * np.pi * month_continuous / 12.0)
    
    return df

def encode_spatial_features(
    df: pd.DataFrame, 
    train_oblasts: list = None, 
    is_train: bool = True
) -> tuple:
    """
    One-hot encodes the spatial 'oblast' feature without data leakage.
    If is_train=True, it builds the list of unique oblasts and drops the first 
    category to prevent collinearity.
    If is_train=False, it uses the provided train_oblasts to align columns.
    """
    df = df.copy()
    
    if is_train:
        # Get sorted list of all unique oblasts in the training set
        train_oblasts = sorted(list(df['oblast'].unique()))
        
    # We drop the first oblast in the list to act as the baseline (prevention of collinearity)
    if not train_oblasts:
        raise ValueError("train_oblasts must be provided if is_train=False")
        
    baseline_oblast = train_oblasts[0]
    encoded_oblasts = train_oblasts[1:]
    
    spatial_columns = []
    for oblast in encoded_oblasts:
        col_name = f"oblast_{oblast}"
        df[col_name] = (df['oblast'] == oblast).astype(int)
        spatial_columns.append(col_name)
        
    return df, train_oblasts, spatial_columns

def prepare_features(
    df: pd.DataFrame, 
    train_oblasts: list = None, 
    is_train: bool = True
) -> tuple:
    """
    Orchestrates concurrency calculation, temporal features extraction, 
    and spatial features encoding.
    """
    # 1. Sort by started_at to ensure concurrency calculation is correct
    df = df.sort_values(by='started_at').reset_index(drop=True)
    
    # 2. Concurrency
    df['concurrency'] = calculate_concurrency(df)
    
    # 3. Temporal
    df = extract_temporal_features(df)
    
    # 4. Spatial encoding
    df, train_oblasts, spatial_cols = encode_spatial_features(df, train_oblasts, is_train)
    
    # 5. Define all model input features
    base_features = [
        'concurrency',
        'sin_hour',
        'cos_hour',
        'sin_dayofweek',
        'cos_dayofweek',
        'sin_month',
        'cos_month'
    ]
    feature_cols = base_features + spatial_cols
    
    return df, train_oblasts, feature_cols

def split_data_stratified(df: pd.DataFrame, train_ratio: float = 0.8) -> tuple:
    """
    Splits the dataset chronologically per region.
    For each region, the first train_ratio fraction of alerts (chronologically)
    goes to the train set, and the rest to the test set.
    """
    df = df.copy()
    # Ensure index is standard range for selection
    df = df.sort_values(by='started_at').reset_index(drop=True)
    
    train_indices = []
    test_indices = []
    
    for region, group in df.groupby('oblast'):
        group_sorted = group.sort_values('started_at')
        n = len(group_sorted)
        split_idx = int(n * train_ratio)
        
        train_indices.extend(group_sorted.index[:split_idx])
        test_indices.extend(group_sorted.index[split_idx:])
        
    train_df = df.loc[train_indices].copy()
    test_df = df.loc[test_indices].copy()
    
    # Sort chronologically
    train_df = train_df.sort_values(by='started_at').reset_index(drop=True)
    test_df = test_df.sort_values(by='started_at').reset_index(drop=True)
    
    return train_df, test_df

