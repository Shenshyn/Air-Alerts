import os
from pathlib import Path
import pandas as pd

# Define paths relative to the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RAW_DATA_PATH = PROJECT_ROOT / "data" / "official_data_en.csv"
DEFAULT_PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed_alerts.csv"

def load_raw_data(raw_path: Path = DEFAULT_RAW_DATA_PATH) -> pd.DataFrame:
    """
    Loads raw CSV data from the given path.
    """
    print(f"Loading raw data from: {raw_path}")
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data file not found at {raw_path}")
    return pd.read_csv(raw_path)

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans, deduplicates, and preprocesses the official air raid alerts dataset.
    - Removes duplicates.
    - Filters for 'oblast' level alerts.
    - Converts started_at and finished_at to datetime.
    - Handles censoring: alerts with null finished_at are marked as event = 0,
      with duration computed up to the max time in the dataset.
    - Filters out duration <= 0.
    - Filters out regions with less than 10 alerts to prevent collinearity issues.
    """
    initial_rows = len(df)
    print(f"Initial record count: {initial_rows:,}")
    
    # 1. Remove duplicate rows globally
    df = df.drop_duplicates()
    dedup_rows = len(df)
    print(f"Records after removing duplicates: {dedup_rows:,} (Removed {initial_rows - dedup_rows:,})")
    
    # 2. Filter to keep only level == 'oblast'
    df = df[df['level'] == 'oblast'].copy()
    level_rows = len(df)
    print(f"Records after filtering for 'oblast' level: {level_rows:,} (Filtered out {dedup_rows - level_rows:,})")
    
    # 3. Convert started_at and finished_at columns to datetime
    df['started_at'] = pd.to_datetime(df['started_at'], errors='coerce', utc=True)
    df['finished_at'] = pd.to_datetime(df['finished_at'], errors='coerce', utc=True)
    
    # Drop rows with NaT in started_at
    valid_start_mask = df['started_at'].notna()
    df = df[valid_start_mask].copy()
    valid_start_rows = len(df)
    if valid_start_rows < level_rows:
        print(f"Removed {level_rows - valid_start_rows:,} rows with invalid started_at dates")
        
    # 4. Handle right-censoring
    # Define censor_time as the maximum timestamp in finished_at (or started_at if all finished_at are null)
    censor_time = df['finished_at'].max()
    if pd.isna(censor_time):
        censor_time = df['started_at'].max()
    print(f"Censor time (max timestamp in dataset): {censor_time}")
    
    # event is 1 if alert is completed (finished_at is not null), and 0 if active/censored (finished_at is null)
    df['event'] = df['finished_at'].notna().astype(int)
    
    # Calculate duration in minutes (filling missing finished_at with censor_time)
    finished_times_filled = df['finished_at'].fillna(censor_time)
    df['duration'] = (finished_times_filled - df['started_at']).dt.total_seconds() / 60.0
    
    # 5. Filter out any rows with duration <= 0
    df = df[df['duration'] > 0].copy()
    positive_duration_rows = len(df)
    print(f"Records with duration > 0: {positive_duration_rows:,} (Removed {valid_start_rows - positive_duration_rows:,} rows with duration <= 0)")
    
    # 6. Filter out low-record oblasts to avoid collinearity/singularity in Cox model
    oblast_counts = df['oblast'].value_counts()
    low_record_oblasts = oblast_counts[oblast_counts < 10].index.tolist()
    if low_record_oblasts:
        print(f"Filtering out regions with fewer than 10 alerts to prevent singularity: {low_record_oblasts}")
        df = df[~df['oblast'].isin(low_record_oblasts)].copy()
        print(f"Remaining records after low-record oblast filtering: {len(df):,}")
        
    return df

def cache_processed_data(df: pd.DataFrame, processed_path: Path = DEFAULT_PROCESSED_DATA_PATH) -> None:
    """
    Saves the cleaned and preprocessed DataFrame to a CSV cache file.
    """
    print(f"Caching processed data to: {processed_path}")
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(processed_path, index=False)
    print("Caching complete!")

def main() -> None:
    print("--- Air Alerts Data Loader ---")
    try:
        raw_df = load_raw_data(DEFAULT_RAW_DATA_PATH)
        df = preprocess_data(raw_df)
        cache_processed_data(df, DEFAULT_PROCESSED_DATA_PATH)
        
        # Print basic stats
        print("\n--- Preprocessing Summary & Statistics ---")
        total_records = len(df)
        print(f"Total remaining records: {total_records:,}")
        
        if total_records > 0:
            censored_count = (df['event'] == 0).sum()
            observed_count = (df['event'] == 1).sum()
            print(f"Observed alerts (completed): {observed_count:,} ({observed_count/total_records:.2%})")
            print(f"Censored alerts (ongoing):   {censored_count:,} ({censored_count/total_records:.2%})")
            
            avg_duration = df['duration'].mean()
            median_duration = df['duration'].median()
            min_duration = df['duration'].min()
            max_duration = df['duration'].max()
            
            print(f"Average alert duration: {avg_duration:.2f} minutes")
            print(f"Median alert duration:  {median_duration:.2f} minutes")
            print(f"Min alert duration:     {min_duration:.2f} minutes")
            print(f"Max alert duration:     {max_duration:.2f} minutes")
            
            print("\nUnique regions (oblasts) represented:")
            regions = df['oblast'].unique()
            print(f"Count: {len(regions)}")
            print(", ".join(sorted(regions)))
        else:
            print("Warning: No records remaining after preprocessing!")
            
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
