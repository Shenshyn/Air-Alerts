import os
from pathlib import Path
import pandas as pd

# Define paths relative to the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RAW_DATA_PATH = PROJECT_ROOT / "data" / "official_data_en.csv"
DEFAULT_PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed_alerts.csv"

def load_and_preprocess_data(raw_path: Path = DEFAULT_RAW_DATA_PATH) -> pd.DataFrame:
    """
    Loads, cleans, and preprocesses the official air raid alerts dataset.
    
    Processing steps:
    1. Load CSV data.
    2. Remove duplicate rows globally.
    3. Filter to keep only level == 'oblast'.
    4. Convert started_at and finished_at columns to datetime.
    5. Drop rows with missing timestamps.
    6. Calculate duration in minutes (finished_at - started_at).
    7. Filter out any rows with duration <= 0.
    8. Create an 'event' column populated with 1 (observed events).
    """
    print(f"Loading raw data from: {raw_path}")
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data file not found at {raw_path}")
        
    df = pd.read_csv(raw_path)
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
    # We specify utc=True because the timezone offsets (+00:00) are present
    df['started_at'] = pd.to_datetime(df['started_at'], errors='coerce', utc=True)
    df['finished_at'] = pd.to_datetime(df['finished_at'], errors='coerce', utc=True)
    
    # Drop rows with NaT in started_at or finished_at
    valid_time_mask = df['started_at'].notna() & df['finished_at'].notna()
    df = df[valid_time_mask].copy()
    valid_time_rows = len(df)
    if valid_time_rows < level_rows:
        print(f"Removed {level_rows - valid_time_rows:,} rows with invalid dates/times")
        
    # 4. Calculate duration in minutes (finished_at - started_at)
    # dt.total_seconds() / 60 gives duration in minutes with floating point precision
    df['duration'] = (df['finished_at'] - df['started_at']).dt.total_seconds() / 60.0
    
    # 5. Filter out any rows with duration <= 0
    df = df[df['duration'] > 0].copy()
    positive_duration_rows = len(df)
    print(f"Records with duration > 0: {positive_duration_rows:,} (Removed {valid_time_rows - positive_duration_rows:,} rows with duration <= 0)")
    
    # 6. Create an 'event' column populated with 1
    df['event'] = 1
    
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
        # Load and preprocess
        df = load_and_preprocess_data(DEFAULT_RAW_DATA_PATH)
        
        # Save to cache
        cache_processed_data(df, DEFAULT_PROCESSED_DATA_PATH)
        
        # Print basic stats
        print("\n--- Preprocessing Summary & Statistics ---")
        total_records = len(df)
        print(f"Total remaining records: {total_records:,}")
        
        if total_records > 0:
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
