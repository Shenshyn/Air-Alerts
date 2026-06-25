import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter

def main():
    # 1. Define paths
    project_root = Path(__file__).resolve().parent.parent
    data_path = project_root / "data" / "processed_alerts.csv"
    plots_dir = project_root / "plots"
    plot_path = plots_dir / "baseline_km.png"

    print(f"Loading data from: {data_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"Processed data file not found at: {data_path}")

    # Load the processed data
    df = pd.read_csv(data_path)

    # Ensure duration_minutes is present
    if "duration_minutes" not in df.columns:
        if "duration" in df.columns:
            df["duration_minutes"] = df["duration"]
        else:
            raise KeyError("Neither 'duration_minutes' nor 'duration' column found in dataset.")

    if "event" not in df.columns:
        raise KeyError("Column 'event' not found in dataset.")

    # 2. Fit a global KaplanMeierFitter
    print("Fitting global Kaplan-Meier model...")
    kmf_global = KaplanMeierFitter()
    kmf_global.fit(df["duration_minutes"], event_observed=df["event"], label="Global (All Regions)")
    global_median = kmf_global.median_survival_time_
    
    # 3. Fit separate Kaplan-Meier curves for key regions
    regions = [
        "Kyiv City",
        "Kharkivska oblast",
        "Lvivska oblast",
        "Odeska oblast",
        "Donetska oblast"
    ]
    
    region_kmfs = {}
    region_medians = {}
    for region in regions:
        print(f"Fitting Kaplan-Meier model for {region}...")
        region_df = df[df["oblast"] == region]
        if len(region_df) == 0:
            print(f"Warning: No records found for region '{region}'")
            continue
        
        kmf_region = KaplanMeierFitter()
        kmf_region.fit(region_df["duration_minutes"], event_observed=region_df["event"], label=region)
        region_kmfs[region] = kmf_region
        region_medians[region] = kmf_region.median_survival_time_

    # 4. Generate the plot
    print("Generating survival curves plot...")
    plt.figure(figsize=(10, 6), dpi=300)
    
    # Set premium styling details
    plt.rcParams['font.sans-serif'] = 'Arial'
    plt.rcParams['font.family'] = 'sans-serif'
    
    # Color palette
    colors = {
        "Kyiv City": "#007ACC",       # Premium blue
        "Kharkivska oblast": "#D81B60", # Deep pink/magenta
        "Lvivska oblast": "#2E7D32",   # Forest green
        "Odeska oblast": "#00838F",     # Deep teal
        "Donetska oblast": "#FF5722"   # Sunset orange
    }

    # Plot global survival curve
    kmf_global.plot_survival_function(
        color="#333333",
        linestyle="--",
        linewidth=2,
        ax=plt.gca(),
        ci_show=False
    )

    # Plot region-specific survival curves
    for region in regions:
        if region in region_kmfs:
            region_kmfs[region].plot_survival_function(
                color=colors[region],
                linewidth=2,
                ax=plt.gca(),
                ci_show=False  # Keep it clean without overlapping confidence bands
            )

    # Apply premium styling
    plt.title("Air Raid Alert Survival Curves (Kaplan-Meier Estimator)", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Duration (minutes)", fontsize=12, labelpad=10)
    plt.ylabel("Survival Probability (Alert Continuing)", fontsize=12, labelpad=10)
    
    plt.xlim(0, 240)
    plt.ylim(0, 1.05)
    
    plt.grid(True, linestyle=":", alpha=0.6, color="#CCCCCC")
    
    # Remove top and right spines for clean aesthetic
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#888888')
    ax.spines['bottom'].set_color('#888888')
    
    plt.legend(frameon=True, facecolor="#F8F9FA", edgecolor="#EAEAEA", fontsize=10, loc="upper right")
    
    # Ensure plots directory exists
    plots_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Plot saved successfully to: {plot_path}")

    # 5. Print median survival times
    print("\n" + "="*50)
    print("MEDIAN SURVIVAL TIMES (Duration in minutes)")
    print("="*50)
    print(f"{'Region':<25} | {'Median Duration (min)':<25}")
    print("-"*50)
    print(f"{'Global (All Regions)':<25} | {global_median:.2f}")
    for region in regions:
        median_time = region_medians.get(region, float('nan'))
        print(f"{region:<25} | {median_time:.2f}")
    print("="*50)

if __name__ == "__main__":
    main()
