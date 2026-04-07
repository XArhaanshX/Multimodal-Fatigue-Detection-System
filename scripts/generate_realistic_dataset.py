# scripts/generate_realistic_dataset.py

import os
import sys
import numpy as np
import pandas as pd

# Add the project root to sys.path to import FEATURE_ORDER
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from backend.ml.feature_schema import FEATURE_ORDER
except ImportError:
    # Fallback if the above fails during execution
    FEATURE_ORDER = [
        "EAR_mean", "EAR_std", "EAR_trend", "BF", "ECD_max", "MAR_max", "YF", 
        "HP_mean", "HP_std", "GD_ratio", "lane_drift_var", "lane_offset_mean", 
        "steering_instability", "correction_freq", "reaction_delay_mean", 
        "steering_reversals", "session_duration", "time_of_day_sin", "time_of_day_cos"
    ]

def generate_samples(count, label):
    """
    Generates realistic samples for a given class label.
    0 = Alert, 1 = Fatigue
    """
    samples = []
    
    for _ in range(count):
        row = {}
        
        # 1. VISION FEATURES
        if label == 0:  # Alert
            ear_mean = np.random.uniform(0.28, 0.35)
            bf = np.random.uniform(0, 3)
            mar_max = np.random.uniform(0.2, 0.5)
        else:  # Fatigue
            ear_mean = np.random.uniform(0.18, 0.26)
            bf = np.random.uniform(2, 7)
            mar_max = np.random.uniform(0.5, 1.3)
            
        # General vision features (common ranges)
        ear_std = np.random.uniform(0.01, 0.06)
        ear_trend = np.random.uniform(-0.01, 0.01)
        ecd_max = np.random.uniform(0.02, 0.30)
        yf = np.random.uniform(0, 2)
        hp_mean = np.random.uniform(-10, 10)
        hp_std = np.random.uniform(0, 20)
        gd_ratio = np.random.uniform(0, 1)

        # Add Noise and Clip
        row["EAR_mean"] = np.clip(ear_mean + np.random.normal(0, 0.01), 0.1, 0.5)
        row["BF"] = np.clip(bf + np.random.normal(0, 0.1), 0, 10)
        row["MAR_max"] = np.clip(mar_max + np.random.normal(0, 0.01), 0.1, 2.0)
        row["EAR_std"] = np.clip(ear_std + np.random.normal(0, 0.005), 0.0, 0.1)
        row["EAR_trend"] = np.clip(ear_trend + np.random.normal(0, 0.001), -0.05, 0.05)
        row["ECD_max"] = np.clip(ecd_max + np.random.normal(0, 0.01), 0.0, 0.5)
        row["YF"] = np.clip(yf + np.random.normal(0, 0.1), 0, 5)
        row["HP_mean"] = np.clip(hp_mean + np.random.normal(0, 0.5), -30, 30)
        row["HP_std"] = np.clip(hp_std + np.random.normal(0, 0.5), 0, 40)
        row["GD_ratio"] = np.clip(gd_ratio + np.random.normal(0, 0.05), 0, 1)

        # 2. TELEMETRY FEATURES (Set to 0 as per instruction)
        row["lane_drift_var"] = 0.0
        row["lane_offset_mean"] = 0.0
        row["steering_instability"] = 0.0
        row["correction_freq"] = 0.0
        row["reaction_delay_mean"] = 0.0
        row["steering_reversals"] = 0.0

        # 3. CONTEXT FEATURES
        session_duration = np.random.uniform(0, 120)
        hour = np.random.uniform(0, 24)
        
        row["session_duration"] = session_duration
        row["time_of_day_sin"] = np.sin(2 * np.pi * hour / 24)
        row["time_of_day_cos"] = np.cos(2 * np.pi * hour / 24)
        
        # Add label
        row["fatigue_label"] = label
        samples.append(row)
        
    return samples

def main():
    print("Generating synthetic fatigue dataset...")
    
    # 1500 Alert, 1500 Fatigue
    alert_samples = generate_samples(1500, 0)
    fatigue_samples = generate_samples(1500, 1)
    
    all_samples = alert_samples + fatigue_samples
    df = pd.DataFrame(all_samples)
    
    # Reorder columns to match FEATURE_ORDER + fatigue_label
    column_order = FEATURE_ORDER + ["fatigue_label"]
    df = df[column_order]
    
    # Ensure data directory exists
    os.makedirs("data", exist_ok=True)
    
    # Save to CSV
    output_path = "data/training_data.csv"
    df.to_csv(output_path, index=False)
    print(f"Dataset saved to {output_path}")
    print(f"Total rows: {len(df)}")
    
    # Simple Verification
    print("\nVerification Summary:")
    print(f"Alert EAR_mean: {df[df['fatigue_label']==0]['EAR_mean'].mean():.4f}")
    print(f"Fatigue EAR_mean: {df[df['fatigue_label']==1]['EAR_mean'].mean():.4f}")
    print(f"Alert BF: {df[df['fatigue_label']==0]['BF'].mean():.4f}")
    print(f"Fatigue BF: {df[df['fatigue_label']==1]['BF'].mean():.4f}")
    
    if len(df) == 3000:
        print("\nSuccess: Dataset contains exactly 3000 rows.")
    else:
        print(f"\nWarning: Row count mismatch ({len(df)})")

if __name__ == "__main__":
    main()
