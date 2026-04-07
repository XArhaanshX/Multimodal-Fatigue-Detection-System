import pandas as pd
import numpy as np
import random
import os

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)

# Ensure the data directory exists
os.makedirs('data', exist_ok=True)

# Feature Order (as per backend/ml/feature_schema.py)
FEATURE_ORDER = [
    "EAR_mean", "EAR_std", "EAR_trend", "BF", "ECD_max", "MAR_max", "YF",
    "HP_mean", "HP_std", "GD_ratio", "lane_drift_var", "lane_offset_mean",
    "steering_instability", "correction_freq", "reaction_delay_mean",
    "steering_reversals", "session_duration", "time_of_day_sin", "time_of_day_cos"
]

def generate_sample(label):
    features = {}
    
    # Introduce some "Noisy" overlap
    # We broaden the ranges to make them less perfectly separable
    
    if label == 0:  # ALERT
        features["EAR_mean"] = random.uniform(0.24, 0.36)  # Expanded from 0.28-0.35
        features["EAR_std"] = random.uniform(0.01, 0.08)   # Expanded
        features["EAR_trend"] = random.uniform(-0.02, 0.02)# Expanded
        features["BF"] = random.uniform(0, 3)              # Expanded (was 0-2)
        features["ECD_max"] = random.uniform(0.05, 0.35)   # Expanded (was 0.05-0.25)
        features["MAR_max"] = random.uniform(0.2, 0.6)     # Expanded (was 0.2-0.5)
        features["YF"] = random.uniform(0, 1.2)            # Expanded (was 0-1)
        features["HP_mean"] = random.uniform(-8, 8)        # Expanded
        features["HP_std"] = random.uniform(0, 15)         # Expanded
        features["GD_ratio"] = random.uniform(0, 0.4)      # Expanded
        
        # Telemetry
        features["lane_drift_var"] = random.uniform(0, 0.3)
        features["lane_offset_mean"] = random.uniform(0, 0.4)
        features["steering_instability"] = random.uniform(0, 0.25)
        features["correction_freq"] = random.uniform(0, 4)
        features["reaction_delay_mean"] = random.uniform(0, 0.45)
        features["steering_reversals"] = random.uniform(0, 3)
        
    else:  # FATIGUED
        features["EAR_mean"] = random.uniform(0.14, 0.28)  # Expanded (was 0.15-0.25) -> Now overlaps with Alert!
        features["EAR_std"] = random.uniform(0.04, 0.18)
        features["EAR_trend"] = random.uniform(-0.07, 0.01)
        features["BF"] = random.uniform(2, 9)              # Overlaps with Alert (2-3)
        features["ECD_max"] = random.uniform(0.3, 1.1)     # Overlaps with Alert (0.3-0.35)
        features["MAR_max"] = random.uniform(0.5, 1.4)     # Overlaps with Alert (0.5-0.6)
        features["YF"] = random.uniform(0.8, 4)            # Overlaps
        features["HP_mean"] = random.uniform(-20, 20)
        features["HP_std"] = random.uniform(8, 35)
        features["GD_ratio"] = random.uniform(0.25, 0.9)   # Overlaps
        
        # Telemetry
        features["lane_drift_var"] = random.uniform(0.2, 1.1) # Overlaps
        features["lane_offset_mean"] = random.uniform(0.2, 0.9)
        features["steering_instability"] = random.uniform(0.15, 0.7)
        features["correction_freq"] = random.uniform(1.5, 8)
        features["reaction_delay_mean"] = random.uniform(0.35, 1.4)
        features["steering_reversals"] = random.uniform(2, 10)
        
    # Add Gaussian Noise to simulate real sensor jitter
    for key in features:
        noise = np.random.normal(0, 0.005) # Subtle noise
        features[key] += noise

    # return as list in FEATURE_ORDER
    return [features[col] for col in FEATURE_ORDER]

def main():
    print("[INFO] Generating Realistic Noisy Training Dataset...")
    data = []
    
    # Generate 2500 alert samples
    for _ in range(2500):
        sample = generate_sample(label=0)
        sample.append(0)  # Label
        data.append(sample)
        
    # Generate 2500 fatigued samples
    for _ in range(2500):
        sample = generate_sample(label=1)
        sample.append(1)  # Label
        data.append(sample)
        
    # Shuffle the dataset
    random.shuffle(data)
    
    # Create DataFrame
    columns = FEATURE_ORDER + ["target"]
    df = pd.DataFrame(data, columns=columns)
    
    # Save to CSV
    output_path = "data/training_data.csv"
    df.to_csv(output_path, index=False)
    print(f"[SUCCESS] Realistic dataset generated with {len(df)} samples: {output_path}")

if __name__ == "__main__":
    main()
