# scripts/generate_training_data.py

import sys
import os
import random
import numpy as np
import pandas as pd

# Add project root to sys.path to import backend modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from backend.ml.feature_schema import FEATURE_ORDER

# Robustness Fix 1: Full Determinism
random.seed(42)
np.random.seed(42)

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
OUTPUT_FILE = os.path.join(DATA_DIR, 'training_data.csv')

def generate_synthetic_dataset(num_samples=3000):
    """
    Generates a synthetic but physiologically realistic dataset for fatigue detection.
    Labels: 0 = Alert (60%), 1 = Fatigue (40% - split between Mild and Severe).
    """
    samples = []
    
    # Label mapping logic for binary classification (Alert=0, Fatigue=1)
    # Alert drivers → 60%
    # Fatigue drivers → 40% (20% Mild, 20% Severe)
    
    num_alert = int(num_samples * 0.6)
    num_fatigue = num_samples - num_alert
    num_mild = num_fatigue // 2
    num_severe = num_fatigue - num_mild

    def get_features(state):
        f = {}
        # VISION FEATURES
        if state == 'alert':
            f["EAR_mean"] = round(random.uniform(0.28, 0.35), 4)
            f["BF"] = round(random.uniform(10, 18), 1)
            f["MAR_max"] = round(random.uniform(0.2, 0.35), 3)
            f["ECD_max"] = round(random.uniform(0.1, 0.25), 2)
            f["YF"] = round(random.uniform(0, 1), 1)
            f["HP_mean"] = round(random.uniform(-3, 3), 1)
            f["HP_std"] = round(random.uniform(1, 4), 2)
            f["GD_ratio"] = round(random.uniform(0.05, 0.2), 2)
            
            # TELEMETRY
            f["lane_drift_var"] = round(random.uniform(0.02, 0.05), 4)
            f["lane_offset_mean"] = round(random.uniform(-0.1, 0.1), 3)
            f["steering_instability"] = round(random.uniform(0.5, 2.0), 2)
            f["correction_freq"] = round(random.uniform(0.1, 0.4), 2)
            f["reaction_delay_mean"] = round(random.uniform(200, 300), 1)
            f["steering_reversals"] = round(random.uniform(1, 4), 1)
            f["fatigue_label"] = 0
            
        elif state == 'mild':
            f["EAR_mean"] = round(random.uniform(0.22, 0.28), 4)
            f["BF"] = round(random.uniform(20, 30), 1)
            f["MAR_max"] = round(random.uniform(0.35, 0.55), 3)
            f["ECD_max"] = round(random.uniform(0.3, 0.6), 2)
            f["YF"] = round(random.uniform(2, 4), 1)
            f["HP_mean"] = round(random.uniform(-5, 5), 1)
            f["HP_std"] = round(random.uniform(3, 8), 2)
            f["GD_ratio"] = round(random.uniform(0.2, 0.5), 2)
            
            # TELEMETRY
            f["lane_drift_var"] = round(random.uniform(0.05, 0.15), 4)
            f["lane_offset_mean"] = round(random.uniform(-0.3, 0.3), 3)
            f["steering_instability"] = round(random.uniform(3.0, 7.0), 2)
            f["correction_freq"] = round(random.uniform(0.5, 1.2), 2)
            f["reaction_delay_mean"] = round(random.uniform(300, 500), 1)
            f["steering_reversals"] = round(random.uniform(5, 12), 1)
            f["fatigue_label"] = 1
            
        else: # severe
            f["EAR_mean"] = round(random.uniform(0.12, 0.22), 4)
            f["BF"] = round(random.uniform(30, 45), 1)
            f["MAR_max"] = round(random.uniform(0.6, 0.9), 3)
            f["ECD_max"] = round(random.uniform(0.8, 2.2), 2)
            f["YF"] = round(random.uniform(5, 12), 1)
            f["HP_mean"] = round(random.uniform(-15, -5), 1) # Head drooping
            f["HP_std"] = round(random.uniform(8, 20), 2)
            f["GD_ratio"] = round(random.uniform(0.5, 1.0), 2)
            
            # TELEMETRY
            f["lane_drift_var"] = round(random.uniform(0.2, 0.5), 4)
            f["lane_offset_mean"] = round(random.uniform(-0.8, 0.8), 3)
            f["steering_instability"] = round(random.uniform(8.0, 15.0), 2)
            f["correction_freq"] = round(random.uniform(1.2, 3.0), 2)
            f["reaction_delay_mean"] = round(random.uniform(500, 1200), 1)
            f["steering_reversals"] = round(random.uniform(15, 30), 1)
            f["fatigue_label"] = 1

        # Shared Features
        f["EAR_std"] = round(random.uniform(0.02, 0.08), 4)
        f["EAR_trend"] = round(random.uniform(-0.01, 0.01), 5)
        f["session_duration"] = round(random.uniform(1, 120), 1)
        
        # Time of day (sin/cos encoding) - Fatigue risk higher at night/early morning
        hour = random.randint(0, 23)
        if state != 'alert' and random.random() > 0.3:
            hour = random.choice([0, 1, 2, 3, 4, 13, 14, 15]) # Circadian dips
            
        f["time_of_day_sin"] = round(np.sin(2 * np.pi * hour / 24), 4)
        f["time_of_day_cos"] = round(np.cos(2 * np.pi * hour / 24), 4)
        
        return [f.get(col, 0) for col in FEATURE_ORDER] + [f["fatigue_label"]]

    for _ in range(num_alert):
        samples.append(get_features('alert'))
    for _ in range(num_mild):
        samples.append(get_features('mild'))
    for _ in range(num_severe):
        samples.append(get_features('severe'))

    # Shuffle dataset
    random.shuffle(samples)
    
    df = pd.DataFrame(samples, columns=FEATURE_ORDER + ["fatigue_label"])
    
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"[SUCCESS] Synthetic dataset generated with {len(df)} samples.")
    print(f"[PATH] {OUTPUT_FILE}")
    print(f"[COLUMNS] {list(df.columns)}")

if __name__ == "__main__":
    generate_synthetic_dataset()
