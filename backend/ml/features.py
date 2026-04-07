# backend/ml/features.py

import time
import numpy as np
from .feature_schema import FEATURE_ORDER

def build_feature_vector(vision_features, telemetry_features=None, session_start_time=None):
    """
    Assembles the 19-dimensional feature vector in the strict PRD order.
    
    Args:
        vision_features (dict): 10 vision metrics.
        telemetry_features (dict): 6 driving metrics from the simulator.
        session_start_time (float): Optional timestamp of session start.
    """
    if session_start_time is None:
        session_start_time = time.time() - 300 # Default 5 mins for placeholder
        
    session_duration = (time.time() - session_start_time) / 60.0 # mins
    
    # Time of day encoding
    now = time.localtime()
    hour = now.tm_hour + now.tm_min / 60.0
    tod_sin = np.sin(2 * np.pi * hour / 24)
    tod_cos = np.cos(2 * np.pi * hour / 24)

    # Safe telemetry access
    tel = telemetry_features if telemetry_features else {}

    # Construct the raw dictionary
    f = {
        # VISION FEATURES (10)
        "EAR_mean": vision_features.get("EAR_mean", 0.0),
        "EAR_std": vision_features.get("EAR_std", 0.0),
        "EAR_trend": vision_features.get("EAR_trend", 0.0),
        "BF": vision_features.get("blink_frequency", 0.0),
        "ECD_max": vision_features.get("ECD_max", 0.0),
        "MAR_max": vision_features.get("MAR_max", 0.0),
        "YF": vision_features.get("yawn_frequency", 0.0),
        "HP_mean": vision_features.get("pitch_mean", 0.0),
        "HP_std": vision_features.get("pitch_std", 0.0),
        "GD_ratio": vision_features.get("gaze_ratio", 0.0),
        
        # TELEMETRY FEATURES (6)
        "lane_drift_var": tel.get("lane_drift_var", 0.0),
        "lane_offset_mean": tel.get("lane_offset_mean", 0.0),
        "steering_instability": tel.get("steering_instability", 0.0),
        "correction_freq": tel.get("correction_freq", 0.0),
        "reaction_delay_mean": tel.get("reaction_delay_mean", 0.0),
        "steering_reversals": tel.get("steering_reversals", 0.0),
        
        # CONTEXT FEATURES (3)
        "session_duration": session_duration,
        "time_of_day_sin": tod_sin,
        "time_of_day_cos": tod_cos
    }

    # Return list in strict order
    return [float(f[col]) for col in FEATURE_ORDER]
