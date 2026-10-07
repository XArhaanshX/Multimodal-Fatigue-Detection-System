# backend/ml/feature_schema.py

"""
Enforces consistent feature ordering between training datasets and real-time inference.
Follows the 19-dimensional feature vector specification from the PRD.
"""

VISION_FEATURES = [
    "EAR_mean",
    "EAR_std",
    "EAR_trend",
    "BF",                  # Blink Frequency
    "ECD_max",             # Eye Closure Duration (max)
    "MAR_max",
    "YF",                  # Yawn Frequency
    "HP_mean",             # Head Pitch Mean
    "HP_std",
    "GD_ratio",            # Gaze Deviation Ratio
]

TELEMETRY_FEATURES = [
    "lane_drift_var",
    "lane_offset_mean",
    "steering_instability",
    "correction_freq",
    "reaction_delay_mean",
    "steering_reversals",
]

CONTEXT_FEATURES = [
    "session_duration",
    "time_of_day_sin",
    "time_of_day_cos",
]

# Fused model (camera + simulator) and driving-only model (simulator without camera)
FEATURE_ORDER = VISION_FEATURES + TELEMETRY_FEATURES + CONTEXT_FEATURES
DRIVING_FEATURE_ORDER = TELEMETRY_FEATURES + CONTEXT_FEATURES
