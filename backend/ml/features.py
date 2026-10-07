# backend/ml/features.py

import time
import numpy as np
from .feature_schema import FEATURE_ORDER, DRIVING_FEATURE_ORDER, TELEMETRY_FEATURES

# Vision pipeline keys -> training column names
VISION_KEYS = {
    "EAR_mean": "EAR_mean",
    "EAR_std": "EAR_std",
    "EAR_trend": "EAR_trend",
    "BF": "blink_frequency",
    "ECD_max": "ECD_max",
    "MAR_max": "MAR_max",
    "YF": "yawn_frequency",
    "HP_mean": "pitch_mean",
    "HP_std": "pitch_std",
    "GD_ratio": "gaze_ratio",
}


def context_features(session_start_time=None):
    """Time-on-task (minutes) and time of day as a point on the unit circle."""
    if session_start_time is None:
        session_start_time = time.time() - 300  # Default 5 mins for placeholder
    now = time.localtime()
    hour = now.tm_hour + now.tm_min / 60.0
    return {
        "session_duration": (time.time() - session_start_time) / 60.0,
        "time_of_day_sin": np.sin(2 * np.pi * hour / 24),
        "time_of_day_cos": np.cos(2 * np.pi * hour / 24),
    }


def _telemetry(telemetry_features):
    tel = telemetry_features or {}
    return {key: tel.get(key, 0.0) for key in TELEMETRY_FEATURES}


def build_feature_vector(vision_features, telemetry_features=None, session_start_time=None):
    """19-dimensional vector for the fused model, in the strict PRD order."""
    f = {col: vision_features.get(key, 0.0) for col, key in VISION_KEYS.items()}
    f.update(_telemetry(telemetry_features))
    f.update(context_features(session_start_time))
    return [float(f[col]) for col in FEATURE_ORDER]


def build_driving_vector(telemetry_features, session_start_time=None):
    """9-dimensional vector for the driving-only model."""
    f = _telemetry(telemetry_features)
    f.update(context_features(session_start_time))
    return [float(f[col]) for col in DRIVING_FEATURE_ORDER]
