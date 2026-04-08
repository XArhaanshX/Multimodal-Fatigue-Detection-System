import threading
import time

# Centralized shared state for telemetry and fatigue results
# This shared object allows the FastAPI server and the Fatigue Pipeline
# thread to communicate without circular imports.
shared_state = {
    "telemetry": {
        "lane_drift_var": 0.0,
        "lane_offset_mean": 0.0,
        "steering_instability": 0.0,
        "correction_freq": 0.0,
        "reaction_delay_mean": 0.0,
        "steering_reversals": 0.0,
        # Raw data for backend.py buffer-based features
        "lane_offset": 0.0,
        "steering_angle": 0.0,
        "steering_correction_hz": 0.0,
        "reaction_delay_ms": 0.0,
        "speed_kmh": 0.0
    },
    "latest_fatigue_score": 0.0,
    "fatigue_state": "normal",
    "alert": False,
    "last_telemetry_time": 0.0,
    "latest_vision_features": {}  # New: Store vision snapshot for WebSocket inference
}

# Thread lock for safe telemetry access
telemetry_lock = threading.Lock()
