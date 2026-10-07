import threading

# Shared between the FastAPI event loop (simulator socket) and the scoring thread
shared_state = {
    "telemetry": {},              # latest aggregated simulator features
    "last_telemetry_time": 0.0,   # when the simulator last sent data
    "latest_fatigue_score": None, # None until the first score of a session
    "fatigue_state": "WAITING",
    "vision_status": "off",       # off | starting | no_camera | no_face | tracking
}

telemetry_lock = threading.Lock()
