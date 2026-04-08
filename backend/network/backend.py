from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import json, time, threading, sys, os
import numpy as np

# Ensure project root is in sys.path for relative imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from run_fatigue_pipeline import start_fatigue_pipeline
    from network.state import shared_state, telemetry_lock
except ImportError as e:
    # Fallback pathing for different execution contexts
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from state import shared_state, telemetry_lock
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from run_fatigue_pipeline import start_fatigue_pipeline

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── Sliding window buffer (30s at 10Hz = 300 samples) ─────
WINDOW = 300
telemetry_buffer = []

def compute_features(buf: list) -> dict:
    """Aggregate last N telemetry samples into feature dict."""
    if len(buf) < 10:
        return {}
    
    # Extract keys and handle missing data
    keys = buf[0].keys()
    arr = {k: [s[k] for s in buf if k in s] for k in keys}
    
    # Check if we have enough data for each key
    if not all(len(arr[k]) > 0 for k in ["lane_offset", "steering_angle"]):
        return {}

    return {
        "lane_offset_mean": float(np.mean(arr["lane_offset"])),
        "lane_drift_var":   float(np.var(arr["lane_offset"])),
        "steering_instability": float(np.std(arr["steering_angle"]) /
                                      (abs(np.mean(arr["steering_angle"])) + 1e-6)),
        "correction_freq":  float(np.mean(arr.get("steering_correction_hz", [0.0]))),
        "reaction_delay_mean":   float(np.mean(arr.get("reaction_delay_ms", [0.0]))),
        "steering_reversals": float(np.sum(np.diff(np.sign(np.diff(arr["steering_angle"]))) != 0)) # Basic count
    }

@app.websocket("/ws/telemetry")
async def telemetry_ws(ws: WebSocket):
    global telemetry_buffer
    await ws.accept()
    print("[INFO] Godot Simulator connected.")
    
    try:
        while True:
            raw = await ws.receive_text()
            data = json.loads(raw)
            
            # 1. Update rolling buffer
            telemetry_buffer.append(data)
            if len(telemetry_buffer) > WINDOW:
                telemetry_buffer.pop(0)

            # 2. Compute higher-level features for the ML model
            features = compute_features(telemetry_buffer)
            
            # 3. Thread-safe update of shared state
            with telemetry_lock:
                if features:
                    shared_state["telemetry"].update(features)
                shared_state["last_telemetry_time"] = time.time()
                
                # Snapshot of current vision-calculated results
                current_score = shared_state["latest_fatigue_score"]
                current_alert = shared_state["alert"]

            # 4. Respond with the LATEST fatigue data from the vision pipeline
            await ws.send_text(json.dumps({
                "fatigue_score": round(float(current_score), 4),
                "alert": bool(current_alert),
                "status": "active"
            }))

    except WebSocketDisconnect:
        print("[INFO] Godot Simulator disconnected.")
    except Exception as e:
        print(f"[ERROR] WebSocket error: {e}")

if __name__ == "__main__":
    import uvicorn
    
    # START PIPELINE THREAD
    print("[INFO] Starting Fatigue Detection Pipeline Thread...")
    pipeline_thread = threading.Thread(
        target=start_fatigue_pipeline,
        args=(shared_state, telemetry_lock),
        daemon=True
    )
    pipeline_thread.start()

    # START FASTAPI SERVER
    print("[INFO] Starting FastAPI Server on ws://0.0.0.0:8001")
    uvicorn.run(app, host="0.0.0.0", port=8001)
