from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import json, time, threading, sys, os
import numpy as np

# Ensure project root is in sys.path for relative imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from run_fatigue_pipeline import start_fatigue_pipeline
    from network.state import shared_state, telemetry_lock
except ImportError:
    # Fallback for alternative execution contexts
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from state import shared_state, telemetry_lock
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from run_fatigue_pipeline import start_fatigue_pipeline

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

try:
    from ml.fatigue_model import FatigueModel
    from ml.features import build_feature_vector
    model = FatigueModel()
except Exception as e:
    print(f"[ERROR] Failed to load Fatigue Model: {e}")
    model = None

def get_fatigue_state(score: float) -> str:
    """Map fatigue probability to state string per PRD thresholds."""
    if score < 0.30: return "normal"
    if score < 0.55: return "mild"
    if score < 0.75: return "severe"
    return "critical"

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
    await ws.accept()
    print("[INFO] Godot Simulator connected.")
    
    last_send_time = 0
    SEND_INTERVAL = 0.67  # Send at ~1.5 Hz
    
    try:
        while True:
            # 1. Receive telemetry from Godot
            data = await ws.receive_json()
            
            # 2. Update rolling buffer & compute aggregated features
            telemetry_buffer.append(data)
            if len(telemetry_buffer) > WINDOW:
                telemetry_buffer.pop(0)

            features = compute_features(telemetry_buffer)
            
            # 3. Thread-safe update of shared state for vision pipeline tracking
            with telemetry_lock:
                if features:
                    shared_state["telemetry"].update(features)
                shared_state["last_telemetry_time"] = time.time()
                vision_snapshot = shared_state.get("latest_vision_features", {}).copy()

            # 4. Multimodal Inference & Rate-Limited Feedback
            try:
                    # Single Source of Truth: Pull stabilized score from shared state
                    with telemetry_lock:
                        fatigue_score = shared_state["latest_fatigue_score"]
                        fatigue_state = shared_state["fatigue_state"]

                    current_time = time.time()
                    if current_time - last_send_time >= SEND_INTERVAL:
                        response = {
                            "type": "fatigue_update",
                            "fatigue_score": round(float(fatigue_score), 4),
                            "fatigue_state": fatigue_state,
                            "timestamp": current_time
                        }
                        await ws.send_json(response)
                        last_send_time = current_time
                        print(f"[DEBUG] websocket_sent_score: {fatigue_score:.4f} | state: {fatigue_state}")

            except Exception as e:
                print(f"[ERROR] Inference failed: {e}")
                await ws.send_json({
                    "type": "error",
                    "message": "fatigue inference failed"
                })

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
