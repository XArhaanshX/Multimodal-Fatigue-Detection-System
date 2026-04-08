from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import json, time, threading, sys, os

# Ensure project root is in sys.path for relative imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from run_fatigue_pipeline import start_fatigue_pipeline
    from network.state import shared_state, telemetry_lock
    from network.telemetry_features import (
        TelemetryWindow,
        compute_window_features,
        extract_direct_features,
        extract_raw_telemetry,
    )
except ImportError as e:
    # Fallback pathing for different execution contexts
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from state import shared_state, telemetry_lock
    from telemetry_features import (
        TelemetryWindow,
        compute_window_features,
        extract_direct_features,
        extract_raw_telemetry,
    )
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from run_fatigue_pipeline import start_fatigue_pipeline

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

telemetry_window = TelemetryWindow(window_size_seconds=30.0)

@app.websocket("/")
@app.websocket("/ws/telemetry")
async def telemetry_ws(ws: WebSocket):
    await ws.accept()
    print("[INFO] Godot Simulator connected.")
    
    try:
        while True:
            raw = await ws.receive_text()
            data = json.loads(raw)
            
            telemetry_window.add_sample(data)
            direct_features = extract_direct_features(data)
            raw_telemetry = extract_raw_telemetry(data)
            window_features = compute_window_features(telemetry_window.get_samples())
            
            # 1. Thread-safe update of shared state
            with telemetry_lock:
                if raw_telemetry:
                    shared_state["telemetry"].update(raw_telemetry)
                if direct_features:
                    shared_state["telemetry"].update(direct_features)
                if window_features:
                    shared_state["telemetry"].update(window_features)
                shared_state["last_telemetry_time"] = time.time()
                
                # Snapshot of current pipeline results
                current_score = shared_state["latest_fatigue_score"]
                current_state = shared_state.get("fatigue_state", "NORMAL")
                current_alert = shared_state["alert"]

            # 2. Respond with the latest fatigue data from the inference pipeline
            await ws.send_text(json.dumps({
                "fatigue_score": round(float(current_score), 4),
                "fatigue_state": current_state,
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
