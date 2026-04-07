from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import json, time, asyncio
import numpy as np

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── Sliding window buffer (30s at 10Hz = 300 samples) ─────
WINDOW = 300
telemetry_buffer = []

# ── Load your trained model here ──────────────────────────
# from lightgbm import Booster
# model = Booster(model_file="fatigue_model.txt")
# STUB: random score for testing until model is ready
def predict_fatigue(features: dict) -> float:
    # Replace with: return float(model.predict([feature_vector])[0])
    import random
    return random.uniform(0, 1)  # STUB

# ── EMA smoother ──────────────────────────────────────────
ema_score = 0.0
ALPHA = 0.3

# ── Hysteresis state ──────────────────────────────────────
current_level = 0
THRESHOLDS_UP   = [0.30, 0.55, 0.75]   # cross up → level 1,2,3
THRESHOLDS_DOWN = [0.22, 0.47, 0.67]   # must drop below to deactivate

def score_to_level(score: float, prev_level: int) -> int:
    if score >= THRESHOLDS_UP[2]:   return 3
    if score >= THRESHOLDS_UP[1]:   return 2 if prev_level >= 2 else max(prev_level, 1 if score >= THRESHOLDS_UP[0] else 0)
    # simpler: just use hysteresis on UP only, allow free fall
    if prev_level == 3 and score < THRESHOLDS_DOWN[2]: return 2
    if prev_level == 2 and score < THRESHOLDS_DOWN[1]: return 1
    if prev_level == 1 and score < THRESHOLDS_DOWN[0]: return 0
    return prev_level

def compute_features(buf: list) -> dict:
    """Aggregate last N telemetry samples into feature dict."""
    if len(buf) < 10:
        return {}
    arr = {k: [s[k] for s in buf if k in s] for k in buf[0]}
    return {
        "lane_offset_mean": float(np.mean(arr["lane_offset"])),
        "lane_drift_var":   float(np.var(arr["lane_offset"])),
        "steering_instability": float(np.std(arr["steering_angle"]) /
                                      (abs(np.mean(arr["steering_angle"])) + 1e-6)),
        "correction_freq":  float(np.mean(arr["steering_correction_hz"])),
        "reaction_delay":   float(np.mean(arr["reaction_delay_ms"])),
        "speed_mean":       float(np.mean(arr["speed_kmh"])),
    }

@app.websocket("/ws/telemetry")
async def telemetry_ws(ws: WebSocket):
    global ema_score, current_level, telemetry_buffer
    await ws.accept()
    print("Godot connected")
    try:
        while True:
            raw = await ws.receive_text()
            print("Rx:", raw)
            data = json.loads(raw)
            telemetry_buffer.append(data)
            if len(telemetry_buffer) > WINDOW:
                telemetry_buffer.pop(0)

            features = compute_features(telemetry_buffer)
            if features:
                raw_score = predict_fatigue(features)
                ema_score = ALPHA * raw_score + (1 - ALPHA) * ema_score
                current_level = score_to_level(ema_score, current_level)

            await ws.send_text(json.dumps({
                "fatigue_level": current_level,
                "p_fatigue": round(ema_score, 3)
            }))
            print("TX: level", current_level, "p_fatigue", round(ema_score,3))

    except WebSocketDisconnect:
        print("Godot disconnected")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)