from contextlib import asynccontextmanager
from pathlib import Path
import asyncio
import os
import sys
import time

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Support both `python -m backend.main` / `uvicorn backend.main:app` and `python backend/main.py`
if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "backend"

from .session_manager import session_manager
from .websocket_manager import manager as ws_manager
from .network.state import shared_state, telemetry_lock
from .network.telemetry_features import TelemetryWindow, extract_direct_features, extract_raw_telemetry, compute_window_features

ROOT = Path(__file__).resolve().parent.parent
TELEMETRY_REPLY_INTERVAL = 0.5  # seconds between fatigue updates sent back to the simulator
SIMULATOR_DRIVER = {"driver_name": "Simulator driver", "driver_phone": "",
                    "emergency_contact_name": "", "emergency_contact_phone": ""}
simulator_connections = 0


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Capture the main event loop for the session manager's thread-safe broadcasts
    session_manager.loop = asyncio.get_running_loop()
    print("[INFO] Server ready on http://localhost:8000")
    yield
    if session_manager.active:
        session_manager.end_session()


app = FastAPI(title="Multimodal Driver Fatigue Detection Backend", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])


class SessionStartRequest(BaseModel):
    driver_name: str
    driver_phone: str
    emergency_contact_name: str
    emergency_contact_phone: str
    camera: bool = True
    simulator: bool = False  # informational: driving data is used whenever the simulator connects


@app.get("/")
async def root():
    return RedirectResponse(url="/dashboard/login.html")


@app.get("/health")
async def health():
    return {"ok": True, "session_active": session_manager.active}


@app.get("/session/status")
async def session_status():
    return {"active": session_manager.active, "session": session_manager.session, "latest": session_manager.latest}


@app.post("/session/start")
async def start_session(request: SessionStartRequest):
    """Starts a session; may end a simulator-started session first, which blocks briefly."""
    data = request.model_dump(exclude={"camera", "simulator"})
    return await asyncio.to_thread(session_manager.start_session, data, request.camera)


@app.post("/session/end")
async def end_session():
    return await asyncio.to_thread(session_manager.end_session)


@app.websocket("/ws/fatigue-score")
async def fatigue_score_ws(websocket: WebSocket):
    """Streams one update per second to the dashboard."""
    await ws_manager.connect(websocket)
    if session_manager.latest:
        await websocket.send_json(session_manager.latest)
    try:
        while True:
            await websocket.receive_text()  # keep-alive; detects disconnects
    except WebSocketDisconnect:
        pass
    finally:
        ws_manager.disconnect(websocket)


@app.websocket("/ws/telemetry")
async def telemetry_ws(websocket: WebSocket):
    """
    Receives simulator telemetry and replies with the latest fatigue score.
    Accepts pre-aggregated features (browser simulator) or raw 10 Hz samples (e.g. a Godot simulator).
    """
    global simulator_connections
    await websocket.accept()
    simulator_connections += 1
    print("[INFO] Simulator connected.")
    # Wait out a reload's previous session shutdown, then start a camera-off session if none is running
    while session_manager.ending:
        await asyncio.sleep(0.1)
    if not session_manager.active:
        session_manager.start_session(dict(SIMULATOR_DRIVER), use_camera=False, started_by_simulator=True)

    raw_window = TelemetryWindow()
    last_reply = 0.0
    try:
        while True:
            data = await websocket.receive_json()
            features = extract_direct_features(data)
            if not features:
                raw_window.add_sample(extract_raw_telemetry(data))
                features = compute_window_features(raw_window.get_samples())
            with telemetry_lock:
                if features:
                    shared_state["telemetry"].update(features)
                    shared_state["last_telemetry_time"] = time.time()
                reply = {
                    "type": "fatigue_update",
                    "fatigue_score": shared_state["latest_fatigue_score"],
                    "fatigue_state": shared_state["fatigue_state"],
                    "vision_status": shared_state["vision_status"],
                }

            now = time.time()
            if now - last_reply >= TELEMETRY_REPLY_INTERVAL:
                reply.update(session_active=session_manager.active, timestamp=now)
                await websocket.send_json(reply)
                last_reply = now
    except WebSocketDisconnect:
        print("[INFO] Simulator disconnected.")
    except Exception as e:
        print(f"[ERROR] Telemetry WebSocket error: {e}")
    finally:
        simulator_connections -= 1
        if simulator_connections == 0 and session_manager.started_by_simulator:
            await asyncio.to_thread(session_manager.end_session)


for mount, folder in (("/sim", "simulator"), ("/dashboard", "dashboard")):
    if (ROOT / folder).is_dir():
        app.mount(mount, StaticFiles(directory=ROOT / folder, html=True), name=folder)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
