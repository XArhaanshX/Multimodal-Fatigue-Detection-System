from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio
import os
import sys

# Ensure backend root is in path if running from parent directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from session_manager import session_manager
from websocket_manager import manager as ws_manager

app = FastAPI(title="Multimodal Driver Fatigue Detection Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SessionStartRequest(BaseModel):
    driver_name: str
    driver_phone: str
    emergency_contact_name: str
    emergency_contact_phone: str

@app.on_event("startup")
async def startup_event():
    # Capture the main event loop for the session manager's thread-safe broadcasts
    session_manager.loop = asyncio.get_event_loop()
    print("[INFO] FastAPI application startup complete.")

@app.post("/session/start")
async def start_session(request: SessionStartRequest):
    """Starts a new driving session and the ML pipeline."""
    result = session_manager.start_session(request.dict())
    return result

@app.post("/session/end")
async def end_session():
    """Ends the current active session and stops the ML pipeline."""
    result = session_manager.end_session()
    return result

@app.websocket("/ws/fatigue-score")
async def fatigue_score_ws(websocket: WebSocket):
    """WebSocket endpoint for real-time fatigue score streaming."""
    await ws_manager.connect(websocket)
    print(f"[INFO] New client connected to fatigue-score stream. Total: {len(ws_manager.active_connections)}")
    try:
        while True:
            # Keep-alive loop to detect disconnects
            await websocket.receive_text()
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
        print(f"[INFO] Client disconnected. Total: {len(ws_manager.active_connections)}")
    except Exception as e:
        print(f"[ERROR] WebSocket error: {e}")
        ws_manager.disconnect(websocket)

if __name__ == "__main__":
    import uvicorn
    # Minimal hackathon setup: Run on port 8000
    print("[INFO] Starting backend server on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
