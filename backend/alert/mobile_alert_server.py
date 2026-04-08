from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse


ROOT_DIR = Path(__file__).resolve().parents[2]
MOBILE_DIR = ROOT_DIR / "mobile"

PHONE_HTML = MOBILE_DIR / "phone-alert.html"
PHONE_JS = MOBILE_DIR / "phone-alert.js"
PHONE_MANIFEST = MOBILE_DIR / "phone-alert.webmanifest"
PHONE_SW = MOBILE_DIR / "phone-sw.js"
PHONE_ICON = MOBILE_DIR / "phone-icon.svg"


app = FastAPI(title="Driver Alert Mobile Hub")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

mobile_clients: set[WebSocket] = set()


def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in ("", None):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_pattern(value: Any) -> list[int]:
    if not isinstance(value, list):
        return []

    pattern: list[int] = []
    for item in value[:8]:
        try:
            number = int(item)
        except (TypeError, ValueError):
            continue
        pattern.append(max(0, min(number, 5000)))
    return pattern


def build_alert_message(payload: dict[str, Any]) -> dict[str, Any]:
    fatigue_score = max(0.0, min(_coerce_float(payload.get("fatigue_score")), 1.0))
    erratic_score = max(0.0, min(_coerce_float(payload.get("erratic_score")), 1.0))
    fatigue_state = str(payload.get("fatigue_state", "")).upper().strip()
    source = str(payload.get("source", "unknown")).strip() or "unknown"
    driver_id = str(payload.get("driver_id", "default")).strip() or "default"

    explicit_alert = payload.get("alert")
    if explicit_alert is None:
        alert = (
            fatigue_score >= 0.55
            or erratic_score >= 0.70
            or fatigue_state in {"HIGH", "CRITICAL"}
        )
    else:
        alert = bool(explicit_alert)

    if fatigue_state not in {"NORMAL", "MILD", "HIGH", "CRITICAL"}:
        if fatigue_score >= 0.75 or erratic_score >= 0.85:
            fatigue_state = "CRITICAL"
        elif fatigue_score >= 0.55 or erratic_score >= 0.70:
            fatigue_state = "HIGH"
        elif fatigue_score >= 0.30 or erratic_score >= 0.45:
            fatigue_state = "MILD"
        else:
            fatigue_state = "NORMAL"

    pattern = _normalize_pattern(payload.get("pattern"))
    if alert and not pattern:
        pattern = [500, 180, 500, 180, 700] if fatigue_state == "CRITICAL" else [250, 120, 250]

    return {
        "type": "driver_alert",
        "alert": alert,
        "state": fatigue_state,
        "fatigue_score": fatigue_score,
        "erratic_score": erratic_score,
        "pattern": pattern,
        "source": source,
        "driver_id": driver_id,
        "timestamp": time.time(),
        "message": str(payload.get("message", "")).strip(),
    }


async def broadcast_to_mobile_clients(payload: dict[str, Any]) -> None:
    stale_clients: list[WebSocket] = []
    for client in mobile_clients:
        try:
            await client.send_json(payload)
        except Exception:
            stale_clients.append(client)

    for client in stale_clients:
        mobile_clients.discard(client)


def _require_asset(path: Path) -> FileResponse:
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Missing asset: {path.name}")
    return FileResponse(path)


@app.get("/")
async def root() -> RedirectResponse:
    return RedirectResponse(url="/phone/", status_code=307)


@app.get("/phone")
@app.get("/phone/")
async def phone() -> FileResponse:
    return _require_asset(PHONE_HTML)


@app.get("/phone/app.js")
async def phone_app() -> FileResponse:
    return _require_asset(PHONE_JS)


@app.get("/phone/manifest.webmanifest")
async def phone_manifest() -> FileResponse:
    return _require_asset(PHONE_MANIFEST)


@app.get("/phone/sw.js")
async def phone_service_worker() -> FileResponse:
    return _require_asset(PHONE_SW)


@app.get("/phone/icon.svg")
async def phone_icon() -> FileResponse:
    return _require_asset(PHONE_ICON)


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "ok": True,
        "mobile_clients": len(mobile_clients),
        "timestamp": time.time(),
    }


@app.post("/api/alert")
async def create_alert(payload: dict[str, Any]) -> dict[str, Any]:
    message = build_alert_message(payload)
    await broadcast_to_mobile_clients(message)
    return {
        "accepted": True,
        "mobile_clients": len(mobile_clients),
        "broadcast": message,
    }


@app.websocket("/ws/mobile")
async def mobile_ws(ws: WebSocket) -> None:
    await ws.accept()
    mobile_clients.add(ws)
    await ws.send_json(
        {
            "type": "connection",
            "connected": True,
            "mobile_clients": len(mobile_clients),
            "timestamp": time.time(),
        }
    )

    try:
        while True:
            raw_message = await ws.receive_text()
            if raw_message.strip().lower() == "ping":
                await ws.send_json({"type": "pong", "timestamp": time.time()})
    except WebSocketDisconnect:
        mobile_clients.discard(ws)
    except Exception:
        mobile_clients.discard(ws)


@app.websocket("/ws/events")
async def events_ws(ws: WebSocket) -> None:
    await ws.accept()
    try:
        while True:
            raw_message = await ws.receive_text()
            payload = json.loads(raw_message)
            message = build_alert_message(payload)
            await broadcast_to_mobile_clients(message)
            await ws.send_json(
                {
                    "accepted": True,
                    "alert": message["alert"],
                    "state": message["state"],
                    "timestamp": message["timestamp"],
                }
            )
    except WebSocketDisconnect:
        return


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.alert.mobile_alert_server:app",
        host="0.0.0.0",
        port=8765,
        reload=True,
    )
