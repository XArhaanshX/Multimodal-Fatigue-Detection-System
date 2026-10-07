import asyncio
import time

from .csv_storage import append_session, update_session
from .monitor import FatigueMonitor
from .network.state import shared_state, telemetry_lock
from .websocket_manager import manager as ws_manager

CRITICAL_THRESHOLD = 0.75
CRITICAL_SECONDS = 10  # continuous seconds above threshold before the emergency flag


def fatigue_state(score):
    if score is None:
        return "WAITING"
    if score < 0.30:
        return "NORMAL"
    if score < 0.55:
        return "MILD"
    if score < CRITICAL_THRESHOLD:
        return "HIGH"
    return "CRITICAL"


class SessionManager:
    def __init__(self):
        self.session = None
        self.monitor = None
        self.loop = None                   # FastAPI event loop, set at startup
        self.started_by_simulator = False  # simulator-started sessions end when it disconnects
        self.ending = False                # True while end_session is stopping the monitor
        self.latest = {}                   # last update broadcast to clients

    @property
    def active(self):
        return self.monitor is not None

    def start_session(self, data: dict, use_camera=True, started_by_simulator=False):
        """Starts a session with the camera on or off; driving data is used whenever the simulator sends it."""
        if self.active:
            if not (self.started_by_simulator and not started_by_simulator):
                return {"status": "error", "message": "Session already in progress"}
            # The dashboard takes over a session the simulator started on its own
            self.end_session()

        data["start_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
        self.session = {
            "driver_name": data.get("driver_name"),
            "emergency_contact_name": data.get("emergency_contact_name"),
            "emergency_contact_phone": data.get("emergency_contact_phone"),
            "camera": use_camera,
            "max_fatigue_score": 0.0,
            "critical_event_triggered": False,
            "seconds_above_threshold": 0,
            "csv_row_index": append_session(data),
        }
        self.started_by_simulator = started_by_simulator
        self.latest = {}
        with telemetry_lock:  # telemetry from a previous session must not count as live
            shared_state.update(telemetry={}, last_telemetry_time=0.0)
        self.monitor = FatigueMonitor(use_camera, self._on_update)
        self.monitor.start()
        print(f"[INFO] Session started (camera {'on' if use_camera else 'off'}).")
        return {"status": "success", "session": self.session}

    def end_session(self):
        """Stops the monitor and saves the session's final metrics."""
        if not self.active:
            return {"status": "error", "message": "No active session to end"}
        self.ending = True
        self.monitor.stop()
        update_session(self.session["csv_row_index"], {
            "end_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "max_fatigue_score": round(self.session["max_fatigue_score"], 4),
            "critical_event_triggered": self.session["critical_event_triggered"],
        })
        with telemetry_lock:
            shared_state.update(latest_fatigue_score=None, fatigue_state="WAITING", vision_status="off")
        self.monitor = None
        self.session = None
        self.started_by_simulator = False
        self.ending = False
        print("[INFO] Session ended.")
        self._broadcast({"session_active": False})
        return {"status": "success"}

    def _on_update(self, update):
        """Called by the monitor thread once per second."""
        session = self.session
        if not session:
            return
        score = update["score"]
        state = fatigue_state(score)

        if score is not None:
            session["max_fatigue_score"] = max(session["max_fatigue_score"], score)
            session["seconds_above_threshold"] = session["seconds_above_threshold"] + 1 if score > CRITICAL_THRESHOLD else 0
            if session["seconds_above_threshold"] >= CRITICAL_SECONDS and not session["critical_event_triggered"]:
                session["critical_event_triggered"] = True
                print("[ALERT] Critical fatigue for 10 s: emergency contact would be notified here.")

        with telemetry_lock:
            shared_state.update(latest_fatigue_score=score, fatigue_state=state, vision_status=update["camera_status"])

        self._broadcast({
            "session_active": True,
            "fatigue_score": None if score is None else round(score, 4),
            "fatigue_state": state,
            "source": update["source"],
            "camera_status": update["camera_status"],
            "simulator_connected": update["simulator_connected"],
            "critical_event_triggered": session["critical_event_triggered"],
        })

    def _broadcast(self, payload):
        self.latest = payload
        if self.loop:
            asyncio.run_coroutine_threadsafe(ws_manager.broadcast(payload), self.loop)


session_manager = SessionManager()
