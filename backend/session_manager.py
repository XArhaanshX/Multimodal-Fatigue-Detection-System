import time
import asyncio
from .csv_storage import append_session, update_session
from .pipeline_interface import start_fatigue_pipeline
from .websocket_manager import manager as ws_manager

class SessionManager:
    def __init__(self):
        self.current_session = None
        self.pipeline_running = False
        self.pipeline_thread = None
        self.stop_event = None
        self.loop = None # Will be set by main.py

    def start_session(self, data: dict):
        """Initializes and starts a fatigue detection session."""
        if self.pipeline_running:
            return {"status": "error", "message": "Session already in progress"}
        
        print("[INFO] Session started")
        
        start_time = time.strftime("%Y-%m-%d %H:%M:%S")
        data["start_time"] = start_time
        
        # Persist to CSV and get row index
        row_index = append_session(data)
        
        # Initialize in-memory state
        self.current_session = {
            "driver_name": data.get("driver_name"),
            "emergency_contact_phone": data.get("emergency_contact_phone"),
            "max_fatigue_score": 0.0,
            "critical_event_triggered": False,
            "fatigue_above_threshold_duration": 0.0,
            "csv_row_index": row_index
        }
        
        # Start ML Pipeline
        self.pipeline_running = True
        self.pipeline_thread, self.stop_event = start_fatigue_pipeline(self.fatigue_callback)
        print("[INFO] Pipeline thread launched")
        
        return {"status": "success", "session": self.current_session}

    def end_session(self):
        """Stops the active session and saves final metrics."""
        if not self.pipeline_running:
            return {"status": "error", "message": "No active session to end"}
        
        # 1. Stop the pipeline thread
        if self.stop_event:
            self.stop_event.set()
        if self.pipeline_thread:
            self.pipeline_thread.join(timeout=5)
            
        # 2. Update CSV with final values
        end_time = time.strftime("%Y-%m-%d %H:%M:%S")
        update_data = {
            "end_time": end_time,
            "max_fatigue_score": round(self.current_session["max_fatigue_score"], 4),
            "critical_event_triggered": self.current_session["critical_event_triggered"]
        }
        update_session(self.current_session["csv_row_index"], update_data)
        
        # 3. Reset state
        self.pipeline_running = False
        self.current_session = None
        self.pipeline_thread = None
        self.stop_event = None
        
        print("[INFO] Session ended")
        return {"status": "success"}

    def fatigue_callback(self, score: float):
        """Handle incoming score from ML pipeline (called from thread)."""
        if not self.current_session:
            return

        # 1. Update max fatigue score observed
        if score > self.current_session["max_fatigue_score"]:
            self.current_session["max_fatigue_score"] = score
            
        # 2. Continuous threshold exceedance logic
        # Pipeline produces scores ≈1Hz, so dt ≈ 1.0s
        if score > 0.75:
            self.current_session["fatigue_above_threshold_duration"] += 1.0
        else:
            self.current_session["fatigue_above_threshold_duration"] = 0.0
            
        # 3. Trigger Critical Fatigue Alert
        if self.current_session["fatigue_above_threshold_duration"] >= 10.0 and not self.current_session["critical_event_triggered"]:
            self.current_session["critical_event_triggered"] = True
            print("\n" + "!" * 45)
            print("CRITICAL FATIGUE DETECTED — CONTACTING EMERGENCY CONTACT")
            print("!" * 45 + "\n")
            # NOTE: Logic for future Twilio SMS call would be inserted here

        # 4. Determine fatigue state mapping
        if score < 0.30: state = "NORMAL"
        elif score < 0.55: state = "MILD"
        elif score < 0.75: state = "HIGH"
        else: state = "CRITICAL"
        
        # 5. Broadcast to WebSocket clients
        payload = {
            "fatigue_score": round(score, 4),
            "fatigue_state": state
        }
        
        # Use thread-safe coroutine scheduling back to the main FastAPI loop
        if self.loop:
            asyncio.run_coroutine_threadsafe(ws_manager.broadcast(payload), self.loop)

session_manager = SessionManager()
