"""Fatigue scoring loop for one session: camera and/or simulator in, one update per second out."""
import os
import threading
import time

import cv2

from .ml.fatigue_model import FatigueModel
from .ml.features import build_feature_vector, build_driving_vector
from .ml.smoothing import EMASmoother
from .network.state import shared_state, telemetry_lock
from .vision.main import get_vision_pipeline, VISION_STATUS

# Set FATIGUE_SHOW_HUD=0 to run without the OpenCV preview window (e.g. headless)
SHOW_HUD = os.environ.get("FATIGUE_SHOW_HUD", "1") != "0"

SCORE_INTERVAL = 1.0           # seconds between scores
TELEMETRY_STALE_SECONDS = 5.0  # simulator counts as disconnected after this
FACE_LOST_SECONDS = 1.5        # camera counts as "no face" after this
MAX_SCORE = 0.95               # prevent extreme spikes

# Demo bias: visible events nudge the score so live demos react quickly. Hand-tuned, not learned.
YAWN_BIAS = 0.15
BLINK_BIAS = 0.01
BIAS_DECAY_PER_SECOND = 0.02


def read_telemetry():
    """Latest simulator features, or None if the simulator isn't sending."""
    with telemetry_lock:
        if time.time() - shared_state["last_telemetry_time"] >= TELEMETRY_STALE_SECONDS:
            return None
        return dict(shared_state["telemetry"])


def camera_status(now):
    if VISION_STATUS["camera_ok"] is None:
        return "starting"
    if not VISION_STATUS["camera_ok"]:
        return "no_camera"
    return "tracking" if now - VISION_STATUS["last_face_time"] < FACE_LOST_SECONDS else "no_face"


class FatigueMonitor:
    """
    Scores fatigue once per second from whatever inputs are available:
    face + driving -> fused model, face only -> fused model (telemetry zeros),
    driving only (camera off or face lost) -> driving-only model.
    """

    def __init__(self, use_camera, on_update):
        self.use_camera = use_camera
        self.on_update = on_update  # called with a status dict once per second
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self, timeout=5.0):
        self._stop.set()
        self._thread.join(timeout)

    def _run(self):
        try:
            self.fused_model = FatigueModel()
            self.driving_model = FatigueModel("fatigue_model_driving.pkl")
        except Exception as e:
            print(f"[ERROR] Model loading failed: {e}")
            return

        self.smoother = EMASmoother(alpha=0.3)
        self.session_start = time.time()
        self.bias = 0.0
        self.score = None
        vision, fresh = None, False
        show_hud = SHOW_HUD and self.use_camera
        frames = None
        if self.use_camera:
            VISION_STATUS.update(camera_ok=None, last_face_time=0.0)
            frames = get_vision_pipeline()

        next_score = time.time() + SCORE_INTERVAL
        try:
            while not self._stop.is_set():
                if frames:
                    features, frame = next(frames)
                    if features is not None:
                        vision, fresh = features, True
                    if show_hud:
                        show_hud = self._show(frame)
                else:
                    time.sleep(0.05)

                now = time.time()
                if now >= next_score:
                    next_score = now + SCORE_INTERVAL
                    self._score(vision, fresh, now)
                    fresh = False
        except Exception as e:
            print(f"[ERROR] Monitor loop error: {e}")
        finally:
            if frames:
                frames.close()
            if show_hud:
                cv2.destroyAllWindows()
            print("[INFO] Monitor stopped.")

    def _score(self, vision, fresh, now):
        status = camera_status(now) if self.use_camera else "off"
        telemetry = read_telemetry()
        face = vision if status == "tracking" else None

        if face is not None:
            probability = self.fused_model.predict(
                build_feature_vector(self._vision_dict(face, now), telemetry, self.session_start))
            probability += self._update_bias(face if fresh else {})
            source = "camera+simulator" if telemetry is not None else "camera"
        elif telemetry is not None:
            probability = self.driving_model.predict(build_driving_vector(telemetry, self.session_start))
            source = "simulator"
        else:
            probability, source = None, None

        if probability is not None:
            self.score = self.smoother.update(min(max(probability, 0.0), MAX_SCORE))

        self.on_update({
            "score": self.score if source else None,
            "source": source,
            "camera_status": status,
            "simulator_connected": telemetry is not None,
        })

    def _vision_dict(self, f, now):
        minutes = max(1.0, (now - self.session_start) / 60.0)
        return {
            "EAR_mean": f["EAR_mean"],
            "EAR_std": f["EAR_std"],
            "EAR_trend": f["EAR_trend"],
            "blink_frequency": f["blink_frequency"],
            "ECD_max": f.get("ECD_max", 0.0),
            "MAR_max": min(f["MAR_max"], 1.5),
            "pitch_mean": f["pitch_mean"],
            "pitch_std": min(f["pitch_std"], 20.0),
            "yawn_frequency": f["yawn_total"] / minutes,
            "gaze_ratio": 0.0,
        }

    def _update_bias(self, f):
        """Yawn/blink events in the latest second add to the bias; otherwise it decays."""
        if f.get("yawn_event_this_window"):
            self.bias += YAWN_BIAS
        else:
            self.bias = max(0.0, self.bias - BIAS_DECAY_PER_SECOND * SCORE_INTERVAL)
        if f.get("blink_event_this_window"):
            self.bias += BLINK_BIAS
        return self.bias

    def _show(self, frame):
        """Draws the HUD window; returns False if no display is available."""
        if self.score is not None:
            cv2.putText(frame, f"Fatigue: {self.score:.2f}", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
        try:
            cv2.imshow("IRoad - Driver Camera", frame)
            cv2.waitKey(1)
            return True
        except cv2.error as e:
            print(f"[WARNING] HUD unavailable, continuing headless: {e}")
            return False
