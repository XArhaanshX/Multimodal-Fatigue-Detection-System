import threading
import time
import os
import sys
import numpy as np
import cv2

# Ensure project root is in sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from ml.fatigue_model import FatigueModel
    from ml.features import build_feature_vector
    from ml.smoothing import EMASmoother
    from vision.main import get_vision_pipeline
except ImportError as e:
    print(f"[ERROR] Import failed in pipeline_interface: {e}")

def start_fatigue_pipeline(callback):
    """
    Launches the fatigue detection pipeline in a background thread.
    Returns (thread, stop_event).
    """
    stop_event = threading.Event()
    
    def pipeline_loop():
        print("[INFO] Pipeline thread launched.")
        try:
            model = FatigueModel()
            # Standard smoothing alpha from stable baseline
            smoother = EMASmoother(alpha=0.3)
        except Exception as e:
            print(f"[ERROR] Pipeline initialization failed: {e}")
            return

        session_start_time = time.time()
        last_inference_time = 0.0
        demo_bias_accum = 0.0
        
        # Static telemetry snapshot for this hackathon layer
        telemetry_snapshot = {
            "lane_drift_var": 0.0,
            "lane_offset_mean": 0.0,
            "steering_instability": 0.0,
            "correction_freq": 0.0,
            "reaction_delay_mean": 0.0,
            "steering_reversals": 0.0
        }

        pipeline = get_vision_pipeline()

        try:
            for features, frame in pipeline:
                if stop_event.is_set():
                    break

                if frame is None:
                    continue

                current_time = time.time()
                
                # --- 1 HZ INFERENCE BLOCK ---
                if features is not None and (current_time - last_inference_time >= 1.0):
                    dt = 1.0  # Fixed interval approximately
                    
                    # 1. Demo Bias Layer Logic
                    yawn_event = features.get("yawn_event_this_window", False)
                    blink_event = features.get("blink_event_this_window", False)
                    
                    if yawn_event:
                        demo_bias_accum += 0.15
                    if blink_event:
                        demo_bias_accum += 0.01
                    
                    # Natural decay if no yawn
                    if not yawn_event:
                        demo_bias_accum = max(0.0, demo_bias_accum - (0.02 * dt))

                    # 2. Build Feature Vector for Model
                    vision_dict = {
                        "EAR_mean": features["EAR_mean"],
                        "EAR_std": features["EAR_std"],
                        "EAR_trend": features["EAR_trend"],
                        "blink_frequency": features["blink_frequency"],
                        "ECD_max": features.get("ECD_max", 0.0),
                        "MAR_max": min(features["MAR_max"], 1.5),
                        "pitch_mean": features["pitch_mean"],
                        "pitch_std": min(features["pitch_std"], 20.0),
                        "perclos": features["perclos"],
                        "yawn_frequency": features["yawn_total"] / max(1.0, (current_time - session_start_time)/60.0),
                        "gaze_ratio": 0.0
                    }
                    
                    vector = build_feature_vector(vision_dict, telemetry_snapshot, session_start_time)
                    
                    # 3. Predict & Smooth
                    model_score = model.predict(vector)
                    bias_adjusted_score = model_score + demo_bias_accum
                    target_score = min(max(bias_adjusted_score, 0.0), 0.95)
                    final_score = smoother.update(target_score)
                    
                    last_inference_time = current_time

                    # 4. Trigger Callback
                    callback(float(final_score))

                # Visual HUD for demo visibility (optional but recommended)
                cv2.imshow("IRoad - Backend ML Pipeline", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except Exception as e:
            print(f"[ERROR] Pipeline runtime error: {e}")
        finally:
            cv2.destroyAllWindows()
            print("[INFO] Pipeline thread terminated.")

    thread = threading.Thread(target=pipeline_loop, daemon=True)
    thread.start()
    return thread, stop_event
