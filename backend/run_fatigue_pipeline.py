import cv2
import sys
import os
import time
import threading
import numpy as np

# Ensure project root is in sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from ml.fatigue_model import FatigueModel
    from ml.features import build_feature_vector
    from ml.smoothing import EMASmoother
    from vision.main import get_vision_pipeline
except ImportError as e:
    print(f"[ERROR] Import failed: {e}")
    sys.exit(1)

def start_fatigue_pipeline(shared_state, telemetry_lock):
    """
    Clean Baseline Fatigue Scoring Pipeline.
    Implements Sections 1-7 (Rollback & Demo Bias).
    """
    print("[INFO] Initializing Clean Baseline Pipeline...")
    try:
        model = FatigueModel()
        # Section 1: Baseline alpha = 0.3
        smoother = EMASmoother(alpha=0.3)
    except Exception as e:
        print(f"[ERROR] System initialization failed: {e}")
        return

    # Initialize Timers and State
    session_start_time = time.time()
    last_inference_time = 0.0
    
    # Section 3 & 4: Persistent Bias Accumulator
    demo_bias_accum = 0.0
    
    # HUD State
    current_state = "normal"
    current_color = (0, 255, 0)

    print("[INFO] Initializing Vision Pipeline (Inference: 1 Hz).")
    pipeline = get_vision_pipeline()

    try:
        for features, frame in pipeline:
            if frame is None:
                continue

            current_time = time.time()
            # --- 1 HZ INFERENCE BLOCK ---
            if features is not None and (current_time - last_inference_time >= 1.0):
                
                dt = 1.0 

                # 1. Telemetry Snapshot
                with telemetry_lock:
                    telemetry_snapshot = shared_state["telemetry"].copy()

                # 2. Section 3: Simple Demo Bias Layer
                yawn_event = features.get("yawn_event_this_window", False)
                blink_event = features.get("blink_event_this_window", False)
                
                if yawn_event:
                    demo_bias_accum += 0.15
                if blink_event:
                    demo_bias_accum += 0.01
                
                # Section 4: Natural Fatigue Recovery (Decay)
                # Decay only if no yawn just happened
                if not yawn_event:
                    demo_bias_accum = max(0.0, demo_bias_accum - (0.02 * dt))

                # 3. Model Prediction Block
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
                
                # STEP 1: Model Output
                model_score = model.predict(vector)
                
                # STEP 2: Bias Adjusted (Section 3)
                bias_adjusted_score = model_score + demo_bias_accum
                
                # STEP 3: Prevent Extreme Spikes (Section 5)
                target_score = min(bias_adjusted_score, 0.95)
                target_score = max(target_score, 0.0)
                
                # STEP 4: EMA Smoothing (Section 1)
                final_score = smoother.update(target_score)
                
                # Update Loop State
                last_inference_time = current_time

                # 4. State Mapping
                if final_score >= 0.75: current_state = "critical"
                elif final_score >= 0.55: current_state = "severe"
                elif final_score >= 0.30: current_state = "mild"
                else: current_state = "normal"

                if current_state == "normal": current_color = (0, 255, 0)
                elif current_state == "mild": current_color = (150, 255, 0)
                elif current_state == "severe": current_color = (0, 165, 255)
                elif current_state == "critical": current_color = (0, 0, 255)
                
                # 5. Shared State Write-back
                with telemetry_lock:
                    shared_state["latest_fatigue_score"] = float(final_score)
                    shared_state["fatigue_state"] = current_state
                    shared_state["alert"] = bool(final_score > 0.55)
                    shared_state["latest_vision_features"] = vision_dict.copy()

                # 6. Section 6: Debug Logging
                print("\n" + "="*45)
                print(f"FATIGUE: {final_score:.4f} ({current_state.upper()})")
                print(f"[LOG] model_score: {model_score:.4f}")
                print(f"[LOG] yawn_event: {yawn_event} | blink_event: {blink_event}")
                print(f"[LOG] bias_accum: {demo_bias_accum:.4f}")
                print(f"[LOG] bias_adjusted_score: {bias_adjusted_score:.4f}")
                print(f"[LOG] final_score: {final_score:.4f}")
                print("="*45)

            # --- HUD OVERLAY ---
            overlay_text = f"Fatigue: {float(shared_state.get('latest_fatigue_score', 0)):.2f} ({current_state})"
            cv2.putText(frame, overlay_text, (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, current_color, 2)

            cv2.imshow("IRoad - Multimodal Fatigue Detector", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"[ERROR] Pipeline runtime error: {e}")
        import traceback; traceback.print_exc()
    finally:
        cv2.destroyAllWindows()
        print("[INFO] Pipeline terminated.")

if __name__ == "__main__":
    from network.state import shared_state, telemetry_lock
    start_fatigue_pipeline(shared_state, telemetry_lock)
