# backend/run_fatigue_pipeline.py

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
    from ml.feature_schema import FEATURE_ORDER
    from ml.smoothing import EMASmoother
    from vision.main import get_vision_pipeline
    from network.telemetry_server import run_server, shared_state, telemetry_lock
except ImportError as e:
    print(f"[ERROR] Import failed: {e}")
    sys.exit(1)

def demo_boost(prob, vision, telemetry):
    """
    Step 1 & 2: Amplifies fatigue signals from visual cues for demo scenarios.
    Step 3: Adds secondary telemetry-based boosts.
    """
    boost = 0.0
    
    # VISION-DOMINANT BOOSTS (Step 2)
    if vision.get("EAR_mean", 1.0) < 0.25:
        boost += 0.55
    if vision.get("blink_frequency", 0.0) > 3.0:
        boost += 0.30
    if vision.get("MAR_max", 0.0) > 0.6:
        boost += 0.40
    if vision.get("yawn_frequency", 0.0) > 0:
        boost += 0.30
    if abs(vision.get("pitch_mean", 0.0)) > 8.0:
        boost += 0.25
        
    # TELEMETRY BOOSTS (Step 3)
    if telemetry.get("lane_drift_var", 0.0) > 0.3:
        boost += 0.10
    if telemetry.get("steering_instability", 0.0) > 0.2:
        boost += 0.10
    if telemetry.get("reaction_delay_mean", 0.0) > 0.5:
        boost += 0.10
        
    return prob + boost

def run_pipeline():
    """
    Final integrated multimodal fatigue detection pipeline.
    Optimized for Demo Mode: High responsiveness to visual fatigue cues.
    """
    print("[INFO] Initializing Optimized Fatigue Detection Model...")
    try:
        model = FatigueModel()
        # Step 5: Faster EMA Response (alpha = 0.65)
        smoother = EMASmoother(alpha=0.65)
    except Exception as e:
        print(f"[ERROR] System initialization failed: {e}")
        return

    # Initialize Contextual Session Timing
    session_start_time = time.time()

    # Start Telemetry WebSocket Server in background
    print("[INFO] Starting Telemetry WebSocket Server on port 8001...")
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()

    print("[INFO] Starting Vision Pipeline (30 FPS Display | 1 Hz Inference). Press 'q' to exit.")
    
    pipeline = get_vision_pipeline()
    
    # State Synchronization & Refinements
    last_inference_time = 0.0
    current_score = 0.0
    current_state = "NORMAL"
    current_color = (0, 255, 0)
    sim_status = "WAITING"

    try:
        for features, frame in pipeline:
            # frame is provided at 30 FPS for the HUD
            if frame is None:
                continue

            # --- 1 HZ INFERENCE BLOCK ---
            current_time = time.time()
            if features is not None and (current_time - last_inference_time >= 1.0):
                
                # Thread-Safe Telemetry Snapshot
                with telemetry_lock:
                    telemetry_snapshot = shared_state["telemetry"].copy()
                    last_msg_time = shared_state["last_telemetry_time"]
                
                # Simulator Connection Health
                if current_time - last_msg_time < 5.0:
                    sim_status = "ACTIVE"
                else:
                    sim_status = "WAITING"

                # 3. Build Vision Dictionary with Clamping
                vision_dict = {
                    "EAR_mean": features["EAR_mean"],
                    "EAR_std": features["EAR_std"],
                    "EAR_trend": features["EAR_trend"],
                    "blink_frequency": features["blink_frequency"],
                    "MAR_max": min(features["MAR_max"], 1.5),
                    "pitch_mean": features["pitch_mean"],
                    "pitch_std": min(features["pitch_std"], 20.0),
                    "perclos": features["perclos"],
                    "yawn_frequency": 0.0, # Placeholder (calculated in vision/main.py if needed)
                    "gaze_ratio": 0.0
                }

                # 4. Feature Vector Fusion & Validation
                vector = build_feature_vector(vision_dict, telemetry_snapshot, session_start_time)
                
                # Strict Validation
                if len(vector) != len(FEATURE_ORDER):
                    print(f"[WARNING] Feature Mismatch! Got {len(vector)}, expected {len(FEATURE_ORDER)}")
                
                # 5. Model Inference
                raw_prob = model.predict(vector)
                
                # STEP 4: APPLY DEMO BOOST
                boosted_prob = demo_boost(raw_prob, vision_dict, telemetry_snapshot)
                boosted_prob = max(0.0, min(1.0, boosted_prob)) # Clamp
                
                # 6. EMA Smoothing (Faster for Demo)
                current_score = smoother.update(boosted_prob)

                # 7. Fatigue State Mapping with Hysteresis (Step 6 thresholds preserved)
                new_state = "NORMAL"
                if current_score >= 0.75:
                    new_state = "CRITICAL"
                elif current_score >= 0.55:
                    new_state = "HIGH"
                elif current_score >= 0.30:
                    new_state = "MILD"
                
                # Apply 0.05 Hysteresis for downgrades
                if current_state == "CRITICAL" and current_score >= 0.70:
                    current_state = "CRITICAL"
                elif current_state == "HIGH" and current_score >= 0.50:
                    current_state = "CRITICAL" if new_state == "CRITICAL" else "HIGH"
                elif current_state == "MILD" and current_score >= 0.25:
                    current_state = new_state if new_state in ["CRITICAL", "HIGH"] else "MILD"
                else:
                    current_state = new_state

                # Update HUD Colors
                if current_state == "NORMAL": current_color = (0, 255, 0)
                elif current_state == "MILD": current_color = (0, 255, 255) # Cyan/Yellowish
                elif current_state == "HIGH": current_color = (0, 165, 255) # Orange
                elif current_state == "CRITICAL": current_color = (0, 0, 255) # Red (BGR)
                
                # 8. Feedback to Simulation
                shared_state["latest_fatigue_score"] = float(current_score)
                shared_state["alert"] = bool(current_score > 0.55)

                # 9. Extended Structured Logging
                print("\n" + "="*40)
                print(f"[DEMO MOD] EAR: {vision_dict['EAR_mean']:.3f} | MAR: {vision_dict['MAR_max']:.3f}")
                print(f"[DEMO MOD] Drift: {telemetry_snapshot['lane_drift_var']:.2f} | Status: {sim_status}")
                print(f"[DEMO MOD] ML Prob: {raw_prob:.4f} | Boosted: {boosted_prob:.4f}")
                print(f"[DEMO MOD] Score: {current_score:.4f} | State: {current_state}")
                print("="*40)
                
                last_inference_time = current_time

            # --- HUD OVERLAY (30 FPS) ---
            overlay_text = f"Fatigue: {current_score:.2f} | State: {current_state} | Sim: {sim_status}"
            cv2.putText(frame, overlay_text, (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, current_color, 2)

            # Display frame
            cv2.imshow("IRoad - Multimodal Fatigue Detector", frame)

            # Safe OpenCV Window Exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[INFO] Quitting application...")
                break

    except KeyboardInterrupt:
        print("\n[INFO] Interrupt received.")
    except Exception as e:
        print(f"[ERROR] Pipeline runtime error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cv2.destroyAllWindows()
        print("[INFO] Cleanup complete. Pipeline terminated.")

if __name__ == "__main__":
    run_pipeline()
