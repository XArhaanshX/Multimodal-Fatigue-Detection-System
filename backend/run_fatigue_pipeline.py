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
    # No longer importing telemetry_server here; state is passed by the caller.
except ImportError as e:
    print(f"[ERROR] Import failed: {e}")
    sys.exit(1)

def apply_progressive_scoring(raw_prob, vision, telemetry, yawn_accumulator, skip_mar_boost):
    """
    Refined Demo Scoring Escalation (Post-Processing).
    Fuses ML base signal with stateful vision-dominant boosts.
    """
    fatigue = raw_prob
    
    # STEP 2: VISION DOMINANT BOOSTS
    ear = vision.get("EAR_mean", 1.0)
    if ear < 0.20:
        fatigue += 0.30
    elif ear < 0.25:
        fatigue += 0.18
        
    # Blink frequency contribution (Capped at 0.15)
    bf = vision.get("blink_frequency", 0.0)
    fatigue += min(0.15, bf * 0.04)
    
    # Head pitch fatigue
    hp = abs(vision.get("pitch_mean", 0.0))
    if hp > 8.0:
        fatigue += 0.12
        
    # STEP 3: STATEFUL YAWN ACCUMULATOR
    # Capped at +0.75 boost
    yawn_boost = min(0.75, 0.25 * yawn_accumulator)
    fatigue += yawn_boost
    
    # STEP 4: PREVENT DOUBLE COUNTING
    # Only apply generic MAR boost if no specific yawn event occurred
    if not skip_mar_boost and vision.get("MAR_max", 0.0) > 0.6:
        fatigue += 0.20
        
    # STEP 5: TELEMETRY CONTRIBUTION (SECONDARY)
    if telemetry.get("lane_drift_var", 0.0) > 0.3:
        fatigue += 0.08
    if telemetry.get("steering_instability", 0.0) > 0.2:
        fatigue += 0.08
        
    return fatigue

def start_fatigue_pipeline(shared_state, telemetry_lock):
    """
    Refactored Integrated Multimodal Fatigue Detection Pipeline.
    Runs as a background thread inside the FastAPI process.
    """
    print("[INFO] Initializing Progressive Fatigue Detection Model...")
    try:
        model = FatigueModel()
        # STEP 8: Moderate Smoothing for Stable Response
        smoother = EMASmoother(alpha=0.55)
    except Exception as e:
        print(f"[ERROR] System initialization failed: {e}")
        return

    # Initialize State Counters
    session_start_time = time.time()
    
    # Yawn Accumulator State
    yawn_accumulator = 0
    prev_yawn_total = 0
    last_yawn_time = time.time()

    print("[INFO] Initializing Vision Pipeline (Inference: 1 Hz).")
    pipeline = get_vision_pipeline()
    
    # State Synchronization
    last_inference_time = 0.0
    current_score = 0.0
    current_state = "NORMAL"
    current_color = (0, 255, 0)
    sim_status = "WAITING"

    try:
        for features, frame in pipeline:
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

                # Detect Yawn Increments and Manage Decay (Step 6)
                new_yawn_total = features.get("yawn_total", 0)
                yawn_event_this_window = features.get("yawn_event_this_window", False)
                
                if new_yawn_total > prev_yawn_total:
                    yawn_accumulator += 1
                    last_yawn_time = current_time # Reset decay timer
                    prev_yawn_total = new_yawn_total
                    print(f"[EVENT] New yawn detected! Accumulator: {yawn_accumulator}")
                elif current_time - last_yawn_time > 30.0:
                    if yawn_accumulator > 0:
                        yawn_accumulator -= 1
                        print(f"[RECOVERY] Yawn accumulator decayed to {yawn_accumulator}")
                    last_yawn_time = current_time # Restart 30s timer

                # 3. Build Vision Dictionary
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
                
                # Update shared state for WebSocket-based inference fusion
                with telemetry_lock:
                    shared_state["latest_vision_features"] = vision_dict.copy()

                # 4. Feature Vector Fusion
                vector = build_feature_vector(vision_dict, telemetry_snapshot, session_start_time)
                
                # CHANGE 4: STRICT SCHEMA VALIDATION
                if len(vector) != len(FEATURE_ORDER):
                    print(f"[WARNING] Feature Mismatch! Expected {len(FEATURE_ORDER)}, got {len(vector)}")
                
                # 5. Model Inference (Step 1)
                raw_prob = model.predict(vector)
                
                # REFINED PROGRESSIVE SCORING
                # skip_mar_boost if a specific yawn was already counted this window
                refined_score = apply_progressive_scoring(
                    raw_prob, 
                    vision_dict, 
                    telemetry_snapshot, 
                    yawn_accumulator,
                    skip_mar_boost=yawn_event_this_window
                )
                
                # STEP 7: CLAMP SCORE
                refined_score = max(0.0, min(1.0, refined_score))
                
                # 6. EMA Smoothing (Step 8)
                current_score = smoother.update(refined_score)

                # 7. Fatigue State Mapping (Step 9 PRD Thresholds)
                new_state = "NORMAL"
                if current_score >= 0.75: new_state = "CRITICAL"
                elif current_score >= 0.55: new_state = "HIGH"
                elif current_score >= 0.30: new_state = "MILD"
                
                # Apply 0.05 Hysteresis
                if current_state == "CRITICAL" and current_score >= 0.70: current_state = "CRITICAL"
                elif current_state == "HIGH" and current_score >= 0.50:
                    current_state = "CRITICAL" if new_state == "CRITICAL" else "HIGH"
                elif current_state == "MILD" and current_score >= 0.25:
                    current_state = new_state if new_state in ["CRITICAL", "HIGH"] else "MILD"
                else: current_state = new_state

                # Update HUD Colors
                if current_state == "NORMAL": current_color = (0, 255, 0)
                elif current_state == "MILD": current_color = (0, 255, 255)
                elif current_state == "HIGH": current_color = (0, 165, 255)
                elif current_state == "CRITICAL": current_color = (0, 0, 255)
                
                # 8. Feedback to Simulation
                shared_state["latest_fatigue_score"] = float(current_score)
                shared_state["fatigue_state"] = current_state
                shared_state["alert"] = bool(current_score > 0.55)

                # 9. Console Logging
                print("\n" + "="*45)
                print(f"[PROGRESSIVE] State: {current_state} | Score: {current_score:.4f}")
                print(f"[PROGRESSIVE] Vision: EAR={vision_dict['EAR_mean']:.2f} | MAR={vision_dict['MAR_max']:.2f} | ECD={vision_dict.get('ECD_max', 0.0):.2f}")
                print(f"[PROGRESSIVE] Tel: Drift={telemetry_snapshot.get('lane_drift_var', 0.0):.2f} | Steering={telemetry_snapshot.get('steering_instability', 0.0):.2f}")
                print(f"[PROGRESSIVE] ML Prob: {raw_prob:.4f} | Sim: {sim_status} | Yawns: {yawn_accumulator} (Total: {new_yawn_total})")
                print("="*45)
                
                last_inference_time = current_time

            # --- HUD OVERLAY (30 FPS) ---
            overlay_text = f"Fatigue: {current_score:.2f} | {current_state} | Y:{yawn_accumulator}"
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
