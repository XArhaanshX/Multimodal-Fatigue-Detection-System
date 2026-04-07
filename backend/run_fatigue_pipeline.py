# backend/run_fatigue_pipeline.py

import cv2
import sys
import os
import time

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

def run_pipeline():
    """
    Optimized live fatigue detection pipeline for 30 FPS display and 1 Hz inference.
    """
    print("[INFO] Initializing Optimized Fatigue Detection Model...")
    try:
        model = FatigueModel()
        smoother = EMASmoother(alpha=0.3)
    except Exception as e:
        print(f"[ERROR] System initialization failed: {e}")
        return

    print("[INFO] Starting Vision Pipeline (30 FPS Display | 1 Hz Inference). Press 'q' to exit.")
    
    pipeline = get_vision_pipeline()
    
    # Optimization State
    last_inference_time = 0.0
    current_score = 0.0
    current_state = "NORMAL"
    current_color = (0, 255, 0)

    try:
        for features, frame in pipeline:
            # frame is always provided (30 FPS)
            if frame is None:
                continue

            # Optimization 5: Limit ML Inference to exactly 1 Hz
            current_time = time.time()
            if features is not None and (current_time - last_inference_time >= 1.0):
                # STEP 1: LOG VISION FEATURES
                print(f"[DEBUG] EAR_mean: {features['EAR_mean']:.3f} | BF: {features['blink_frequency']:.1f} | MAR: {features['MAR_max']:.3f} | P_STD: {features['pitch_std']:.1f}")

                # 1. Build Feature Vector (19-dim)
                vision_dict = {
                    "EAR_mean": features["EAR_mean"],
                    "EAR_std": features["EAR_std"],
                    "EAR_trend": features["EAR_trend"],
                    "blink_frequency": features["blink_frequency"],
                    "MAR_max": features["MAR_max"],
                    "pitch_mean": features["pitch_mean"],
                    "pitch_std": features["pitch_std"],
                    "ECD_max": features["perclos"], # Proxy for ECD
                    "yawn_frequency": 0.0,
                    "gaze_ratio": 0.0
                }
                
                # STEP 3 & 4: CLAMP FEATURES (to preserve ML stability)
                vision_dict["pitch_std"] = min(vision_dict["pitch_std"], 20.0)
                vision_dict["MAR_max"] = min(vision_dict["MAR_max"], 1.5)

                vector = build_feature_vector(vision_dict)
                
                # STEP 2: LOG FEATURE VECTOR
                print(f"[DEBUG] Feature Vector: {[round(v, 4) for v in vector]}")
                
                # 2. ML Prediction (Binary LightGBM)
                raw_score = model.predict(vector)
                
                # STEP 3: LOG RAW MODEL OUTPUT
                print(f"[DEBUG] Raw Model Score: {raw_score:.4f}")

                # STEP 4: IMPROVE FALLBACK FATIGUE HEURISTIC (Step 3 & 4 from prompt)
                # response directly to eye closure (EAR) and blink rate
                fatigue_from_ear = max(0, (0.30 - features["EAR_mean"]) * 5)
                fatigue_from_blink = min(1, features["blink_frequency"] / 40)
                
                # Combine using 70/30 weighting
                fallback_score = (0.7 * fatigue_from_ear) + (0.3 * fatigue_from_blink)
                fallback_score = min(1.0, max(0.0, fallback_score)) # Clamp

                if raw_score < 0.01:
                    raw_score = float(fallback_score)
                    print(f"[DEBUG] Using Fallback Heuristic: {raw_score:.4f}")
                
                # 3. EMA Smoothing
                current_score = smoother.update(raw_score)

                # 4. State Mapping (PRD Thresholds)
                if current_score < 0.30:
                    current_state, current_color = "NORMAL", (0, 255, 0)
                elif current_score < 0.55:
                    current_state, current_color = "MILD", (0, 255, 255)
                elif current_score < 0.75:
                    current_state, current_color = "HIGH", (0, 165, 255)
                else:
                    current_state, current_color = "CRITICAL", (0, 0, 255)
                
                print(f"Fatigue: {current_score:.2f} | State: {current_state}")
                last_inference_time = current_time

            # Optimization 6: Adjust HUD position (20, 80) to avoid covering vision markers
            overlay_text = f"Fatigue: {current_score:.2f} | State: {current_state}"
            cv2.putText(frame, overlay_text, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, current_color, 2)

            # Display frame (High FPS)
            cv2.imshow("IRoad - Multimodal Fatigue Detector", frame)

            # Safe OpenCV Window Exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[INFO] Quitting application...")
                break

    except KeyboardInterrupt:
        print("\n[INFO] Interrupt received.")
    finally:
        cv2.destroyAllWindows()
        print("[INFO] Cleanup complete. Pipeline terminated.")

if __name__ == "__main__":
    run_pipeline()
