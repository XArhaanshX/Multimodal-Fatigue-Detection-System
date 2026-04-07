import cv2
import time
import numpy as np
try:
    from .camera import start_camera
    from .facemesh import FaceMeshDetector
    from .features import eye_aspect_ratio, mouth_aspect_ratio, EYE_CLOSED_THRESHOLD, EYE_OPEN_THRESHOLD
    from .headpose import estimate_head_pose
    from .buffer import RollingFeatureBuffer
    from .feature_extractor import VisionFeatureExtractor, build_feature_vector
except (ImportError, ValueError):
    from vision.camera import start_camera
    from vision.facemesh import FaceMeshDetector
    from vision.features import eye_aspect_ratio, mouth_aspect_ratio, EYE_CLOSED_THRESHOLD, EYE_OPEN_THRESHOLD
    from vision.headpose import estimate_head_pose
    from vision.buffer import RollingFeatureBuffer
    from vision.feature_extractor import VisionFeatureExtractor, build_feature_vector


LEFT_EYE  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

MOUTH_MAR_INDICES = [13, 14, 82, 87, 312, 317, 61, 291]

def get_vision_pipeline():
    """
    Robust generator that yields (features, frame) indefinitely.
    Implements Step 2: Hysteresis-based blink detection.
    """
    detector = FaceMeshDetector()
    feature_buffer = RollingFeatureBuffer(window_size_seconds=30.0)
    # Using EYE_CLOSED_THRESHOLD as the base threshold for extractor
    extractor = VisionFeatureExtractor(ear_threshold=EYE_CLOSED_THRESHOLD)

    blink_total   = 0
    eye_was_closed = False # For Step 2 hysteresis
    prev_pitch = 0.0         
    last_aggregation_time = time.time()
    frame_count = 0

    # Safety wrapper
    while True:
        try:
            for raw_frame in start_camera():
                current_time = time.time()
                
                # Handling missing camera frame
                if raw_frame is None:
                    frame = np.zeros((240, 320, 3), dtype=np.uint8)
                    cv2.putText(frame, "CAMERA NOT READY", (50, 120), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    yield None, frame
                    continue
                
                frame_count += 1
                
                # Optimized resolution
                frame = cv2.resize(raw_frame, (320, 240))
                h, w, _ = frame.shape
                
                blink_event_this_frame = False

                # Process FaceMesh every 3rd frame
                if frame_count % 3 == 0:
                    landmarks = detector.get_landmarks(frame)

                    if landmarks is not None:
                        coords = []
                        for lm in landmarks:
                            x, y = int(lm.x * w), int(lm.y * h)
                            coords.append((x, y))

                        # Landmarks -> Features (EAR, MAR, Pitch)
                        left_eye_coords  = [coords[i] for i in LEFT_EYE]
                        right_eye_coords = [coords[i] for i in RIGHT_EYE]
                        ear = (eye_aspect_ratio(left_eye_coords) + eye_aspect_ratio(right_eye_coords)) / 2.0

                        m_pts = [coords[i] for i in MOUTH_MAR_INDICES]
                        mar = mouth_aspect_ratio(m_pts[0], m_pts[1], m_pts[2], m_pts[3], 
                                                 m_pts[4], m_pts[5], m_pts[6], m_pts[7])

                        pitch, yaw, roll = estimate_head_pose(landmarks, frame)

                        pitch_delta = abs(pitch - prev_pitch)
                        prev_pitch = pitch
                        
                        # Step 2: Hysteresis Blink Detection
                        if ear < EYE_CLOSED_THRESHOLD:
                            eye_was_closed = True
                        elif ear > EYE_OPEN_THRESHOLD and eye_was_closed:
                            blink_total += 1
                            blink_event_this_frame = True
                            eye_was_closed = False

                        is_currently_closed = (ear < EYE_CLOSED_THRESHOLD)
                        feature_buffer.add_frame(ear, mar, pitch, blink_event_this_frame)

                        # HUD Drawing
                        def draw_box(pts, color, frame, padding=3):
                            x_coords = [p[0] for p in pts]
                            y_coords = [p[1] for p in pts]
                            cv2.rectangle(frame, 
                                          (min(x_coords) - padding, min(y_coords) - padding),
                                          (max(x_coords) + padding, max(y_coords) + padding), 
                                          color, 1)

                        eye_color = (0, 0, 255) if is_currently_closed else (0, 255, 0)
                        draw_box(left_eye_coords, eye_color, frame)
                        draw_box(right_eye_coords, eye_color, frame)

                        mouth_color = (0, 0, 255) if mar > 0.6 else (0, 255, 0)
                        draw_box(m_pts, mouth_color, frame)

                        cv2.putText(frame, f"EAR: {ear:.2f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, eye_color, 1)
                        cv2.putText(frame, f"MAR: {mar:.2f}", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.4, mouth_color, 1)
                        cv2.putText(frame, f"Blinks: {blink_total}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

                # 1-second feature aggregation check
                yielded_features = None
                if current_time - last_aggregation_time >= 1.0:
                    window = feature_buffer.get_window()
                    features = extractor.compute_features(window)
                    if features:
                        features["blink_total"] = blink_total
                        yielded_features = features
                    last_aggregation_time = current_time

                # yield
                yield yielded_features, frame

        except Exception as e:
            print(f"[ERROR] Pipeline error: {e}. Retrying...")
            time.sleep(1)

if __name__ == "__main__":
    print("[INFO] Starting Vision Module (Step 2 Hysteresis)...")
    for features, frame in get_vision_pipeline():
        if features:
            print(f"EAR_mean: {features['EAR_mean']:.3f} | Blinks: {features['blink_total']}")

        cv2.imshow("IRoad Driver Monitor", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    
    cv2.destroyAllWindows()