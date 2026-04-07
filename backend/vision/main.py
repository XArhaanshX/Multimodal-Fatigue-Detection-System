import cv2
import time
import numpy as np
from camera import start_camera
from facemesh import FaceMeshDetector
from features import eye_aspect_ratio, mouth_aspect_ratio, EAR_THRESHOLD
from headpose import estimate_head_pose
from buffer import RollingFeatureBuffer
from feature_extractor import VisionFeatureExtractor, build_feature_vector


LEFT_EYE  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

MOUTH_MAR_INDICES = [13, 14, 82, 87, 312, 317, 61, 291]

detector = FaceMeshDetector()
feature_buffer = RollingFeatureBuffer(window_size_seconds=30.0)
extractor = VisionFeatureExtractor(ear_threshold=EAR_THRESHOLD)

blink_total   = 0
closed_frames_counter = 0
prev_pitch = 0.0         
last_aggregation_time = time.time()

for frame in start_camera():
    landmarks = detector.get_landmarks(frame)
    current_time = time.time()
    
   
    blink_event_this_frame = False

    if landmarks is not None:
        h, w, _ = frame.shape

       
        coords = []
        for lm in landmarks:
            x, y = int(lm.x * w), int(lm.y * h)
            coords.append((x, y))

        # --- Features (EAR, MAR, Pitch) ---
        left_eye_coords  = [coords[i] for i in LEFT_EYE]
        right_eye_coords = [coords[i] for i in RIGHT_EYE]
        ear = (eye_aspect_ratio(left_eye_coords) + eye_aspect_ratio(right_eye_coords)) / 2.0

        m_pts = [coords[i] for i in MOUTH_MAR_INDICES]
        mar = mouth_aspect_ratio(m_pts[0], m_pts[1], m_pts[2], m_pts[3], 
                                 m_pts[4], m_pts[5], m_pts[6], m_pts[7])

        pitch, yaw, roll = estimate_head_pose(landmarks, frame)

        pitch_delta = abs(pitch - prev_pitch)
        prev_pitch = pitch
        
    
        if pitch_delta > 10.0:
            closed_frames_counter = 0
        else:
            if ear < EAR_THRESHOLD:
                closed_frames_counter += 1
            else:
                if closed_frames_counter >= 3:
                    blink_total += 1
                    blink_event_this_frame = True 
                closed_frames_counter = 0

        is_currently_closed = (ear < EAR_THRESHOLD)

        feature_buffer.add_frame(ear, mar, pitch, blink_event_this_frame)


        def draw_box(pts, color, frame, padding=5):
            x_coords = [p[0] for p in pts]
            y_coords = [p[1] for p in pts]
            cv2.rectangle(frame, 
                          (min(x_coords) - padding, min(y_coords) - padding),
                          (max(x_coords) + padding, max(y_coords) + padding), 
                          color, 2)

        # 1. Eye Indicators (Red on current closing and not only on blink)
        eye_color = (0, 0, 255) if is_currently_closed else (0, 255, 0)
        draw_box(left_eye_coords, eye_color, frame)
        draw_box(right_eye_coords, eye_color, frame)

        # 2. Mouth Indicator
        mouth_color = (0, 0, 255) if mar > 0.6 else (0, 255, 0)
        draw_box(m_pts, mouth_color, frame)

        # 3. Status Display
        cv2.putText(frame, f"EAR: {ear:.2f}", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, eye_color, 2)
        cv2.putText(frame, f"MAR: {mar:.2f}", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, mouth_color, 2)
        cv2.putText(frame, f"Pitch: {pitch:.1f}", (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"Blinks: {blink_total}", (30, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if is_currently_closed or blink_event_this_frame:
            cv2.putText(frame, "BLINK!", (w-150, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)

    if current_time - last_aggregation_time >= 1.0:
        window = feature_buffer.get_window()
        features = extractor.compute_features(window)
        
        if features:
            f_vector = build_feature_vector(features)
            print("\nVision Feature Vector (1s update):")
            print([round(x, 4) for x in f_vector])
            
            print("\nReadable metrics:")
            print(f"EAR_mean:    {features['EAR_mean']:.3f}")
            print(f"Blink_rate:  {features['blink_frequency']:.1f} blinks/min")
            print(f"MAR_max:     {features['MAR_max']:.3f}")
            print(f"Pitch_mean:  {features['pitch_mean']:.1f}")
            print(f"Pitch_std:   {features['pitch_std']:.2f}")
            print(f"PERCLOS:     {features['perclos']:.1%}")
            print("-" * 40)

        last_aggregation_time = current_time

    cv2.imshow("Driver Monitor", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()