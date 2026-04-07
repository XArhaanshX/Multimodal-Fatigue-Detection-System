import cv2
import numpy as np


MODEL_POINTS = np.array([
    (0.0,    0.0,    0.0),       # Tip of the nose
    (0.0, -330.0,  -65.0),       # Chin
    (-225.0, 170.0, -135.0),     # Left eye
    (225.0, 170.0, -135.0),      # Right eye
    (-150.0, -150.0, -125.0),    # Left mouth corner
    (150.0, -150.0, -125.0),     # Right mouth corner
], dtype=np.float64)

# nose: 1, chin: 152, L-eye: 33, R-eye: 263, L-mouth: 61, R-mouth: 291
POSE_LANDMARKS = [1, 152, 33, 263, 61, 291]


_prev_rotation_vector = np.zeros((3, 1), dtype=np.float64)

DEBUG_POSE = False

def estimate_head_pose(landmarks, frame):
    """
    Estimates raw head pitch using solvePnP and Euler decomposition.
    Accepts raw FaceMesh landmarks and converts them to pixel coordinates.
    """
    global _prev_rotation_vector
    
    if landmarks is None:
        return 0.0, 0.0, 0.0

    h, w, _ = frame.shape

    try:
        points_2d = []
        for idx in POSE_LANDMARKS:
            lm = landmarks[idx]
            points_2d.append((lm.x * w, lm.y * h))
        
        image_points = np.array(points_2d, dtype=np.float64)
        
        if len(image_points) != 6:
            return 0.0, 0.0, 0.0
            
    except Exception as e:
        return 0.0, 0.0, 0.0

    focal_length = w  
    center = (w / 2.0, h / 2.0)
    camera_matrix = np.array([
        [focal_length, 0,            center[0]],
        [0,            focal_length, center[1]],
        [0,            0,            1.0      ],
    ], dtype=np.float64)

    dist_coeffs = np.zeros((4, 1))

    success, rotation_vector, translation_vector = cv2.solvePnP(
        MODEL_POINTS,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )

    if not success:
        # Reuse the previous rotation vector
        rotation_vector = _prev_rotation_vector
    else:
        # Update the previous rotation vector
        _prev_rotation_vector = rotation_vector

    # Convert rotation vector to rotation matrix
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    
    # Extract Euler angles using RQ decomposition
    res = cv2.RQDecomp3x3(rotation_matrix)
    angles = res[0] # angles in degrees (X, Y, Z)
    
    pitch = float(angles[0])
    
    if DEBUG_POSE:
        print(f"ROT_VEC: {rotation_vector.flatten()}")
        print(f"RAW_PITCH: {pitch:.2f}")

    return pitch, float(angles[1]), float(angles[2])

    return pitch, float(angles[1]), float(angles[2])