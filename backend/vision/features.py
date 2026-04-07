import numpy as np

EAR_THRESHOLD = 0.20

#  Euclidean Distance Helper
def euclidean(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

# Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    """
    Standard EAR formula: (vertical1 + vertical2) / (2 * horizontal)
    """
    A = euclidean(eye[1], eye[5])
    B = euclidean(eye[2], eye[4])
    C = euclidean(eye[0], eye[3])

    return (A + B) / (2.0 * C)

# Mouth Aspect Ratio (MAR)
def mouth_aspect_ratio(p13, p14, p82, p87, p312, p317, p61, p291):
    """
    Refined MAR formula using 3 vertical pairs and 1 horizontal width.
    MAR = (d(13,14) + d(82,87) + d(312,317)) / (2 * d(61,291))
    """
    v1 = euclidean(p13, p14)
    v2 = euclidean(p82, p87)
    v3 = euclidean(p312, p317)
    width = euclidean(p61, p291)

    if width == 0:
        return 0.0

    return (v1 + v2 + v3) / (2.0 * width)