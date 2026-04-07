import cv2
import time

def start_camera():
    cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    while True:
        if not cap.isOpened():
            print("[WARNING] Camera failed to open. Retrying...")
            cap = cv2.VideoCapture(0)
            time.sleep(1.0)
            continue

        ret, frame = cap.read()
        if not ret:
            print("[WARNING] Failed to capture frame. Camera may be busy.")
            yield None
            time.sleep(0.1) # Prevent CPU flooding
            continue

        yield frame

    cap.release()