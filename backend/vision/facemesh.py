import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class FaceMeshDetector:
    def __init__(self):
        # Path to the .task model file
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, 'utils', 'models', 'face_landmarker.task')

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"MediaPipe model not found at: {model_path}")

        # Configure Face Landmarker
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)

    def get_landmarks(self, frame):
        """
        Processes a frame and returns a list of normalized landmarks for the first face.
        Each landmark has x, y, z properties.
        """
        # Convert OpenCV BGR to RGB (MediaPipe expects RGB)
        rgb_frame = frame[:, :, ::-1]
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Perform detection
        result = self.detector.detect(mp_image)
        
        if not result.face_landmarks:
            return None
            
        # Return the list of NormalizedLandmark for the first detected face
        return result.face_landmarks[0]