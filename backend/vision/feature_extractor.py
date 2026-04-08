import numpy as np
try:
    from .features import EYE_CLOSED_THRESHOLD
except (ImportError, ValueError):
    from vision.features import EYE_CLOSED_THRESHOLD

EAR_THRESHOLD = EYE_CLOSED_THRESHOLD

class VisionFeatureExtractor:
    """
    Stabilized feature extraction summarizing 30 seconds of driver behavior.
    Includes Task 8 range validation and Task 9 Python float normalization.
    """

    def __init__(self, ear_threshold=EAR_THRESHOLD):
        self.ear_threshold = ear_threshold

    def compute_features(self, frame_window):
        """
        Calculates vision features from a list of frames.
        :param frame_window: List of feature dictionaries (timestamp, EAR, MAR, pitch, blink)
        :return: A dictionary of aggregated and validated features.
        """
        if not frame_window or len(frame_window) < 1:
            return None

        # Extract values for processing
        timestamps = np.array([f["timestamp"] for f in frame_window])
        ear_vals = np.array([f["EAR"] for f in frame_window])
        mar_vals = np.array([f["MAR"] for f in frame_window])
        pitch_vals = np.array([f["pitch"] for f in frame_window])
        blinks = np.array([f["blink"] for f in frame_window])

        # 1. EAR Features (Mean, Std, Trend)
        ear_mean = float(np.mean(ear_vals))
        ear_std = float(np.std(ear_vals))
        
        # EAR_trend: Linear slope using polyfit (normalized timestamps)
        if len(timestamps) > 1:
            t_normalized = timestamps - timestamps[0]
            ear_trend = float(np.polyfit(t_normalized, ear_vals, 1)[0])
        else:
            ear_trend = 0.0

        # 2. Blink Features (Blinks per minute)
        blink_count = int(np.sum(blinks))
        duration_sec = float(timestamps[-1] - timestamps[0])
        blink_frequency = float((blink_count / duration_sec) * 60.0) if duration_sec > 0 else 0.0

        # 3. Eye Closure Duration (max contiguous closed duration in seconds)
        ecd_max = 0.0
        closed_start = None
        previous_timestamp = None
        for timestamp, ear in zip(timestamps, ear_vals):
            if ear < self.ear_threshold:
                if closed_start is None:
                    closed_start = float(timestamp)
            elif closed_start is not None:
                end_timestamp = float(previous_timestamp if previous_timestamp is not None else timestamp)
                ecd_max = max(ecd_max, end_timestamp - closed_start)
                closed_start = None
            previous_timestamp = float(timestamp)
        if closed_start is not None and previous_timestamp is not None:
            ecd_max = max(ecd_max, previous_timestamp - closed_start)

        # 4. Mouth Features (Max MAR)
        mar_max = float(np.max(mar_vals))

        # 5. Head Pose Features (Mean & Std Pitch)
        pitch_mean = float(np.mean(pitch_vals))
        pitch_std = float(np.std(pitch_vals))

        # 6. PERCLOS initialization adn defination
        perclos = float(np.mean(ear_vals < self.ear_threshold))

        self._validate_ranges({
            "EAR_mean": ear_mean,
            "blink_frequency": blink_frequency,
            "ECD_max": ecd_max,
            "MAR_max": mar_max,
            "pitch_mean": pitch_mean,
            "pitch_std": pitch_std
        })

        return {
            "EAR_mean": ear_mean,
            "EAR_std": ear_std,
            "EAR_trend": ear_trend,
            "blink_frequency": blink_frequency,
            "ECD_max": ecd_max,
            "MAR_max": mar_max,
            "pitch_mean": pitch_mean,
            "pitch_std": pitch_std,
            "perclos": perclos
        }

    def _validate_ranges(self, metrics):
        """
        Logs warnings if features fall outside physiological ranges defined in Task 8.
        """
        # EAR_mean → 0.2 – 0.35 when eyes open
        if not (0.15 <= metrics["EAR_mean"] <= 0.40):
            print(f"[WARNING] EAR_mean out of range: {metrics['EAR_mean']:.3f} (Expected 0.2 - 0.35)")
        # Blink_rate → 0 – 60 blinks/min
        if not (0 <= metrics["blink_frequency"] <= 60):
            print(f"[WARNING] Blink_rate out of range: {metrics['blink_frequency']:.1f} (Expected 0 - 60)")
        # ECD_max -> 0 - 2.5 sec
        if not (0.0 <= metrics["ECD_max"] <= 2.5):
            print(f"[WARNING] ECD_max abnormal: {metrics['ECD_max']:.3f} (Expected 0.0 - 2.5)")
        # MAR_max → 0.2 – 1.0
        if not (0.2 <= metrics["MAR_max"] <= 1.0):
            print(f"[WARNING] MAR_max abnormal: {metrics['MAR_max']:.3f} (Expected 0.2 - 1.0)")
        # Pitch_mean → -45 to +45
        if not (-45.0 <= metrics["pitch_mean"] <= 45.0):
            print(f"[WARNING] Pitch_mean extreme: {metrics['pitch_mean']:.1f} (Expected ±45)")
        # Pitch_std → < 20
        if metrics["pitch_std"] > 20.0:
            print(f"[WARNING] High Pitch Noise: {metrics['pitch_std']:.1f} (Expected < 20)")

def build_feature_vector(features):
    """
    Task 9: Constructs the final Python list feature vector with guaranteed float types.
    F = [EAR_mean, EAR_std, EAR_trend, blink_frequency, MAR_max, pitch_mean, pitch_std]
    """
    if features is None:
        return [0.0] * 7

    return [
        float(features["EAR_mean"]),
        float(features["EAR_std"]),
        float(features["EAR_trend"]),
        float(features["blink_frequency"]),
        float(features["MAR_max"]),
        float(features["pitch_mean"]),
        float(features["pitch_std"])
    ]
