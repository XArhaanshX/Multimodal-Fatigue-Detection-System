import time
import numpy as np


DIRECT_TELEMETRY_KEYS = (
    "lane_drift_var",
    "lane_offset_mean",
    "steering_instability",
    "correction_freq",
    "reaction_delay_mean",
    "steering_reversals",
)


RAW_TELEMETRY_KEYS = (
    "lane_offset",
    "steering_angle",
    "steering_correction_hz",
    "reaction_delay_ms",
    "speed_kmh",
)


class TelemetryWindow:
    """
    Maintains a 30-second time-based telemetry buffer.
    """

    def __init__(self, window_size_seconds=30.0):
        self.window_size_seconds = float(window_size_seconds)
        self.samples = []

    def add_sample(self, sample, timestamp=None):
        sample_time = time.time() if timestamp is None else float(timestamp)
        self.samples.append({"timestamp": sample_time, "data": dict(sample)})
        self._cleanup(sample_time)

    def get_samples(self):
        self._cleanup()
        return list(self.samples)

    def _cleanup(self, current_time=None):
        now = time.time() if current_time is None else float(current_time)
        cutoff = now - self.window_size_seconds
        while self.samples and self.samples[0]["timestamp"] < cutoff:
            self.samples.pop(0)


def extract_direct_features(sample):
    """
    Preserves telemetry already aggregated by the simulator.
    """
    features = {}
    for key in DIRECT_TELEMETRY_KEYS:
        if key in sample:
            features[key] = float(sample[key])
    return features


def extract_raw_telemetry(sample):
    """
    Preserves raw simulator telemetry used for rolling aggregation.
    """
    values = {}
    for key in RAW_TELEMETRY_KEYS:
        if key in sample:
            values[key] = float(sample[key])
    return values


def compute_window_features(samples):
    """
    Aggregates raw telemetry when lane offset and steering angle are available.
    """
    if len(samples) < 2:
        return {}

    raw_samples = [entry["data"] for entry in samples]
    if not all("lane_offset" in sample for sample in raw_samples):
        return {}
    if not all("steering_angle" in sample for sample in raw_samples):
        return {}

    lane_offset = np.array([float(sample["lane_offset"]) for sample in raw_samples], dtype=float)
    steering_angle = np.array([float(sample["steering_angle"]) for sample in raw_samples], dtype=float)

    steering_mean = float(np.mean(steering_angle))
    steering_std = float(np.std(steering_angle))
    if abs(steering_mean) < 1e-3:
        steering_instability = steering_std
    else:
        steering_instability = steering_std / abs(steering_mean)

    steering_reversals = 0.0
    if len(steering_angle) >= 3:
        steering_reversals = float(np.sum(np.diff(np.sign(np.diff(steering_angle))) != 0))

    return {
        "lane_offset_mean": float(np.mean(lane_offset)),
        "lane_drift_var": float(np.var(lane_offset)),
        "steering_instability": float(steering_instability),
        "correction_freq": _mean_optional(raw_samples, "steering_correction_hz"),
        "reaction_delay_mean": _mean_optional(raw_samples, "reaction_delay_ms"),
        "steering_reversals": steering_reversals,
    }


def _mean_optional(samples, key):
    values = [float(sample[key]) for sample in samples if key in sample]
    if not values:
        return 0.0
    return float(np.mean(values))
