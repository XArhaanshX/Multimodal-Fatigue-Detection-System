import time

class RollingFeatureBuffer:
    """
    Stores a rolling window of frame-level vision features (EAR, MAR, Pitch, Blinks).
    Automatically discards entries older than the specified window size (30 seconds).
    """

    def __init__(self, window_size_seconds=30.0):
        self.window_size = window_size_seconds
        self.buffer = []

    def add_frame(self, ear, mar, pitch, blink):
        """
        Adds a new frame's data to the buffer.
        :param ear: Eye Aspect Ratio (float)
        :param mar: Mouth Aspect Ratio (float)
        :param pitch: Head pitch in degrees (float)
        :param blink: Whether a blink was detected in this frame (bool)
        """
        current_time = time.time()
        entry = {
            "timestamp": current_time,
            "EAR": ear,
            "MAR": mar,
            "pitch": pitch,
            "blink": blink
        }
        self.buffer.append(entry)
        self._cleanup()

    def get_window(self):
        """
        Returns all frames within the last 30 seconds.
        """
        self._cleanup()
        return self.buffer

    def _cleanup(self):
        """
        Removes frames older than the window size from the buffer.
        """
        cutoff_time = time.time() - self.window_size
        # Keep only frames with timestamp > cutoff_time
        while self.buffer and self.buffer[0]["timestamp"] < cutoff_time:
            self.buffer.pop(0)

    def __len__(self):
        return len(self.buffer)
