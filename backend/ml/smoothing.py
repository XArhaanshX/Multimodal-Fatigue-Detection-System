# backend/ml/smoothing.py

class EMASmoother:
    """
    Implements an Exponential Moving Average for stabilizing fatigue scores.
    PRD alpha = 0.3.
    """
    def __init__(self, alpha=0.3):
        self.alpha = alpha
        self.smoothed_value = None

    def update(self, new_value):
        """
        Updates the EMA with a new value and returns the current smoothed result.
        """
        if self.smoothed_value is None:
            self.smoothed_value = new_value
        else:
            self.smoothed_value = (self.alpha * new_value) + ((1 - self.alpha) * self.smoothed_value)
        return float(self.smoothed_value)
