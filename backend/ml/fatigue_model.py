# backend/ml/fatigue_model.py

import os
import joblib
import numpy as np
from .features import build_feature_vector

class FatigueModel:
    """
    Real-time inference class for the Multimodal Driver Fatigue Detection System.
    Loads a calibrated LightGBM model and predicts P(fatigue).
    """
    def __init__(self, model_path=None):
        if model_path is None:
            # Resolve default path relative to this file
            base_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(base_dir, "model", "fatigue_model.pkl")
            
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}. Run train_model.py first.")
            
        print(f"[INFO] Loading Fatigue Model: {model_path}")
        self.model = joblib.load(model_path)

    def predict(self, feature_vector):
        """
        Predicts the probability of fatigue.
        Args:
            feature_vector (list): 19-dimensional feature vector.
        Returns:
            float: P(fatigue) in range [0, 1].
        """
        X = np.array(feature_vector).reshape(1, -1)
        
        # predict_proba returns [P(class 0), P(class 1)]
        # We return P(class 1) which is P(fatigue)
        probs = self.model.predict_proba(X)
        fatigue_prob = float(probs[0][1])
        
        return fatigue_prob

if __name__ == "__main__":
    # --- TASK 5: INFERENCE TEST SCRIPT ---
    print("[TEST] Initializing FatigueModel...")
    try:
        fm = FatigueModel()
        
        # 1. Create a sample vision feature dictionary (Physiologically realistic for Mild Fatigue)
        sample_vision = {
            "EAR_mean": 0.25,
            "EAR_std": 0.04,
            "EAR_trend": -0.002,
            "blink_frequency": 25.0,
            "ECD_max": 0.45,
            "MAR_max": 0.42,
            "yawn_frequency": 3.0,
            "pitch_mean": -2.0,
            "pitch_std": 4.5,
            "gaze_ratio": 0.35
        }
        
        # 2. Build the 19-dimensional feature vector
        vector = build_feature_vector(sample_vision)
        print(f"[TEST] Feature Vector (19-dim): {vector}")
        
        # 3. Run prediction
        p_fatigue = fm.predict(vector)
        
        # 4. Print result
        print(f"[TEST] Predicted P(fatigue): {p_fatigue:.4f}")
        
    except Exception as e:
        print(f"[ERROR] Inference test failed: {e}")
