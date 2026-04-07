# backend/ml/train_model.py

import os
import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
from .feature_schema import FEATURE_ORDER

# Hyperparameters from PRD/User Request
LGBM_PARAMS = {
    "n_estimators": 200,
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "binary",
    "random_state": 42,
    "verbosity": -1
}

def train_fatigue_model(data_path="data/training_data.csv", model_dir="backend/ml/model"):
    """
    Retrains the fatigue detection model with scaling, calibration, and validation.
    """
    print(f"[INFO] Loading dataset: {data_path}")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Training data not found at {data_path}")

    df = pd.read_csv(data_path)
    
    # Extract features using strict order
    X = df[FEATURE_ORDER]
    y = df["fatigue_label"]

    print(f"[INFO] Dataset shape: {df.shape}")
    print("[INFO] Splitting dataset into train/test (80/20) with stratification")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("[INFO] Building Pipeline: StandardScaler -> LightGBM")
    # We wrap the Pipeline in CalibratedClassifierCV to ensure calibration is applied
    # to the scaled features during each fold of the cross-validation.
    base_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", lgb.LGBMClassifier(**LGBM_PARAMS))
    ])

    print("[INFO] Applying Platt Scaling Calibration (Sigmoid)...")
    calibrated_model = CalibratedClassifierCV(
        estimator=base_pipeline,
        method="sigmoid",
        cv=5
    )
    
    print("[INFO] Training calibrated model...")
    calibrated_model.fit(X_train, y_train)

    # Evaluation
    print("\n--- MODEL EVALUATION ---")
    y_pred = calibrated_model.predict(X_test)
    y_prob = calibrated_model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)

    print(f"Accuracy: {acc:.4f}")
    print(f"ROC-AUC:  {roc_auc:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Probability Statistics
    print("\n--- PROBABILITY STATISTICS ---")
    print(f"Min probability:  {np.min(y_prob):.4f}")
    print(f"Max probability:  {np.max(y_prob):.4f}")
    print(f"Mean probability: {np.mean(y_prob):.4f}")

    # Feature Importance
    # To get feature importance from CalibratedClassifierCV, we access underlying estimators
    print("\n--- TOP 10 FEATURE IMPORTANCE ---")
    importances = []
    for calibrator in calibrated_model.calibrated_classifiers_:
        # calibrator.base_estimator is the Pipeline
        # pipeline.named_steps["clf"] is the LGBMClassifier
        importances.append(calibrator.estimator.named_steps["clf"].feature_importances_)
    
    avg_importance = np.mean(importances, axis=0)
    feat_imp = pd.Series(avg_importance, index=FEATURE_ORDER).sort_values(ascending=False)
    print(feat_imp.head(10))

    # Goal Verification
    print("\n--- VERIFICATION ---")
    if roc_auc > 0.75 and acc > 0.70:
        print("[SUCCESS] Model meets performance goals (>0.75 ROC-AUC, >0.70 Accuracy).")
    else:
        print("[WARNING] Model performance is below desired threshold.")

    # Export
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        
    model_path = os.path.join(model_dir, "fatigue_model.pkl")
    joblib.dump(calibrated_model, model_path)
    print(f"\n[EXPORT] Saved calibrated pipeline to: {model_path}")

if __name__ == "__main__":
    train_fatigue_model()
