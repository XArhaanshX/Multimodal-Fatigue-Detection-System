import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
import pickle
import os

# Ensure model directory exists
os.makedirs('backend/ml/model', exist_ok=True)

def train_model():
    data_path = 'data/training_data.csv'
    if not os.path.exists(data_path):
        print(f"[ERROR] Training data not found at {data_path}")
        return

    print(f"[INFO] Loading training data from {data_path}...")
    df = pd.read_csv(data_path)
    
    X = df.drop(columns=['target'])
    y = df['target']
    
    # Feature names check (should match FEATURE_ORDER)
    print(f"[INFO] Training on {len(X.columns)} features.")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("[INFO] Training LightGBM classifier...")
    # Hyperparameters from PRD/Prompt
    clf = lgb.LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        importance_type='gain'
    )
    
    clf.fit(X_train, y_train)
    
    # Predictions
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)
    
    print("\n" + "="*30)
    print("      MODEL PERFORMANCE")
    print("="*30)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("="*30)
    
    # Export Model
    model_path = 'backend/ml/model/fatigue_model.pkl'
    print(f"[INFO] Exporting model to {model_path}...")
    with open(model_path, 'wb') as f:
        pickle.dump(clf, f)
    print("[SUCCESS] Model training and export complete.")

if __name__ == "__main__":
    train_model()
