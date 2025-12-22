import numpy as np
import pickle
import os
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# =========================================================
# CONFIG
# =========================================================
DATA_DIR = r"C:\Users\GIRIDHAR\OneDrive\Desktop\FRA\processed_fra_pytorch"

train_path = os.path.join(DATA_DIR, "train.npz")
val_path   = os.path.join(DATA_DIR, "val.npz")
test_path  = os.path.join(DATA_DIR, "test.npz")

# =========================================================
# LOAD DATA
# =========================================================
def load_split(path):
    data = np.load(path, allow_pickle=True)
    X = data["X"]
    P = data["P"]
    meta = data["meta"]
    y_fault = data["y_fault"]
    y_cond = data["y_cond"]
    return X, P, meta, y_fault, y_cond

print("\nLoading dataset...")
X_train, P_train, M_train, y_train, _ = load_split(train_path)
X_val,   P_val,   M_val,   y_val,   _ = load_split(val_path)
X_test,  P_test,  M_test,  y_test,  _ = load_split(test_path)

print("TRAIN:", X_train.shape)
print("VAL:",   X_val.shape)
print("TEST:",  X_test.shape)

# =========================================================
# MERGE FEATURES (Magnitude + Phase + Metadata)
# =========================================================
def merge_features(X, P, M):
    return np.concatenate([X, P, M], axis=1)

X_train_full = merge_features(X_train, P_train, M_train)
X_val_full   = merge_features(X_val,   P_val,   M_val)
X_test_full  = merge_features(X_test,  P_test,  M_test)

print("Final feature size:", X_train_full.shape)

# =========================================================
# TRAIN XGBOOST MODEL
# =========================================================
print("\n======================")
print(" TRAINING XGBOOST MODEL")
print("======================\n")

model = XGBClassifier(
    n_estimators=350,          # number of trees
    learning_rate=0.05,        # slower learning → better accuracy
    max_depth=6,               # deeper trees → capture FRA patterns
    subsample=0.9,
    colsample_bytree=0.7,
    eval_metric="mlogloss",
    tree_method="hist",        # fastest CPU method
    random_state=42
)

model.fit(
    X_train_full, y_train,
    eval_set=[(X_val_full, y_val)],
    verbose=True
)

# =========================================================
# EVALUATE
# =========================================================
print("\nEvaluating on validation set...")
val_pred = model.predict(X_val_full)
print("VAL Accuracy:", accuracy_score(y_val, val_pred))
print("\nVAL Report:\n", classification_report(y_val, val_pred))

print("\nEvaluating on TEST set...")
test_pred = model.predict(X_test_full)
print("TEST Accuracy:", accuracy_score(y_test, test_pred))
print("\nTEST Report:\n", classification_report(y_test, test_pred))

print("\nConfusion Matrix:\n", confusion_matrix(y_test, test_pred))

# =========================================================
# SAVE MODEL
# =========================================================
MODEL_PATH = os.path.join(DATA_DIR, "xgboost_fault_classifier.pkl")
with open(MODEL_PATH, "wb") as f:
    pickle.dump(model, f)

print("\nModel saved to:", MODEL_PATH)
print("\n🎉 XGBoost Training Complete!")
