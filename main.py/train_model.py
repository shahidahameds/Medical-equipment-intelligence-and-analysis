
import pandas as pd
import numpy as np
import joblib
import json

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)

# ==========================================
# 1. PROJECT PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_DIR.mkdir(exist_ok=True)

DEVICE_FILE = BASE_DIR / "devices-1681209661.csv"
EVENT_FILE = BASE_DIR / "events-1681209680.csv"

# ==========================================
# 2. LOAD DATASETS
# ==========================================

print("\nLoading datasets...")

devices = pd.read_csv(DEVICE_FILE, low_memory=False)
events = pd.read_csv(EVENT_FILE, low_memory=False)

print("Devices shape:", devices.shape)
print("Events shape:", events.shape)

# ==========================================
# 3. CREATE FAILURE TARGET
# ==========================================

print("\nCreating failure target...")

events["reason"] = (
    events["reason"].fillna("").astype(str)
)

failure_mask = events["reason"].str.contains(
    "device failure",
    case=False,
    regex=False,
    na=False
)

failed_device_ids = set(
    events.loc[failure_mask, "device_id"].dropna()
)

devices["failure"] = (
    devices["id"].isin(failed_device_ids).astype(int)
)

positive_count = int(devices["failure"].sum())
negative_count = len(devices) - positive_count

print("\nTarget distribution:")
print(devices["failure"].value_counts())

print("Positive devices:", positive_count)
print("Negative devices:", negative_count)

print(
    "Failure percentage:",
    round(positive_count / len(devices) * 100, 3),
    "%"
)

# ==========================================
# 4. SELECT FEATURES
# ==========================================

FEATURES = [
    "classification",
    "implanted",
    "quantity_in_commerce",
    "risk_class",
    "manufacturer_id"
]

X = devices[FEATURES].copy()
y = devices["failure"].copy()

X["quantity_in_commerce"] = pd.to_numeric(
    X["quantity_in_commerce"],
    errors="coerce"
)

categorical_features = [
    "classification",
    "implanted",
    "risk_class",
    "manufacturer_id"
]

numeric_features = ["quantity_in_commerce"]

# ==========================================
# 5. PREPROCESSING
# ==========================================

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    (
        "numeric",
        numeric_transformer,
        numeric_features
    ),
    (
        "categorical",
        categorical_transformer,
        categorical_features
    )
])

# ==========================================
# 6. RANDOM FOREST MODEL
# ==========================================

random_forest = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced",
    min_samples_leaf=2,
    n_jobs=-1
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", random_forest)
])

# ==========================================
# 7. TRAIN / VALIDATION / TEST SPLIT
# ==========================================

print("\nSplitting dataset...")

# 20% reserved for final testing
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 25% of remaining 80% = 20% of total
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val,
    y_train_val,
    test_size=0.25,
    random_state=42,
    stratify=y_train_val
)

print("Training samples:", len(X_train))
print("Validation samples:", len(X_val))
print("Testing samples:", len(X_test))

# ==========================================
# 8. TRAIN MODEL
# ==========================================

print("\nTraining Random Forest model...")

pipeline.fit(X_train, y_train)

print("Training completed!")

# ==========================================
# 9. THRESHOLD SELECTION
# ==========================================

print("\n========== VALIDATION THRESHOLD ANALYSIS ==========")

val_probabilities = pipeline.predict_proba(X_val)[:, 1]

candidate_thresholds = [
    0.10, 0.20, 0.30, 0.40, 0.50,
    0.60, 0.70, 0.80, 0.90
]

best_threshold = 0.50
best_f1 = -1

for threshold in candidate_thresholds:

    val_predictions = (
        val_probabilities >= threshold
    ).astype(int)

    precision = precision_score(
        y_val, val_predictions, zero_division=0
    )

    recall = recall_score(
        y_val, val_predictions, zero_division=0
    )

    f1 = f1_score(
        y_val, val_predictions, zero_division=0
    )

    print(
        f"\nThreshold: {threshold:.2f}"
        f"\nPrecision: {precision:.4f}"
        f"\nRecall: {recall:.4f}"
        f"\nF1 Score: {f1:.4f}"
    )

    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

print("\nSelected threshold:", best_threshold)
print("Best validation F1:", round(best_f1, 4))

# ==========================================
# 10. FINAL TEST EVALUATION
# ==========================================

print("\n========== FINAL TEST EVALUATION ==========")

y_probability = pipeline.predict_proba(X_test)[:, 1]

y_pred = (
    y_probability >= best_threshold
).astype(int)

metrics = {
    "Accuracy": accuracy_score(y_test, y_pred),
    "Precision": precision_score(
        y_test, y_pred, zero_division=0
    ),
    "Recall": recall_score(
        y_test, y_pred, zero_division=0
    ),
    "F1 Score": f1_score(
        y_test, y_pred, zero_division=0
    ),
    "ROC AUC": roc_auc_score(y_test, y_probability),
    "PR AUC": average_precision_score(y_test, y_probability)
}

for metric, value in metrics.items():
    print(f"{metric}: {value:.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(
    classification_report(
        y_test, y_pred, zero_division=0
    )
)

# ==========================================
# 11. FEATURE IMPORTANCE
# ==========================================

print("\nCalculating feature importance...")

feature_names = (
    pipeline.named_steps["preprocessor"]
    .get_feature_names_out()
)

importance_values = (
    pipeline.named_steps["model"]
    .feature_importances_
)

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance_values
}).sort_values(
    by="Importance",
    ascending=False
)

print("\nTop 15 Important Features:")
print(importance_df.head(15).to_string(index=False))

importance_df.to_csv(
    MODEL_DIR / "feature_importance.csv",
    index=False
)

# ==========================================
# 12. SAVE TRAINED MODEL
# ==========================================

model_path = MODEL_DIR / "random_forest_pipeline.joblib"

joblib.dump(pipeline, model_path)

print("\nModel saved:", model_path)

# ==========================================
# 13. SAVE METADATA AND METRICS
# ==========================================

metadata = {
    "model_name": "Random Forest",
    "target": "Explicit device failure event",
    "target_definition": (
        "A device is labelled 1 if at least one linked "
        "event reason contains 'device failure', "
        "case-insensitively; otherwise 0."
    ),
    "features": FEATURES,
    "decision_threshold": float(best_threshold),
    "training_samples": len(X_train),
    "validation_samples": len(X_val),
    "testing_samples": len(X_test),
    "positive_devices": positive_count,
    "negative_devices": negative_count,
    "metrics": {
        key: float(value)
        for key, value in metrics.items()
    }
}

with open(
    MODEL_DIR / "model_metadata.json",
    "w"
) as file:
    json.dump(metadata, file, indent=4)

pd.DataFrame([metrics]).to_csv(
    MODEL_DIR / "model_metrics.csv",
    index=False
)

print("\n========================================")
print("MODEL TRAINING COMPLETED SUCCESSFULLY!")
print("========================================")

print("\nSaved files:")
print("1. random_forest_pipeline.joblib")
print("2. model_metadata.json")
print("3. model_metrics.csv")
print("4. feature_importance.csv")
