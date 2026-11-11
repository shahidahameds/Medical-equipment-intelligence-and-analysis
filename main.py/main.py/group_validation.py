
import pandas as pd
import joblib
import json

from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
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
    confusion_matrix
)

# ==========================================
# 1. PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_DIR.mkdir(exist_ok=True)

# ==========================================
# 2. LOAD DATA
# ==========================================

print("\nLoading datasets...")

devices = pd.read_csv(
    BASE_DIR / "devices-1681209661.csv",
    low_memory=False
)

events = pd.read_csv(
    BASE_DIR / "events-1681209680.csv",
    low_memory=False
)

# ==========================================
# 3. CREATE TARGET
# ==========================================

events["reason"] = events["reason"].fillna("").astype(str)

failure_mask = events["reason"].str.contains(
    "device failure",
    case=False,
    regex=False,
    na=False
)

failed_ids = set(
    events.loc[failure_mask, "device_id"].dropna()
)

devices["failure"] = (
    devices["id"].isin(failed_ids).astype(int)
)

# ==========================================
# 4. FEATURES AND GROUPS
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

# Treat missing manufacturer IDs as one group
groups = (
    X["manufacturer_id"]
    .fillna("__MISSING_MANUFACTURER__")
    .astype(str)
)

print("\nTotal devices:", len(X))
print("Unique manufacturer groups:", groups.nunique())
print("Failure devices:", int(y.sum()))

# ==========================================
# 5. GROUP-BASED TEST SPLIT
# ==========================================

# Hold out manufacturers for final testing
test_splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_val_idx, test_idx = next(
    test_splitter.split(X, y, groups)
)

X_train_val = X.iloc[train_val_idx]
y_train_val = y.iloc[train_val_idx]
groups_train_val = groups.iloc[train_val_idx]

X_test = X.iloc[test_idx]
y_test = y.iloc[test_idx]
groups_test = groups.iloc[test_idx]

# Split remaining manufacturers into train and validation
val_splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.25,
    random_state=42
)

train_idx, val_idx = next(
    val_splitter.split(
        X_train_val,
        y_train_val,
        groups_train_val
    )
)

X_train = X_train_val.iloc[train_idx]
y_train = y_train_val.iloc[train_idx]
groups_train = groups_train_val.iloc[train_idx]

X_val = X_train_val.iloc[val_idx]
y_val = y_train_val.iloc[val_idx]
groups_val = groups_train_val.iloc[val_idx]

print("\nGROUP SPLIT RESULTS")
print("Training devices:", len(X_train))
print("Validation devices:", len(X_val))
print("Testing devices:", len(X_test))

print("\nUnique manufacturer groups:")
print("Training:", groups_train.nunique())
print("Validation:", groups_val.nunique())
print("Testing:", groups_test.nunique())

print("\nFailure distribution:")
print("Training:", y_train.value_counts().to_dict())
print("Validation:", y_val.value_counts().to_dict())
print("Testing:", y_test.value_counts().to_dict())

# Check no manufacturer overlap
assert set(groups_train).isdisjoint(set(groups_val))
assert set(groups_train).isdisjoint(set(groups_test))
assert set(groups_val).isdisjoint(set(groups_test))

# ==========================================
# 6. PREPROCESSING AND MODEL
# ==========================================

categorical_features = [
    "classification",
    "implanted",
    "risk_class",
    "manufacturer_id"
]

numeric_features = ["quantity_in_commerce"]

preprocessor = ColumnTransformer([
    (
        "numeric",
        SimpleImputer(strategy="median"),
        numeric_features
    ),
    (
        "categorical",
        Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]),
        categorical_features
    )
])

classifier = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced",
    min_samples_leaf=2,
    n_jobs=-1
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", classifier)
])

# ==========================================
# 7. TRAIN
# ==========================================

print("\nTraining group-based model...")

pipeline.fit(X_train, y_train)

# ==========================================
# 8. VALIDATION THRESHOLD
# ==========================================

val_prob = pipeline.predict_proba(X_val)[:, 1]

thresholds = [
    0.10, 0.20, 0.30, 0.40, 0.50,
    0.60, 0.70, 0.80, 0.90
]

best_threshold = 0.50
best_f1 = -1

print("\nVALIDATION THRESHOLD RESULTS")

for threshold in thresholds:

    pred = (val_prob >= threshold).astype(int)

    precision = precision_score(
        y_val, pred, zero_division=0
    )

    recall = recall_score(
        y_val, pred, zero_division=0
    )

    f1 = f1_score(
        y_val, pred, zero_division=0
    )

    print(
        f"Threshold: {threshold:.2f} | "
        f"Precision: {precision:.4f} | "
        f"Recall: {recall:.4f} | "
        f"F1: {f1:.4f}"
    )

    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

print("\nSelected threshold:", best_threshold)

# ==========================================
# 9. UNSEEN MANUFACTURER TEST
# ==========================================

print("\nUNSEEN MANUFACTURER TEST")

test_prob = pipeline.predict_proba(X_test)[:, 1]

test_pred = (
    test_prob >= best_threshold
).astype(int)

print("Threshold:", best_threshold)
print("Accuracy:", accuracy_score(y_test, test_pred))
print("Precision:", precision_score(
    y_test, test_pred, zero_division=0
))
print("Recall:", recall_score(
    y_test, test_pred, zero_division=0
))
print("F1:", f1_score(
    y_test, test_pred, zero_division=0
))

print("Confusion Matrix:")
print(confusion_matrix(y_test, test_pred))

# ROC-AUC and PR-AUC require both classes
if y_test.nunique() == 2:
    print("ROC-AUC:", roc_auc_score(y_test, test_prob))
    print("PR-AUC:", average_precision_score(y_test, test_prob))
else:
    print("ROC-AUC and PR-AUC unavailable: test set has one class.")

# ==========================================
# 10. SAVE RESULTS
# ==========================================

joblib.dump(
    pipeline,
    MODEL_DIR / "random_forest_group_test.joblib"
)

metadata = {
    "model": "Random Forest",
    "evaluation": "Manufacturer-group holdout",
    "features": FEATURES,
    "decision_threshold": float(best_threshold),
    "training_manufacturers": int(groups_train.nunique()),
    "validation_manufacturers": int(groups_val.nunique()),
    "testing_manufacturers": int(groups_test.nunique()),
    "training_devices": len(X_train),
    "validation_devices": len(X_val),
    "testing_devices": len(X_test)
}

with open(
    MODEL_DIR / "group_test_metadata.json",
    "w"
) as file:
    json.dump(metadata, file, indent=4)

print("\nGroup validation completed!")
print("Model and metadata saved.")
