
import pandas as pd
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
    confusion_matrix
)

# ==========================================
# 1. PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent.parent

MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_DIR.mkdir(exist_ok=True)

DEVICE_FILE = BASE_DIR / "devices-1681209661.csv"
EVENT_FILE = BASE_DIR / "events-1681209680.csv"

MODEL_DIR.mkdir(exist_ok=True)

DEVICE_FILE = BASE_DIR / "devices-1681209661.csv"
EVENT_FILE = BASE_DIR / "events-1681209680.csv"

# ==========================================
# 2. LOAD DATA
# ==========================================

print("\nLoading datasets...")

devices = pd.read_csv(DEVICE_FILE, low_memory=False)
events = pd.read_csv(EVENT_FILE, low_memory=False)

# ==========================================
# 3. CREATE TARGET
# ==========================================

print("\nCreating failure target...")

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

y = devices["failure"]

print("Target distribution:")
print(y.value_counts())

# ==========================================
# 4. FEATURES
# ==========================================

all_features = [
    "classification",
    "implanted",
    "quantity_in_commerce",
    "risk_class",
    "manufacturer_id"
]

X = devices[all_features].copy()

X["quantity_in_commerce"] = pd.to_numeric(
    X["quantity_in_commerce"],
    errors="coerce"
)

# ==========================================
# 5. SAME TRAIN / VALIDATION / TEST SPLIT
# ==========================================

X_train_val, X_test, y_train_val, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_val,
    y_train_val,
    test_size=0.25,
    random_state=42,
    stratify=y_train_val
)

print("\nTraining:", len(X_train))
print("Validation:", len(X_val))
print("Testing:", len(X_test))

# ==========================================
# 6. MODEL TRAINING FUNCTION
# ==========================================

def train_and_evaluate(model_name, features):

    print("\n" + "=" * 55)
    print("MODEL:", model_name)
    print("=" * 55)

    categorical_features = [
        col for col in features
        if col != "quantity_in_commerce"
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

    # Select only the requested features
    X_train_model = X_train[features]
    X_val_model = X_val[features]
    X_test_model = X_test[features]

    # Train
    print("\nTraining model...")
    pipeline.fit(X_train_model, y_train)

    # ======================================
    # VALIDATION THRESHOLD SELECTION
    # ======================================

    val_probabilities = pipeline.predict_proba(
        X_val_model
    )[:, 1]

    thresholds = [
        0.10, 0.20, 0.30, 0.40, 0.50,
        0.60, 0.70, 0.80, 0.90
    ]

    best_threshold = 0.50
    best_f1 = -1

    print("\nValidation threshold analysis:")

    for threshold in thresholds:

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
            f"Threshold: {threshold:.2f} | "
            f"Precision: {precision:.4f} | "
            f"Recall: {recall:.4f} | "
            f"F1: {f1:.4f}"
        )

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    print("\nSelected threshold:", best_threshold)
    print("Validation F1:", round(best_f1, 4))

    # ======================================
    # FINAL TEST EVALUATION
    # ======================================

    test_probabilities = pipeline.predict_proba(
        X_test_model
    )[:, 1]

    test_predictions = (
        test_probabilities >= best_threshold
    ).astype(int)

    metrics = {
        "Model": model_name,
        "Threshold": best_threshold,
        "Accuracy": accuracy_score(y_test, test_predictions),
        "Precision": precision_score(
            y_test, test_predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_test, test_predictions, zero_division=0
        ),
        "F1 Score": f1_score(
            y_test, test_predictions, zero_division=0
        ),
        "ROC AUC": roc_auc_score(
            y_test, test_probabilities
        ),
        "PR AUC": average_precision_score(
            y_test, test_probabilities
        )
    }

    print("\nFINAL TEST RESULTS:")

    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, test_predictions))

    # Save each model separately
    filename = (
        "random_forest_with_manufacturer.joblib"
        if "With" in model_name
        else "random_forest_without_manufacturer.joblib"
    )

    joblib.dump(
        pipeline,
        MODEL_DIR / filename
    )

    return metrics, best_threshold


# ==========================================
# 7. RUN BOTH EXPERIMENTS
# ==========================================

results = []

# Model A: With manufacturer ID
features_with = all_features

metrics_a, threshold_a = train_and_evaluate(
    "With Manufacturer ID",
    features_with
)

results.append(metrics_a)

# Model B: Without manufacturer ID
features_without = [
    "classification",
    "implanted",
    "quantity_in_commerce",
    "risk_class"
]

metrics_b, threshold_b = train_and_evaluate(
    "Without Manufacturer ID",
    features_without
)

results.append(metrics_b)

# ==========================================
# 8. SAVE COMPARISON
# ==========================================

comparison = pd.DataFrame(results)

comparison.to_csv(
    MODEL_DIR / "manufacturer_model_comparison.csv",
    index=False
)

print("\n" + "=" * 65)
print("FINAL MODEL COMPARISON")
print("=" * 65)

print(comparison.to_string(index=False))

print("\nComparison saved successfully!")

print("\nSaved model files:")
print("1. random_forest_with_manufacturer.joblib")
print("2. random_forest_without_manufacturer.joblib")
print("3. manufacturer_model_comparison.csv")
