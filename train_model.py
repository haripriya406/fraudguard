import hashlib
import os
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from fraudguard_ml import build_pipeline, infer_positive_label, resolve_target


ROOT = Path(__file__).resolve().parent
DATA_PATH = Path(os.environ.get("FRAUDGUARD_DATASET_PATH", ROOT / "data" / "excel.csv"))
MODEL_PATH = ROOT / "models" / "fraudguard_model.joblib"
TARGET_COLUMN = os.environ.get("FRAUD_TARGET_COLUMN", "is_fraudulent").strip()


def fingerprint(path):
    digest = hashlib.sha256()
    with path.open("rb") as dataset_file:
        for chunk in iter(lambda: dataset_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    if not DATA_PATH.is_file():
        raise SystemExit(
            f"Dataset not found: {DATA_PATH}. Add your CSV before training."
        )
    if not TARGET_COLUMN:
        raise SystemExit(
            "Set FRAUD_TARGET_COLUMN to the exact CSV target column name before training."
        )

    try:
        frame = pd.read_csv(DATA_PATH)
    except (OSError, UnicodeDecodeError, pd.errors.ParserError, ValueError) as error:
        raise SystemExit(f"Could not read dataset: {error}") from error

    if TARGET_COLUMN not in frame.columns:
        raise SystemExit(
            f"Target column '{TARGET_COLUMN}' was not found. "
            f"Available columns: {', '.join(map(str, frame.columns))}"
        )
    if frame.empty:
        raise SystemExit("The CSV contains no transaction rows.")

    missing_target_rows = int(frame[TARGET_COLUMN].isna().sum())
    frame = frame.loc[frame[TARGET_COLUMN].notna()].copy()
    print(f"Removed {missing_target_rows} rows with missing target labels.")
    print(f"Remaining dataset size: {len(frame)} rows.")
    if frame.empty:
        raise SystemExit("No rows with valid target labels remain for training.")

    try:
        positive_label = infer_positive_label(frame[TARGET_COLUMN])
        y = resolve_target(frame[TARGET_COLUMN], positive_label)
    except ValueError as error:
        raise SystemExit(str(error)) from error

    if y.value_counts().min() < 2:
        raise SystemExit("At least two examples of each class are required for a train/test split.")

    X = frame.drop(columns=[TARGET_COLUMN]).copy()
    empty_columns = X.columns[X.isna().all()].tolist()
    if empty_columns:
        X = X.drop(columns=empty_columns)
    if X.shape[1] == 0:
        raise SystemExit(
            "No usable input features remain after excluding the target and fully empty columns."
        )

    numeric_columns = X.select_dtypes(include="number").columns.tolist()
    categorical_columns = [column for column in X.columns if column not in numeric_columns]
    feature_types = {
        column: "numeric" if column in numeric_columns else "categorical"
        for column in X.columns
    }

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y,
        )
    except ValueError as error:
        raise SystemExit(f"Could not create a stratified train/test split: {error}") from error

    pipeline = build_pipeline(numeric_columns, categorical_columns)
    try:
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
    except (ValueError, TypeError) as error:
        raise SystemExit(
            f"Training failed. Check that the input columns contain valid data: {error}"
        ) from error

    metrics = {
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "precision": round(float(precision_score(y_test, predictions, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, predictions, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, predictions, zero_division=0)), 4),
    }
    bundle = {
        "pipeline": pipeline,
        "feature_columns": list(X.columns),
        "feature_types": feature_types,
        "target_column": TARGET_COLUMN,
        "positive_label": str(positive_label),
        "metrics": metrics,
        "dataset_fingerprint": fingerprint(DATA_PATH),
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)

    print(f"Trained on {len(X_train)} rows; evaluated on {len(X_test)} held-out rows.")
    print(f"Fraud label: {positive_label}")
    if empty_columns:
        print("Ignored fully empty feature columns: " + ", ".join(map(str, empty_columns)))
    print("Held-out evaluation metrics:")
    for name, value in metrics.items():
        print(f"  {name.capitalize()}: {value:.4f}")
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
