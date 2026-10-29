import base64
import hashlib
import io
import os
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

from fraudguard_ml import infer_positive_label, resolve_target


ROOT = Path(__file__).resolve().parent
DATA_PATH = Path(os.environ.get("FRAUDGUARD_DATASET_PATH", ROOT / "data" / "excel.csv"))
MODEL_PATH = ROOT / "models" / "fraudguard_model.joblib"
TARGET_COLUMN = os.environ.get("FRAUD_TARGET_COLUMN", "is_fraudulent").strip()
app = Flask(__name__)


def file_fingerprint(path):
    digest = hashlib.sha256()
    with path.open("rb") as dataset_file:
        for chunk in iter(lambda: dataset_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_chart(labels, counts):
    figure, axis = plt.subplots(figsize=(6, 3.2))
    figure.patch.set_facecolor("#111a2e")
    axis.set_facecolor("#111a2e")
    colors = ["#f97386" if str(label) == "FRAUD" else "#5a9bff" for label in labels]
    bars = axis.bar(labels, counts, color=colors, width=0.56)
    axis.set_ylabel("Transactions", color="#a8b4cf")
    axis.tick_params(colors="#a8b4cf")
    axis.spines[["top", "right", "bottom", "left"]].set_visible(False)
    axis.grid(axis="y", color="#27334c", linewidth=0.7)
    axis.set_axisbelow(True)
    axis.bar_label(bars, color="#e8edfa", padding=4)
    figure.tight_layout()
    image = io.BytesIO()
    figure.savefig(image, format="png", dpi=140, facecolor=figure.get_facecolor())
    plt.close(figure)
    return base64.b64encode(image.getvalue()).decode("ascii")


def make_feature_chart(frame, target_column):
    features = frame.drop(columns=[target_column])
    columns = list(features.columns[:3])
    if not columns:
        return None

    figure, axes = plt.subplots(1, len(columns), figsize=(6, 2.5), squeeze=False)
    figure.patch.set_facecolor("#111a2e")
    for axis, column in zip(axes[0], columns):
        axis.set_facecolor("#111a2e")
        series = features[column].dropna()
        if pd.api.types.is_numeric_dtype(features[column]):
            axis.hist(series, bins=12, color="#8c79ff", edgecolor="#111a2e")
            axis.set_ylabel("Rows", color="#a8b4cf", fontsize=8)
        else:
            counts = series.astype(str).value_counts().head(6).sort_values()
            axis.barh(counts.index, counts.values, color="#5798ff")
            axis.set_ylabel("")
            axis.tick_params(axis="y", labelsize=7)
        axis.set_title(str(column)[:22], color="#dce4f4", fontsize=8, pad=7)
        axis.tick_params(colors="#a8b4cf", labelsize=7)
        axis.spines[["top", "right", "bottom", "left"]].set_visible(False)
        axis.grid(axis="x" if not pd.api.types.is_numeric_dtype(features[column]) else "y",
                  color="#27334c", linewidth=0.6)
        axis.set_axisbelow(True)

    figure.tight_layout()
    image = io.BytesIO()
    figure.savefig(image, format="png", dpi=140, facecolor=figure.get_facecolor())
    plt.close(figure)
    return base64.b64encode(image.getvalue()).decode("ascii")


def load_dashboard_data():
    state = {
        "dataset_ready": False,
        "dataset_message": "",
        "total_transactions": None,
        "fraud_count": None,
        "legitimate_count": None,
        "fraud_percent": None,
        "missing_values": None,
        "columns": [],
        "chart": None,
        "feature_chart": None,
        "class_counts": None,
        "model_ready": False,
        "model_message": "Train a model to enable transaction predictions.",
        "metrics": None,
        "model": None,
        "feature_columns": [],
        "feature_types": {},
    }

    if not DATA_PATH.is_file():
        state["dataset_message"] = (
            f"Dataset not found at {DATA_PATH}. Add your CSV there, then restart the app."
        )
        return state

    try:
        frame = pd.read_csv(DATA_PATH)
    except (OSError, UnicodeDecodeError, pd.errors.ParserError, ValueError) as error:
        state["dataset_message"] = f"Could not read the dataset: {error}"
        return state

    state.update(
        dataset_ready=True,
        total_transactions=len(frame),
        missing_values=int(frame.isna().sum().sum()),
        columns=[str(column) for column in frame.columns],
    )
    if frame.empty:
        state["dataset_message"] = "The CSV has headers but no transaction rows."
        state["dataset_ready"] = False
        return state

    if not TARGET_COLUMN:
        state["dataset_message"] = (
            "Dataset loaded. Set FRAUD_TARGET_COLUMN to the exact name of the "
            "column containing fraud/legitimate labels."
        )
        return state
    if TARGET_COLUMN not in frame.columns:
        state["dataset_message"] = (
            f"Target column '{TARGET_COLUMN}' was not found. Available columns: "
            + ", ".join(state["columns"])
        )
        return state

    labeled_target = frame.loc[frame[TARGET_COLUMN].notna(), TARGET_COLUMN]
    try:
        positive_label = infer_positive_label(labeled_target)
        target = resolve_target(labeled_target, positive_label)
    except ValueError as error:
        state["dataset_message"] = str(error)
        return state

    fraud_count = int(target.sum())
    legitimate_count = int(len(target) - fraud_count)
    state.update(
        fraud_count=fraud_count,
        legitimate_count=legitimate_count,
        fraud_percent=round(fraud_count / len(target) * 100, 2),
        class_counts={"FRAUD": fraud_count, "LEGITIMATE": legitimate_count},
        dataset_message=(
            f"Dataset loaded with target column '{TARGET_COLUMN}'. "
            f"{len(frame) - len(labeled_target)} rows with missing target labels are excluded "
            "from label counts. Review the column names and label counts before training."
        ),
    )
    state["chart"] = make_chart(["FRAUD", "LEGITIMATE"], [fraud_count, legitimate_count])
    state["feature_chart"] = make_feature_chart(frame, TARGET_COLUMN)

    if not MODEL_PATH.is_file():
        state["model_message"] = "Dataset is configured. Run train_model.py to train the model."
        return state

    try:
        bundle = joblib.load(MODEL_PATH)
    except (OSError, ValueError, EOFError, KeyError, AttributeError) as error:
        state["model_message"] = f"Saved model could not be loaded: {error}"
        return state

    if bundle.get("dataset_fingerprint") != file_fingerprint(DATA_PATH):
        state["model_message"] = "The CSV has changed since training. Run train_model.py again."
        return state
    if bundle.get("target_column") != TARGET_COLUMN:
        state["model_message"] = "The saved model uses a different target. Run train_model.py again."
        return state

    state.update(
        model_ready=True,
        model_message="Model is trained and matches the current dataset.",
        model=bundle["pipeline"],
        feature_columns=bundle["feature_columns"],
        feature_types=bundle["feature_types"],
        metrics=bundle["metrics"],
    )
    return state


@app.get("/")
def index():
    state = load_dashboard_data()
    return render_template("index.html", **{key: value for key, value in state.items() if key != "model"})


@app.post("/predict")
def predict():
    state = load_dashboard_data()
    if not state["model_ready"]:
        return jsonify({"error": state["model_message"]}), 409

    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Submit transaction details as a JSON object."}), 400

    values = {}
    for column in state["feature_columns"]:
        raw_value = payload.get(column)
        if raw_value is None or (isinstance(raw_value, str) and not raw_value.strip()):
            values[column] = np.nan
            continue
        if state["feature_types"][column] == "numeric":
            try:
                values[column] = float(raw_value)
            except (TypeError, ValueError):
                return jsonify({"error": f"'{column}' must be a number."}), 400
        else:
            values[column] = str(raw_value).strip()

    try:
        row = pd.DataFrame([values], columns=state["feature_columns"])
        prediction = int(state["model"].predict(row)[0])
        probabilities = state["model"].predict_proba(row)[0]
        positive_probability = float(probabilities[list(state["model"].classes_).index(1)])
    except (ValueError, TypeError, KeyError, AttributeError) as error:
        app.logger.exception("Transaction prediction failed")
        return jsonify({"error": f"Could not analyze this transaction: {error}"}), 500

    return jsonify(
        {
            "prediction": "FRAUD" if prediction == 1 else "LEGITIMATE",
            "fraud_probability": round(positive_probability * 100, 2),
            "legitimate_probability": round((1 - positive_probability) * 100, 2),
        }
    )


if __name__ == "__main__":
    app.run(debug=os.environ.get("FLASK_DEBUG") == "1")
