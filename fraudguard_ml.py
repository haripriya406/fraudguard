import os

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


FRAUD_LABELS = {"fraud", "fraudulent", "scam", "1", "1.0", "true", "yes"}


def infer_positive_label(target):
    classes = list(pd.Series(target).dropna().unique())
    if len(classes) != 2:
        raise ValueError(
            f"Target column must contain exactly two non-missing classes; found {len(classes)}."
        )

    configured = os.environ.get("FRAUD_POSITIVE_LABEL")
    if configured is not None:
        matches = [label for label in classes if str(label).strip() == configured.strip()]
        if len(matches) != 1:
            raise ValueError(
                "FRAUD_POSITIVE_LABEL must exactly match one of the target values: "
                + ", ".join(map(str, classes))
            )
        return matches[0]

    fraud_matches = [
        label for label in classes if str(label).strip().lower() in FRAUD_LABELS
    ]
    if len(fraud_matches) == 1:
        return fraud_matches[0]

    numeric_classes = pd.to_numeric(pd.Series(classes), errors="coerce")
    if not numeric_classes.isna().any() and set(numeric_classes.tolist()) == {0, 1}:
        return classes[numeric_classes.tolist().index(1)]

    raise ValueError(
        "Could not identify the fraud label from the target values "
        f"({', '.join(map(str, classes))}). Set FRAUD_POSITIVE_LABEL to the exact fraud value."
    )


def resolve_target(target, positive_label):
    present = target.notna()
    if not present.all():
        raise ValueError(
            f"Target column has {int((~present).sum())} missing values. "
            "Remove or label those rows before training."
        )
    return (target == positive_label).astype("int8")


def build_pipeline(numeric_columns, categorical_columns):
    transformers = []
    if numeric_columns:
        numeric_pipeline = Pipeline(
            steps=[("imputer", SimpleImputer(strategy="median"))]
        )
        transformers.append(("numeric", numeric_pipeline, numeric_columns))
    if categorical_columns:
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        transformers.append(("categorical", categorical_pipeline, categorical_columns))

    if not transformers:
        raise ValueError("No usable input features remain after excluding the target column.")

    from sklearn.ensemble import RandomForestClassifier

    return Pipeline(
        steps=[
            ("preprocessor", ColumnTransformer(transformers=transformers)),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=250,
                    class_weight="balanced_subsample",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )
