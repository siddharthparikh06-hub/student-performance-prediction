"""
Student Performance Prediction - Machine Learning Pipeline
Module: ml/train_model.py

Handles dataset loading, validation, preprocessing, model training
(Linear Regression and SVR), evaluation, and artifact serialization.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

DEFAULT_FEATURES: List[str] = ["G1", "G2", "studytime", "failures", "absences"]
DEFAULT_TARGET: str = "G3"


def detect_delimiter(filepath: Path) -> str:
    """
    Detects whether a CSV file is comma or semicolon delimited.
    Ensures single-column parsing anomalies are avoided.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        first_line = f.readline().strip()

    if not first_line:
        raise ValueError(f"Dataset file is empty or contains no header: {filepath}")

    semicolon_count = first_line.count(";")
    comma_count = first_line.count(",")

    if semicolon_count > comma_count:
        return ";"
    return ","


def load_and_validate_data(
    filepath: str,
    features: Optional[List[str]] = None,
    target: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Loads CSV data from disk and performs strict validation.

    Checks:
    - File existence and non-empty status.
    - Delimiter detection (comma and semicolon support).
    - Presence of required feature and target columns.
    - Missing/null value detection.
    - Numeric data type validation.

    Returns:
        Tuple of (X, y) as (pd.DataFrame, pd.Series)
    """
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file not found at: {filepath}")

    if path.stat().st_size == 0:
        raise ValueError(f"Dataset file is empty: {filepath}")

    features_to_use = features if features is not None else DEFAULT_FEATURES
    target_to_use = target if target is not None else DEFAULT_TARGET
    required_columns = list(features_to_use) + [target_to_use]

    delimiter = detect_delimiter(path)
    df = pd.read_csv(path, sep=delimiter)

    # Guard against parsing anomalies resulting in single composite columns
    if len(df.columns) == 1 and (";" in df.columns[0] or "," in df.columns[0]):
        fallback_delim = "," if delimiter == ";" else ";"
        df = pd.read_csv(path, sep=fallback_delim)
        if len(df.columns) == 1 and (";" in df.columns[0] or "," in df.columns[0]):
            raise ValueError(
                f"Failed to parse CSV delimiter correctly for {filepath}. Resulted in a single unseparated column."
            )

    if df.empty:
        raise ValueError(f"Dataset is empty or contains no data rows: {filepath}")

    # Validate required columns existence
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Dataset is missing required column(s): {missing_cols}. Found columns: {list(df.columns)}"
        )

    working_df = df[required_columns].copy()

    # Validate absence of null / NaN values
    if working_df.isnull().any().any():
        null_counts = working_df.isnull().sum()
        cols_with_nulls = null_counts[null_counts > 0].to_dict()
        raise ValueError(
            f"Dataset contains null or missing values in required columns: {cols_with_nulls}"
        )

    # Validate and enforce numeric types
    for col in required_columns:
        if not pd.api.types.is_numeric_dtype(working_df[col]):
            coerced = pd.to_numeric(working_df[col], errors="coerce")
            if coerced.isnull().any():
                invalid_vals = working_df.loc[coerced.isnull(), col].tolist()
                raise ValueError(
                    f"Column '{col}' contains non-numeric value(s): {invalid_vals[:5]}"
                )
            working_df[col] = coerced

    X = working_df[features_to_use].astype(float)
    y = working_df[target_to_use].astype(float)

    return X, y


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Computes MAE, RMSE, and R2 regression metrics in a version-compatible manner.
    """
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))

    return {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "r2": round(r2, 4),
    }


def build_linear_regression_pipeline() -> Pipeline:
    """Creates a Scikit-learn Pipeline with StandardScaler and Linear Regression."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression()),
    ])


def build_svr_pipeline(
    kernel: str = "rbf",
    c_param: float = 1.0,
    epsilon: float = 0.1,
) -> Pipeline:
    """Creates a Scikit-learn Pipeline with StandardScaler and SVR."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVR(kernel=kernel, C=c_param, epsilon=epsilon)),
    ])


def train_and_evaluate(
    data_path: str,
    models_dir: str,
    features: Optional[List[str]] = None,
    target: Optional[str] = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict[str, any]:
    """
    Loads dataset, splits into reproducible train/test sets, trains Linear Regression and SVR pipelines,
    evaluates both models on held-out test data, and saves model pipelines and metrics.

    Args:
        data_path: Path to dataset CSV file.
        models_dir: Directory where trained models and metrics.json will be saved.
        features: Optional list of feature names (defaults to DEFAULT_FEATURES).
        target: Optional target column name (defaults to DEFAULT_TARGET).
        test_size: Fraction of data reserved for test set (default: 0.2).
        random_state: Random seed for train-test split reproducibility (default: 42).

    Returns:
        Dictionary containing evaluation summary and saved artifact file paths.
    """
    features_to_use = features if features is not None else DEFAULT_FEATURES
    target_to_use = target if target is not None else DEFAULT_TARGET

    # 1. Load and validate data
    X, y = load_and_validate_data(data_path, features=features_to_use, target=target_to_use)

    # 2. Reproducible Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    # 3. Instantiate pipelines
    lr_pipeline = build_linear_regression_pipeline()
    svr_pipeline = build_svr_pipeline()

    # 4. Fit models on training data only
    lr_pipeline.fit(X_train, y_train)
    svr_pipeline.fit(X_train, y_train)

    # 5. Evaluate on the same held-out test set
    lr_preds = lr_pipeline.predict(X_test)
    svr_preds = svr_pipeline.predict(X_test)

    lr_metrics = calculate_metrics(y_test.to_numpy(), lr_preds)
    svr_metrics = calculate_metrics(y_test.to_numpy(), svr_preds)

    # 6. Prepare output paths
    out_dir = Path(models_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    lr_model_path = out_dir / "linear_regression.joblib"
    svr_model_path = out_dir / "svr.joblib"
    metrics_path = out_dir / "metrics.json"

    # 7. Serialize trained pipelines
    joblib.dump(lr_pipeline, lr_model_path)
    joblib.dump(svr_pipeline, svr_model_path)

    # 8. Compile and serialize metrics JSON
    metrics_data = {
        "features": features_to_use,
        "target": target_to_use,
        "split_settings": {
            "test_size": test_size,
            "random_state": random_state,
            "train_samples": int(len(X_train)),
            "test_samples": int(len(X_test)),
        },
        "models": {
            "linear_regression": {
                "name": "Linear Regression",
                "metrics": lr_metrics,
            },
            "svr": {
                "name": "Support Vector Regression (SVR)",
                "metrics": svr_metrics,
            },
        },
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=2)

    # 9. Print concise training summary
    print("=" * 60)
    print("ML TRAINING & EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Dataset path:      {data_path}")
    print(f"Features:          {features_to_use}")
    print(f"Target:            {target_to_use}")
    print(f"Train / Test size: {len(X_train)} / {len(X_test)} samples")
    print("-" * 60)
    print("Model Evaluation (Held-out Test Set):")
    print(f"  [Linear Regression] -> MAE: {lr_metrics['mae']}, RMSE: {lr_metrics['rmse']}, R2: {lr_metrics['r2']}")
    print(f"  [SVR]               -> MAE: {svr_metrics['mae']}, RMSE: {svr_metrics['rmse']}, R2: {svr_metrics['r2']}")
    print("-" * 60)
    print(f"Artifacts saved:")
    print(f"  - {lr_model_path}")
    print(f"  - {svr_model_path}")
    print(f"  - {metrics_path}")
    print("=" * 60)

    return {
        "metrics": metrics_data,
        "artifacts": {
            "linear_regression": str(lr_model_path),
            "svr": str(svr_model_path),
            "metrics": str(metrics_path),
        },
    }


if __name__ == "__main__":
    import sys

    base_dir = Path(__file__).resolve().parent.parent
    default_dataset = base_dir / "dataset" / "student_data.csv"
    if not default_dataset.is_file():
        alt_dataset = base_dir / "dataset" / "student-mat.csv"
        if alt_dataset.is_file():
            default_dataset = alt_dataset
    default_models_dir = base_dir / "models"

    dataset_arg = sys.argv[1] if len(sys.argv) > 1 else str(default_dataset)
    models_arg = sys.argv[2] if len(sys.argv) > 2 else str(default_models_dir)

    try:
        train_and_evaluate(data_path=dataset_arg, models_dir=models_arg)
    except Exception as exc:
        print(f"Error executing training pipeline: {exc}", file=sys.stderr)
        sys.exit(1)
