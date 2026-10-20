"""
Unit Tests for Machine Learning Pipeline
Module: tests/test_train_model.py

Tests data validation, delimiter handling, pipeline training,
evaluation metrics, and artifact generation using isolated synthetic fixtures.
"""

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ml.train_model import (
    DEFAULT_FEATURES,
    DEFAULT_TARGET,
    calculate_metrics,
    detect_delimiter,
    load_and_validate_data,
    train_and_evaluate,
)


class TestTrainModelPipeline(unittest.TestCase):
    def setUp(self):
        """Create a temporary directory for test fixtures and outputs."""
        self.test_dir = tempfile.mkdtemp()
        self.models_dir = os.path.join(self.test_dir, "models")
        os.makedirs(self.models_dir, exist_ok=True)

    def tearDown(self):
        """Clean up temporary directories and files."""
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _create_synthetic_csv(
        self,
        filename: str = "sample.csv",
        sep: str = ",",
        n_samples: int = 50,
        missing_col: bool = False,
        with_nulls: bool = False,
        non_numeric: bool = False,
        empty: bool = False,
    ) -> str:
        """Helper to create synthetic student datasets for testing."""
        filepath = os.path.join(self.test_dir, filename)

        if empty:
            Path(filepath).touch()
            return filepath

        np.random.seed(42)
        data = {
            "G1": np.random.randint(0, 20, size=n_samples),
            "G2": np.random.randint(0, 20, size=n_samples),
            "studytime": np.random.randint(1, 5, size=n_samples),
            "failures": np.random.randint(0, 4, size=n_samples),
            "absences": np.random.randint(0, 30, size=n_samples),
            "G3": np.random.randint(0, 20, size=n_samples),
        }

        df = pd.DataFrame(data)

        if missing_col:
            df = df.drop(columns=["absences"])

        if with_nulls:
            df.loc[0, "G1"] = np.nan

        if non_numeric:
            df["G2"] = df["G2"].astype(str)
            df.loc[0, "G2"] = "invalid_score"

        df.to_csv(filepath, sep=sep, index=False)
        return filepath

    def test_detect_delimiter_comma_and_semicolon(self):
        """Test reliable detection between comma and semicolon separated CSVs."""
        comma_file = self._create_synthetic_csv("comma.csv", sep=",")
        semi_file = self._create_synthetic_csv("semi.csv", sep=";")

        self.assertEqual(detect_delimiter(Path(comma_file)), ",")
        self.assertEqual(detect_delimiter(Path(semi_file)), ";")

    def test_load_and_validate_data_valid(self):
        """Test valid loading and column separation."""
        csv_file = self._create_synthetic_csv("valid.csv", sep=",", n_samples=30)
        X, y = load_and_validate_data(csv_file)

        self.assertEqual(len(X), 30)
        self.assertEqual(list(X.columns), DEFAULT_FEATURES)
        self.assertEqual(y.name, DEFAULT_TARGET)

    def test_load_and_validate_data_semicolon(self):
        """Test valid loading of semicolon-delimited CSV."""
        csv_file = self._create_synthetic_csv("semi.csv", sep=";", n_samples=30)
        X, y = load_and_validate_data(csv_file)

        self.assertEqual(len(X), 30)
        self.assertEqual(list(X.columns), DEFAULT_FEATURES)
        self.assertEqual(y.name, DEFAULT_TARGET)

    def test_load_and_validate_missing_column(self):
        """Test that missing required columns raise a descriptive ValueError."""
        csv_file = self._create_synthetic_csv("missing.csv", missing_col=True)
        with self.assertRaises(ValueError) as ctx:
            load_and_validate_data(csv_file)
        self.assertIn("missing required column", str(ctx.exception).lower())

    def test_load_and_validate_null_values(self):
        """Test that null/missing values in required columns raise ValueError."""
        csv_file = self._create_synthetic_csv("nulls.csv", with_nulls=True)
        with self.assertRaises(ValueError) as ctx:
            load_and_validate_data(csv_file)
        self.assertIn("null or missing values", str(ctx.exception).lower())

    def test_load_and_validate_non_numeric_values(self):
        """Test that non-numeric values in numeric columns raise ValueError."""
        csv_file = self._create_synthetic_csv("non_numeric.csv", non_numeric=True)
        with self.assertRaises(ValueError) as ctx:
            load_and_validate_data(csv_file)
        self.assertIn("non-numeric", str(ctx.exception).lower())

    def test_load_and_validate_empty_file(self):
        """Test that empty CSV file raises ValueError."""
        csv_file = self._create_synthetic_csv("empty.csv", empty=True)
        with self.assertRaises(ValueError) as ctx:
            load_and_validate_data(csv_file)
        self.assertIn("empty", str(ctx.exception).lower())

    def test_load_and_validate_nonexistent_file(self):
        """Test that non-existent file path raises FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            load_and_validate_data(os.path.join(self.test_dir, "nonexistent.csv"))

    def test_calculate_metrics(self):
        """Test MAE, RMSE, and R2 calculation accuracy."""
        y_true = np.array([10.0, 15.0, 20.0])
        y_pred = np.array([11.0, 14.0, 20.0])
        metrics = calculate_metrics(y_true, y_pred)

        self.assertIn("mae", metrics)
        self.assertIn("rmse", metrics)
        self.assertIn("r2", metrics)
        self.assertAlmostEqual(metrics["mae"], (1.0 + 1.0 + 0.0) / 3.0, places=3)

    def test_train_and_evaluate_end_to_end(self):
        """Test complete ML training pipeline, metric generation, and model artifact serialization."""
        csv_file = self._create_synthetic_csv("train_sample.csv", n_samples=60)

        result = train_and_evaluate(
            data_path=csv_file,
            models_dir=self.models_dir,
            test_size=0.2,
            random_state=42,
        )

        # Verify returned dictionary
        self.assertIn("metrics", result)
        self.assertIn("artifacts", result)

        lr_path = Path(result["artifacts"]["linear_regression"])
        svr_path = Path(result["artifacts"]["svr"])
        metrics_path = Path(result["artifacts"]["metrics"])

        # Check files exist on disk
        self.assertTrue(lr_path.is_file(), "linear_regression.joblib should exist")
        self.assertTrue(svr_path.is_file(), "svr.joblib should exist")
        self.assertTrue(metrics_path.is_file(), "metrics.json should exist")

        # Verify metrics.json content
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics_json = json.load(f)

        self.assertEqual(metrics_json["features"], DEFAULT_FEATURES)
        self.assertEqual(metrics_json["target"], DEFAULT_TARGET)
        self.assertEqual(metrics_json["split_settings"]["test_size"], 0.2)
        self.assertEqual(metrics_json["split_settings"]["random_state"], 42)
        self.assertEqual(metrics_json["split_settings"]["train_samples"], 48)
        self.assertEqual(metrics_json["split_settings"]["test_samples"], 12)

        for model_key in ["linear_regression", "svr"]:
            self.assertIn(model_key, metrics_json["models"])
            m = metrics_json["models"][model_key]["metrics"]
            self.assertIsInstance(m["mae"], float)
            self.assertIsInstance(m["rmse"], float)
            self.assertIsInstance(m["r2"], float)

        # Verify saved model pipelines can be loaded and used for prediction
        loaded_lr = joblib.load(lr_path)
        loaded_svr = joblib.load(svr_path)

        dummy_input = pd.DataFrame([[12.0, 14.0, 2.0, 0.0, 4.0]], columns=DEFAULT_FEATURES)
        lr_pred = loaded_lr.predict(dummy_input)
        svr_pred = loaded_svr.predict(dummy_input)

        self.assertEqual(len(lr_pred), 1)
        self.assertEqual(len(svr_pred), 1)


if __name__ == "__main__":
    unittest.main()
