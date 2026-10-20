"""
Unit and Integration Tests for College CGPA Prediction Module
Module: tests/test_college.py

Tests data loading, duplicate removal, pipeline training,
evaluation metrics, saved model artifacts, and Flask college API endpoints.
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
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from app import create_app
from ml.train_college_model import (
    DEFAULT_COLLEGE_FEATURES,
    DEFAULT_COLLEGE_TARGET,
    calculate_metrics,
    detect_delimiter,
    load_and_validate_college_data,
    train_and_evaluate_college,
)


class TestCollegePipelineAndAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Prepare temporary directory, synthetic datasets, and test models."""
        cls.test_dir = tempfile.mkdtemp()
        cls.empty_models_dir = os.path.join(cls.test_dir, "empty_models")
        cls.loaded_models_dir = os.path.join(cls.test_dir, "loaded_models")
        os.makedirs(cls.empty_models_dir, exist_ok=True)
        os.makedirs(cls.loaded_models_dir, exist_ok=True)

        # Train minimal dummy college pipelines for Flask tests
        X_dummy = pd.DataFrame({"Study_Hours": [1.0, 3.0, 5.0, 7.0, 9.0]})
        y_dummy = pd.Series([2.0, 4.0, 6.0, 8.0, 9.5])

        lr_pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]).fit(X_dummy, y_dummy)

        svr_pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVR(kernel="rbf")),
        ]).fit(X_dummy, y_dummy)

        joblib.dump(lr_pipe, os.path.join(cls.loaded_models_dir, "linear_regression.joblib"))
        joblib.dump(svr_pipe, os.path.join(cls.loaded_models_dir, "svr.joblib"))

        cls.sample_metrics = {
            "domain": "College CGPA Prediction",
            "features": ["Study_Hours"],
            "target": "CGPA",
            "target_range": [0.0, 10.0],
            "training_metadata": {
                "raw_train_samples": 5,
                "deduplicated_train_samples": 5,
                "duplicates_removed": 0,
                "test_samples": 5,
                "feature_notes": "Test synthetic metadata.",
            },
            "models": {
                "linear_regression": {"name": "Linear Regression", "metrics": {"mae": 0.25, "rmse": 0.30, "r2": 0.98}},
                "svr": {"name": "Support Vector Regression (SVR)", "metrics": {"mae": 0.28, "rmse": 0.32, "r2": 0.97}},
            },
        }
        with open(os.path.join(cls.loaded_models_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(cls.sample_metrics, f)

    @classmethod
    def tearDownClass(cls):
        """Clean up isolated test fixtures."""
        shutil.rmtree(cls.test_dir, ignore_errors=True)

    def _create_synthetic_csv(
        self,
        filename: str,
        n_samples: int = 30,
        duplicates: int = 0,
        missing_col: bool = False,
        non_numeric: bool = False,
        with_nulls: bool = False,
        empty: bool = False,
    ) -> str:
        filepath = os.path.join(self.test_dir, filename)
        if empty:
            Path(filepath).touch()
            return filepath

        np.random.seed(42)
        study_hours = np.linspace(1.0, 10.0, n_samples)
        cgpa = np.clip(1.5 + 0.85 * study_hours + np.random.normal(0, 0.2, n_samples), 0.0, 10.0)

        df = pd.DataFrame({"Study_Hours": study_hours, "CGPA": cgpa})

        if duplicates > 0:
            dup_rows = df.iloc[:duplicates].copy()
            df = pd.concat([df, dup_rows], ignore_index=True)

        if missing_col:
            df = df.drop(columns=["CGPA"])

        if non_numeric:
            df["Study_Hours"] = df["Study_Hours"].astype(str)
            df.loc[0, "Study_Hours"] = "not_a_number"

        if with_nulls:
            df.loc[0, "CGPA"] = np.nan

        df.to_csv(filepath, index=False)
        return filepath

    # -------------------------------------------------------------------------
    # ML Pipeline Tests
    # -------------------------------------------------------------------------
    def test_load_and_validate_college_data_valid_and_deduplication(self):
        """Test valid loading and exact duplicate row removal."""
        csv_file = self._create_synthetic_csv("train_dup.csv", n_samples=25, duplicates=3)
        X, y, dups_removed = load_and_validate_college_data(csv_file, drop_duplicates=True)

        self.assertEqual(dups_removed, 3)
        self.assertEqual(len(X), 25)
        self.assertEqual(len(y), 25)
        self.assertEqual(list(X.columns), DEFAULT_COLLEGE_FEATURES)
        self.assertEqual(y.name, DEFAULT_COLLEGE_TARGET)

    def test_load_and_validate_college_data_missing_column(self):
        """Test missing CGPA column raises descriptive ValueError."""
        csv_file = self._create_synthetic_csv("missing.csv", missing_col=True)
        with self.assertRaises(ValueError) as ctx:
            load_and_validate_college_data(csv_file)
        self.assertIn("missing required column", str(ctx.exception).lower())

    def test_load_and_validate_college_data_non_numeric(self):
        """Test non-numeric values in Study_Hours raise ValueError."""
        csv_file = self._create_synthetic_csv("non_num.csv", non_numeric=True)
        with self.assertRaises(ValueError) as ctx:
            load_and_validate_college_data(csv_file)
        self.assertIn("non-numeric", str(ctx.exception).lower())

    def test_load_and_validate_college_data_nulls(self):
        """Test null/missing values in required columns raise ValueError."""
        csv_file = self._create_synthetic_csv("nulls.csv", with_nulls=True)
        with self.assertRaises(ValueError) as ctx:
            load_and_validate_college_data(csv_file)
        self.assertIn("null or missing", str(ctx.exception).lower())

    def test_train_and_evaluate_college_end_to_end(self):
        """Test complete college training and evaluation pipeline on separate train/test files."""
        train_csv = self._create_synthetic_csv("train_e2e.csv", n_samples=40, duplicates=2)
        test_csv = self._create_synthetic_csv("test_e2e.csv", n_samples=15, duplicates=0)
        output_dir = os.path.join(self.test_dir, "e2e_models")

        result = train_and_evaluate_college(
            train_path=train_csv,
            test_path=test_csv,
            models_dir=output_dir,
        )

        lr_path = Path(result["artifacts"]["linear_regression"])
        svr_path = Path(result["artifacts"]["svr"])
        metrics_path = Path(result["artifacts"]["metrics"])

        self.assertTrue(lr_path.is_file())
        self.assertTrue(svr_path.is_file())
        self.assertTrue(metrics_path.is_file())

        with open(metrics_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        self.assertEqual(meta["training_metadata"]["duplicates_removed"], 2)
        self.assertEqual(meta["training_metadata"]["deduplicated_train_samples"], 40)
        self.assertEqual(meta["training_metadata"]["test_samples"], 15)
        self.assertIn("linear_regression", meta["models"])
        self.assertIn("svr", meta["models"])

        # Test model prediction clamp
        loaded_lr = joblib.load(lr_path)
        pred = loaded_lr.predict(pd.DataFrame({"Study_Hours": [5.0]}))
        self.assertEqual(len(pred), 1)

    # -------------------------------------------------------------------------
    # Flask College API Tests
    # -------------------------------------------------------------------------
    def test_college_health_without_models(self):
        """Test college health reports degraded when models are missing."""
        app = create_app({"TESTING": True, "COLLEGE_MODELS_DIR": self.empty_models_dir})
        client = app.test_client()

        resp = client.get("/api/college/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["status"], "degraded")
        self.assertFalse(data["models_available"])

    def test_college_metrics_without_models(self):
        """Test college metrics returns 503 when missing."""
        app = create_app({"TESTING": True, "COLLEGE_MODELS_DIR": self.empty_models_dir})
        client = app.test_client()

        resp = client.get("/api/college/metrics")
        self.assertEqual(resp.status_code, 503)

    def test_college_predict_without_models(self):
        """Test college predict returns 503 when missing."""
        app = create_app({"TESTING": True, "COLLEGE_MODELS_DIR": self.empty_models_dir})
        client = app.test_client()

        resp = client.post("/api/college/predict", json={"Study_Hours": 5.0})
        self.assertEqual(resp.status_code, 503)

    def test_college_health_and_metrics_with_models(self):
        """Test college health and metrics endpoints with loaded models."""
        app = create_app({"TESTING": True, "COLLEGE_MODELS_DIR": self.loaded_models_dir})
        client = app.test_client()

        health_resp = client.get("/api/college/health")
        self.assertEqual(health_resp.status_code, 200)
        self.assertEqual(health_resp.get_json()["status"], "healthy")

        metrics_resp = client.get("/api/college/metrics")
        self.assertEqual(metrics_resp.status_code, 200)
        self.assertEqual(metrics_resp.get_json()["domain"], "College CGPA Prediction")

    def test_college_predict_valid_payload_and_metadata(self):
        """Test college prediction with valid study hours and optional metadata."""
        app = create_app({"TESTING": True, "COLLEGE_MODELS_DIR": self.loaded_models_dir})
        client = app.test_client()

        payload = {
            "Study_Hours": 6.5,
            "student_name": "Alex Smith",
            "student_id": "CS-2026-042",
            "department": "Computer Science",
            "semester": "Semester 4",
        }
        resp = client.post("/api/college/predict", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()

        self.assertIn("predictions", data)
        self.assertIn("linear_regression", data["predictions"])
        self.assertIn("svr", data["predictions"])
        self.assertIn("metadata", data)
        self.assertEqual(data["metadata"]["student_name"], "Alex Smith")
        self.assertEqual(data["metadata"]["student_id"], "CS-2026-042")

        # Bounds verification
        self.assertTrue(0.0 <= data["predictions"]["linear_regression"] <= 10.0)
        self.assertTrue(0.0 <= data["predictions"]["svr"] <= 10.0)

    def test_college_predict_bounds_clamping(self):
        """Test that extreme study hours produce predictions clamped between 0.0 and 10.0."""
        app = create_app({"TESTING": True, "COLLEGE_MODELS_DIR": self.loaded_models_dir})
        client = app.test_client()

        # Extreme high within allowed [0, 24] range
        resp = client.post("/api/college/predict", json={"Study_Hours": 24.0})
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertLessEqual(data["predictions"]["linear_regression"], 10.0)
        self.assertLessEqual(data["predictions"]["svr"], 10.0)

        # Extreme low
        resp_low = client.post("/api/college/predict", json={"Study_Hours": 0.0})
        self.assertEqual(resp_low.status_code, 200)
        data_low = resp_low.get_json()
        self.assertGreaterEqual(data_low["predictions"]["linear_regression"], 0.0)
        self.assertGreaterEqual(data_low["predictions"]["svr"], 0.0)

    def test_college_predict_invalid_inputs(self):
        """Test validation failures: non-json, missing Study_Hours, non-numeric, out-of-range."""
        app = create_app({"TESTING": True, "COLLEGE_MODELS_DIR": self.loaded_models_dir})
        client = app.test_client()

        # Non-JSON
        r1 = client.post("/api/college/predict", data="plain text", content_type="text/plain")
        self.assertEqual(r1.status_code, 400)

        # Missing Study_Hours
        r2 = client.post("/api/college/predict", json={"student_name": "Bob"})
        self.assertEqual(r2.status_code, 400)
        self.assertIn("Missing required feature", r2.get_json()["error"])

        # Non-numeric Study_Hours
        r3 = client.post("/api/college/predict", json={"Study_Hours": "five_hours"})
        self.assertEqual(r3.status_code, 400)
        self.assertIn("must be a valid number", r3.get_json()["error"])

        # Negative Study_Hours
        r4 = client.post("/api/college/predict", json={"Study_Hours": -1.5})
        self.assertEqual(r4.status_code, 400)
        self.assertIn("between 0.0 and 24.0", r4.get_json()["error"])

        # Excessive Study_Hours (> 24)
        r5 = client.post("/api/college/predict", json={"Study_Hours": 25.0})
        self.assertEqual(r5.status_code, 400)
        self.assertIn("between 0.0 and 24.0", r5.get_json()["error"])


if __name__ == "__main__":
    unittest.main()
