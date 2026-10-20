import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret-key-student-marks")
    DATASET_DIR = BASE_DIR / "dataset"
    MODELS_DIR = BASE_DIR / "models"
    COLLEGE_MODELS_DIR = MODELS_DIR / "college"

    # School dataset
    DATASET_PATH = (
        DATASET_DIR / "student_data.csv"
        if (DATASET_DIR / "student_data.csv").is_file()
        else DATASET_DIR / "student-mat.csv"
    )

    # College dataset
    COLLEGE_TRAIN_DATASET_PATH = DATASET_DIR / "train - Sheet1.csv"
    COLLEGE_TEST_DATASET_PATH = DATASET_DIR / "test - Sheet1.csv"
