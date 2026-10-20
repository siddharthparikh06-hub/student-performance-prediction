# Student Performance Prediction System

A web-based machine learning application to predict student academic scores and compare the performance of Linear Regression and Support Vector Regression (SVR) using standard regression metrics (MAE, RMSE, R²).

## Project Structure

```text
student-performance-prediction/
├── app.py                  # Flask application entry point
├── config.py               # Application configuration and paths
├── requirements.txt        # Python package dependencies
├── README.md               # Project documentation
├── ml/
│   ├── __init__.py         # ML package indicator
│   └── train_model.py      # Model training, evaluation, and serialization
├── models/
│   └── .gitkeep            # Directory for saved trained models and metrics
├── dataset/
│   └── .gitkeep            # Directory for dataset CSV files
├── templates/
│   └── base.html           # Base HTML layout template
├── static/
│   ├── css/
│   │   └── style.css       # Core stylesheet
│   └── js/
│       └── dashboard.js    # Interactive charts and client logic
└── tests/
    └── .gitkeep            # Unit and integration tests
```

## Technologies
- **Backend:** Python, Flask
- **Machine Learning & Data:** Pandas, NumPy, Scikit-learn, Joblib
- **Frontend:** HTML5, CSS3, JavaScript, Chart.js
