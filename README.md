# Phishing URL Detection System

A comprehensive system for detecting phishing URLs using machine learning techniques. This project implements various algorithms to analyze URL characteristics and identify potential phishing attempts.

## Project Structure

```
Phishing-Url-Detection/
├── src/
│   ├── models/           # Machine learning model implementations
│   │   ├── ml_training.py
│   │   └── retrain.py
│   ├── utils/           # Helper functions and utilities
│   │   └── feature.py
│   ├── data/            # Dataset and data processing
│   │   └── phishing.csv
│   └── web/             # Web interface
│       ├── app.py
│       ├── templates/
│       └── static/
├── pickle/              # Saved model files
├── visualizations/      # Analysis visualizations
├── results/            # Performance metrics
├── requirements.txt    # Project dependencies
└── README.md          # Documentation
```

## Getting Started

1. Set up the Python environment:
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Train the detection models:
```bash
python src/models/ml_training.py
```

4. Launch the web interface:
```bash
python src/web/app.py
```

## Model Performance

The system employs multiple machine learning algorithms for phishing detection. The Gradient Boosting Classifier demonstrates the best performance:
- Accuracy: 97.4%
- F1-score: 97.7%
- Recall: 99.4%
- Precision: 98.6%

Complete performance metrics are available in `results/model_results.csv`.

## Key Features

- Advanced URL feature analysis
- Multiple machine learning models
- Comprehensive performance evaluation
- Interactive web interface
- Flexible model retraining

## Technical Requirements

- Python 3.8 or higher
- scikit-learn
- pandas
- numpy
- Flask
- XGBoost
- CatBoost
- Additional requirements in requirements.txt
