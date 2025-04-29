# Phishing URL Detector

A machine learning-based web application that detects potential phishing URLs using various features and provides a detailed analysis.

## Features

- Real-time URL analysis
- Detailed feature analysis
- Threat level visualization
- Modern and responsive UI
- Machine learning-based detection
=======
# Phishing URL Detection System

A comprehensive system for detecting phishing URLs using machine learning techniques. This project implements various algorithms to analyze URL characteristics and identify potential phishing attempts.

## Project Structure

```
Phishing-URL-Detection/
├── app.py              # Main Flask application
├── feature.py          # Feature extraction module
├── requirements.txt    # Python dependencies
├── retrain.py         # Model training script
├── phishing.csv       # Training dataset
├── pickle/            # Directory containing the trained model
│   └── model.pkl      # Trained model file
└── templates/         # HTML templates
    └── index.html     # Main application template
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Phishing-URL-Detection.git
cd Phishing-URL-Detection
```

2. Install the required dependencies:
=======
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
3. 
```bash
pip install -r requirements.txt
```

3. Make sure you have the model file:
   - The model should be in the `pickle/model.pkl` file
   - If you don't have it, you can train the model using:
   ```bash
   python retrain.py
   ```

## Usage

1. Start the Flask application:
```bash
python app.py
```

2. Open your web browser and navigate to:
```
http://localhost:5000
```

3. Enter a URL to analyze its safety

## Features Analyzed

The application analyzes various features of URLs including:
- URL length and structure
- Domain age and registration
- SSL certificate presence
- IP address usage
- Redirects and forwarding
- JavaScript and iframe usage
- Form handling
- And many more...

## Technologies Used

- Python
- Flask
- scikit-learn
- BeautifulSoup4
- Bootstrap 5
- Font Awesome

## Contributing

Feel free to submit issues and enhancement requests!
=======
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
