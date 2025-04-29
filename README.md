# Phishing URL Detector

A machine learning-based web application that detects potential phishing URLs using various features and provides a detailed analysis.

## Features

- Real-time URL analysis
- Detailed feature analysis
- Threat level visualization
- Modern and responsive UI
- Machine learning-based detection

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
