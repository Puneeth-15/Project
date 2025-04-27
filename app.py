#importing required libraries

from flask import Flask, request, render_template, flash, jsonify
import numpy as np
import pandas as pd
from sklearn import metrics 
import warnings
import pickle
import os
import json
warnings.filterwarnings('ignore')
from feature import FeatureExtraction

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Load the model
try:
    with open("pickle/model.pkl", "rb") as file:
        gbc = pickle.load(file)
except Exception as e:
    print(f"Error loading model: {e}")
    gbc = None

def get_feature_analysis(features):
    feature_names = [
        "Using IP Address", "Long URL", "Short URL", "Symbol @", "Redirecting",
        "Prefix/Suffix", "Sub Domains", "HTTPS", "Domain Registration Length",
        "Favicon", "Non-Standard Port", "HTTPS Domain URL", "Request URL",
        "Anchor URL", "Links in Script Tags", "Server Form Handler",
        "Info Email", "Abnormal URL", "Website Forwarding", "Status Bar Customization",
        "Disable Right Click", "Using Popup Window", "Iframe Redirection",
        "Age of Domain", "DNS Recording", "Website Traffic", "Page Rank",
        "Google Index", "Links Pointing to Page", "Stats Report"
    ]
    
    analysis = []
    for i, feature in enumerate(features):
        status = "safe" if feature == 1 else "warning" if feature == 0 else "danger"
        analysis.append({
            "name": feature_names[i],
            "status": status,
            "value": feature
        })
    return analysis

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            url = request.form["url"]
            if not url:
                flash("Please enter a URL", "error")
                return render_template("index.html", xx=-1)
            
            obj = FeatureExtraction(url)
            features = obj.getFeaturesList()
            x = np.array(features).reshape(1,30)
            
            if gbc is None:
                flash("Model not loaded properly. Please contact administrator.", "error")
                return render_template("index.html", xx=-1)
            
            y_pred = gbc.predict(x)[0]
            y_pro_phishing = gbc.predict_proba(x)[0,0]
            y_pro_non_phishing = gbc.predict_proba(x)[0,1]
            
            # Calculate threat level (inverted logic)
            threat_level = y_pro_non_phishing * 100  # Now using non-phishing probability
            if threat_level > 70:
                status = "safe"
                prediction = f"This URL appears to be safe ({threat_level:.2f}% confidence)"
            elif threat_level > 30:
                status = "warning"
                prediction = f"Exercise caution with this URL ({threat_level:.2f}% confidence)"
            else:
                status = "danger"
                prediction = f"This URL appears to be unsafe ({threat_level:.2f}% confidence)"
            
            # Get feature analysis
            feature_analysis = get_feature_analysis(features)
            
            return render_template('index.html', 
                                xx=round(threat_level/100,2), 
                                url=url, 
                                prediction=prediction,
                                threat_level=threat_level,
                                status=status,
                                feature_analysis=feature_analysis)
        except Exception as e:
            flash(f"An error occurred: {str(e)}", "error")
            return render_template("index.html", xx=-1)
    
    return render_template("index.html", xx=-1)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)