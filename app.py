#importing required libraries

from flask import Flask, request, render_template, flash, jsonify, session
import numpy as np
import pandas as pd
from sklearn import metrics 
import warnings
import pickle
import os
import json
from datetime import datetime
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
            "value": feature,
            "description": get_feature_description(feature_names[i], feature)
        })
    return analysis

def get_feature_description(feature_name, value):
    descriptions = {
        "Using IP Address": {
            1: "URL uses a domain name instead of an IP address",
            0: "URL uses an IP address directly, which is suspicious"
        },
        "Long URL": {
            1: "URL length is within normal range",
            0: "URL is unusually long, which is a common phishing tactic"
        },
        "Short URL": {
            1: "URL is not using a URL shortening service",
            0: "URL is using a URL shortening service, which can hide the true destination"
        },
        "Symbol @": {
            1: "No @ symbol in URL",
            0: "Contains @ symbol, which can be used to hide the true domain"
        },
        "Redirecting": {
            1: "No suspicious redirects detected",
            0: "Contains suspicious redirects"
        },
        "Prefix/Suffix": {
            1: "No suspicious prefix/suffix",
            0: "Contains suspicious prefix/suffix"
        },
        "Sub Domains": {
            1: "Normal number of subdomains",
            0: "Unusually high number of subdomains"
        },
        "HTTPS": {
            1: "Uses HTTPS protocol",
            0: "Does not use HTTPS protocol"
        },
        "Domain Registration Length": {
            1: "Domain registration period is normal",
            0: "Domain registration period is suspiciously short"
        },
        "Favicon": {
            1: "Favicon is from the same domain",
            0: "Favicon is from a different domain"
        },
        "Non-Standard Port": {
            1: "Uses standard port",
            0: "Uses non-standard port"
        },
        "HTTPS Domain URL": {
            1: "HTTPS domain matches the URL",
            0: "HTTPS domain does not match the URL"
        },
        "Request URL": {
            1: "Request URL is from the same domain",
            0: "Request URL is from a different domain"
        },
        "Anchor URL": {
            1: "Anchor URL is from the same domain",
            0: "Anchor URL is from a different domain"
        },
        "Links in Script Tags": {
            1: "No suspicious links in script tags",
            0: "Contains suspicious links in script tags"
        },
        "Server Form Handler": {
            1: "Form handler is from the same domain",
            0: "Form handler is from a different domain"
        },
        "Info Email": {
            1: "No suspicious email patterns",
            0: "Contains suspicious email patterns"
        },
        "Abnormal URL": {
            1: "URL structure is normal",
            0: "URL structure is abnormal"
        },
        "Website Forwarding": {
            1: "No suspicious forwarding",
            0: "Contains suspicious forwarding"
        },
        "Status Bar Customization": {
            1: "No status bar customization",
            0: "Contains status bar customization"
        },
        "Disable Right Click": {
            1: "Right click is enabled",
            0: "Right click is disabled"
        },
        "Using Popup Window": {
            1: "No popup windows",
            0: "Uses popup windows"
        },
        "Iframe Redirection": {
            1: "No iframe redirection",
            0: "Contains iframe redirection"
        },
        "Age of Domain": {
            1: "Domain age is normal",
            0: "Domain is suspiciously new"
        },
        "DNS Recording": {
            1: "DNS records are normal",
            0: "DNS records are suspicious"
        },
        "Website Traffic": {
            1: "Website traffic is normal",
            0: "Website traffic is suspiciously low"
        },
        "Page Rank": {
            1: "Page rank is normal",
            0: "Page rank is suspiciously low"
        },
        "Google Index": {
            1: "Website is indexed by Google",
            0: "Website is not indexed by Google"
        },
        "Links Pointing to Page": {
            1: "Normal number of external links",
            0: "Suspiciously low number of external links"
        },
        "Stats Report": {
            1: "Website statistics are normal",
            0: "Website statistics are suspicious"
        }
    }
    return descriptions.get(feature_name, {}).get(value, "No description available")

def validate_url(url):
    if not url:
        return False, "Please enter a URL"
    if not url.startswith(('http://', 'https://')):
        return False, "URL must start with http:// or https://"
    return True, None

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            url = request.form["url"]
            is_valid, error_message = validate_url(url)
            
            if not is_valid:
                flash(error_message, "error")
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
            
            # Store scan history in session
            if 'scan_history' not in session:
                session['scan_history'] = []
            
            scan_entry = {
                'url': url,
                'status': status,
                'score': round(threat_level),
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'features': feature_analysis
            }
            
            session['scan_history'] = [scan_entry] + session['scan_history'][:19]  # Keep last 20 scans
            session.modified = True
            
            return render_template('index.html', 
                                xx=round(threat_level/100,2), 
                                url=url, 
                                prediction=prediction,
                                threat_level=threat_level,
                                status=status,
                                feature_analysis=feature_analysis,
                                scan_history=session.get('scan_history', []))
        except Exception as e:
            flash(f"An error occurred: {str(e)}", "error")
            return render_template("index.html", xx=-1)
    
    return render_template("index.html", xx=-1, scan_history=session.get('scan_history', []))

@app.route("/clear-history", methods=["POST"])
def clear_history():
    session.pop('scan_history', None)
    return jsonify({"status": "success"})

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)