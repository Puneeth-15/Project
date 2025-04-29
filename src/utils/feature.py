"""
URL Feature Extraction Module

This module implements a comprehensive set of features for analyzing URLs to detect
potential phishing attempts. It includes various URL characteristics analysis methods
and feature extraction techniques.

Author: [Your Name]
Date: [Current Date]
"""

import ipaddress
import re
import urllib.request
from bs4 import BeautifulSoup
import socket
import requests
from googlesearch import search
import whois
from datetime import date, datetime
import time
from dateutil.parser import parse as date_parse
from urllib.parse import urlparse
import urllib3
import difflib
import tld
import numpy as np

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class URLFeatureExtractor:
    """
    A class to extract various features from URLs for phishing detection.
    Implements multiple feature extraction methods and analysis techniques.
    """
    
    def __init__(self):
        """Initialize the feature extractor with common patterns and thresholds."""
        self.suspicious_words = [
            'login', 'signin', 'account', 'banking', 'secure',
            'verify', 'confirm', 'update', 'security', 'alert'
        ]
        
        self.suspicious_tlds = [
            'tk', 'ml', 'ga', 'cf', 'gq', 'xyz', 'top', 'work',
            'site', 'online', 'click', 'loan', 'money'
        ]
    
    def extract_features(self, url):
        """
        Extract all features from a given URL.
        
        Args:
            url (str): The URL to analyze
            
        Returns:
            dict: Dictionary containing all extracted features
        """
        features = {}
        
        # Basic URL features
        features.update(self._extract_basic_features(url))
        
        # Domain features
        features.update(self._extract_domain_features(url))
        
        # Content features
        features.update(self._extract_content_features(url))
        
        # Security features
        features.update(self._extract_security_features(url))
        
        return features
    
    def _extract_basic_features(self, url):
        """Extract basic URL characteristics."""
        features = {}
        
        # URL length
        features['url_length'] = len(url)
        
        # Number of dots
        features['num_dots'] = url.count('.')
        
        # Number of hyphens
        features['num_hyphens'] = url.count('-')
        
        # Number of underscores
        features['num_underscores'] = url.count('_')
        
        # Number of slashes
        features['num_slashes'] = url.count('/')
        
        # Number of question marks
        features['num_question_marks'] = url.count('?')
        
        # Number of equal signs
        features['num_equal_signs'] = url.count('=')
        
        # Number of at signs
        features['num_at_signs'] = url.count('@')
        
        # Number of and signs
        features['num_and_signs'] = url.count('&')
        
        # Number of exclamation marks
        features['num_exclamation_marks'] = url.count('!')
        
        # Number of spaces
        features['num_spaces'] = url.count(' ')
        
        # Number of tildes
        features['num_tildes'] = url.count('~')
        
        # Number of commas
        features['num_commas'] = url.count(',')
        
        # Number of plus signs
        features['num_plus_signs'] = url.count('+')
        
        # Number of asterisks
        features['num_asterisks'] = url.count('*')
        
        # Number of hashtags
        features['num_hashtags'] = url.count('#')
        
        # Number of dollar signs
        features['num_dollar_signs'] = url.count('$')
        
        # Number of percent signs
        features['num_percent_signs'] = url.count('%')
        
        return features
    
    def _extract_domain_features(self, url):
        """Extract domain-related features."""
        features = {}
        
        try:
            # Parse URL
            parsed_url = urlparse(url)
            domain = parsed_url.netloc
            
            # Extract TLD
            tld_info = tld.get_tld(url, as_object=True)
            features['tld'] = tld_info.tld
            
            # Check if TLD is suspicious
            features['suspicious_tld'] = 1 if tld_info.tld in self.suspicious_tlds else 0
            
            # Domain length
            features['domain_length'] = len(domain)
            
            # Number of subdomains
            features['num_subdomains'] = len(domain.split('.')) - 1
            
            # Check if domain contains IP address
            features['is_ip'] = 1 if re.match(r'\d+\.\d+\.\d+\.\d+', domain) else 0
            
            # Check if domain contains suspicious words
            features['suspicious_words'] = 1 if any(word in domain.lower() for word in self.suspicious_words) else 0
            
            # Try to get WHOIS information
            try:
                whois_info = whois.whois(domain)
                features['domain_age'] = self._calculate_domain_age(whois_info.creation_date)
            except:
                features['domain_age'] = -1
            
        except:
            # If URL parsing fails, set default values
            features.update({
                'tld': '',
                'suspicious_tld': 0,
                'domain_length': 0,
                'num_subdomains': 0,
                'is_ip': 0,
                'suspicious_words': 0,
                'domain_age': -1
            })
        
        return features
    
    def _extract_content_features(self, url):
        """Extract content-related features."""
        features = {}
        
        try:
            # Try to get page content
            response = requests.get(url, timeout=5)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Check for forms
            features['has_forms'] = 1 if len(soup.find_all('form')) > 0 else 0
            
            # Check for iframes
            features['has_iframes'] = 1 if len(soup.find_all('iframe')) > 0 else 0
            
            # Check for JavaScript
            features['has_javascript'] = 1 if len(soup.find_all('script')) > 0 else 0
            
            # Check for external resources
            features['has_external_resources'] = 1 if len(soup.find_all(['img', 'link', 'script'], src=True)) > 0 else 0
            
        except:
            # If content extraction fails, set default values
            features.update({
                'has_forms': 0,
                'has_iframes': 0,
                'has_javascript': 0,
                'has_external_resources': 0
            })
        
        return features
    
    def _extract_security_features(self, url):
        """Extract security-related features."""
        features = {}
        
        # Check for HTTPS
        features['has_https'] = 1 if url.startswith('https://') else 0
        
        # Check for SSL certificate
        try:
            response = requests.get(url, timeout=5)
            features['has_ssl'] = 1 if response.url.startswith('https://') else 0
        except:
            features['has_ssl'] = 0
        
        return features
    
    def _calculate_domain_age(self, creation_date):
        """Calculate domain age in days."""
        if not creation_date:
            return -1
        
        if isinstance(creation_date, list):
            creation_date = creation_date[0]
        
        try:
            age = (datetime.now() - creation_date).days
            return age
        except:
            return -1

def main():
    """Main function to demonstrate feature extraction."""
    extractor = URLFeatureExtractor()
    
    # Example URL
    url = "https://example.com"
    
    # Extract features
    features = extractor.extract_features(url)
    
    # Print features
    print("\nExtracted Features:")
    for feature, value in features.items():
        print(f"{feature}: {value}")

if __name__ == "__main__":
    main()
