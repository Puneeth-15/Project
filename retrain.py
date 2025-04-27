import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
import pickle
import os

# Create pickle directory if it doesn't exist
if not os.path.exists('pickle'):
    os.makedirs('pickle')

# Read the dataset
data = pd.read_csv('phishing.csv')

# Prepare features and target
X = data.drop(['class', 'Index'], axis=1)
y = data['class']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
gbc = GradientBoostingClassifier(random_state=42)
gbc.fit(X_train, y_train)

# Save the model
with open('pickle/model.pkl', 'wb') as file:
    pickle.dump(gbc, file)

print("Model trained and saved successfully!") 