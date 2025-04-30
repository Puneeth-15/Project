"""
Phishing URL Detection Model Training Module

This module implements a comprehensive approach to training multiple machine learning models
for phishing URL detection. It includes data preprocessing, model training, evaluation,
and visualization components.

Author: [Your Name]
Date: [Current Date]
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics 
import warnings
import os
import pickle
from datetime import datetime
warnings.filterwarnings('ignore')

#Loading data into dataframe
data = pd.read_csv("phishing.csv")
data.head()

#Shape of dataframe
print("Shape of dataframe:", data.shape)

#Listing the features of the dataset
print("\nFeatures of the dataset:")
print(data.columns)

#Information about the dataset
print("\nDataset information:")
print(data.info())

# nunique value in columns
print("\nUnique values in columns:")
print(data.nunique())

#droping index column
data = data.drop(['Index'],axis = 1)

#description of dataset
print("\nDataset description:")
print(data.describe().T)

#Correlation heatmap
plt.figure(figsize=(15,15))
sns.heatmap(data.corr(), annot=True)
plt.title("Correlation Heatmap")
plt.savefig('correlation_heatmap.png')
plt.close()

#pairplot for particular features
df = data[['PrefixSuffix-', 'SubDomains', 'HTTPS','AnchorURL','WebsiteTraffic','class']]
sns.pairplot(data = df,hue="class",corner=True)
plt.savefig('feature_pairplot.png')
plt.close()

# Phishing Count in pie chart
data['class'].value_counts().plot(kind='pie',autopct='%1.2f%%')
plt.title("Phishing Count")
plt.savefig('phishing_count.png')
plt.close()

# Splitting the dataset into dependent and independent features
X = data.drop(["class"], axis=1)

# Transform target labels to be in the range [0, 1]
data['class'] = data['class'].map({-1: 0, 1: 1})

y = data["class"]

# Splitting the dataset into train and test sets: 80-20 split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating holders to store the model performance results
ML_Model = []
accuracy = []
f1_score = []
recall = []
precision = []

#function to call for storing the results
def storeResults(model, a,b,c,d):
  ML_Model.append(model)
  accuracy.append(round(a, 3))
  f1_score.append(round(b, 3))
  recall.append(round(c, 3))
  precision.append(round(d, 3))
  
  # Save results after each model
  result = pd.DataFrame({ 'ML Model' : ML_Model,
                        'Accuracy' : accuracy,
                        'f1_score' : f1_score,
                        'Recall'   : recall,
                        'Precision': precision,
                      })
  result.to_csv('model_results.csv', index=False)

# Ensure the directory exists
os.makedirs("pickle", exist_ok=True)

# Linear regression model 
from sklearn.linear_model import LogisticRegression

# instantiate the model
log = LogisticRegression()

# fit the model 
log.fit(X_train,y_train)

#predicting the target value from the model for the samples
y_train_log = log.predict(X_train)
y_test_log = log.predict(X_test)

#computing the accuracy, f1_score, Recall, precision of the model performance
acc_train_log = metrics.accuracy_score(y_train,y_train_log)
acc_test_log = metrics.accuracy_score(y_test,y_test_log)
print("\nLogistic Regression Results:")
print("Accuracy on training Data: {:.3f}".format(acc_train_log))
print("Accuracy on test Data: {:.3f}".format(acc_test_log))

f1_score_train_log = metrics.f1_score(y_train,y_train_log)
f1_score_test_log = metrics.f1_score(y_test,y_test_log)
print("\nf1_score on training Data: {:.3f}".format(f1_score_train_log))
print("f1_score on test Data: {:.3f}".format(f1_score_test_log))

recall_score_train_log = metrics.recall_score(y_train,y_train_log)
recall_score_test_log = metrics.recall_score(y_test,y_test_log)
print("\nRecall on training Data: {:.3f}".format(recall_score_train_log))
print("Recall on test Data: {:.3f}".format(recall_score_test_log))

precision_score_train_log = metrics.precision_score(y_train,y_train_log)
precision_score_test_log = metrics.precision_score(y_test,y_test_log)
print("\nPrecision on training Data: {:.3f}".format(precision_score_train_log))
print("Precision on test Data: {:.3f}".format(precision_score_test_log))

#computing the classification report of the model
print("\nClassification Report:")
print(metrics.classification_report(y_test, y_test_log))

#storing the results
storeResults('Logistic Regression',acc_test_log,f1_score_test_log,recall_score_train_log,precision_score_train_log)

# Save the model
pickle.dump(log, open("pickle/logistic_regression.pkl", "wb"))

# K-Nearest Neighbors Classifier model
from sklearn.neighbors import KNeighborsClassifier

# instantiate the model
knn = KNeighborsClassifier(n_neighbors=1)

# fit the model 
knn.fit(X_train,y_train)

#predicting the target value from the model for the samples
y_train_knn = knn.predict(X_train)
y_test_knn = knn.predict(X_test)

#computing the accuracy,f1_score,Recall,precision of the model performance
acc_train_knn = metrics.accuracy_score(y_train,y_train_knn)
acc_test_knn = metrics.accuracy_score(y_test,y_test_knn)
print("\nK-Nearest Neighbors Results:")
print("Accuracy on training Data: {:.3f}".format(acc_train_knn))
print("Accuracy on test Data: {:.3f}".format(acc_test_knn))

f1_score_train_knn = metrics.f1_score(y_train,y_train_knn)
f1_score_test_knn = metrics.f1_score(y_test,y_test_knn)
print("\nf1_score on training Data: {:.3f}".format(f1_score_train_knn))
print("f1_score on test Data: {:.3f}".format(f1_score_test_knn))

recall_score_train_knn = metrics.recall_score(y_train,y_train_knn)
recall_score_test_knn = metrics.recall_score(y_test,y_test_knn)
print("\nRecall on training Data: {:.3f}".format(recall_score_train_knn))
print("Recall on test Data: {:.3f}".format(recall_score_test_knn))

precision_score_train_knn = metrics.precision_score(y_train,y_train_knn)
precision_score_test_knn = metrics.precision_score(y_test,y_test_knn)
print("\nPrecision on training Data: {:.3f}".format(precision_score_train_knn))
print("Precision on test Data: {:.3f}".format(precision_score_test_knn))

#computing the classification report of the model
print("\nClassification Report:")
print(metrics.classification_report(y_test, y_test_knn))

#storing the results
storeResults('K-Nearest Neighbors',acc_test_knn,f1_score_test_knn,recall_score_train_knn,precision_score_train_knn)

# Save the model
pickle.dump(knn, open("pickle/knn.pkl", "wb"))

# Support Vector Classifier model 
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

# defining parameter range
param_grid = {'gamma': [0.1],'kernel': ['rbf','linear']}

svc = GridSearchCV(SVC(), param_grid)

# fitting the model for grid search
svc.fit(X_train, y_train)

#predicting the target value from the model for the samples
y_train_svc = svc.predict(X_train)
y_test_svc = svc.predict(X_test)

#computing the accuracy, f1_score, Recall, precision of the model performance
acc_train_svc = metrics.accuracy_score(y_train,y_train_svc)
acc_test_svc = metrics.accuracy_score(y_test,y_test_svc)
print("\nSupport Vector Machine Results:")
print("Accuracy on training Data: {:.3f}".format(acc_train_svc))
print("Accuracy on test Data: {:.3f}".format(acc_test_svc))

f1_score_train_svc = metrics.f1_score(y_train,y_train_svc)
f1_score_test_svc = metrics.f1_score(y_test,y_test_svc)
print("\nf1_score on training Data: {:.3f}".format(f1_score_train_svc))
print("f1_score on test Data: {:.3f}".format(f1_score_test_svc))

recall_score_train_svc = metrics.recall_score(y_train,y_train_svc)
recall_score_test_svc = metrics.recall_score(y_test,y_test_svc)
print("\nRecall on training Data: {:.3f}".format(recall_score_train_svc))
print("Recall on test Data: {:.3f}".format(recall_score_test_svc))

precision_score_train_svc = metrics.precision_score(y_train,y_train_svc)
precision_score_test_svc = metrics.precision_score(y_test,y_test_svc)
print("\nPrecision on training Data: {:.3f}".format(precision_score_train_svc))
print("Precision on test Data: {:.3f}".format(precision_score_test_svc))

#computing the classification report of the model
print("\nClassification Report:")
print(metrics.classification_report(y_test, y_test_svc))

#storing the results
storeResults('Support Vector Machine',acc_test_svc,f1_score_test_svc,recall_score_train_svc,precision_score_train_svc)

# Save the model
pickle.dump(svc, open("pickle/svm.pkl", "wb"))

# Naive Bayes Classifier Model
from sklearn.naive_bayes import GaussianNB

# instantiate the model
nb=  GaussianNB()

# fit the model 
nb.fit(X_train,y_train)

#predicting the target value from the model for the samples
y_train_nb = nb.predict(X_train)
y_test_nb = nb.predict(X_test)

#computing the accuracy, f1_score, Recall, precision of the model performance
acc_train_nb = metrics.accuracy_score(y_train,y_train_nb)
acc_test_nb = metrics.accuracy_score(y_test,y_test_nb)
print("\nNaive Bayes Classifier Results:")
print("Accuracy on training Data: {:.3f}".format(acc_train_nb))
print("Accuracy on test Data: {:.3f}".format(acc_test_nb))

f1_score_train_nb = metrics.f1_score(y_train,y_train_nb)
f1_score_test_nb = metrics.f1_score(y_test,y_test_nb)
print("\nf1_score on training Data: {:.3f}".format(f1_score_train_nb))
print("f1_score on test Data: {:.3f}".format(f1_score_test_nb))

recall_score_train_nb = metrics.recall_score(y_train,y_train_nb)
recall_score_test_nb = metrics.recall_score(y_test,y_test_nb)
print("\nRecall on training Data: {:.3f}".format(recall_score_train_nb))
print("Recall on test Data: {:.3f}".format(recall_score_test_nb))

precision_score_train_nb = metrics.precision_score(y_train,y_train_nb)
precision_score_test_nb = metrics.precision_score(y_test,y_test_nb)
print("\nPrecision on training Data: {:.3f}".format(precision_score_train_nb))
print("Precision on test Data: {:.3f}".format(precision_score_test_nb))

#computing the classification report of the model
print("\nClassification Report:")
print(metrics.classification_report(y_test, y_test_nb))

#storing the results
storeResults('Naive Bayes Classifier',acc_test_nb,f1_score_test_nb,recall_score_train_nb,precision_score_train_nb)

# Save the model
pickle.dump(nb, open("pickle/naive_bayes.pkl", "wb"))

# Decision Tree Classifier model 
from sklearn.tree import DecisionTreeClassifier

# instantiate the model 
tree = DecisionTreeClassifier(max_depth=30)

# fit the model 
tree.fit(X_train, y_train)

#predicting the target value from the model for the samples
y_train_tree = tree.predict(X_train)
y_test_tree = tree.predict(X_test)

#computing the accuracy, f1_score, Recall, precision of the model performance
acc_train_tree = metrics.accuracy_score(y_train,y_train_tree)
acc_test_tree = metrics.accuracy_score(y_test,y_test_tree)
print("\nDecision Tree Results:")
print("Accuracy on training Data: {:.3f}".format(acc_train_tree))
print("Accuracy on test Data: {:.3f}".format(acc_test_tree))

f1_score_train_tree = metrics.f1_score(y_train,y_train_tree)
f1_score_test_tree = metrics.f1_score(y_test,y_test_tree)
print("\nf1_score on training Data: {:.3f}".format(f1_score_train_tree))
print("f1_score on test Data: {:.3f}".format(f1_score_test_tree))

recall_score_train_tree = metrics.recall_score(y_train,y_train_tree)
recall_score_test_tree = metrics.recall_score(y_test,y_test_tree)
print("\nRecall on training Data: {:.3f}".format(recall_score_train_tree))
print("Recall on test Data: {:.3f}".format(recall_score_test_tree))

precision_score_train_tree = metrics.precision_score(y_train,y_train_tree)
precision_score_test_tree = metrics.precision_score(y_test,y_test_tree)
print("\nPrecision on training Data: {:.3f}".format(precision_score_train_tree))
print("Precision on test Data: {:.3f}".format(precision_score_test_tree))

#computing the classification report of the model
print("\nClassification Report:")
print(metrics.classification_report(y_test, y_test_tree))

#storing the results
storeResults('Decision Tree',acc_test_tree,f1_score_test_tree,recall_score_train_tree,precision_score_train_tree)

# Save the model
pickle.dump(tree, open("pickle/decision_tree.pkl", "wb"))

# Random Forest Classifier Model
from sklearn.ensemble import RandomForestClassifier

# instantiate the model
forest = RandomForestClassifier(n_estimators=10)

# fit the model 
forest.fit(X_train,y_train)

#predicting the target value from the model for the samples
y_train_forest = forest.predict(X_train)
y_test_forest = forest.predict(X_test)

#computing the accuracy, f1_score, Recall, precision of the model performance
acc_train_forest = metrics.accuracy_score(y_train,y_train_forest)
acc_test_forest = metrics.accuracy_score(y_test,y_test_forest)
print("\nRandom Forest Results:")
print("Accuracy on training Data: {:.3f}".format(acc_train_forest))
print("Accuracy on test Data: {:.3f}".format(acc_test_forest))

f1_score_train_forest = metrics.f1_score(y_train,y_train_forest)
f1_score_test_forest = metrics.f1_score(y_test,y_test_forest)
print("\nf1_score on training Data: {:.3f}".format(f1_score_train_forest))
print("f1_score on test Data: {:.3f}".format(f1_score_test_forest))

recall_score_train_forest = metrics.recall_score(y_train,y_train_forest)
recall_score_test_forest = metrics.recall_score(y_test,y_test_forest)
print("\nRecall on training Data: {:.3f}".format(recall_score_train_forest))
print("Recall on test Data: {:.3f}".format(recall_score_test_forest))

precision_score_train_forest = metrics.precision_score(y_train,y_train_forest)
precision_score_test_forest = metrics.precision_score(y_test,y_test_forest)
print("\nPrecision on training Data: {:.3f}".format(precision_score_train_forest))
print("Precision on test Data: {:.3f}".format(precision_score_test_forest))

#computing the classification report of the model
print("\nClassification Report:")
print(metrics.classification_report(y_test, y_test_forest))

#storing the results
storeResults('Random Forest',acc_test_forest,f1_score_test_forest,recall_score_train_forest,precision_score_train_forest)

# Save the model
pickle.dump(forest, open("pickle/random_forest.pkl", "wb"))

# Gradient Boosting Classifier Model
from sklearn.ensemble import GradientBoostingClassifier

# instantiate the model
gbc = GradientBoostingClassifier(max_depth=4,learning_rate=0.7)

# fit the model 
gbc.fit(X_train,y_train)

#predicting the target value from the model for the samples
y_train_gbc = gbc.predict(X_train)
y_test_gbc = gbc.predict(X_test)

#computing the accuracy, f1_score, Recall, precision of the model performance
acc_train_gbc = metrics.accuracy_score(y_train,y_train_gbc)
acc_test_gbc = metrics.accuracy_score(y_test,y_test_gbc)
print("\nGradient Boosting Classifier Results:")
print("Accuracy on training Data: {:.3f}".format(acc_train_gbc))
print("Accuracy on test Data: {:.3f}".format(acc_test_gbc))

f1_score_train_gbc = metrics.f1_score(y_train,y_train_gbc)
f1_score_test_gbc = metrics.f1_score(y_test,y_test_gbc)
print("\nf1_score on training Data: {:.3f}".format(f1_score_train_gbc))
print("f1_score on test Data: {:.3f}".format(f1_score_test_gbc))

recall_score_train_gbc = metrics.recall_score(y_train,y_train_gbc)
recall_score_test_gbc = metrics.recall_score(y_test,y_test_gbc)
print("\nRecall on training Data: {:.3f}".format(recall_score_train_gbc))
print("Recall on test Data: {:.3f}".format(recall_score_test_gbc))

precision_score_train_gbc = metrics.precision_score(y_train,y_train_gbc)
precision_score_test_gbc = metrics.precision_score(y_test,y_test_gbc)
print("\nPrecision on training Data: {:.3f}".format(precision_score_train_gbc))
print("Precision on test Data: {:.3f}".format(precision_score_test_gbc))

#computing the classification report of the model
print("\nClassification Report:")
print(metrics.classification_report(y_test, y_test_gbc))

#storing the results
storeResults('Gradient Boosting Classifier',acc_test_gbc,f1_score_test_gbc,recall_score_train_gbc,precision_score_train_gbc)

# Save the model
pickle.dump(gbc, open("pickle/gradient_boosting.pkl", "wb"))

#  catboost Classifier Model
from catboost import CatBoostClassifier

# instantiate the model
cat = CatBoostClassifier(learning_rate  = 0.1)

# fit the model 
cat.fit(X_train,y_train)

#predicting the target value from the model for the samples
y_train_cat = cat.predict(X_train)
y_test_cat = cat.predict(X_test)

#computing the accuracy, f1_score, Recall, precision of the model performance
acc_train_cat  = metrics.accuracy_score(y_train,y_train_cat)
acc_test_cat = metrics.accuracy_score(y_test,y_test_cat)
print("\nCatBoost Classifier Results:")
print("Accuracy on training Data: {:.3f}".format(acc_train_cat))
print("Accuracy on test Data: {:.3f}".format(acc_test_cat))

f1_score_train_cat = metrics.f1_score(y_train,y_train_cat)
f1_score_test_cat = metrics.f1_score(y_test,y_test_cat)
print("\nf1_score on training Data: {:.3f}".format(f1_score_train_cat))
print("f1_score on test Data: {:.3f}".format(f1_score_test_cat))

recall_score_train_cat = metrics.recall_score(y_train,y_train_cat)
recall_score_test_cat = metrics.recall_score(y_test,y_test_cat)
print("\nRecall on training Data: {:.3f}".format(recall_score_train_cat))
print("Recall on test Data: {:.3f}".format(recall_score_test_cat))

precision_score_train_cat = metrics.precision_score(y_train,y_train_cat)
precision_score_test_cat = metrics.precision_score(y_test,y_test_cat)
print("\nPrecision on training Data: {:.3f}".format(precision_score_train_cat))
print("Precision on test Data: {:.3f}".format(precision_score_test_cat))

#computing the classification report of the model
print("\nClassification Report:")
print(metrics.classification_report(y_test, y_test_cat))

#storing the results
storeResults('CatBoost Classifier',acc_test_cat,f1_score_test_cat,recall_score_train_cat,precision_score_train_cat)

# Save the model
pickle.dump(cat, open("pickle/catboost.pkl", "wb"))

#  XGBoost Classifier Model
from xgboost import XGBClassifier

# instantiate the model
xgb = XGBClassifier()

# fit the model 
xgb.fit(X_train,y_train)

#predicting the target value from the model for the samples
y_train_xgb = xgb.predict(X_train)
y_test_xgb = xgb.predict(X_test)

#computing the accuracy, f1_score, Recall, precision of the model performance
acc_train_xgb = metrics.accuracy_score(y_train,y_train_xgb)
acc_test_xgb = metrics.accuracy_score(y_test,y_test_xgb)
print("\nXGBoost Classifier Results:")
print("Accuracy on training Data: {:.3f}".format(acc_train_xgb))
print("Accuracy on test Data: {:.3f}".format(acc_test_xgb))

f1_score_train_xgb = metrics.f1_score(y_train,y_train_xgb)
f1_score_test_xgb = metrics.f1_score(y_test,y_test_xgb)
print("\nf1_score on training Data: {:.3f}".format(f1_score_train_xgb))
print("f1_score on test Data: {:.3f}".format(f1_score_test_xgb))

recall_score_train_xgb = metrics.recall_score(y_train,y_train_xgb)
recall_score_test_xgb = metrics.recall_score(y_test,y_test_xgb)
print("\nRecall on training Data: {:.3f}".format(recall_score_train_xgb))
print("Recall on test Data: {:.3f}".format(recall_score_test_xgb))

precision_score_train_xgb = metrics.precision_score(y_train,y_train_xgb)
precision_score_test_xgb = metrics.precision_score(y_test,y_test_xgb)
print("\nPrecision on training Data: {:.3f}".format(precision_score_train_xgb))
print("Precision on test Data: {:.3f}".format(precision_score_test_xgb))

#computing the classification report of the model
print("\nClassification Report:")
print(metrics.classification_report(y_test, y_test_xgb))

#storing the results
storeResults('XGBoost Classifier',acc_test_xgb,f1_score_test_xgb,recall_score_train_xgb,precision_score_train_xgb)

# Save the model
pickle.dump(xgb, open("pickle/xgboost.pkl", "wb"))

# Multi-layer Perceptron Classifier Model
from sklearn.neural_network import MLPClassifier

# instantiate the model
mlp = MLPClassifier()

# fit the model 
mlp.fit(X_train,y_train)

#predicting the target value from the model for the samples
y_train_mlp = mlp.predict(X_train)
y_test_mlp = mlp.predict(X_test)

#computing the accuracy, f1_score, Recall, precision of the model performance
acc_train_mlp  = metrics.accuracy_score(y_train,y_train_mlp)
acc_test_mlp = metrics.accuracy_score(y_test,y_test_mlp)
print("\nMulti-layer Perceptron Results:")
print("Accuracy on training Data: {:.3f}".format(acc_train_mlp))
print("Accuracy on test Data: {:.3f}".format(acc_test_mlp))

f1_score_train_mlp = metrics.f1_score(y_train,y_train_mlp)
f1_score_test_mlp = metrics.f1_score(y_test,y_test_mlp)
print("\nf1_score on training Data: {:.3f}".format(f1_score_train_mlp))
print("f1_score on test Data: {:.3f}".format(f1_score_test_mlp))

recall_score_train_mlp = metrics.recall_score(y_train,y_train_mlp)
recall_score_test_mlp = metrics.recall_score(y_test,y_test_mlp)
print("\nRecall on training Data: {:.3f}".format(recall_score_train_mlp))
print("Recall on test Data: {:.3f}".format(recall_score_test_mlp))

precision_score_train_mlp = metrics.precision_score(y_train,y_train_mlp)
precision_score_test_mlp = metrics.precision_score(y_test,y_test_mlp)
print("\nPrecision on training Data: {:.3f}".format(precision_score_train_mlp))
print("Precision on test Data: {:.3f}".format(precision_score_test_mlp))

#computing the classification report of the model
print("\nClassification Report:")
print(metrics.classification_report(y_test, y_test_mlp))

#storing the results
storeResults('Multi-layer Perceptron',acc_test_mlp,f1_score_test_mlp,recall_score_train_mlp,precision_score_train_mlp)

# Save the model
pickle.dump(mlp, open("pickle/mlp.pkl", "wb"))

#creating dataframe
result = pd.DataFrame({ 'ML Model' : ML_Model,
                        'Accuracy' : accuracy,
                        'f1_score' : f1_score,
                        'Recall'   : recall,
                        'Precision': precision,
                      })

# dispalying total result
print("\nFinal Results:")
print(result)

#Sorting the datafram on accuracy
sorted_result=result.sort_values(by=['Accuracy', 'f1_score'],ascending=False).reset_index(drop=True)
# dispalying total result
print("\nSorted Results:")
print(sorted_result)

# Save the best model (Gradient Boosting Classifier)
pickle.dump(gbc, open("pickle/model.pkl", "wb"))

#checking the feature improtance in the model
plt.figure(figsize=(9,7))
n_features = X_train.shape[1]
plt.barh(range(n_features), gbc.feature_importances_, align='center')
plt.yticks(np.arange(n_features), X_train.columns)
plt.title("Feature importances using permutation on full model")
plt.xlabel("Feature importance")
plt.ylabel("Feature")
plt.savefig('feature_importance.png')
plt.close() 