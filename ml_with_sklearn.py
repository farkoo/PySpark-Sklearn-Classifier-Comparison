# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 17:41:09 2023

@author: ACER
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time

start_time = time.time()

# Read the training data CSV files
# train_features_file = "C:\\uni\\2 - second term\\Cloud Computing\\Project\\braintumor\\pca_vgg19_X_train - Copy.csv"
# train_labels_file = "C:\\uni\\2 - second term\\Cloud Computing\\Project\\braintumor\\y_train - Copy.csv"
train_features_file = "C:\\uni\\2 - second term\\Cloud Computing\\Project\\eye_csv\\pca_X_train - Copy.csv"
train_labels_file = "C:\\uni\\2 - second term\\Cloud Computing\\Project\\eye_csv\\y_train - Copy.csv"

train_features = pd.read_csv(train_features_file)
train_labels = pd.read_csv(train_labels_file)

# Read the test data CSV files
# test_features_file = "C:\\uni\\2 - second term\\Cloud Computing\\Project\\braintumor\\pca_vgg19_X_test - Copy.csv"
# test_labels_file = "C:\\uni\\2 - second term\\Cloud Computing\\Project\\braintumor\\y_test - Copy.csv"
test_features_file = "C:\\uni\\2 - second term\\Cloud Computing\\Project\\eye_csv\\pca_X_test - Copy.csv"
test_labels_file = "C:\\uni\\2 - second term\\Cloud Computing\\Project\\eye_csv\\y_test - Copy.csv"

test_features = pd.read_csv(test_features_file)
test_labels = pd.read_csv(test_labels_file)

# Prepare the feature vectors and labels
X_train = train_features.values[:, 1:]
y_train = train_labels.values[:, 1:].ravel()

X_test = test_features.values[:, 1:]
y_test = test_labels.values[:, 1:].ravel()

data_preparation_time = time.time() - start_time

#%% Random Forest Classifier
print("Random Forest")

start_model = time.time()

# Create the random forest classifier
rf = RandomForestClassifier()

# Fit the classifier to the training data
rf.fit(X_train, y_train)

# Make predictions on the test data
predictions = rf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)

elapsed_time = time.time() - start_model
print("Elapsed time: {:.2f} seconds".format(data_preparation_time + elapsed_time))
 
#%% Gradient-Boosted Tree Classifier -- binary classification
print("Gradient-Boosted Tree")

start_model = time.time()

# Create the Gradient-Boosted Tree Classifier
gbt = GradientBoostingClassifier()

# Fit the classifier to the training data
gbt.fit(X_train, y_train)

# Make predictions on the test data
predictions = gbt.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)

elapsed_time = time.time() - start_model
print("Elapsed time: {:.2f} seconds".format(data_preparation_time + elapsed_time))

#%% Decision Tree Classifier
print("Decision Tree")

start_model = time.time()

# Create the Decision Tree Classifier
dt = DecisionTreeClassifier()

# Fit the classifier to the training data
dt.fit(X_train, y_train)

# Make predictions on the test data
predictions = dt.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)

elapsed_time = time.time() - start_model
print("Elapsed time: {:.2f} seconds".format(data_preparation_time + elapsed_time))

#%% Support Vector Machine Classifier -- binary only
print("Support Vector Machine")

start_model = time.time()

# Create the Support Vector Machine Classifier
svm = SVC()

# Fit the classifier to the training data
svm.fit(X_train, y_train)

# Make predictions on the test data
predictions = svm.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)

elapsed_time = time.time() - start_model
print("Elapsed time: {:.2f} seconds".format(data_preparation_time + elapsed_time))

#%% Multilayer Perceptron Classifier
print("Multilayer Perceptron")

start_model = time.time()

# Create the 
layers = [10, 5]  # Customize the hidden layer sizes as per your requirement
mlp = MLPClassifier(hidden_layer_sizes=layers)

# Fit the classifier to the training data
mlp.fit(X_train, y_train)

# Make predictions on the test data
predictions = mlp.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)

elapsed_time = time.time() - start_model
print("Elapsed time: {:.2f} seconds".format(data_preparation_time + elapsed_time))

#%% Logistic Regression Classifier
print("Logistic Regression")

start_model = time.time()

# Create the Logistic Regression Classifier
lr = LogisticRegression()

# Fit the classifier to the training data
lr.fit(X_train, y_train)

# Make predictions on the test data
predictions = lr.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)

elapsed_time = time.time() - start_model
print("Elapsed time: {:.2f} seconds".format(data_preparation_time + elapsed_time))

#%% Naive Bayes Classifier  -- nonnegative only
print("Naive Bayes")

start_model = time.time()

# Create the Naive Bayes Classifier
nb = GaussianNB()

# Fit the classifier to the training data
nb.fit(X_train, y_train)

# Make predictions on the test data
predictions = nb.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)

elapsed_time = time.time() - start_model
print("Elapsed time: {:.2f} seconds".format(data_preparation_time + elapsed_time))
