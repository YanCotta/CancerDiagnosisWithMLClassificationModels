# Kernel SVM
# Advanced SVM using kernel trick for non-linear classification
# Best used for: Non-linear classification problems
# Advantages: Can handle non-linear relationships, versatile kernel functions
# Limitations: Computationally intensive, kernel selection can be challenging

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Training the Kernel SVM model
# Kernel Options:
# - 'rbf': Radial Basis Function (most common)
# - 'poly': Polynomial kernel
# - 'sigmoid': Sigmoid kernel
# Parameters:
# - gamma: Kernel coefficient ('scale', 'auto' or float)
# - C: Regularization parameter
from sklearn.svm import SVC
classifier = SVC(
    kernel='rbf',
    random_state=0,
    # gamma='scale',  # Uncomment to adjust kernel coefficient
    # C=1.0          # Uncomment to adjust regularization
)
classifier.fit(X_train, y_train)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)