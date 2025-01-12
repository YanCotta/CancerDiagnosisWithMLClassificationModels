# Decision Tree Classification
# A tree-structured classifier where nodes represent decisions
# Best used for: Problems requiring interpretable results
# Advantages: Easy to understand, handles non-linear relationships
# Limitations: Can overfit, unstable (small changes can give different trees)

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

# Training Decision Tree
# Parameters explained:
# - criterion: 'entropy' or 'gini' for split quality
# - max_depth: Controls tree complexity
# - min_samples_split: Min samples before splitting
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(
    criterion='entropy',
    random_state=0
    # max_depth=3,  # Uncomment to limit tree depth
    # min_samples_split=5  # Uncomment to control splitting
)
classifier.fit(X_train, y_train)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)

# Optional: Visualize the decision tree
# from sklearn.tree import export_graphviz
# import graphviz
# dot_data = export_graphviz(classifier, filled=True, rounded=True,
#                           feature_names=feature_names,  # Add your feature names
#                           class_names=['0', '1'])  # Add your class names
# graph = graphviz.Source(dot_data)
# graph.render("decision_tree_visualization", view=True)