"""
This file re-runs the codes from the KNN-classifier in Chapter 1 for iris dataset
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

iris_data = load_iris()
print(iris_data.keys())
# data = X covariate values
# feature_names = column names of X
# target = y classification
# DESCR = dataset description
# target_names = names of classified features for 0,1,2 in target
# filename, data_module = just specifications of the file

X = iris_data['data']
y = iris_data['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True,  stratify=y, random_state=123)

iris_df = pd.DataFrame(X_train, columns=iris_data["feature_names"])
#group_plot = pd.plotting.scatter_matrix(iris_df, c = y_train)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

print(knn.score(X_test, y_test))
y_pred = knn.predict(X_test)
print("Predicted values: ", y_pred)
print("Observed values: ", y_test)

## Input from the book
X_new = np.array([[5, 2.9,1,0.2]])
print("Output for the new prediction observation: ", iris_data['target_names'][knn.predict(X_new)])

## My own input check
X_my = np.array([[9,3,5.8,2.1], [5.2, 3.4,1.5,0.3]])
print("Output for my check-up: ", iris_data['target_names'][knn.predict(X_my)])