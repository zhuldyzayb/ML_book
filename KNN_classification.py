"""
This file explores the KNN classifier in scikit-learn through running the algorithm
and reconstructing the version of it by myrself.
The dataset used was taken from kaggle.
1. First, I begin with uploading the data and cleaning it.
2. I run the knn algorithm from scikit-learn and recreate the version of it
    by myself for comparison and better understanding.
3. Note that here I am assuming that the dataset is fully numeric
    (even if there are categorical variables, we assume they are converted to integers)
4. In the distance function I have implemented Minkowski distance for
    it is the default in KNeighborsClassifier.
5. First I fix the value of k = (10,5) and calculate score of the classification algorithm
    then I use elbow method and k-fold cross-validation to find the suitable k. For this, I
    simply use the KNeighborsClassifier function as it is faster than my rough implementation.
6. Note that here I have not performed any variable selection, I am just assuming that the
    variables are all statistically significant. I do understand that it is not necessarily
    the case, but the purpose here is to understand the technique behind the knn and not
    do a full model prediction (already familiar from linear regression).
7. Last but not least, I just scooped all the codes here together, but for better
    understanding it is better to run each block separately
"""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV

## 1. Data cleaning
df = pd.read_csv("breast-cancer-wisconsin-data.txt", header=None)
col_names = ['Id', 'Clump_thickness', 'Uniformity_Cell_Size', 'Uniformity_Cell_Shape', 'Marginal_Adhesion',
             'Single_Epithelial_Cell_Size', 'Bare_Nuclei', 'Bland_Chromatin', 'Normal_Nucleoli', 'Mitoses', 'Class']
df.columns = col_names
print(df.describe())
df['Bare_Nuclei'] = pd.to_numeric(df['Bare_Nuclei'], errors='coerce')
df = df.drop('Id', axis=1)
print(df.isna().sum())
df = df.dropna()
X = df.drop("Class", axis=1)
y = df['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=123, test_size=0.3)

## 2. Model run
k=5

## Scikit-learn version
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
print("The test score for knn is: ", knn.score(X_test, y_test))

## My version
def my_dist(point1, point2):
    p = len(point1)
    if len(point2)!=p:
        print("Error: the length of vectors must match.")
    else:
        dist = (abs(point1 - point2))**p
        return (pow( sum(dist), 1/p))

def my_knn_predict(train_data, train_labels, test_point, k):
    btw_dist = []
    for i in range(len(train_data)):
        btw_dist.append(my_dist(test_point, train_data.iloc[i]))
    sorted_dist = sorted(zip(btw_dist, train_labels), key=lambda x: x[0])
    knearest_labels = [pair[1] for pair in sorted_dist][:k]
    return Counter(knearest_labels).most_common()[0][0]

y_pred = []
for i in range(len(X_test)):
    y_pred.append(my_knn_predict(X_train, y_train, X_test.iloc[i], k))

my_score = sum(y1 == y2 for y1, y2 in zip(y_pred, y_test))/len(y_pred)
print("Test score for my version is: ", my_score)
## 96.09 with KNN and 97.07 with my alg for k=10
## 97.56 for both KNN and my alg with k=5

## 3. Find suitable k

## Elbow method: k vs error_rate
error_rate = []
for i in range(1,21):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred = knn.predict(X_test)
    error_rate.append(np.mean(pred!=y_test))
res = pd.DataFrame({'k':range(1,21), 'error':error_rate})
print(res)
#res.plot('error')
#plt.show()
## lowest error rate seems t be at k=5

## Cross validation:
# fix the same range as for elbow method; here I use 5 blocks for CV
params = {'n_neighbors': range(1, 21)}
cv = KFold(n_splits=5, shuffle=True, random_state=123)
grid_search = GridSearchCV(KNeighborsClassifier(), params, cv=cv)
grid_search.fit(X_train, y_train)
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_:.4f}")