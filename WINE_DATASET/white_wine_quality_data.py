#import the files required
from google.colab import files
import io
uploaded =files.upload()
for fn in uploaded.keys():
   print('User uploaded file "{name}" with length {length} bytes'.format(name=fn, length=len(uploaded[fn])))

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

white_wine = pd.read_csv('winequality-white.csv', sep=';')
print("Rows, columns: " + str(white_wine.shape))

white_wine.sample(5)

white_wine.isna().sum()

white_wine.dtypes

"""#Box plot for white wine quality data"""

fig = plt.figure(figsize=(24,10))
features = ["total sulfur dioxide", "residual sugar", "volatile acidity", "free sulfur dioxide",
            "chlorides", "fixed acidity", "citric acid","sulphates","density","alcohol"]
for i in range(10):
    ax1 = fig.add_subplot(2,5,i+1)
    sns.barplot(x='quality', y=features[i],data=white_wine)

"""1.   As chlorides level decreases, quality increases
2.  Fixed acidity and sulphates has no impact on Quality
3.  As free sulfur dioxide increases, quality increases

#Correlation matrix plot for white wine quality data
"""

corr = white_wine.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr, annot=True, vmin=-1.0, vmax=1.0)
plt.show()

"""
1. PH value has a negative relationship with fixed acidity.
2.  Alcohol has a negative relationship with density, residual sugar, and total sulfur dioxide.
3.  Citric acid and Free Sulfur Dioxide has almost no impact to quality. 
4.  Density has a negative relationship with quality.
5.  Density has a positive relationship with residual sugar.
6.  Alcohol and sulfate have a positive relationship with quality.
7.  Free sulfur and total sulfur also have a positive relationship.
"""

white_wine['quality'].unique()

#Getting an idea about the distribution of wine quality 
p = sns.countplot(data=white_wine, x = 'quality')

correlations = white_wine.corr()['quality'].sort_values(ascending=False)
correlations.plot(kind='bar')

print(correlations)

print(abs(correlations) > 0.1)

"""Target variable "quality" (range from 3 to 9), values 5 and 6 make up more than 60% of the dataset indicating that this class is unbalanced.

quality >= 7 as good

quality < 7 as bad
"""

white_wine['quality'] = ['Good' if x>=7 else 'bad' for x in white_wine['quality']]
white_wine.head()

from sklearn.preprocessing import LabelEncoder
labelencoder_y = LabelEncoder()
white_wine['quality'] = labelencoder_y.fit_transform(white_wine['quality'].astype(str))

white_wine.quality.unique()

sns.countplot(white_wine['quality'])

from sklearn.model_selection import train_test_split, cross_val_score
X = white_wine.drop('quality',axis = 1)
Y = white_wine['quality']

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 42)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
lr = LogisticRegression()
lr.fit(X_train , Y_train)
lr_predict = lr.predict(X_test)
lr_accuracy_score = accuracy_score(Y_test, lr_predict)
LR = (lr_accuracy_score*100)

from sklearn.neighbors import KNeighborsClassifier
classifier_knn = KNeighborsClassifier()
classifier_knn.fit(X_train , Y_train)
knn_predict = classifier_knn.predict(X_test)
knn_accuracy_score = accuracy_score(Y_test, knn_predict)
KNN = (knn_accuracy_score*100)

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(X_train,Y_train)
dt_predict = dt.predict(X_test)
dt_accuracy_score = accuracy_score(Y_test, dt_predict)
DT = (dt_accuracy_score*100)

from sklearn.svm import SVC
svc = SVC()
svc.fit(X_train, Y_train)
svc_predict = svc.predict(X_test)
svc_accuracy_score = accuracy_score(Y_test, svc_predict)
SVM = (svc_accuracy_score*100)

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(X_train, Y_train)
rf_predict = rf.predict(X_test)
rf_accuracy_score = accuracy_score(Y_test, rf_predict)
RN = (rf_accuracy_score*100)

result = pd.DataFrame({
    'MODEL': ['Support Vector Machines', 'KNN', 'Logistic Regression',
              'Random Forest','Decision Tree'],
    'SCORE': [SVM, KNN, LR,RN, DT]})
result = result.sort_values(by='SCORE', ascending=False)
result = result.set_index('SCORE')
result.head(5)

new_wine_metrics = [[7.8, 0.60, 0.04, 2.3, 0.092, 15.0, 54.0,0.44,0.67,50.24, 0.79700]]
print("Decision Tree : ",dt.predict(new_wine_metrics))
print("Logistic Regression : ",lr.predict(new_wine_metrics))
print("KNN : ",classifier_knn.predict(new_wine_metrics))
print("Random forest : ",rf.predict(new_wine_metrics))
print("SVM : ",svc.predict(new_wine_metrics))

"""Observations:Although Logistic Regression did wrong prediction.Easily it classifies that weather a wine is good or bad in quality using Random forest algorithm."""
