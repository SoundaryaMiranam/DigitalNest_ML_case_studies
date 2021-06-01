#Load the required libraries 
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

red_wine = pd.read_csv('winequality-red.csv')

print("Rows, columns: " + str(red_wine.shape))

red_wine.isna().sum()

red_wine.describe()

"""There is notably a large difference between 75th %tile and max values of predictors “residual sugar”,”free sulfur dioxide”,”total sulfur dioxide”.

#Box plot for red wine quality data
"""

fig = plt.figure(figsize=(24,10))
features = ["total sulfur dioxide", "residual sugar", "volatile acidity", "free sulfur dioxide",
            "chlorides", "fixed acidity", "citric acid","sulphates","density","alcohol"]
for i in range(10):
    ax1 = fig.add_subplot(2,5,i+1)
    sns.barplot(x='quality', y=features[i],data=red_wine)

"""1.   fixed acidity does not give any specification to classify the quality.
1.   there is no significant effect of residual sugar on quality of wine.
2.   downing trend in the volatile acidity as we go higher the quality.
2.    decreasing trend of chlorides with the increase in the quality of wine.
3.    increasing trend of citric acid. That is, as we go higher in quality of wine the composition of citric acid in wine also increases.
3.     the increasing trend of sulphates as we go higher in quality of wine.

4.   Both the free sulphur dioxide and total sulphur dioxide are comparatively more in the 5th and 6th quality wine

#Correlation matrix plot for red wine quality data
"""

corr = red_wine.corr()
sns.heatmap(corr , xticklabels=corr.columns.values , yticklabels=corr.columns.values)
plt.show()

"""1.   “density” has strong positive correlation with “residual sugar” whereas it has strong negative correlation with “alcohol”.
2.   “free sulphur dioxide” and “citric acid” has almost no correlation with “quality”.
3.  'alcohol, sulphates & fixed_acidity' has positive corelation with 'quality'.
"""

sns.countplot(x='quality',data=red_wine)

red_wine.quality.unique()

red_wine.quality.value_counts()

correlations = red_wine.corr()['quality'].sort_values(ascending=False)
correlations.plot(kind='bar')

print(abs(correlations) > 0.2)

"""From all the values,alcohol, sulphates, citric_acid and volatile_acidity which effect the quality of wine.So, these are helpful during feature selection."""

#0 ==> 3, 4, 5 ==>Bad
#1 ==> 6, 7, 8 ==> Good

bins_ = (2,6,8)
labels_ = ['bad','good']
red_wine['quality']=pd.cut(red_wine['quality'],bins=bins_,labels=labels_)

from sklearn.preprocessing import LabelEncoder
labelencoder_y = LabelEncoder()
red_wine['quality'] = labelencoder_y.fit_transform(red_wine['quality'].astype(str))

red_wine.quality.unique()

sns.countplot(red_wine['quality'])

"""#Machine Learning on red wine quality data"""

from sklearn.model_selection import train_test_split, cross_val_score
X = red_wine.drop('quality',axis = 1)
Y = red_wine['quality']

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)

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

"""
Observations:Looked over the key factors that determine and affects the quality of the red wine.The vast majority of wines get a quality rating of five or six,while the dataset must be more populated with bad wines (>4) to study and make predications.There seem not to be any excellent wines (>8) on this dataset.

Random forest classifier model has around 90% accuracy, which is  better than the other models.This analysis helps to understand by modifying the variables, it is possible to increase the quality of the wine in the market to obtain more profits."""
