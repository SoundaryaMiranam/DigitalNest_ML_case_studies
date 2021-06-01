#import the files required
from google.colab import files
import io
uploaded =files.upload()
for fn in uploaded.keys():
   print('User uploaded file "{name}" with length {length} bytes'.format(name=fn, length=len(uploaded[fn])))

#import the files required
from google.colab import files
import io
upload =files.upload()
for fn in upload.keys():
   print('User uploaded file "{name}" with length {length} bytes'.format(name=fn, length=len(upload[fn])))

# Commented out IPython magic to ensure Python compatibility.
#Load the required libraries 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns

test_data = pd.read_csv('test_set.csv')
print("Rows, columns: " + str(test_data.shape))

Loan_prediction_data = pd.read_csv('train_set.csv')
print("Rows, columns: " + str(Loan_prediction_data.shape))

Loan_prediction_data.head()

Loan_prediction_data.dtypes

Loan_prediction_data.isnull().sum().sort_values(ascending = False)

missing = Loan_prediction_data.isnull().sum()
missing = missing[missing > 0]
missing.sort_values(inplace=True)
missing.plot.bar()
print("Number of attributes having missing values " + str(len(missing)))

Loan_prediction_data.describe()

"""#Data cleaning """

print(Loan_prediction_data['Gender'].value_counts())
Loan_prediction_data['Gender'].fillna('Male', inplace=True)

print(Loan_prediction_data['Married'].value_counts())
Loan_prediction_data['Married'].fillna('Yes', inplace=True)

print(Loan_prediction_data['Dependents'].value_counts())
Loan_prediction_data['Dependents'].fillna('0', inplace=True)

print(Loan_prediction_data['Self_Employed'].value_counts())
Loan_prediction_data['Self_Employed'].fillna('NO', inplace=True)

print(Loan_prediction_data['Credit_History'].value_counts())
Loan_prediction_data.Credit_History.fillna(1.0, inplace=True)

Loan_prediction_data.isnull().sum().sort_values(ascending = False)

print(Loan_prediction_data['Loan_Amount_Term'].value_counts())

print("Median of 'Loan_Amount_Term':",Loan_prediction_data['Loan_Amount_Term'].median())
print("Mode of 'Loan_Amount_Term':",Loan_prediction_data['Loan_Amount_Term'].mode())

Loan_prediction_data['Loan_Amount_Term'].fillna('360.0', inplace=True)

"""The most occurring value is 360 which is nothing but 30 years.There is no difference between median and mode values.Here it replaced with value (360.0). """

Loan_prediction_data["LoanAmount"] = Loan_prediction_data["LoanAmount"].replace(np.nan,Loan_prediction_data["LoanAmount"].mean())

Loan_prediction_data['LoanAmount'].hist(bins=50)
plt.show()

""" Loan_ID should be unique. So if there n number of rows, there should be n number of unique Loan_ID’s. Let us check for that. If there are any duplicate values we can remove that."""

Loan_prediction_data.apply(lambda x: len(x.unique()))

"""#Exploratory data analysis

1) Univariate Analysis

numercial data - box plot and histogram
"""

Loan_prediction_data.boxplot(column='ApplicantIncome')
plt.show()

Loan_prediction_data['ApplicantIncome'].hist(bins=50)
plt.show()

Loan_prediction_data['LoanAmount'].hist(bins=50)
plt.show()

Loan_prediction_data.boxplot(column='LoanAmount')
plt.show()

"""there are outliers in both the columns.

2) bivarient analysis

categorical data - count plot
"""

sns.countplot(y ='Gender' , hue = 'Loan_Status', data = Loan_prediction_data)
plt.show()

sns.countplot(y ='Married' , hue = 'Loan_Status', data = Loan_prediction_data)
plt.show()

sns.countplot(y ='Self_Employed' , hue = 'Loan_Status', data = Loan_prediction_data)
plt.show()

sns.countplot(y ='Credit_History' , hue = 'Loan_Status', data = Loan_prediction_data)
plt.show()

sns.countplot(y ='Property_Area' , hue = 'Loan_Status', data = Loan_prediction_data)
plt.show()

sns.countplot(y ='Loan_Amount_Term' , hue = 'Loan_Status', data = Loan_prediction_data)
plt.show()

"""1) More males tend to take loan than females.

2) Married people are more on loan than unmarried people.

3) Self-employed people take less loans than those are not self-employed.

4) credit history shows that high number of people pay back their loans.

5) Semiurban obtain more loan, folowed by Urban and then rural. This is logical!

6) Most of the people opt for 360 cyclic loan term which is pay back within a year of time.
"""

grid = sns.FacetGrid(Loan_prediction_data, row = 'Gender', col = 'Married', height = 3.5, aspect=1.8)
grid.map_dataframe(plt.hist, 'ApplicantIncome')
grid.set_axis_labels('ApplicantIncome', 'Count')
plt.show()

grid = sns.FacetGrid(Loan_prediction_data, row = 'Gender', col = 'Dependents', height = 3.5, aspect=1.8)
grid.map_dataframe(plt.hist, 'ApplicantIncome')
grid.set_axis_labels('ApplicantIncome', 'Count')
plt.show()

grid = sns.FacetGrid(Loan_prediction_data, row = 'Gender', col = 'Education', height = 3.5, aspect=1.8)
grid.map_dataframe(plt.hist, 'ApplicantIncome')
grid.set_axis_labels('ApplicantIncome', 'Count')
plt.show()

grid = sns.FacetGrid(Loan_prediction_data, row = 'Married', col = 'Dependents', height = 3.5, aspect=1.8)
grid.map_dataframe(plt.hist, 'ApplicantIncome')
grid.set_axis_labels('ApplicantIncome', 'Count')
plt.show()

grid = sns.FacetGrid(Loan_prediction_data, row = 'Married', col = 'Education', height = 3.5, aspect=1.8)
grid.map_dataframe(plt.hist, 'ApplicantIncome')
grid.set_axis_labels('ApplicantIncome', 'Count')
plt.show()

grid = sns.FacetGrid(Loan_prediction_data, row = 'Married', col = 'Credit_History', height = 3.5, aspect=1.8)
grid.map_dataframe(plt.hist, 'ApplicantIncome')
grid.set_axis_labels('ApplicantIncome', 'Count')
plt.show()

grid = sns.FacetGrid(Loan_prediction_data, row = 'Education', col = 'Dependents', height = 3.5, aspect=1.8)
grid.map_dataframe(plt.hist, 'ApplicantIncome')
grid.set_axis_labels('ApplicantIncome', 'Count')
plt.show()

grid = sns.FacetGrid(Loan_prediction_data, row = 'Education', col = 'Credit_History', height = 3.5, aspect=1.8)
grid.map_dataframe(plt.hist, 'ApplicantIncome')
grid.set_axis_labels('ApplicantIncome', 'Count')
plt.show()

grid = sns.FacetGrid(Loan_prediction_data, row = 'Self_Employed', col = 'Dependents', height = 3.5, aspect=1.8)
grid.map_dataframe(plt.hist, 'ApplicantIncome')
grid.set_axis_labels('ApplicantIncome', 'Count')
plt.show()

grid = sns.FacetGrid(Loan_prediction_data, row = 'Self_Employed', col = 'Education', height = 3.5, aspect=1.8)
grid.map_dataframe(plt.hist, 'ApplicantIncome')
grid.set_axis_labels('ApplicantIncome', 'Count')
plt.show()

grid = sns.FacetGrid(Loan_prediction_data, row = 'Property_Area', col = 'Dependents', height = 3.5, aspect=1.8)
grid.map_dataframe(plt.hist, 'ApplicantIncome')
grid.set_axis_labels('ApplicantIncome', 'Count')
plt.show()

"""1. Males generally have the highest income. Explicitly, Males that are married have greater income that unmarried male.
2. No one is dependent and a male tremendously has more income. 
3. A graduate who is a male has more income.
4. Not married and no one is dependent on such has more income. Also, Married and no one dependent has greater income with a decreasing effect as the dependents increases.
5. A graduate and married individual has more income.
6. Married and has a good credit history depicts more income. Also, Not married but has a good credit history follows in the hierarchy.
7. A graduate with no one dependent has more income.
8. Educated with good credit history depicts a good income. Also, not a graduate and have a good credit history can be traced to having a better income than a fellow with no degree
9. No one is dependent and self-employed has more income
10. A graduate but not self-employed has more income.
11. No one is dependent and have property in urban, rural and semiurban has more income.

Correlation matrix plot
"""

sns.heatmap(Loan_prediction_data.corr())
plt.show()

"""#Machine learning"""

#drop all the object types features
Loan_prediction_data = Loan_prediction_data.drop(['Loan_ID'], axis=1)

Loan_prediction_data.columns

from sklearn.preprocessing import LabelEncoder
 var_mod = ['Gender','Married','Dependents','Education','Self_Employed','Property_Area','Loan_Status']
 le = LabelEncoder()
 for i in var_mod:
     Loan_prediction_data[i] = le.fit_transform(Loan_prediction_data[i])
 Loan_prediction_data.dtypes

# Splitting traing data
X = Loan_prediction_data.drop('Loan_Status', axis=1)
y = Loan_prediction_data.Loan_Status

X

y

test_data.isnull().sum().sort_values(ascending = False)

print(test_data['Gender'].value_counts())
test_data['Gender'].fillna('Male', inplace=True)

print(test_data['Dependents'].value_counts())
test_data['Dependents'].fillna('0', inplace=True)

print(test_data['Self_Employed'].value_counts())
test_data['Self_Employed'].fillna('NO', inplace=True)

print(test_data['Credit_History'].value_counts())
test_data.Credit_History.fillna(1.0, inplace=True)

print(test_data['Loan_Amount_Term'].value_counts())

print("Median of 'Loan_Amount_Term':",test_data['Loan_Amount_Term'].median())
print("Mode of 'Loan_Amount_Term':",test_data['Loan_Amount_Term'].mode())

test_data['Loan_Amount_Term'].fillna('360.0', inplace=True)

test_data["LoanAmount"] = test_data["LoanAmount"].replace(np.nan,test_data["LoanAmount"].mean())

test = test_data.drop(['Loan_ID'], axis=1)

var_mod = ['Gender','Married','Dependents','Education','Self_Employed','Property_Area']
 le = LabelEncoder()
 for i in var_mod:
     test[i] = le.fit_transform(test[i])
 test.dtypes

# Splitting the dataset into the Training set and Test set
 from sklearn.model_selection import train_test_split
 X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 1/3, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_val = sc.fit_transform(X_val)

# Fitting LogisticRegression to the Training set
from sklearn.linear_model import LogisticRegression
LR_classifier = LogisticRegression(random_state = 0)
LR_classifier.fit(X_train, y_train)
y_pred = LR_classifier.predict(X_val)

# Measuring Accuracy
from sklearn import metrics
print('The accuracy of Logistic Regression is: ', metrics.accuracy_score(y_pred, y_val))

# Fitting K-NN to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_val)

# Measuring Accuracy
from sklearn import metrics
print('The accuracy of KNN is: ', metrics.accuracy_score(y_pred, y_val))

# Fitting SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_val)

# Measuring Accuracy
from sklearn import metrics
print('The accuracy of SVM is: ', metrics.accuracy_score(y_pred, y_val))

# Fitting Decision Tree Classification to the Training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_val)

# Measuring Accuracy
from sklearn import metrics
print('The accuracy of Decision Tree Classifier is: ', metrics.accuracy_score(y_pred, y_val))

# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_val)

# Measuring Accuracy
from sklearn import metrics
print('The accuracy of Random Forest Classification is: ', metrics.accuracy_score(y_pred, y_val))

"""Training dataset results:

The accuracy of Logistic Regression is: 82.43 %

The accuracy of KNN is: 79.51 %

The accuracy of SVM is: 81.95 %

The accuracy of Decision Tree Classifier is: 72.19 %

The accuracy of Random Forest Classification is: 78.53 %

#Testing
"""

# predict on new set with logistic regression
prediction = LR_classifier.predict(test)

test_data['Loan_Status_Prediction'] = prediction 
test_data['Loan_Status_Prediction'] = test_data['Loan_Status_Prediction'].map({1: 'Yes', 0: 'No'})
test_data.head(10)
