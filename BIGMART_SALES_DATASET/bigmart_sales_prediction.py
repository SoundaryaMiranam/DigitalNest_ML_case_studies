#Load the required libraries 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns

train_data = pd.read_csv('Train.csv')
print("Shape of train data - (Rows, columns): " + str(train_data.shape))

test_data = pd.read_csv('Test.csv')
print("Shape of test data - (Rows, columns): " + str(test_data.shape))

test_set = test_data.copy()

train_data.head()

train_data.info()

train_data.describe()

train_data.isnull().sum()/len(train_data)*100

# Plotting the percentage of missing values
total = train_data.isnull().sum().sort_values(ascending = False)
percent_total = (train_data.isnull().sum()/train_data.isnull().count()).sort_values(ascending=False)*100
missing = pd.concat([total, percent_total], axis=1, keys=["Total", "Percentage"])
missing_data = missing[missing['Total']>0]

plt.figure(figsize=(5,5))
sns.set(style="whitegrid")
sns.barplot(x=missing_data.index, y=missing_data['Percentage'], data = missing_data)
plt.title('Percentage of missing data by feature')
plt.xlabel('Features', fontsize=14)
plt.ylabel('Percentage', fontsize=14)
plt.show()

"""summary:
1. Item_Visibility has a min value of zero,which isn't possible because when a product is being sold at a store, the visibility cannot be 0.
2. Outlet_Establishment_Years vary from 1985 to 2009. The values might not be apt in this form. Rather, if we can convert them to how old the particular store is, it should have a better impact on sales.
3. There are some missing values which have to be treated.
4. Cleaning Item_Fat_Content column with two groups as low and regular.
5. Categorizating  Item_Visibility with 'Low Viz', 'Viz' and 'High Viz'.

#Exploratory data analysis
"""

num_features = train_data.select_dtypes(include=[np.number])
num_features.dtypes
#out of 12 there are only 5 numeric variables.

"""1. Univariate Analysis

a) Numerical features:
"""

corr=num_features.corr()
corr
corr['Item_Outlet_Sales'].sort_values(ascending=False)

"""b) Categorical features:"""

sns.countplot(x ='Item_Fat_Content', data = train_data)
plt.show()

sns.countplot(x ='Item_Type', data = train_data)
plt.xticks(rotation=90)
plt.show()

sns.countplot(x ='Outlet_Size', data = train_data)
plt.show()

sns.countplot(x ='Outlet_Location_Type', data = train_data)
plt.show()

sns.countplot(x ='Outlet_Type', data = train_data)
plt.xticks(rotation=90)
plt.show()

"""2. Bivariate Analysis

a) Numerical features:
"""

plt.figure(figsize=(12,7))
plt.xlabel("Item_Weight")
plt.ylabel("Item_Outlet_Sales")
plt.title("Item_Weight and Item_Outlet_Sales Analysis")
plt.plot(train_data.Item_Weight, train_data["Item_Outlet_Sales"],'.', alpha = 0.3)
plt.show()

Outlet_Establishment_Year_pivot = train_data.pivot_table(index='Outlet_Establishment_Year', values="Item_Outlet_Sales", aggfunc=np.median)
Outlet_Establishment_Year_pivot.plot(kind='bar', color='blue',figsize=(12,7))
plt.xlabel("Outlet_Establishment_Year")
plt.ylabel("Sqrt Item_Outlet_Sales")
plt.title("Impact of Outlet_Establishment_Year on Item_Outlet_Sales")
plt.show()

"""b) Categorical features:"""

train_data.pivot_table(index='Item_Fat_Content', values="Item_Outlet_Sales", aggfunc=np.median)

Item_Fat_Content_pivot = train_data.pivot_table(index='Item_Fat_Content', values="Item_Outlet_Sales", aggfunc=np.median)
Item_Fat_Content_pivot.plot(kind='bar', color='blue',figsize=(12,7))
plt.xlabel("Item_Fat_Content")
plt.ylabel("Item_Outlet_Sales")
plt.title("Impact of Item_Fat_Content on Item_Outlet_Sales")
plt.show()

train_data.pivot_table(values='Outlet_Type', columns='Outlet_Identifier',aggfunc=lambda x:x.mode())

Outlet_Identifier_pivot = train_data.pivot_table(index='Outlet_Identifier', values="Item_Outlet_Sales", aggfunc=np.median)
Outlet_Identifier_pivot.plot(kind='bar', color='blue',figsize=(12,7))
plt.xlabel("Outlet_Identifier")
plt.ylabel("Item_Outlet_Sales")
plt.title("Impact of Outlet_Identifier on Item_Outlet_Sales")
plt.show()

train_data.pivot_table(values='Outlet_Type',columns='Outlet_Size',aggfunc=lambda x:x.mode())

Outlet_Size_pivot = train_data.pivot_table(index='Outlet_Size', values="Item_Outlet_Sales", aggfunc=np.median)
Outlet_Size_pivot.plot(kind='bar', color='blue',figsize=(12,7))
plt.xlabel("Outlet_Size")
plt.ylabel("Item_Outlet_Sales")
plt.title("Impact of Outlet_Size on Item_Outlet_Sales")
plt.show()

Outlet_Location_Type_pivot = train_data.pivot_table(index='Outlet_Location_Type', values="Item_Outlet_Sales", aggfunc=np.median)
Outlet_Location_Type_pivot.plot(kind='bar', color='blue',figsize=(12,7))
plt.xlabel("Outlet_Location_Type")
plt.ylabel("Item_Outlet_Sales")
plt.title("Impact of Outlet_Location_Type on Item_Outlet_Sales")
plt.show()

"""#Data Pre-processing"""

# Filling the null values with the mean value
train_data['Item_Weight'] = train_data['Item_Weight'].fillna(train_data['Item_Weight'].mean())

print(train_data['Outlet_Size'].unique())

print(train_data['Outlet_Size'].value_counts())

var = train_data.pivot_table(values = 'Outlet_Size', columns = 'Outlet_Type', aggfunc = (lambda x:x.mode()))
var

#Filling the null values with the 'medium' value from the above table
train_data['Outlet_Size'] = train_data['Outlet_Size'].fillna('Medium')

train_data.isnull().sum().sort_values(ascending = False)

print(train_data['Outlet_Establishment_Year'].unique())

#The data is from 2013 
train_data['Outlet_Age'] = 2013 - train_data['Outlet_Establishment_Year']
train_data.head()

print(train_data['Item_Fat_Content'].unique())

train_data['Item_Fat_Content'] = train_data['Item_Fat_Content'].replace(['low fat', 'LF'], 'Low Fat')
train_data['Item_Fat_Content'] = train_data['Item_Fat_Content'].replace('reg', 'Regular')

train_data['Item_Fat_Content'].value_counts()

train_data['Item_Visibility'].value_counts()

train_data['Item_Visibility'].hist(bins=20)
plt.show()

# The minimum value of the item visibility feature is zero(0)
# Replace the minimum value with the 2nd minimum value of the feature, as item visibility cannot be zero
train_data['Item_Visibility'] = train_data['Item_Visibility'].replace(0.000000,0.003574698)

train_data['Item_Visibility_bins'] = pd.cut(train_data['Item_Visibility'], [0.000, 0.065, 0.13, 0.2], labels=['Low Viz', 'Viz', 'High Viz'])

train_data['Item_Visibility_bins'].value_counts()

train_data['Item_Visibility_bins'] = train_data['Item_Visibility_bins'].fillna('Low Viz')

train_data['Item_Visibility_bins'].value_counts()

train_data.isnull().sum().sort_values(ascending = False)

train_data['Item_Identifier'].value_counts(50)

"""These are the three catagories how the items are identified ['FD':'Food','NC':'Non-Consumable','DR':'Drinks']"""

train_data['Item_Type'].value_counts()

#grouping Item Id
train_data['Item_Id']=train_data['Item_Identifier'].str[:2]
train_data.groupby(['Item_Id','Item_Type'])['Item_Outlet_Sales'].count()

#Get the first two characters of ID:
train_data['Item_Type_Combined'] = train_data['Item_Identifier'].apply(lambda x: x[0:2])
#Rename them to more intuitive categories:
train_data['Item_Type_Combined'] = train_data['Item_Type_Combined'].map({'FD':'Food','NC':'Non-Consumable','DR':'Drinks'})
train_data['Item_Type_Combined'].value_counts()

from sklearn.preprocessing import LabelEncoder
var_mod = ['Item_Fat_Content','Outlet_Size','Outlet_Location_Type','Item_Visibility_bins','Item_Type_Combined']
le = LabelEncoder()
for i in var_mod:
     train_data[i] = le.fit_transform(train_data[i])
train_data.dtypes

#create dummies for outlet type
dummy = pd.get_dummies(train_data['Outlet_Type'])
dummy.head()

train_data = pd.concat([train_data, dummy], axis=1)

# got to drop all the object types features
train_data = train_data.drop(['Item_Identifier', 'Item_Type', 'Outlet_Identifier', 'Outlet_Type','Outlet_Establishment_Year','Item_Id'], axis=1)

train_data.columns

"""#Data Pre-processing - test set"""

train_data.head()

test_data.isnull().sum()/len(test_data)*100

test_data['Item_Weight'] = test_data['Item_Weight'].fillna(test_data['Item_Weight'].mean())

print(test_data['Outlet_Size'].unique())

print(test_data['Outlet_Size'].value_counts())

test_data['Outlet_Size'] = test_data['Outlet_Size'].fillna('Medium')

print(test_data['Outlet_Establishment_Year'].unique())

test_data['Outlet_Age'] = 2021 - test_data['Outlet_Establishment_Year']
test_data.head()

test_data.isnull().sum().sort_values(ascending = False)

print(test_data['Item_Fat_Content'].unique())

test_data['Item_Fat_Content'] = test_data['Item_Fat_Content'].replace(['low fat', 'LF'], 'Low Fat')

test_data['Item_Fat_Content'] = test_data['Item_Fat_Content'].replace('reg', 'Regular')

test_data['Item_Fat_Content'].value_counts()

test_data['Item_Visibility'].hist(bins=20)
plt.show()

test_data['Item_Visibility'] = test_data['Item_Visibility'].replace(0.000000,0.003574698)

test_data['Item_Visibility_bins'] = pd.cut(test_data['Item_Visibility'], [0.000, 0.065, 0.13, 0.2], labels=['Low Viz', 'Viz', 'High Viz'])

test_data['Item_Visibility_bins'].isnull().sum()

test_data['Item_Visibility_bins'] = test_data['Item_Visibility_bins'].fillna('Low Viz')

test_data.isnull().sum().sort_values(ascending = False)

test_data['Item_Type_Combined'] = test_data['Item_Identifier'].apply(lambda x: x[0:2])

test_data['Item_Type_Combined'] = test_data['Item_Type_Combined'].map({'FD':'Food','NC':'Non-Consumable','DR':'Drinks'})
test_data['Item_Type_Combined'].value_counts()

from sklearn.preprocessing import LabelEncoder
var_mod = ['Item_Fat_Content','Outlet_Size','Outlet_Location_Type','Item_Visibility_bins','Item_Type_Combined']
le = LabelEncoder()
for i in var_mod:
     test_data[i] = le.fit_transform(test_data[i])
test_data.dtypes

dummy_var = pd.get_dummies(test_data['Outlet_Type'])
dummy_var.head()

test_data = pd.concat([test_data, dummy_var], axis=1)

test_data = test_data.drop(['Item_Identifier', 'Item_Type', 'Outlet_Identifier', 'Outlet_Type','Outlet_Establishment_Year'], axis=1)

test_data.columns

test_data.head()

test_data.isnull().sum().sort_values(ascending = False)

X = train_data.drop('Item_Outlet_Sales', axis=1)
y = train_data.Item_Outlet_Sales

"""#Machine learning"""

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 1/3, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_val = sc.fit_transform(X_val)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error

reg_lin=LogisticRegression(max_iter=2000)
reg_lin.fit(X_train,y_train)
reg_lin.score(X_train,y_train)

reg_lin_predict = reg_lin.predict(X_val)

rmse_lin=np.sqrt(mean_squared_error(reg_lin_predict,y_val))
print('RMSE for Linear Regression:{0:.2f}'.format(rmse_lin))


accuracy_lin = reg_lin.score(X_val,y_val)
print('Accuracy of the Linear Regression model:',accuracy_lin*100,'%' )

from sklearn.linear_model import RidgeCV
reg_rid=RidgeCV(cv=10)
reg_rid.fit(X_train,y_train)
reg_rid.score(X_train,y_train)

reg_rid_predict = reg_rid.predict(X_val)

rmse_rid=np.sqrt(mean_squared_error(reg_rid_predict,y_val))
print('RMSE for Ridge Regression:{0:.2f}'.format(rmse_rid))

accuracy_rid = reg_rid.score(X_val,y_val)
print('Accuracy of the Ridge Regression model:',accuracy_rid*100,'%' )

from sklearn.linear_model import Lasso
reg_lo=Lasso(alpha=0.01)
reg_lo.fit(X_train,y_train)
reg_lo.score(X_train,y_train)

reg_lo_predict = reg_lo.predict(X_val)

rmse_lo=np.sqrt(mean_squared_error(reg_lo_predict,y_val))
print('RMSE for Lasso Regression:{0:.2f}'.format(rmse_lo))

accuracy_lo = reg_lo.score(X_val,y_val)
print('Accuracy of the Lasso Regression model:',accuracy_lo*100,'%' )

from sklearn.ensemble import RandomForestRegressor
reg_rfr=RandomForestRegressor(random_state=0)
reg_rfr.fit(X_train,y_train)
reg_rfr.score(X_train,y_train)
reg_rfr_predict = reg_rfr.predict(X_val)

rmse_rfr=np.sqrt(mean_squared_error(reg_rfr_predict,y_val))
print('RMSE for Random Forest Regression:{0:.2f}'.format(rmse_rfr))

accuracy_rfr = reg_rfr.score(X_val,y_val)
print('Accuracy of the Random Forest Regression model:',accuracy_rfr*100,'%' )

from sklearn.tree import DecisionTreeRegressor
reg_dt = DecisionTreeRegressor(random_state=0)
reg_dt.fit(X_train,y_train)
reg_dt.score(X_train,y_train)
reg_dt_predict = reg_dt.predict(X_val)

rmse_dt = np.sqrt(mean_squared_error(reg_dt_predict,y_val))
print('RMSE for Decision Tree Regression:{0:.2f}'.format(rmse_dt))

accuracy_dt = reg_dt.score(X_val,y_val)
print('Accuracy of the Decision Tree Regression model:',accuracy_dt*100,'%' )
