#Load the required libraries 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns

#read the data from the file
housing_data = pd.read_csv('HousingData.csv')
print("Rows, columns: " + str(housing_data.shape))

housing_data.sample(10)

"""#Statistical analysis"""

housing_data.dtypes

housing_data.isnull().sum()

"""We can see that there are columns that have missing values which are 'CRIM', 'ZN', 'INDUS', 'CHAS', 'AGE', 'LSTAT'."""

housing_data.describe()

housing_data.rename(columns={'MEDV':'PRICE'}, inplace=True)

"""#Box plot for Boston housing data

Here we can see that the variables ‘chas’, 'TAX' and ‘rad’ are non numeric others are numeric.
"""

fig = plt.figure(figsize=(40,5))
features = ["TAX","RAD","CHAS"]
for i in range(3):
    ax1 = fig.add_subplot(1,3,i+1)
    sns.countplot(x=features[i],data=housing_data)

"""As you can see CHAS is skewed.There is just one bar which is dominating and other one have very less rows. """

housing_data.hist(['CRIM', 'ZN', 'INDUS', 'NOX', 'RM', 'AGE', 'DIS', 'PTRATIO', 'B', 'LSTAT'], figsize=(18,15))
plt.show()

"""CRIM has outliers in it beyond point 40.

#Correlation matrix for Boston housing data
"""

corr = housing_data.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr, annot=True, vmin=-1.0, vmax=1.0)
plt.show()

"""1.  RM has a strong positive correlation with PRICE where as LSTAT has a high negative correlation with PRICE.
2.  The features RAD, TAX have a correlation of 0.91. These feature pairs are strongly correlated to each other.
3. The features DIS and AGE also have a strong correlation of -0.75.
4. CRIM is strongly associated with variables RAD and TAX.
5. INDUS is strongly correlated with NOX ,which shows that industrial areas has nitrogen oxides concentration.
"""

sns.regplot(y="PRICE",x="LSTAT", data=housing_data, fit_reg= True)
plt.title("Relationship between Lower Status Population and Price")
plt.show()

sns.regplot(y="PRICE",x="CRIM", data=housing_data, fit_reg= True)
plt.title("Relationship between Crime rate and Price")
plt.show()

sns.regplot(y="PRICE",x="NOX", data=housing_data, fit_reg= True)
plt.title("Relationship between Nitric Oxide concentration and Price")
plt.show()

"""1)We can see a strong negative correlation between lower status population and price.
2)As Crime rate increases the rate of House decreases.
3)As Nitric Oxide concentration increases the rate of House decreases.

#Feature Selection for Boston housing data
"""

correlations = housing_data.corr()['PRICE'].sort_values(ascending=False)
correlations.plot(kind='bar')

print(correlations)

print(abs(correlations) >= 0.40)

"""These featurs are effecting the target variable:
'RM', 'NOX', 'TAX', 'INDUS', 'PTRATIO', 'LSTAT'.

#Handling missing values of Boston housing data
"""

housing_data["CRIM"] = housing_data["CRIM"].replace(np.nan,housing_data["CRIM"].median())
housing_data["ZN"] = housing_data["ZN"].replace(np.nan,housing_data["ZN"].median())
housing_data["INDUS"] = housing_data["INDUS"].replace(np.nan,housing_data["INDUS"].mean()) 
housing_data["CHAS"] = housing_data["CHAS"].replace(np.nan,housing_data["CHAS"].median())
housing_data["AGE"] = housing_data["AGE"].replace(np.nan,housing_data["AGE"].median())
housing_data["LSTAT"] = housing_data["LSTAT"].replace(np.nan,housing_data["LSTAT"].median())

#housing_data.CHAS = housing_data.CHAS.astype(int)

"""#Machine learning"""

from sklearn.model_selection import train_test_split, cross_val_score
X = housing_data.drop('PRICE',axis = 1)
Y = housing_data['PRICE']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)

from sklearn.linear_model import LinearRegression
import sklearn.metrics as sm
from sklearn import linear_model, metrics
regressor = LinearRegression()
regressor.fit(X_train,Y_train)
regressor_predict = regressor.predict(X_test)
accuracy = regressor.score(X_test,Y_test)
print('Accuracy of the model:',accuracy*100,'%' )

print("Regressor model performance:")
print("Mean absolute error(MAE) =", (round(sm.mean_absolute_error(Y_test, regressor_predict), 2)))
print("Mean squared error(MSE) =", (round(sm.mean_squared_error(Y_test, regressor_predict), 2)))
print("Median absolute error =", (round(sm.median_absolute_error(Y_test, regressor_predict), 2)))
print("Explain variance score =",( round(sm.explained_variance_score(Y_test, regressor_predict), 2)))
print("R2 score =", (round(sm.r2_score(Y_test, regressor_predict), 2)))

plt.style.use('fivethirtyeight')
plt.scatter(regressor.predict(X_train), regressor.predict(X_train) - Y_train, color = "green", s = 20, label = 'Train data')
plt.scatter(regressor.predict(X_test), regressor.predict(X_test) - Y_test, color = "blue", s = 20, label = 'Test data')
plt.hlines(y = 0, xmin = 0, xmax = 50, linewidth = 2)
plt.legend(loc = 'upper right')
plt.title("Residual errors")
plt.show()
