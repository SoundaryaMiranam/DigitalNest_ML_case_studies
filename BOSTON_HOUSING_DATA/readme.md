# Boston_housing_data
# Introduction:

The Boston Housing Dataset is a derived from information collected by the U.S. Census Service concerning housing in the area of Boston MA. The following describes the dataset columns:

CRIM - per capita crime rate by town

ZN - proportion of residential land zoned for lots over 25,000 sq.ft.

INDUS - proportion of non-retail business acres per town.

CHAS - Charles River dummy variable (1 if tract bounds river; 0 otherwise)

NOX - nitric oxides concentration (parts per 10 million)

RM - average number of rooms per dwelling

AGE - proportion of owner-occupied units built prior to 1940

DIS - weighted distances to five Boston employment centres

RAD - index of accessibility to radial highways

TAX - full-value property-tax rate per $10,000

PTRATIO - pupil-teacher ratio by town

B - 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town

LSTAT - % lower status of the population

MEDV - Median value of owner-occupied homes in $1000'

# Purpose:

The objective is to predict the value of prices of the house using the given features.The prices of the house indicated by the variable MEDV is our target variable and the remaining are the feature variables of a house dataset.

# Exploratory Data Analysis:

There two variables cloumns which are:
1)Categorical variables: Bar plot('TAX','RAD','CHAS(after processing)')
2)Continuous variables: Histogram('CRIM', 'ZN', 'INDUS', 'NOX', 'RM', 'AGE', 'DIS', 'PTRATIO', 'B', 'LSTAT')

Box plot for Boston housing data:
CRIM has outliers in it which lies beyond point 40.
Most of the columns are skewed except 'RM' weather it be numeric or non-numeric.

Correlation matrix plot for Boston housing data:
1.  RM has a strong positive correlation with PRICE where as LSTAT has a high negative correlation with PRICE.
2.  The features RAD, TAX have a correlation of 0.91. These feature pairs are strongly correlated to each other.
3. The features DIS and AGE also have a strong correlation of -0.75.
4. CRIM is strongly associated with variables RAD and TAX.
5. INDUS is strongly correlated with NOX ,which shows that industrial areas has nitrogen oxides concentration.

Additional points:

 1)Population of lower status increases/decreases with decrease/increase in price.

 2)As Crime rate increases the rate of House decreases.

 3)As Nitric Oxide concentration increases the rate of House decreases.

Specifically, there are also missing observations for some columns.
Imputed missing Values with sensible values.(mean and median).


# Machine learning:

Based on the problem statement we need to create a supervised ML Regression model, as the target variable is Continuous.
Here linear regression is applied which gives 65% accuracy.So, the model isn't very good for predicting the housing prices.

Regressor model performance:

Mean absolute error(MAE) = 3.15

Mean squared error(MSE) = 25.11

Median absolute error = 2.25

Explain variance score = 0.66

R2 score = 0.66
