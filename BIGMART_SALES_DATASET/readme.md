# BigMart_sales_data
# Introduction:

From the challange hosted at: https://datahack.analyticsvidhya.com/contest/practice-problem-big-mart-sales-iii/


About Company: The data scientists at BigMart have collected 2013 sales data for 1559 products across 10 stores in different cities. Also, certain attributes of each product and store have been defined.BigMart will try to understand the properties of products and stores which play a key role in increasing sales. Please note that the data may have missing values as some stores might not report all the data due to technical glitches. Hence, it will be required to treat them accordingly.
 
Data We have train (8523) and test (5681) data set, train data set has both input and output variable(s).Need to predict the sales for test data set.

Item_Identifier : Unique product ID.

Item_Weight : Weight of product.

Item_Fat_Content : Whether the product is low fat or not.

Item_Visibility : The % of total display area of all products in a store allocated to the particular product.

Item_Type : The category to which the product belongs.

Item_MRP : Maximum Retail Price (list price) of the product.

Outlet_Identifier : Unique store ID.

Outlet_Establishment_Year : The year in which store was established.

Outlet_Size : The size of the store in terms of ground area covered.

Outlet_Location_Type : The type of place in which the store is located.

Outlet_Type : Whether the outlet is just a grocery store or some sort of supermarket.

Item_Outlet_Sales : Sales of the product in the particulat store.

# Purpose:
The aim is to build a predictive model and find out the sales of each product at a particular store.This is a supervised machine learning problem with target variable(Item_Outlet_Sales).

# Exploratory Data Analysis:

Data cleaning:

1. columns which have missing values are:(Item_Weight,Outlet_Size).Imputed missing Values with sensible values as most counts of values and mode resplectively.

2. Item_Visibility has a min value of zero,which isn't possible because when a product is being sold at a store, the visibility cannot be 0.

3. Outlet_Establishment_Years vary from 1985 to 2009. The values might not be apt in this form. Rather, if we can convert them to how old the particular store is, it should have a better impact on sales.

4. Cleaning Item_Fat_Content column with two groups as low and regular.

5. Categorizating  Item_Visibility with 'Low Viz', 'Viz' and 'High Viz'.

6. Grouping Item_Type with Item_Id based on the object's first two character ('FD':'Food','NC':'Non-Consumable','DR':'Drinks').

7. Applying label encoding on the categorical variables and one-hot encoding. 

Data visualization:


1. Univariate Analysis

a) Numerical features:

1.  Item_MRP have the most positive correlation and the Item_Visibility have the lowest correlation with our target variable.

b) Categorical features:

1. Item_Type has 16 different types of unique values and it has high number for categorical variable.

2. There seems to be less number of stores with size equals to “High”.

3. Bigmart is a brand of medium and small size city compare to densely populated area.

4. There seems like Supermarket Type2 , Grocery Store and Supermarket Type3 all have low numbers of stores.

2. Bivariate Analysis

a) Numerical features:

1. As Item_Weight had a low correlation with our target variable. This plot shows there relation.

2. There seems to be no relation between the year of store establishment and the sales for the items.

b) Categorical features:

1. Low Fat products seem to higher sales than the Regular products.

2. Out of 10- There are 2 Groceries strore, 6 Supermarket Type1, 1Supermarket Type2, and 1 Supermarket Type3.

3. Most of the stores are of Supermarket Type1 of size High and they do not have best results. whereas Supermarket Type3 (OUT027) is a Medium size store and have best results.

4. Tier 2 cities have the higher sales than the Tier 1 and Tier 2.

Data Pre-processing:

1. choosing the Fat content, item vizibility bins, outlet size, loc type and type for LABEL ENCODER.

2. create dummies for outlet type.

3. drop all the object types features.

4. Feature Scaling.

# Machine learning:

This is a supervised regression problem, as the target variable is numerical. Here logistic, Ridge, Lasso Regression which gives 56.41 % accuracy, Random Forest Regression which gives 56.63 % accuracy, Decision Tree Regression which gives 20.04 % accuracy.
So this doesn't predict sales on the test dataset.

# Future Improvements:
Hyper-parameter Tuning and Gradient Boosting.
