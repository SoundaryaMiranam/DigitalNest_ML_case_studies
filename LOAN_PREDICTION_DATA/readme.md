# Loan-prediction-dataset

# Introduction:

From the challange hosted at: https://datahack.analyticsvidhya.com/contest/practice-problem-loan-prediction-iii/

About Company:
Dream Housing Finance company deals in all home loans. They have presence across all urban, semi urban and rural areas. Customer first apply for home loan after that company validates the customer eligibility for loan.

Case:
Company wants to automate the loan eligibility process (real time) based on customer detail provided while filling online application form. These details are Gender, Marital Status, Education, Number of Dependents, Income, Loan Amount, Credit History and others. To automate this process, they have given a problem to identify the customers segments, those are eligible for loan amount so that they can specifically target these customers. Here they have provided a partial data set.

There are 2 data sets that are given. One is training data and one is testing data.The features in this datasets are as follows:

Loan ID  -  unique loan ID.

Gender - male or female.

Married - Applicant who is married is represented by Y and not married is represented as N.

Dependents - the number of people dependent on the applicant who has taken loan has been provided.

Education - It is either non -graduate or graduate.

Self_Employed - An applicant who is self employed is represented by Y and the one who is not is represented by N.

Applicant Income - The income of an Applicant.

Co-Applicant income - This represents the income of co-applicant.

Loan Amount - This amount represents the loan amount in thousands.

Loan_Amount_Term - This represents the number of months required to repay the loan.

Credit_History - A credit history is a record of a borrower’s responsible repayment of debts. It suggests → 1 denotes that the credit history is good and 0 otherwise.

Property_Area - The area where they belong to, here there are 3 types: Urban or Semi Urban or Rural.

Loan_Status - The applicant is eligible for loan it’s yes represented by Y else it’s no represented by N.

# Purpose:
The problem is to predict which of the customers will have their loan paid or not with given features.The variable Loan_Status is our target variable and the remaining are the feature variables of the dataset.


# Exploratory Data Analysis:

Data cleaning:
columns which have missing values are:(Credit_History, Self_Employed, LoanAmount, Dependents, Loan_Amount_Term, Gender, Married).Imputed missing Values with sensible values.(most counts of value, mean, mode and median)
 
1. categorical columns - fill the missing values with most counts of value.(Credit_History, Self_Employed, Dependents, Gender, Married). 
2. numerical columns -  fill it with median, mode or most counts of value.(Loan_Amount_Term, LoanAmount).

Data visualization:

Univariate Analysis for loan prediction data:

1. Numerical Variables - Box plot
Applicant Income and LoanAmount has outliers in it.

Bivariate Analysis for loan prediction data:

1. categorical variables - count plot  
1) More males tend to take loan than females.

2) Married people are more on loan than unmarried people.

3) Self-employed people take less loans than those are not self-employed.

4) credit history shows that high number of people pay back their loans.

5) Semiurban obtain more loan, folowed by Urban and then rural.Which is understandable.

6) Most of the people opt for 360 cyclic loan term which is pay back within a year of time.   

7) Males generally have the highest income. Explicitly, Males that are married have greater income that unmarried male.

8) No one is dependent and a male tremendously has more income. 

9) A graduate who is a male has more income.

10) Not married and no one is dependent on such has more income. Also, Married and no one dependent has greater income with a decreasing effect as the dependents increases.

11) A graduate and married individual has more income.

12) Married and has a good credit history depicts more income. Also, Not married but has a good credit history follows in the hierarchy.

13) A graduate with no one dependent has more income.

14) Educated with good credit history depicts a good income. Also, not a graduate and have a good credit history can be traced to having a better income than a fellow with no degree

15) No one is dependent and self-employed has more income

16) A graduate but not self-employed has more income.

17) No one is dependent and have property in urban, rural and semiurban has more income.

2. Correlation matrix plot - 

No correlations are extremely high. The correlations between LoanAmount and ApplicantIncome can be explained.


# Machine learning:

This is a supervised classification problem, as the target variable is categorical.
Here  logistic Regression works well to training set which gives 82.43 % accuracy.So we will use it to predict loan status on the test dataset.
