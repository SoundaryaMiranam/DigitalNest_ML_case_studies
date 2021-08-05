# Wine-quality-data
# Introduction:
The dataset used is Wine Quality Data set from UCI Machine
Learning Repository.The datasets are related
to the Portuguese "Vinho Verde" wine.Due to privacy and logistic issues, only
physicochemical (inputs) and sensory (the output)
variables are available.There is no data about grape
types, wine brand, wine selling price, etc.
Inputs for given data:
1. Fixed Acidity: acid that contributes to the conservation of wine.
2. Volatile Acidity: Amount of acetic acid in wine at high levels can lead to an unpleasant taste of vinegar.
3. Citric Acid: found in small amounts, can add “freshness” and flavor to wines.
4. Residual sugar: amount of sugar remaining after the end of the fermentation.
5. Chlorides: amount of salt in wine.
6.  Free Sulfur Dioxide: it prevents the increase of microbes and the oxidation of the wine.
7. Total Sulfur Dioxide: it shows the aroma and taste of the wine.
8.  Density: density of water, depends on the percentage of alcohol and amount of sugar.
9. pH: describes how acid or basic a wine is on a scale of 0 to 14.
10. Sulfates: additive that acts as antimocrobian and antioxidant.
11. Alcohol: percentage of alcohol present in the wine.

# Info about wine:

Red wine is made from dark red and black grapes. The color usually ranges from various shades of red, brown and violet. This is produced with whole grapes including the skin which adds to the color and flavor of red wines, giving it a rich flavor.

White wine is made from white grapes with no skins or seeds. The color is usually straw-yellow, yellow-green, or yellow-gold. Most white wines have a light and fruity flavor as compared to richer red wines

# Purpose:

Predict the quality of wine on a scale of 0–10 given a set of features as inputs(fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free
sulphur dioxide, total sulphur dioxide, density, pH,
sulphates, alcohol) and classification is used to
classify the wine as good or bad.
# Exploratory analysis on both the wines:

Statistical analysis - 
75% of the data consists of white wine remaining 25% is red wine data.Describe function shows that mean value is less than the median value of each column.60% of the data is classified with good wine quality.There were differences in high quality wines when analyzing the density and fixed acidity for red wines and residual sugar and density for white wines.

Correlation matrix plot - There are few features which are correlated.Fixed acidity the best correlation for red wines and residual sugar the best correlation for white wines.
Box plot - Better quality red and white wines have shown decreased level of chlorides and increase in the levels of Citric acid and sulphates.

From the above analysis, good quality red wines have more freshness/flavor and antioxidants in them.

Note:Domain knowledge about wines suggests that we shouldn’t mix them together for classification.

# Machine Learning on red wine dataset:
 
Data visualization:
Histogram plot - Horizontal bar represents maximum range of that data and width of box represents spread of that data.Trends found are - 

fixed acidity : No significant effect
volatile acidity : Decreasing
citric acid : Increasing
residual sugar : No significant effect
chlorides : Decreasing
free sulphur dioxide : No significant effect
total sulphur dioxide : No significant effect
sulphates : Increasing
alcohol : Increasing

Correlation matrix plot - Few feature have shown relation with each other(quality,alcohol, sulphates, citric_acid and volatile_acidity).

Machine Learning:
Quality of red wine is distributed in range of (3 to 8).
Quality range and meaning:

0 ==> 3, 4, 5 ==>Bad

1 ==> 6, 7, 8 ==> Good

Normalise the data with standard scaling because different scales of features may impact the performance of the machine learning models.

Random forest classifier model has around 80% accuracy, which is  better than the other models.


# Machine learning on white wine dataset:

Data visualization:
Histogram plot - Horizontal bar represents maximum range of that data and width of box represents spread of that data.Trends found are - 

1.   As chlorides level decreases, quality increases
2.  Fixed acidity and sulphates has no impact on Quality
3.  As free sulfur dioxide increases, quality increases

Correlation matrix plot - Few feature have shown relation with each other.

1. PH value has a negative relationship with fixed acidity.
2.  Alcohol has a negative relationship with density, residual sugar, and total sulfur dioxide.
3.  Citric acid and Free Sulfur Dioxide has almost no impact to quality. 
4.  Density has a negative relationship with quality.
5.  Density has a positive relationship with residual sugar.
6.  Alcohol and sulfate have a positive relationship with quality.
7.  Free sulfur and total sulfur also have a positive relationship.

Machine Learning:
Quality of white wine is distributed in range of (3 to 9).
Quality range and meaning:

1 ==> quality (>= 7) as good

0 ==> quality (< 7) as bad

Normalise the data with standard scaling.

Random forest classifier model has around 90% accuracy, which is  better than the other models.
