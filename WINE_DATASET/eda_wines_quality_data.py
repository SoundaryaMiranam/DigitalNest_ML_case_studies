#Load the required libraries 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline
from matplotlib.ticker import PercentFormatter
import seaborn as sns

#read the data from the file
red_wine = pd.read_csv('winequality-red.csv')
white_wine = pd.read_csv('winequality-white.csv', sep=';')

# store wine type as an attribute
red_wine['wine_type'] = 'red' 
white_wine['wine_type'] = 'white'

#Viewing the data
red_wine.sample(5)

#Viewing the data
white_wine.sample(5)

# bucket wine quality scores into qualitative quality labels

red_wine['quality_label'] = red_wine['quality'].apply(lambda value: 'low' 
                                                          if value <= 5 else 'medium' 
                                                              if value <= 7 else 'high')
red_wine['quality_label'] = pd.Categorical(red_wine['quality_label'], 
                                           categories=['low', 'medium', 'high'])

red_wine.head()

white_wine['quality_label'] = white_wine['quality'].apply(lambda value: 'low' 
                                                              if value <= 5 else 'medium' 
                                                                  if value <= 7 else 'high')
white_wine['quality_label'] = pd.Categorical(white_wine['quality_label'], 
                                             categories=['low', 'medium', 'high'])

white_wine.head()

# merge red and white wine datasets
wines = pd.concat([red_wine, white_wine]) 
# re-shuffle records just to randomize data points
wines = wines.sample(frac=1, random_state=42).reset_index(drop=True)

wines.head()

print(wines.shape)

"""#Statistical analysis"""

data = wines["wine_type"].value_counts(normalize=True)
data.plot(kind='bar',figsize=(7,5))
plt.title("Wine type proportion",fontsize= 16)
plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
plt.show()

"""75% of the data consists of white wine remaining 25% is red wine data."""

wines.describe()

"""Mean value is less than the median value of each column.
There is a large difference between the 75th% tile and max values of residual sugar, free sulfur dioxide & total sulfur dioxide.
"""

subset_attributes = ['alcohol', 'volatile acidity', 'pH', 'quality']
ls = round(wines[wines['quality_label'] == 'low'][subset_attributes].describe(),2)
ms = round(wines[wines['quality_label'] == 'medium'][subset_attributes].describe(),2)
hs = round(wines[wines['quality_label'] == 'high'][subset_attributes].describe(),2)
pd.concat([ls, ms, hs], axis=1, keys=['Low Quality Wine', 'Medium Quality Wine', 'High Quality Wine'])

"""60% of the
data is classified with good wine quality and
40% with bad quality of wine.

"""

subset_attributes = ['residual sugar', 'total sulfur dioxide', 'sulphates', 'alcohol','fixed acidity','volatile acidity', 'quality']
rs = round(red_wine[subset_attributes].describe(),2)
ws = round(white_wine[subset_attributes].describe(),2)

pd.concat([rs, ws], axis=1, keys=['Red Wine Statistics', 'White Wine Statistics'])

"""The distribution of wine quality are similar between the reds and whites. In this sample, there were no evaluations lower than 3 or greater than 9.
Red wine usually is less acid than white wine and opposite is true when we look at volatile_acidity feature. The 'residual_sugar' in average is almost 3 times bigger for a white wine.

#Correlation matrix plot for wines dataset
"""

corr = wines.corr()
sns.heatmap(corr , xticklabels=corr.columns.values , yticklabels=corr.columns.values)
plt.show()

wines.corr()

"""1.   Density has a strong positive correlation with residual sugar, whereas it has a strong negative correlation with alcohol.
2.   Density & fixed acidity has positive correlation.
3.   pH & fixed acidity has negative correlation.
4.   Citric acid & fixed acidity has positive correlation.
5.   Free sulphur dioxide & total sulphur dioxide has positive correlation.
6.   Citric acid & volatile acidity has negative correlation.

#Box plot for wines dataset
"""

fig = plt.figure(figsize=(24,10))
features = ["total sulfur dioxide", "residual sugar", "volatile acidity", "total sulfur dioxide", "chlorides", "fixed acidity", "citric acid","sulphates"]

for i in range(8):
    ax1 = fig.add_subplot(2,4,i+1)
    sns.barplot(x='quality', y=features[i],data=wines, hue='wine_type', palette='rocket')

"""1. The volatile acidity of red wine decreases with better quality.
2. Better quality red and white wines have shown decreased level of chlorides in them, meaning less amount of salt.
3. There is an increase in the levels of Citric acid and sulphates in higher quality of red wine, which could mean that good quality red wines have more freshness/flavor and antioxidants in them.

Observations:It was possible to find subtle differences in high quality wines when analyzing the density and fixed acidity for red wines and residual sugar and density for white wines.It was not clear what differentiates a low quality wine from a medium quality. The few wines classified with low quality may be a limitation of this data set.
"""
