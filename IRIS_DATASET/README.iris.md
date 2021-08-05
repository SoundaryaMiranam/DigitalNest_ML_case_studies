Exploratory analysis and Machine learning on Iris data

Introduction:

The Iris dataset was used in R.A. Fisher's classic 1936 paper, The Use of Multiple Measurements in Taxonomic Problems, and can also be found on the UCI Machine Learning Repository. It consists of 150 samples of three species of Iris (Iris setosa, Iris virginica and Iris versicolor).Four features from each sample, the length and the width of the sepals and petals.

Purpose:

With given features (sepal length, sepal width, petal length and petal width) study, classify ML algo. and predict the iris flower into one of the three species (Setosa, Virginica and Versicolor).

Statistical analysis:

Mean and median - For all the species, the mean and median of its features are found to be pretty close. This indicates that data is nearly symmetrically distributed.

Standard deviation - It indicates how widely the data is spread about the mean.

Histogram plot - Horizontal bar represents maximum range of that data and width of box represents spread of that data. For both petal length and petal width there seems to be a group of data points that have smaller values than the others.

Box plot - Statistical tool used for outlier detection in the data. Horizontal bar represents maximum range of that data and width of box represents spread of that data, isolated points are to be the outliers.

Scatter plot - Used to observe relationships between variables.Setosa is very well separated than that of Versicolor and Virginica.

Machine learning: Comparative study of performance of supervised ML algo. and predict the species of a new sample.

A)ML on the samples - Splitting the dataset based on samples (150) by 70% & 30%.SVM  gives very good accuracy among other algorithms. Decision tree and KNN gives same accuracy during training.

B) ML on petals and sepals - Splitting the dataset based on features (sepal, petal). Using Petals over Sepal for training the data gives a much better accuracy.
