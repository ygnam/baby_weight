#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 11:53:58 2019

@author: Yun Gon Nam (Team 5)
"""

"This script is written for analysis to make model to predic newborn baby weight."


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split # train/test split
from sklearn.neighbors import KNeighborsRegressor # KNN for Regression
import statsmodels.formula.api as smf # regression modeling
import sklearn.metrics # more metrics for model performance evaluation
from sklearn.model_selection import cross_val_score # k-folds cross validation
from sklearn.linear_model import LinearRegression

#importing
file = 'birthweight_feature_set-1.xlsx'
baby_data = pd.read_excel(file)

print(baby_data)

### Exploratory Data Analysis(EDA) ###
## Numerical EDA ##
# Check what kind of variables are there.
baby_data.columns


# Displaying the first rows of the DataFrame
print(baby_data.head(30))

# Dimensions of the DataFrame
baby_data.shape


# Information about each variable
baby_data.info()


# Descriptive statistics and rounding up by two digits.
baby_data.describe().round(2)


baby_data.sort_values('bwght', ascending = False)
pd.set_option('display.max_columns', 30)


# Replace NaN with median since the data is skewed.
for col in baby_data:
       
    if baby_data[col].isnull().any():
        
        col_median = baby_data[col].median()
        
        baby_data[col] = baby_data[col].fillna(col_median).astype(int)


# Confirming to see if there are any remaining missing values.
baby_data.isna().sum()


## Flaging for outliers.

baby_quantiles = baby_data.loc[:, :].quantile([0.05,
                                                0.40,
                                                0.60,
                                                0.80,
                                                0.95])
    
print(baby_quantiles)

# Outlier flags(not for race since they are all binary data)
# Not nmaps and fmaps because this is out of control. 
mage_hi = 61
meduc_hi = 17
monpre_hi = 5
npvis_hi = 17.4
fage_hi = 58.25
feduc_hi = 17
#omaps_hi = 9
#fmaps_hi = 10
cigs_hi = 21
drink_hi = 10


mage_low = 27
meduc_low = 11
monpre_low = 1
npvis_low = 6
fage_low = 26
feduc_low = 11
#omaps_low = 4
#fmaps_low = 8
cigs_low = 1
drink_low = 0

# High Outliers

baby_data['out_mage_hi'] = 0

for val in enumerate(baby_data.loc[ : , 'mage']):    
    if val[1] >= mage_hi:
        baby_data.loc[val[0], 'out_mage_hi'] = 1
        
baby_data['meduc_hi'] = 0

for val in enumerate(baby_data.loc[ : , 'meduc']):    
    if val[1] >= meduc_hi:
        baby_data.loc[val[0], 'out_meduc_hi'] = 1
        
baby_data['out_monpre_hi'] = 0

for val in enumerate(baby_data.loc[ : , 'monpre']):    
    if val[1] >= monpre_hi:
        baby_data.loc[val[0], 'out_monpure_hi'] = 1
        
baby_data['out_npvis_hi'] = 0

for val in enumerate(baby_data.loc[ : , 'npvis']):    
    if val[1] >= npvis_hi:
        baby_data.loc[val[0], 'out_npvis_hi'] = 1
        
baby_data['out_fage_hi'] = 0

for val in enumerate(baby_data.loc[ : , 'fage']):    
    if val[1] >= fage_hi:
        baby_data.loc[val[0], 'out_fage_hi'] = 1
        
baby_data['out_feduc_hi'] = 0

for val in enumerate(baby_data.loc[ : , 'feduc']):    
    if val[1] >= feduc_hi:
        baby_data.loc[val[0], 'out_feduc_hi'] = 1
        
baby_data['out_cigs_hi'] = 0

for val in enumerate(baby_data.loc[ : , 'cigs']):    
    if val[1] >= cigs_hi:
        baby_data.loc[val[0], 'out_cigs_hi'] = 1
        
baby_data['out_drink_hi'] = 0

for val in enumerate(baby_data.loc[ : , 'drink']):    
    if val[1] >= drink_hi:
        baby_data.loc[val[0], 'out_drink_hi'] = 1



# low outliers

baby_data['out_mage_low'] = 0

for val in enumerate(baby_data.loc[ : , 'mage']):    
    if val[1] >= mage_low:
        baby_data.loc[val[0], 'out_mage_low'] = 1
        
baby_data['out_meduc_low'] = 0

for val in enumerate(baby_data.loc[ : , 'meduc']):    
    if val[1] >= mage_low:
        baby_data.loc[val[0], 'out_meduc_low'] = 1
        
baby_data['out_monpre_low'] = 0

for val in enumerate(baby_data.loc[ : , 'monpre']):    
    if val[1] >= mage_low:
        baby_data.loc[val[0], 'out_monpre_low'] = 1
        
baby_data['out_npvis_low'] = 0

for val in enumerate(baby_data.loc[ : , 'npvis']):    
    if val[1] >= mage_low:
        baby_data.loc[val[0], 'out_npvis_low'] = 1
        
baby_data['out_fage_low'] = 0

for val in enumerate(baby_data.loc[ : , 'fage']):    
    if val[1] >= mage_low:
        baby_data.loc[val[0], 'out_fage_low'] = 1
        
        
# May not make these new variables since this is the majority
baby_data['out_cigs_low'] = 0

for val in enumerate(baby_data.loc[ : , 'cigs']):    
    if val[1] >= mage_low:
        baby_data.loc[val[0], 'out_cigs_low'] = 1
        
baby_data['out_drink_low'] = 0

for val in enumerate(baby_data.loc[ : , 'drink']):    
    if val[1] >= mage_low:
        baby_data.loc[val[0], 'out_drink_low'] = 1
        
## Trying to find best variables(features) via Recursive Feature Elimination(RFE).
array = baby_data.values
X = array[:, 0:17]
Y = array[:, 17]
# Import the necessary libraries first
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
# Feature extraction
test = SelectKBest(score_func=chi2, k=4)
fit = test.fit(X, Y)

# Summarize scores
np.set_printoptions(precision=3)
print(fit.scores_)

features = fit.transform(X)
# Summarize selected features
print(features[0:5,:])
# Import your necessary dependencies
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
# Feature extraction
model = LogisticRegression()
rfe = RFE(model, 14)
fit = rfe.fit(X, Y)
print("Num Features: %s" % (fit.n_features_))
print("Selected Features: %s" % (fit.support_))
print("Feature Ranking: %s" % (fit.ranking_))
"""Despite using the top 14 variables, the R-square value is below 0.7.
   Therefore, we decided to use another method to find a good model."""
        
## Seeing what correlates with bwght. Trying to look for high correlation numbers. ##
baby_data.corr()

# Due to many columns, it's difficult to view each one of them since on the console, not all of
# them are visiable. Therefore, need to display all columns.
pd.set_option('display.max_columns', 100)

# Use heatmap to better understand the correlation between variables
sns.heatmap(baby_data.corr(), square=True, cmap='RdYlGn')

# Linear regression model based on the highest correlations.
lm_baby = smf.ols(formula = """bwght ~ baby_data['mage']
                                       + baby_data['fage']
                                       + baby_data['cigs']
                                       + baby_data['drink']
                                       + baby_data['out_fage_hi']
                                       + baby_data['out_cigs_hi']
                                       + baby_data['out_drink_hi']
                                       """,
                  data = baby_data)

results = lm_baby.fit()

# Printing Summary Statistics
print(results.summary())
"""From our new model, for OLS. the R-squared is 0.721 and the Adjusted R-squared value is 0.711."""

##Predictions on our final model using the following parameters during our train/test split:

# Import necessary modules
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# Create feature and target arrays
X = baby_data[['mage', 'fage', 'cigs', 'drink', 'out_fage_hi', 'out_cigs_hi', 'out_drink_hi']]
y = baby_data.bwght

# Train/Test Split for Regression 

# Import necessary modules
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.10, random_state=508)

# Create the regressor: reg_all
reg_all = LinearRegression()

# Fit the regressor to the training data
reg_all.fit(X_train, y_train)

# Predict on the test data: y_pred
y_pred = reg_all.predict(X_test)


# Compute and print R^2 and RMSE for OLS
print("R^2: {}".format(reg_all.score(X_test, y_test)))
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error: {}".format(rmse))
""" Despite the results on the model we had previously, the OLS regression score for R^2 turns out
    to be 0.638 with a Root Mean Squared Error of 305.913."""

# KNN score
from sklearn.neighbors import KNeighborsRegressor # KNN for Regression

# Instantiate
knn_reg = KNeighborsRegressor(algorithm = 'auto',
                              n_neighbors = 10)

# Fit
knn_reg.fit(X_train, y_train)

# Predict
y_pred = knn_reg.predict(X_test)

# Score
y_score_knn = knn_reg.score(X_test, y_test)

print(y_score_knn)    
"""We have a score of about 0.225 for KNN which is much lower than OLS."""

# Make sure that everything was correctly imported
X.shape
y.shape

# Make sure the rows are the same
X_train.shape
y_train.shape

X_test.shape
y_test.shape

# Split into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, baby_data.bwght, test_size = 0.10,
                                                    random_state=508)

# Create a k-NN classifier with 10 neighbors: knn => based on the previous plot,
# it states that knn score is highest when 10 <= k <= 14
knn = KNeighborsClassifier(n_neighbors = 10)

# Fit the classifier to the training data
knn.fit(X_train ,y_train)

# Print the accuracy
print(knn.score(X_test, y_test))

# Find the best k value
# Setup arrays to store train and test accuracies
neighbors = np.arange(1, 16)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

# Loop over different values of k
for i, k in enumerate(neighbors):
    # Setup a k-NN Classifier with k neighbors: knn
    knn = KNeighborsClassifier(n_neighbors=k)

    # Fit the classifier to the training data
    knn.fit(X_train,y_train)
    
    #Compute accuracy on the training set
    train_accuracy[i] = knn.score(X_train, y_train)

    #Compute accuracy on the testing set
    test_accuracy[i] = knn.score(X_test, y_test)

# Generate plot
plt.title('k-NN: Varying Number of Neighbors')
plt.plot(neighbors, test_accuracy, label = 'Testing Accuracy')
plt.plot(neighbors, train_accuracy, label = 'Training Accuracy')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.show()

# Exporting into excel file for final prediction model.
X.to_excel('Team5.xlsx')