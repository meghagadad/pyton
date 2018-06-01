# -*- coding: utf-8 -*-
"""
Created on Thu May 31 16:35:58 2018

@author: MeghaGadad
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score

#read two files for test and train
train = pd.read_csv('Flaveria.csv')
test = pd.read_csv('Flaveria.csv')

#get dummie values
train = pd.get_dummies(train)
test =  pd.get_dummies(test)

#specify features and target
features = ['N Level','Species','Plant Weight(g)']
features = train.drop('Plant Weight(g)', axis=1)

#devide our dataaset into train and test
X_train = train[list(features)].values

y_train = train['Plant Weight(g)'].values

X_test =  test[list(features)].values

y_test =  test['Plant Weight(g)'].values

###########Training and Making Predictions using DecisionTreeRegressor #############
from sklearn.tree import DecisionTreeRegressor  
regr = DecisionTreeRegressor() 

# Fit regression model 
regr.fit(X_train, y_train)

# Make predictions using the testing set
y_pred = regr.predict(X_test)

#calculaate R2 score
print('R_2 score:', r2_score(y_test, y_pred))

#let's compare our predicted values with the actual values
df=pd.DataFrame({'Actual':y_test, 'Predicted':y_pred})  
#print(df)

#evaluate performance of the regression algorithm
#from sklearn import metrics  
#print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
#print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
#print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred))) 

