# -*- coding: utf-8 -*-
"""
Created on Wed May 30 14:58:00 2018

@author: MeghaGadad
"""

import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

#load dataset
df = pd.read_csv('Flaveria.csv')#['N Level','Species','Plant Weight(g)']
df = pd.get_dummies(df)

# #############################################################################
# Fit LinearRegression models
from sklearn.linear_model import LinearRegression

data = df.copy()

data_X = df.loc[:, df.columns != 'Plant Weight(g)']
target_y = df.iloc[:,0]

from sklearn.model_selection import train_test_split

data_X_train, data_X_test, target_y_train, target_y_test = train_test_split(data_X, target_y, shuffle=True,
                                                    test_size=0.5, random_state=0)

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(data_X_train, target_y_train)

# Make predictions using the testing set
target_y_pred = regr.predict(data_X_test)

# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(target_y_test, target_y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score or r2score: %.2f' % r2_score(target_y_test, target_y_pred))