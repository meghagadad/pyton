# -*- coding: utf-8 -*-
"""
Created on Wed May 30 12:12:18 2018

@author: MeghaGadad
"""

import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

#load dataset
df = pd.read_csv('Flaveria.csv')#['N Level','Species','Plant Weight(g)']
#df = pd.read_csv(url)
df.head()
df = pd.get_dummies(df)
#print(df)
df.columns
#df.drop(labels=['N level_L', 'species_bidentis'], axis=1,inplace=True)
#print(df.max() - df.min())

from sklearn.linear_model import LinearRegression
#data = df.drop(['Plant Weight(g)'], axis=1)
data = df.copy()

data_X = df.loc[:, df.columns != 'Plant Weight(g)']
#data = df.iloc[:, 1:2]
print(data)
target_y = df.iloc[:,0]

# Split the data into training/testing sets
data_X_train = data_X[:-10]
data_X_test = data_X[-10:]

# Split the targets into training/testing sets
target_y_train = target_y[:-10]
target_y_test = target_y[-10:]

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
print('Variance score: %.2f' % r2_score(target_y_test, target_y_pred))

# Plot outputs
#plt.scatter(data_X_test, target_y_test,  color='black')
#plt.plot(data_X_test, target_y_pred, color='blue', linewidth=3)

#plt.xticks(())
#plt.yticks(())

#plt.show()


#lr = LinearRegression(fit_intercept=True)
#lr.fit(data, target)

#from sklearn.metrics import mean_squared_error

# R^2
#print('R2 score using linear regression before split into train and test')
#print(lr.score(data, target)) 

#predictions = lr.predict(data)
#mse = mean_squared_error(target, predictions)
#rmse = np.sqrt(mse)
#print(rmse)

#from sklearn.model_selection import train_test_split

#X_train, X_test, y_train, y_test = train_test_split(data, target, shuffle=True,
#                                                    test_size=0.5, random_state=0)
#lr_split = LinearRegression(fit_intercept=True)
#lr_split.fit(X_train, y_train)

#functions to Calculate ISE and OSE
def calc_ISE(data_X_train, target_y_train, model):
#   '''returns the in-sample R^2 and RMSE; assumes model already fit.'''
    predictions = model.predict(data_X_train)
    mse = mean_squared_error(target_y_train, predictions)
    rmse = np.sqrt(mse)
    return model.score(data_X_train, target_y_train), rmse
    
def calc_OSE(data_X_test, target_y_test, model):
#    '''returns the out-of-sample R^2 and RMSE; assumes model already fit.'''
    predictions = model.predict(data_X_test)
    mse = mean_squared_error(target_y_test, predictions)
    rmse = np.sqrt(mse)
    return model.score(data_X_test, target_y_test), rmse
    is_r2, ise = calc_ISE(data_X_train, target_y_train, regr)
    os_r2, ose = calc_OSE(data_X_test, target_y_test, regr)

#Calculate In-Sample and Out-of-Sample R^2 and Error
is_r2, ise = calc_ISE(data_X_train, target_y_train, regr)
os_r2, ose = calc_OSE(data_X_test, target_y_test, regr)

# show dataset sizes
data_list = (('R^2_in', is_r2), ('R^2_out', os_r2), 
            ('ISE', ise), ('OSE', ose))
for item in data_list:
   print('{:10}: {}'.format(item[0], item[1]))
