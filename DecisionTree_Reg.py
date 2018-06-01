# -*- coding: utf-8 -*-
"""
Created on Wed May 30 21:45:20 2018

@author: MeghaGadad
"""



# Import the necessary modules and libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error

# Create a dataset
df = pd.read_csv('Flaveria.csv')#['N Level','Species','Plant Weight(g)']
df = pd.get_dummies(df)
data = df.copy()

#define fetures and target value
X = data.drop('Plant Weight(g)', axis=1)  
y = data['Plant Weight(g)']  

#divide our data into training and test sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True,
                                                   test_size=0.20, random_state=20)

# Split data in train set and test set
#n_samples = X.shape[0]

#X_train, y_train = X[:n_samples // 2], y[:n_samples // 2]
#X_test, y_test = X[n_samples // 2:], y[n_samples // 2:]

# Training and Making Predictions
from sklearn.tree import DecisionTreeRegressor  
regr = DecisionTreeRegressor() 

# Fit regression model 
regr.fit(X_train, y_train)  


# Make predictions using the testing set
y_pred = regr.predict(X_test)

#calculaate R2 score
from sklearn.metrics import r2_score
print('R_2 score:', r2_score(y_test, y_pred))

#let's compare our predicted values with the actual values
df=pd.DataFrame({'Actual':y_test, 'Predicted':y_pred})  
#print(df)

#evaluate performance of the regression algorithm
#from sklearn import metrics  
#print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
#print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
#print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred))) 



