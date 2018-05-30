# -*- coding: utf-8 -*-
"""
Created on Wed May 30 16:19:09 2018

@author: MeghaGadad
"""

import pandas as pd
import numpy as np
#from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error, r2_score

file = pd.read_csv('Flaveria.csv')
filedummies = pd.get_dummies(file);
y = []
X = []
for data in filedummies.get_values():
    y.append(data[0])    
    X.append(np.array(data[1:]))
     


dictionary = {}
X_train = X.copy
X_index = []
X_test = []
y_test = []

for index, value in enumerate(X):
    key = tuple(value)
    if key not in dictionary:
        dictionary[key] = y[index]
        X_index.append(index)
        X_test.append(value)
        y_test.append(y[index])
#        
        
#X_train = np.delete(X, X_index)

X_train = [x for i,x in enumerate(X) if i not in X_index]
y_train = [x for i,x in enumerate(y) if i not in X_index]
#print('X train is')
#print(X_train)




#Ridge Regression Test v0.2#####################################
from sklearn import linear_model
reg = linear_model.BayesianRidge()
reg.fit(X_train, y_train)

prediction = reg.predict (X_test)
print("R2 score from redge regression")
print('Variance score or r2 score: %.2f' % r2_score(y_test, prediction))