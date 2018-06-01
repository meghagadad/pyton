# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 12:21:24 2018

@author: MeghaGadad
"""

import csv
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score


train = "Flaveria.csv"

train = pd.read_csv(train,delim_whitespace =False);
train = pd.get_dummies(train);

#test file
test = "Flaveria.csv"

test = pd.read_csv(test,delim_whitespace =False);
test = pd.get_dummies(test);

y = []
X = []
for val in train.get_values():
    y.append(val[0])    
    X.append(np.array(val[1:]));

test_y = []
test_X = []
for val1 in test.get_values():
    test_y.append(val1[0])    
    test_X.append(np.array(val[1:]));


X_train = X
y_train = y
X_test = test_X
y_test = test_y

#using Ridge Regression
from sklearn import linear_model
reg = linear_model.BayesianRidge()

#fit the module
reg.fit(X_train, y_train)

#make prediction
prediction = reg.predict (X_test)

print(r2_score(y_test, prediction))