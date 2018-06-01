# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 14:21:08 2018

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

###########Training and Making Predictions using Lasso
from sklearn import linear_model
reg = linear_model.LassoLars(alpha=.1)
reg.fit(X_train, y_train)

prediction = reg.predict (X_test)

print(r2_score(y_test, prediction))