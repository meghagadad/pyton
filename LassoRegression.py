# -*- coding: utf-8 -*-
"""
Created on Sun May 27 17:03:17 2018

@author: MeghaGadad
"""

import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
#from numpy import random, float
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# #############################################################################
# load dataset to play with
df = pd.read_csv('Flaveria.csv')
df = pd.get_dummies(df)

data = df.copy()

X = df.loc[:, df.columns != 'Plant Weight(g)']
y = df.iloc[:,0]


# Split data in train set and test set
n_samples = X.shape[0]

X_train, y_train = X[:n_samples // 2], y[:n_samples // 2]
X_test, y_test = X[n_samples // 2:], y[n_samples // 2:]

#from sklearn.model_selection import train_test_split

#data_X_train, data_X_test, target_y_train, target_y_test = train_test_split(data_X, target_y, shuffle=True,
   #                                                 test_size=0.5, random_state=0)


# #############################################################################
# Lasso
from sklearn.linear_model import Lasso

alpha = 0.1
lasso = Lasso(alpha=alpha)

y_pred_lasso = lasso.fit(X_train, y_train).predict(X_test)
r2_score_lasso = r2_score(y_test, y_pred_lasso)
#print(lasso)
print("r^2 on test data using Lasso : %f" % r2_score_lasso)

# #############################################################################
# ElasticNet
from sklearn.linear_model import ElasticNet

enet = ElasticNet(alpha=alpha, l1_ratio=0.7)

y_pred_enet = enet.fit(X_train, y_train).predict(X_test)
r2_score_enet = r2_score(y_test, y_pred_enet)
#print(enet)
print("r^2 on test data using ElasticNet : %f" % r2_score_enet)






