# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 17:51:15 2022

@author: emrec
"""
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV

#%%
iris = load_iris()

x = iris.data
y = iris.target
#%% Grid search CV with logistic regression
x = x[:100,:]
y = y[:100]

#%%
x = (x-np.min(x))/(np.max(x)-np.min(x))

#%% train test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.3)

from sklearn.linear_model import LogisticRegression

grid = {"C":np.logspace(-3,3,7),"penalty":['l1','l2'],"solver":["liblinear"]}

logreg =LogisticRegression(solver="liblinear")
logreg_cv = GridSearchCV(logreg, grid, cv = 10)
logreg_cv.fit(x_train,y_train)

print("tuned hyperparameters: (best parameters):",logreg_cv.best_params_)
print("accuracy:",logreg_cv.best_score_)

#%%

logreg2 = LogisticRegression(C=1,penalty="l2")
logreg2.fit(x_train,y_train)
print("score: ", logreg2.score(x_test,y_test))