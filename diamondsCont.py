# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 11:12:24 2018

@author: yz55966
"""
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
#%matplotlib inline

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline

df = pd.read_csv('I:\Intro to Python\Project 1\diamonds.csv', )
df = df.drop('Unnamed: 0', axis = 1)

dummy = df[['cut', 'color', 'clarity']]
numerical = df[['carat', 'depth', 'table', 'x', 'y', 'z']]

y = df['price']
X = df.drop('price', axis = 1)

X = pd.get_dummies(X, columns=dummy)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

from sklearn.linear_model import Ridge
pipeline = make_pipeline(MinMaxScaler(), Ridge())
grid_values = [{'ridge__alpha': [0.01, 0.1, 1, 10, 100, 1000]}]
grid_reg = GridSearchCV(pipeline, param_grid = grid_values, cv = 5)
grid_reg.fit(X_train, y_train)

print('Grid best parameter (max. accuracy): ', grid_reg.best_params_)
print('Grid best score (accuracy): ', grid_reg.best_score_)

np.array(grid_reg.cv_results_['mean_test_score']).reshape(6,1)

# regression evaluation
print("Mean squared error (linear model): {:.2f}".format(mean_squared_error(y_test, grid_reg.predict(X_test))))
print("r2_score (linear model): {:.2f}".format(r2_score(y_test, grid_reg.predict(X_test))))


from sklearn.linear_model import Lasso
pipeline = make_pipeline(MinMaxScaler(), Lasso(max_iter = 10000))
grid_values = [{'lasso__alpha': [0.01, 0.1, 1, 10, 100, 1000]}]
grid_reg = GridSearchCV(pipeline, param_grid = grid_values, cv = 5)
grid_reg.fit(X_train, y_train)

print('Grid best parameter (max. accuracy): ', grid_reg.best_params_)
print('Grid best score (accuracy): ', grid_reg.best_score_)

np.array(grid_reg.cv_results_['mean_test_score']).reshape(6,1)

# regression evaluation
print("Mean squared error (linear model): {:.2f}".format(mean_squared_error(y_test, grid_reg.predict(X_test))))
print("r2_score (linear model): {:.2f}".format(r2_score(y_test, grid_reg.predict(X_test))))





















