# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 14:30:36 2018

@author: yz55966
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#%matplotlib inline

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import validation_curve
from sklearn.pipeline import make_pipeline

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.externals import joblib 

df = pd.read_csv('I:\Intro to Python\Project 1\diamonds.csv')
df = df.drop('Unnamed: 0', axis = 1)
df['price'] = pd.cut(df['price'], [0,1000,4000,10000,100000], labels = ['low', 'medium', 'high', 'luxury'])
#Now lets assign a labels to our quality variable
label = LabelEncoder()
df['price_ind'] = label.fit_transform(df['price'])
mapping = pd.Series(label.transform(label.classes_), index = label.classes_)

dict(zip(label.classes_, label.transform(label.classes_)))

dict = {'low': 1, 'medium': 2, 'high': 3, 'luxury': 4}
df['price'] = df['price'].map(dict)

# 1. summary statistics
df.describe()
df.isnull().any()
df.isnull().sum()
df = df.dropna()
df.info()

['carat', 'cut', 'color', 'clarity', 'depth', 'table', 'price', 'x', 'y', 'z']

df['cut'].value_counts()
plt.subplots(figsize=(6,6))
sns.countplot('cut',data=df,palette='RdYlGn_r',edgecolor=sns.color_palette('dark',7))
plt.xticks()
plt.title('Count Plot')
plt.show()

df['color'].value_counts()
plt.subplots(figsize=(6,6))
sns.countplot('color',data=df,palette='RdYlGn_r',edgecolor=sns.color_palette('dark',7))
plt.xticks()
plt.title('Count Plot')
plt.show()

df['clarity'].value_counts()
plt.subplots(figsize=(6,6))
sns.countplot('clarity',data=df,palette='RdYlGn_r',edgecolor=sns.color_palette('dark',7))
plt.xticks()
plt.title('Count Plot')
plt.show()

df['carat'].describe()
sns.distplot(df['carat'])
sns.boxplot(df['carat'])

fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'carat', y = 'price', data = df)

df['depth'].describe()
sns.distplot(df['depth'])
sns.boxplot(df['depth'])

fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'depth', y = 'price', data = df)

df['table'].describe()
sns.distplot(df['table'])
sns.boxplot(df['table'])

fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'table', y = 'price', data = df)

dummy = df[['cut', 'color', 'clarity']]
numerical = df[['carat', 'depth', 'table', 'x', 'y', 'z']]

# two_way correlation
from matplotlib import cm
cmap = cm.get_cmap('gnuplot')
scatter = pd.scatter_matrix(numerical, c= y, marker = 'o', s=40, 
                            hist_kwds={'bins':15}, figsize=(9,9), cmap=cmap)

pd.tools.plotting.scatter_matrix(numerical)

#(2) correlation matrix
numerical.corr()

#(3) summary statistics
numerical.describe()

#2. Preprocessing
X = df[['carat', 'depth', 'table', 'x', 'y', 'z', 'cut', 'color', 'clarity']]
y = df['price']

#(1) dummy variables
X = pd.get_dummies(X, columns=dummy)

# y = pd.get_dummies(y)

#(2) train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 3. knn
from sklearn.neighbors import KNeighborsClassifier
pipeline = make_pipeline(MinMaxScaler(), KNeighborsClassifier())
grid_values = {'kneighborsclassifier__n_neighbors': [1, 3, 5, 10, 15, 25, 30, 35]}
grid_clf = GridSearchCV(pipeline, param_grid = grid_values, 
                        scoring = 'f1_weighted')
grid_clf.fit(X_train, y_train)

print('Grid best parameter (max. accuracy): ', grid_clf.best_params_)
print('Grid best score (accuracy): ', grid_clf.best_score_)

print('accuracy score: ', accuracy_score(y_test, grid_clf.predict(X_test)))
print('precision score: ', precision_score(y_test, grid_clf.predict(X_test), average = 'weighted'))
print('recall score: ', recall_score(y_test, grid_clf.predict(X_test), average = 'weighted'))
print('f1 score: ', f1_score(y_test, grid_clf.predict(X_test), average = 'weighted'))

confusion_mc = confusion_matrix(y_test, grid_clf.predict(X_test))
df_cm = pd.DataFrame(confusion_mc, 
                     index = [i for i in range(0,4)], columns = [i for i in range(0,4)])

plt.figure(figsize=(5.5,4))
sns.heatmap(df_cm, annot=True)
plt.title('KNN \nF1 Score:{0:.3f}'.format(f1_score(y_test, grid_clf.predict(X_test), average = 'weighted')))
plt.ylabel('True label')
plt.xlabel('Predicted label')

print(classification_report(y_test, grid_clf.predict(X_test)))

param_range = [1, 3, 5, 10, 15, 25, 30, 35]
train_scores, test_scores = validation_curve(pipeline, X, y,
                                            param_name='kneighborsclassifier__n_neighbors',
                                            param_range=param_range, cv=3)
plt.figure()

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.title('Validation Curve with KNN')
plt.xlabel('Number of Neighbors')
plt.ylabel('Score')
plt.ylim(0.0, 1.1)
lw = 2

plt.semilogx(param_range, train_scores_mean, label='Training score',
            color='darkorange', lw=lw)

plt.fill_between(param_range, train_scores_mean - train_scores_std,
                train_scores_mean + train_scores_std, alpha=0.2,
                color='darkorange', lw=lw)

plt.semilogx(param_range, test_scores_mean, label='Cross-validation score',
            color='navy', lw=lw)

plt.fill_between(param_range, test_scores_mean - test_scores_std,
                test_scores_mean + test_scores_std, alpha=0.2,
                color='navy', lw=lw)

plt.legend(loc='best')
plt.show()


# 4. logistic regression
from sklearn.linear_model import LogisticRegression
pipeline = make_pipeline(MinMaxScaler(), LogisticRegression())
grid_values = {'logisticregression__penalty': ['l1', 'l2'], 
               'logisticregression__C': [1000, 500, 100, 10, 1, 0.1, 0.01]}
grid_clf = GridSearchCV(pipeline, param_grid = grid_values, scoring = 'f1_weighted')
grid_clf.fit(X_train, y_train)

y_decision_fn_scores = grid_clf.decision_function(X_test)

print('Grid best parameter (max. accuracy): ', grid_clf.best_params_)
print('Grid best score (accuracy): ', grid_clf.best_score_)

print('accuracy score: ', accuracy_score(y_test, grid_clf.predict(X_test)))
print('precision score: ', precision_score(y_test, grid_clf.predict(X_test), average = 'weighted'))
print('recall score: ', recall_score(y_test, grid_clf.predict(X_test), average = 'weighted'))
print('f1 score: ', f1_score(y_test, grid_clf.predict(X_test), average = 'weighted'))

confusion_mc = confusion_matrix(y_test, grid_clf.predict(X_test))
df_cm = pd.DataFrame(confusion_mc, 
                     index = [i for i in range(0,4)], columns = [i for i in range(0,4)])

plt.figure(figsize=(5.5,4))
sns.heatmap(df_cm, annot=True)
plt.title('Logistic Regression \nF1 Score:{0:.3f}'.format(f1_score(y_test, grid_clf.predict(X_test), average = 'weighted')))
plt.ylabel('True label')
plt.xlabel('Predicted label')

print(classification_report(y_test, grid_clf.predict(X_test)))

param_range = [1000, 500, 100, 10, 1, 0.1, 0.01]
train_scores, test_scores = validation_curve(pipeline, X, y,
                                            param_name='logisticregression__C',
                                            param_range=param_range, cv=3)
plt.figure()

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.title('Validation Curve with Logistic Regression')
plt.xlabel('Regularization Parameter C')
plt.ylabel('Score')
plt.ylim(0.0, 1.01)
lw = 2

plt.semilogx(param_range, train_scores_mean, label='Training score',
            color='darkorange', lw=lw)

plt.fill_between(param_range, train_scores_mean - train_scores_std,
                train_scores_mean + train_scores_std, alpha=0.2,
                color='darkorange', lw=lw)

plt.semilogx(param_range, test_scores_mean, label='Cross-validation score',
            color='navy', lw=lw)

plt.fill_between(param_range, test_scores_mean - test_scores_std,
                test_scores_mean + test_scores_std, alpha=0.2,
                color='navy', lw=lw)

plt.legend(loc='best')
plt.show()


# 5. SVC
from sklearn.svm import SVC
pipeline = make_pipeline(MinMaxScaler(), SVC())
grid_values = [{'svc__C': [0.01, 0.1, 1, 10, 100, 1000], 'svc__kernel': ['linear']}, 
                  {'svc__C': [0.01, 0.1, 1, 10, 100, 1000], 
                   'svc__gamma': [0.0001, 0.001, 0.01, 0.1, 1, 10], 'svc__kernel': ['rbf']}]
grid_clf = GridSearchCV(pipeline, param_grid = grid_values, scoring = 'f1_weighted')
grid_clf.fit(X_train, y_train)

y_decision_fn_scores = grid_clf.decision_function(X_test)

print('Grid best parameter (max. accuracy): ', grid_clf.best_params_)
print('Grid best score (accuracy): ', grid_clf.best_score_)

print('accuracy score: ', accuracy_score(y_test, grid_clf.predict(X_test)))
print('precision score: ', precision_score(y_test, grid_clf.predict(X_test), average = 'weighted'))
print('recall score: ', recall_score(y_test, grid_clf.predict(X_test), average = 'weighted'))
print('f1 score: ', f1_score(y_test, grid_clf.predict(X_test), average = 'weighted'))

confusion_mc = confusion_matrix(y_test, grid_clf.predict(X_test))
df_cm = pd.DataFrame(confusion_mc, 
                     index = [i for i in range(0,3)], columns = [i for i in range(0,3)])

plt.figure(figsize=(5.5,4))
sns.heatmap(df_cm, annot=True)
plt.title('SVC \nAccuracy:{0:.3f}'.format(f1_score(y_test, grid_clf.predict(X_test), average = 'weighted')))
plt.ylabel('True label')
plt.xlabel('Predicted label')

print(classification_report(y_test, grid_clf.predict(X_test)))

# C
param_range = [0.01, 0.1, 1, 10, 100, 1000]
train_scores, test_scores = validation_curve(pipeline, X, y,
                                            param_name='svc__C',
                                            param_range=param_range, cv=3)
plt.figure()

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.title('Validation Curve with SVC')
plt.xlabel('Regularization Parameter C')
plt.ylabel('Score')
plt.ylim(0.0, 1.01)
lw = 2

plt.semilogx(param_range, train_scores_mean, label='Training score',
            color='darkorange', lw=lw)

plt.fill_between(param_range, train_scores_mean - train_scores_std,
                train_scores_mean + train_scores_std, alpha=0.2,
                color='darkorange', lw=lw)

plt.semilogx(param_range, test_scores_mean, label='Cross-validation score',
            color='navy', lw=lw)

plt.fill_between(param_range, test_scores_mean - test_scores_std,
                test_scores_mean + test_scores_std, alpha=0.2,
                color='navy', lw=lw)

plt.legend(loc='best')
plt.show()

# gamma
param_range = [0.0001, 0.001, 0.01, 0.1, 1, 10]
train_scores, test_scores = validation_curve(pipeline, X, y,
                                            param_name='svc__gamma',
                                            param_range=param_range, cv=3)
plt.figure()

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.title('Validation Curve with SVC')
plt.xlabel('gamma')
plt.ylabel('Score')
plt.ylim(0.0, 1.01)
lw = 2

plt.semilogx(param_range, train_scores_mean, label='Training score',
            color='darkorange', lw=lw)

plt.fill_between(param_range, train_scores_mean - train_scores_std,
                train_scores_mean + train_scores_std, alpha=0.2,
                color='darkorange', lw=lw)

plt.semilogx(param_range, test_scores_mean, label='Cross-validation score',
            color='navy', lw=lw)

plt.fill_between(param_range, test_scores_mean - test_scores_std,
                test_scores_mean + test_scores_std, alpha=0.2,
                color='navy', lw=lw)

plt.legend(loc='best')
plt.show()


# 6. Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
pipeline = make_pipeline(MinMaxScaler(), RandomForestClassifier(n_estimators=100))
grid_values = { 'randomforestclassifier__max_features' : ['auto', None, 'log2'],
                  'randomforestclassifier__max_depth': [None, 7, 5, 3, 1]}
grid_clf = GridSearchCV(pipeline, param_grid = grid_values, scoring = 'f1_weighted')
grid_clf.fit(X_train, y_train)

print('Grid best parameter (max. accuracy): ', grid_clf.best_params_)
print('Grid best score (accuracy): ', grid_clf.best_score_)

print('accuracy score: ', accuracy_score(y_test, grid_clf.predict(X_test)))
print('precision score: ', precision_score(y_test, grid_clf.predict(X_test), average = 'weighted'))
print('recall score: ', recall_score(y_test, grid_clf.predict(X_test), average = 'weighted'))
print('f1 score: ', f1_score(y_test, grid_clf.predict(X_test), average = 'weighted'))

confusion_mc = confusion_matrix(y_test, grid_clf.predict(X_test))
df_cm = pd.DataFrame(confusion_mc, 
                     index = [i for i in range(0,3)], columns = [i for i in range(0,3)])

plt.figure(figsize=(5.5,4))
sns.heatmap(df_cm, annot=True)
plt.title('Random Forest \nAccuracy:{0:.3f}'.format(f1_score(y_test, grid_clf.predict(X_test), average = 'weighted')))
plt.ylabel('True label')
plt.xlabel('Predicted label')

print(classification_report(y_test, grid_clf.predict(X_test)))

# Depth
param_range = [9, 7, 5, 3, 1]
train_scores, test_scores = validation_curve(pipeline, X, y,
                                            param_name='randomforestclassifier__max_depth',
                                            param_range=param_range, cv=3)
plt.figure()

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.title('Validation Curve with SVC')
plt.xlabel('Maximal Depth')
plt.ylabel('Score')
plt.ylim(0.0, 1.01)
lw = 2

plt.semilogx(param_range, train_scores_mean, label='Training score',
            color='darkorange', lw=lw)

plt.fill_between(param_range, train_scores_mean - train_scores_std,
                train_scores_mean + train_scores_std, alpha=0.2,
                color='darkorange', lw=lw)

plt.semilogx(param_range, test_scores_mean, label='Cross-validation score',
            color='navy', lw=lw)

plt.fill_between(param_range, test_scores_mean - test_scores_std,
                test_scores_mean + test_scores_std, alpha=0.2,
                color='navy', lw=lw)

plt.legend(loc='best')
plt.show()

# gamma
param_range = [0.0001, 0.001, 0.01, 0.1, 1, 10]
train_scores, test_scores = validation_curve(pipeline, X, y,
                                            param_name='svc__gamma',
                                            param_range=param_range, cv=3)
plt.figure()

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.title('Validation Curve with SVC')
plt.xlabel('gamma')
plt.ylabel('Score')
plt.ylim(0.0, 1.01)
lw = 2

plt.semilogx(param_range, train_scores_mean, label='Training score',
            color='darkorange', lw=lw)

plt.fill_between(param_range, train_scores_mean - train_scores_std,
                train_scores_mean + train_scores_std, alpha=0.2,
                color='darkorange', lw=lw)

plt.semilogx(param_range, test_scores_mean, label='Cross-validation score',
            color='navy', lw=lw)

plt.fill_between(param_range, test_scores_mean - test_scores_std,
                test_scores_mean + test_scores_std, alpha=0.2,
                color='navy', lw=lw)

plt.legend(loc='best')
plt.show()





































