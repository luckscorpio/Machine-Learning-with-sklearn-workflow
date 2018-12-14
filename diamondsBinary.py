# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 13:22:02 2018

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
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('I:\Intro to Python\Project 1\diamonds.csv', )
df = df.drop('Unnamed: 0', axis = 1)
df['price'] = pd.cut(df['price'], [0,10000,100000], labels = ['low', 'high'])

dict = {'low': 0,  'high': 1}
df['price'] = df['price'].map(dict)

count_classes = pd.value_counts(df['price'], sort = True).sort_index()
count_classes.plot(kind = 'bar')

sn.distplot(df['price'])
sn.pairplot(df)

dummy = df[['cut', 'color', 'clarity']]
numerical = df[['carat', 'depth', 'table', 'x', 'y', 'z']]

y = df['price']
X = df.drop('price', axis = 1)

X = pd.get_dummies(X, columns=dummy)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)



df.isnull().sum()
ts.info()
print(df.dtypes)
df.diff()
terror.rename(columns={'iyear':'Year','imonth':'Month'},inplace=True)

# count values
print('Regions with Highest Terrorist Attacks:',terror['Region'].value_counts().index[0])
print('Maximum people killed in an attack are:',terror['Killed'].max(),'that took place in',terror.loc[terror['Killed'].idxmax()].Country)

# count plot
plt.subplots(figsize=(15,6))
sns.countplot('Year',data=terror,palette='RdYlGn_r',edgecolor=sns.color_palette('dark',7))
plt.xticks(rotation=90)
plt.title('Number Of Terrorist Activities Each Year')
plt.show()

# ordered by number of counts
plt.subplots(figsize=(15,6))
sns.countplot('Region',data=terror,palette='RdYlGn',edgecolor=sns.color_palette('dark',7),order=terror['Region'].value_counts().index)
plt.xticks(rotation=90)
plt.title('Number Of Terrorist Activities By Region')
plt.show()

#explore data
fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'volatile acidity', data = wine)































