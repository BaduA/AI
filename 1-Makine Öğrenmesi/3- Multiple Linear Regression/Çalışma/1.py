# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder

veriler = pd.read_csv("odev_tenis.csv")
input = veriler.iloc[:,0:4].values
output = veriler.iloc[:,-1].values


outlook = veriler.iloc[:,0:1].values
windy = veriler.iloc[:,3:4].values
rest = veriler.iloc[:,1:3]
play = veriler.iloc[:,-1:].values

ohe = preprocessing.OneHotEncoder()
outlook = ohe.fit_transform(veriler.iloc[:,0:1]).toarray()

le = preprocessing.LabelEncoder()

windy = le.fit_transform(veriler.iloc[:,3:4])
play = le.fit_transform(veriler.iloc[:,-1:])

outlook = pd.DataFrame(data=outlook,index=range(14),columns=["overcast","rainy","sunny"])
windy = pd.DataFrame(data=windy,index=range(14),columns=["windy"])
play = pd.DataFrame(data=play,index=range(14),columns=["play"])

result = pd.concat([outlook,rest],axis=1)
result = pd.concat([result,windy],axis=1)

from sklearn.model_selection import train_test_split

x_tr,x_te,y_tr,y_te = train_test_split(result,play,random_state=0)


from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_tr,y_tr)
y_pred = lr.predict(x_te)


import statsmodels.api as sm
X = np.append(arr=np.ones((14,1)).astype(int),values = result,axis=1)
X_l = result.iloc[:,[0,1,2,3,4,5]].values
print(sm.OLS(play,X_l).fit().summary())

X = np.append(arr=np.ones((14,1)).astype(int),values = result,axis=1)
X_l = result.iloc[:,[0,1,2,4,5]].values
print(sm.OLS(play,X_l).fit().summary())

X = np.append(arr=np.ones((14,1)).astype(int),values = result,axis=1)
X_l = result.iloc[:,[0,1,2,5]].values
print(sm.OLS(play,X_l).fit().summary())

X = np.append(arr=np.ones((14,1)).astype(int),values = result,axis=1)
X_l = result.iloc[:,[0,1,2]].values
print(sm.OLS(play,X_l).fit().summary())

X = np.append(arr=np.ones((14,1)).astype(int),values = result,axis=1)
X_l = result.iloc[:,[0,1]].values
print(sm.OLS(play,X_l).fit().summary())



x_tr,x_te,y_tr,y_te = train_test_split(X_l,play,random_state=0)


from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_tr,y_tr)
y_pred = lr.predict(x_te)























