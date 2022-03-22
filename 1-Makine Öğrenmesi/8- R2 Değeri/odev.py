# Yöntemler: MLR, PR, SVR, DT, RR
# Model çıkar, yöntem başarılarını karşılaştır, 10 yıl tecrübeli 100 puan almış CEO ve aynı özelliklere sahip bir müdürün maaşlarını 5 yöntemle tahmin edip sonuçlarını yorumla.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import statsmodels.api as sm

veriler = pd.read_csv("maaslar_yeni.csv")
veriler = veriler.iloc[:,2:]

input = veriler.iloc[:,0:3].values
output = veriler.iloc[:,-1:].values

x_tr,x_te,y_tr,y_te = train_test_split(input,output,random_state=0)

# ♂ Linear Regression
lr = LinearRegression()
lr.fit(x_tr,y_tr)
pred = lr.predict(x_te)
R2_lr = r2_score(y_te, lr.predict(x_te))
# print(R2_lr)
model = sm.OLS(lr.predict(x_te),x_te).fit()
print(model.summary())




# │ Polynomial Regression
poly = PolynomialFeatures(degree=2)
xtr_poly = poly.fit_transform(x_tr)
xte_poly = poly.fit_transform(x_te)
poly_reg = LinearRegression()
poly_reg.fit(xtr_poly,y_tr)
pred = poly_reg.predict(xte_poly)
R2_po = r2_score(poly_reg.predict(xte_poly),y_te)

model2 = sm.OLS(poly_reg.predict(xte_poly),xte_poly).fit()
print(model2.summary())



# ♦ Support Vector Regression
sc = StandardScaler()
xtr_sc = sc.fit_transform(x_tr)
ytr_sc = sc.fit_transform(y_tr)
xte_sc = sc.fit_transform(x_te)
yte_sc = sc.fit_transform(y_te)

svr = SVR(kernel="rbf")
svr.fit(xtr_sc,ytr_sc)
R2_svr = r2_score(yte_sc,svr.predict(xte_sc))
# print("SVR R2")
# print(R2_svr)

model3 = sm.OLS(svr.predict(xte_sc),xte_sc).fit()
print(model3.summary())



# DT Regression
dt = DecisionTreeRegressor(random_state=0)
dt.fit(x_tr,y_tr)
R2_dt = r2_score(y_te,dt.predict(x_te))
# print("DT R2")
# print(R2_dt)
"""0.255345502417066"""

model4 = sm.OLS(dt.predict(x_te),x_te).fit()
print(model4.summary())


# RF Regression
rf = RandomForestRegressor(n_estimators=10,random_state=0)
rf.fit(x_tr,y_tr)
R2_rf = r2_score(y_te,rf.predict(x_te))
# print("RF R2")
# print(R2_rf)

model5 = sm.OLS(rf.predict(x_te),x_te).fit()
print(model5.summary())


















































































