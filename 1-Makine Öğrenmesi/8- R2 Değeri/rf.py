# kütüphaneler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# veri yükleme
veriler = pd.read_csv("maaslar.csv")

# DataFrame slicing
x = veriler.iloc[:,1:2]
y = veriler.iloc[:,2:]

# NumPy array
X = x.values
Y = y.values

# Lin Reg
# Doğrusal Model Oluşturma

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X,Y)



# Poly REG
# Non Linear Model Oluşturma

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=2)
x_poly = poly_reg.fit_transform(X)
lin_reg = LinearRegression()
lin_reg.fit(x_poly,y)

# 4th degree polynomial regression

poly_reg2 = PolynomialFeatures(degree=4)
x_poly2 = poly_reg2.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly2,y)




# Görselleştirme
plt.scatter(X,Y,color="red")
plt.plot(x,lr.predict(X),color="blue")
plt.show()

plt.scatter(X,Y)
plt.plot(x,lin_reg.predict(x_poly))
plt.show()

plt.scatter(X,Y)
plt.plot(x,lin_reg2.predict(x_poly2))
plt.show()

# Tahmin ölçümü
a1 = lin_reg.predict(poly_reg.fit_transform([[6.6]]))
a2 = lin_reg2.predict(poly_reg2.fit_transform([[6.6]]))


# veri ölçeklenmesi
from sklearn.preprocessing import StandardScaler

sc1 = StandardScaler()
x_sc = sc1.fit_transform(X)
sc2 = StandardScaler()
y_sc = sc2.fit_transform(Y)

from sklearn.svm import SVR
svr = SVR(kernel ="rbf")
svr.fit(x_sc,y_sc)

plt.scatter(x_sc,y_sc,color="green")
plt.plot(x_sc,svr.predict(x_sc))
plt.show()

from sklearn.tree import DecisionTreeRegressor
d_re = DecisionTreeRegressor(random_state=0)
d_re.fit(X,Y)

plt.scatter(X,Y)
plt.plot(X,d_re.predict(X))
plt.show()

d1 = d_re.predict([[6.6]])
d2 = d_re.predict([[11]])
d3 = d_re.predict([[200]])


from sklearn.ensemble import RandomForestRegressor
RF = RandomForestRegressor(n_estimators=10,random_state=0)
# n_estimators = kaç tane decision tree kullanılacak değeri.
RF.fit(X,Y.ravel())

R1 = RF.predict([[7.6]])









