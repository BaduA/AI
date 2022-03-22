# kütüphaneler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# veri yükleme
veriler = pd.read_csv("CSV/maaslar.csv")

x = veriler.iloc[:,1:2]
y = veriler.iloc[:,2:]
X = x.values
Y = y.values

# Lin Reg
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X,Y)

plt.scatter(X,Y,color="red")

# Poly REG
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=2)
x_poly = poly_reg.fit_transform(X)
print(x_poly)

lin_reg = lr
lin_reg.fit(x_poly,y)

plt.scatter(X,Y)
plt.plot(x,lin_reg.predict(x_poly))
plt.show()

a1 = lin_reg.predict(poly_reg.fit_transform([[6.6]]))

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)
x_poly = poly_reg.fit_transform(X)
print(x_poly)

lin_reg = lr
lin_reg.fit(x_poly,y)

plt.scatter(X,Y)
plt.plot(x,lin_reg.predict(x_poly))
plt.show()

a2 = lin_reg.predict(poly_reg.fit_transform([[6.6]]))


































