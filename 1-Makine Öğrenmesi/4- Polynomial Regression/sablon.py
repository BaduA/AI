# kütüphaneler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# veri yükleme
veriler = pd.read_csv("CSV/maaslar.csv")

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



a1 = lin_reg.predict(poly_reg.fit_transform([[6.6]]))
a2 = lin_reg2.predict(poly_reg2.fit_transform([[6.6]]))


























