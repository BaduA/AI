# kütüphaneler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# veri yükleme
veriler = pd.read_csv("CSV/satislar.csv")

# veri ön işleme
aylar= veriler[["Aylar"]]
satislar = veriler[["Satislar"]]


# train,test 
from sklearn.model_selection import train_test_split
x_tr,x_te,y_tr,y_te = train_test_split(aylar,satislar,test_size=0.33,random_state=0)
"""
# Scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_tr = sc.fit_transform(x_tr)
X_te = sc.fit_transform(x_te)

Y_tr = sc.fit_transform(y_tr)
Y_te = sc.fit_transform(y_te)
"""
# Modeling
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_tr,y_tr)
predict = lr.predict(x_te)

# Görselleştirme
x_tr = x_tr.sort_index()
y_tr = y_tr.sort_index()

plt.plot(x_tr,y_tr)
plt.plot(x_te,predict)

plt.title("Aylara Göre Satış")
plt.xlabel("Aylar")
plt.ylabel("Satışlar")




