# kütüphaneler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# veri yükleme
veriler = pd.read_csv("CSV/veriler.csv")


# Kategorik Donüsüm: Kategorik -> Numeric
ulke = veriler.iloc[:,0:1].values
print("*****************")
print(ulke)
print("*****************")

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
ulke[:,0] = le.fit_transform(veriler.iloc[:,0])
print(ulke)

ohe = preprocessing.OneHotEncoder()
ulke = ohe.fit_transform(ulke).toarray()
print(ulke)

c = veriler.iloc[:,-1:].values
# print("*********************")
# print(veriler.iloc[:,-1:])
# print("*********************")
# print(veriler.iloc[:,-1])

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
c[:,-1] = le.fit_transform(veriler.iloc[:,-1])
print(c)

ohe = preprocessing.OneHotEncoder()
c = ohe.fit_transform(c).toarray()
print(c)

# Data Framing
Yas = veriler.iloc[:,1:4].values

sonuc = pd.DataFrame(data=ulke,index=range(22),columns=["fr","tr","us"])  # DataFrame'ler index ve kolon başlıklarına sahiptir
sonuc2 = pd.DataFrame(data=Yas,index=range(22),columns=["boy","kilo","yas"])
cinsiyet = veriler.iloc[:,-1].values
sonuc3 = pd.DataFrame(data=c[:,:1],index=range(22),columns=["cinsiyet"])

# concatting
s = pd.concat([sonuc,sonuc2],axis=1)
s2 = pd.concat([s,sonuc3],axis=1)

# train,test 
from sklearn.model_selection import train_test_split
x_tr,x_te,y_tr,y_te = train_test_split(s,sonuc3,test_size=0.33,random_state=0)
 
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_tr,y_tr)

y_pred = regressor.predict(x_te)




boy = s2.iloc[:,3:4].values
sol = s2.iloc[:,:3]
sag = s2.iloc[:,4:]

veri = pd.concat([sol,sag],axis=1)
x_tr,x_te,y_tr,y_te = train_test_split(veri,boy,test_size=0.33,random_state=0)

regressor.fit(x_tr,y_tr)
y_pred = regressor.predict(x_te)


import statsmodels.api as sm

X = np.append(arr=np.ones((22,1)).astype(int),values=veri,axis=1)

X_l = veri.iloc[:,[0,1,2,3,4,5]].values
X_l = np.array(X_l,dtype=float) #bir anlamı yok sanırım
model = sm.OLS(boy,X_l).fit()
print(model.summary())

X_l = veri.iloc[:,[0,1,2,3,5]].values
X_l = np.array(X_l,dtype=float) #bir anlamı yok sanırım
model = sm.OLS(boy,X_l).fit()
print(model.summary())












