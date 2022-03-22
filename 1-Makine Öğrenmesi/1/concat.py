# kütüphaneler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# veri yükleme
veriler = pd.read_csv("csv/eksikveriler.csv")

# veri ön işleme
boy= veriler[["boy"]]
boykilo = veriler[["boy","kilo"]]

# eksikveriler
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan,strategy="mean")
Yas = veriler.iloc[:,1:4].values
imputer = imputer.fit(Yas[:,2:])
Yas[:,2:] = imputer.transform(Yas[:,2:])

# Kategorik Donüsüm: Kategorik -> Numeric
ulke = veriler.iloc[:,0:1].values

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
ulke[:,0] = le.fit_transform(veriler.iloc[:,0])
print(ulke)

ohe = preprocessing.OneHotEncoder()
ulke = ohe.fit_transform(ulke).toarray()
print(ulke)

# Data Framing
sonuc = pd.DataFrame(data=ulke,index=range(22),columns=["fr","tr","us"])  # DataFrame'ler index ve kolon başlıklarına sahiptir
sonuc2 = pd.DataFrame(data=Yas,index=range(22),columns=["boy","kilo","yas"])
cinsiyet = veriler.iloc[:,-1].values
sonuc3 = pd.DataFrame(data=cinsiyet,index=range(22),columns=["cinsiyet"])

# concatting
s = pd.concat([sonuc,sonuc2],axis=1)
s2 = pd.concat([s,sonuc3],axis=1)

# train,test 
from sklearn.model_selection import train_test_split
x_tr,x_te,y_tr,y_te = train_test_split(s,sonuc3,test_size=0.33,random_state=0)
 
# Scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_tr = sc.fit_transform(x_tr)
X_te = sc.fit_transform(x_te)








