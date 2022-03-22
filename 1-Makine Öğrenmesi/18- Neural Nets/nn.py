# kütüphaneler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# veri yükleme
veriler = pd.read_csv("Churn_Modelling.csv")

# veri ön işleme
X = veriler.iloc[:,3:13].values
Y = veriler.iloc[:,-1:].values


Xsol = veriler.iloc[:,3:4]
Xsag = veriler.iloc[:,6:13]
country = X[:,1:2]
cinsiyet = X[:,2:3]




# Kategorik Donüsüm: Kategorik -> Numeric

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
cinsiyet = le.fit_transform(cinsiyet)

ohe = preprocessing.OneHotEncoder()
country = ohe.fit_transform(country).toarray()

# Data Framing
country = pd.DataFrame(data=country,index=range(10000),columns=["fr","de","sp"])  # DataFrame'ler index ve kolon başlıklarına sahiptir
cinsiyet = pd.DataFrame(data=cinsiyet,index=range(10000),columns=["cinsiyet"])

# concatting
s = pd.concat([Xsol,country],axis=1)
s2 = pd.concat([s,cinsiyet],axis=1)
s3 = pd.concat([s2,Xsag],axis=1)

# train,test 
from sklearn.model_selection import train_test_split
x_tr,x_te,y_tr,y_te = train_test_split(s3,Y,test_size=0.33,random_state=0)
 
# Scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_tr = sc.fit_transform(x_tr)
X_te = sc.fit_transform(x_te)


import keras

from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()
classifier.add(Dense(6,kernel_initializer="random_uniform",activation="relu",input_dim=12))
classifier.add(Dense(6,kernel_initializer="random_uniform",activation="relu"))
classifier.add(Dense(1,kernel_initializer="random_uniform",activation="sigmoid"))

classifier.compile(optimizer="adam", loss="binary_crossentropy", metrics = ["accuracy"])

classifier.fit(X_tr,y_tr,epochs=50)
pred = classifier.predict(X_te)


pred = (pred >= 0.5)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_te,pred)
print(cm)







"""
Dense kernel_initializer="random_normal"
Initializer that generates tensors with a normal distribution.

Arguments

mean: a python scalar or a scalar tensor. Mean of the random values to generate.
stddev: a python scalar or a scalar tensor. Standard deviation of the random values to generate.
seed: A Python integer. An initializer created with a given seed will always produce the same random tensor for a given shape and dtype.

---------------------------------------------------------------

Dense kernel_initializer="random_uniform"
Initializer that generates tensors with a uniform distribution.

Arguments

minval: A python scalar or a scalar tensor. Lower bound of the range of random values to generate (inclusive).
maxval: A python scalar or a scalar tensor. Upper bound of the range of random values to generate (exclusive).
seed: A Python integer. An initializer created with a given seed will always produce the same random tensor for a given shape and dtype.

---------------------------------------------------------------

Dense kernel_initializer="truncated_normal
Initializer that generates a truncated normal distribution.

Arguments

mean: a python scalar or a scalar tensor. Mean of the random values to generate.
stddev: a python scalar or a scalar tensor. Standard deviation of the random values to generate before truncation.
seed: A Python integer. An initializer created with a given seed will always produce the same random tensor for a given shape and dtype.
"""





































