# kütüphaneler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


# veri yükleme
veriler = pd.read_csv("veriler.csv")

x = veriler.iloc[:,1:4].values
y = veriler.iloc[:,4:].values


# train,test 
from sklearn.model_selection import train_test_split
x_tr,x_te,y_tr,y_te = train_test_split(x,y,test_size=0.33,random_state=0)
 
# Scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_tr = sc.fit_transform(x_tr)
X_te = sc.transform(x_te)

# Logistic Regression
from sklearn.linear_model import LogisticRegression

logr = LogisticRegression(random_state=0)
logr.fit(X_tr,y_tr)
predict = logr.predict(X_te)


cm = confusion_matrix(y_te, predict)
print(cm)


from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1,metric="minkowski")
knn.fit(X_tr,y_tr)
pred_knn = knn.predict(X_te)

cm_knn = confusion_matrix(y_te, pred_knn)
print(cm_knn)



































