# kütüphaneler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# veri yükleme
veriler = pd.read_csv("veriler.csv")

x = veriler.iloc[:,1:4].values # bağımsız değişken
y = veriler.iloc[:,4:].values # bağımlı değişken


# train,test 
from sklearn.model_selection import train_test_split
x_tr,x_te,y_tr,y_te = train_test_split(x,y,test_size=0.33,random_state=0)
 
# Scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_tr = sc.fit_transform(x_tr)
X_te = sc.transform(x_te)

# Sınıflandırma
# Logistic Regression
from sklearn.linear_model import LogisticRegression

logr = LogisticRegression(random_state=0)
logr.fit(X_tr,y_tr)
predict = logr.predict(X_te)

cm = confusion_matrix(y_te, predict)
print("LR")
print(cm)

# KNN
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1,metric="minkowski")
knn.fit(X_tr,y_tr)

pred_knn = knn.predict(X_te)

cm_knn = confusion_matrix(y_te, pred_knn)
print("KNN")
print(cm_knn)

# Support Vector
from sklearn.svm import SVC
svc = SVC(kernel ="rbf") # poly>rbf>linear bu kümede
svc.fit(X_tr,y_tr)

pred_svc = svc.predict(X_te)

cm_svc = confusion_matrix(y_te, pred_svc)
print("SVM")
print(cm_svc)


# Naive Bayes

from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(X_tr,y_tr)

pred_nb = nb.predict(X_te)

cm_nb = confusion_matrix(y_te,pred_nb)
print("NB")
print(cm_nb)

# Bernoulli     -> Binary
# Gaussian      -> Continious bir değerse
# Multinominal  -> İnteger sayılarla adlandırılıyorsa, araba markanız gibi.

from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier(criterion="entropy")
dtc.fit(X_tr,y_tr)

pred_dtc = dtc.predict(X_te)

cm_dtc = confusion_matrix(y_te,pred_dtc)
print("DTC")
print(cm_dtc)


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=10,criterion="entropy")
rfc.fit(X_tr,y_tr)

pred_rfc = rfc.predict(X_te)

cm_rfc = confusion_matrix(y_te,pred_rfc)
print("RFC")
print(cm_rfc)












