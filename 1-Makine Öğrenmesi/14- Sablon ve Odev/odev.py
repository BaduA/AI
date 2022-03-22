# kütüphaneler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import statsmodels.api as sm # **
from sklearn.metrics import r2_score # **

# veri yükleme
veriler = pd.read_excel("Iris.xls")

x = veriler.iloc[:,0:4].values # bağımsız değişken
y = veriler.iloc[:,4:].values # bağımlı değişken


# train,test  xx
from sklearn.model_selection import train_test_split
x_tr,x_te,y_tr,y_te = train_test_split(x,y,test_size=0.3,random_state=0)

# Scaling xx
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_tr = sc.fit_transform(x_tr)
X_te = sc.fit_transform(x_te)

# Sınıflandırme

# 1. Logistic Regression
from sklearn.linear_model import LogisticRegression
LR = LogisticRegression()
LR.fit(X_tr,y_tr)
pred_lr = LR.predict(X_te)

cm_lr = confusion_matrix(y_te,pred_lr)
print("CM-LR")
print(cm_lr)

# 2. Support Vector Classifier
from sklearn.svm import SVC
svc = SVC(kernel="poly")
svc.fit(X_tr,y_tr)
pred_svc = svc.predict(X_te)

cm_svc = confusion_matrix(y_te,pred_svc)
print("CM-SVC")
print(cm_svc)


# 3. KNN
from sklearn.neighbors import KNeighborsClassifier # **
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_tr,y_tr)
pred_knn = knn.predict(X_te)

cm_knn = confusion_matrix(y_te,pred_knn)
print("CM-KNN")
print(cm_knn)

# 5. Naive Bayes
from sklearn.naive_bayes import GaussianNB #**
nb = GaussianNB()
nb.fit(X_tr,y_tr)
pred_nb = nb.predict(X_te)

cm_nb = confusion_matrix(y_te,pred_nb)
print("CM-NB")
print(cm_nb)

# 6. Decision Tree
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(random_state=0)
dt.fit(X_tr,y_tr)
pred_dt = dt.predict(X_te)

cm_dt = confusion_matrix(y_te, pred_dt)
print("CM-DT")
print(cm_dt)

# RF Regression
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=10,random_state=0)
rf.fit(X_tr,y_tr)
pred_rf = rf.predict(X_te)

cm_rf = confusion_matrix(y_te, pred_rf)
print("CM-RF")
print(cm_rf)











