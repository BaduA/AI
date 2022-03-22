# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


veriler = pd.read_csv("musteriler.csv")

X = veriler.iloc[:,3:].values

# K Means
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2, init="k-means++")
kmeans.fit(X)

print(kmeans.cluster_centers_)

sonuclar = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i,init="k-means++",random_state=123)
    kmeans.fit(X)
    sonuclar.append(kmeans.inertia_)

plt.plot(range(1,11),sonuclar)
plt.show()

kmeans = KMeans(n_clusters=4, init="k-means++",random_state=123)
pred_y = kmeans.fit_predict(X)

plt.scatter(X[pred_y==0,0],X[pred_y==0,1],s=100,c="red")
plt.scatter(X[pred_y==1,0],X[pred_y==1,1],s=100,c="blue")
plt.scatter(X[pred_y==2,0],X[pred_y==2,1],s=100,c="green")
plt.scatter(X[pred_y==3,0],X[pred_y==3,1],s=100,c="black")
plt.show()





from sklearn.cluster import AgglomerativeClustering
"""
affinity: Noktalar arası mesafe ölçümü
linkage : kümeler arası mesafe ölçümü
"""

agg = AgglomerativeClustering(n_clusters=3,affinity="euclidean",linkage="ward")
pred_y = agg.fit_predict(X)
print(pred_y)
a = X[pred_y==0,0]

plt.scatter(X[pred_y==0,0],X[pred_y==0,1],s=100,c="red")
plt.scatter(X[pred_y==1,0],X[pred_y==1,1],s=100,c="blue")
plt.scatter(X[pred_y==2,0],X[pred_y==2,1],s=100,c="green")
plt.show()






















