# Apriori

# Importing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


veriler = pd.read_csv("sepet.csv",header = None)
lista = []
if veriler.iloc[1,6:7].values != np.nan:
    print("PİPİ")

for i in range(7501):
    list_i = []
    for n in veriler.iloc[i,:].values:
        if type(n) == str:
            list_i.append(n)
    lista.append(list_i)
    
from apyori import apriori
rules = apriori(lista,min_support=0.01,min_confidence=0.2,min_lift = 3,min_length = 2)

print(list(rules))

























