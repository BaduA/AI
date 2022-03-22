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
print(Yas[:,2:])
print(Yas)
imputer = imputer.fit(Yas[:,2:])
Yas[:,2:] = imputer.transform(Yas[:,2:])




