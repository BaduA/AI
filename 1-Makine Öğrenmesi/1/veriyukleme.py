# kütüphaneler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# veri yükleme
veriler = pd.read_csv("veriler.csv")
print(veriler)

# veri ön işleme
boy= veriler[["boy"]]
print(boy)

boykilo = veriler[["boy","kilo"]]













