# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

veriler = pd.read_csv("Restaurant_Reviews.csv")



import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

ps = PorterStemmer()

durma = nltk.download("stopwords")


derlem = []
for i in range(999):
    yorum = re.sub("[^a-zA-Z]"," ",veriler["Review"][i])
    yorum = yorum.lower()
    yorum = yorum.split()
    yorum = [ps.stem(kelime) for kelime in yorum if not kelime in set(stopwords.words("english"))]
    yorum =" ".join(yorum)
    derlem.append(yorum)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=2000)
X = cv.fit_transform(derlem).toarray()
Y = veriler.iloc[:,1].values

from sklearn.model_selection import train_test_split

x_tr,x_te,y_tr,y_te = train_test_split(X,Y,test_size=0.20,random_state=0)



from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(x_tr,y_tr)
pred = gnb.predict(x_te)


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_te,pred)
print(cm)





print(yorum)
























































