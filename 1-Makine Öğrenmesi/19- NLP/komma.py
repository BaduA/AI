import re
r = open('Restaurant_Reviews.csv', 'r')
w = open('Restaurant_Reviews_No_Commas.csv', 'a')
w.write(r.readline()) #ilk satırı yaz
skipfirstline = r.readlines()[1:]
for line in skipfirstline:
    line = re.sub(',', '',line)
    line = line[:-2]+','+line[-2:]
    #print(line)
    w.write(line)
r.close()
w.close()# -*- coding: utf-8 -*-

