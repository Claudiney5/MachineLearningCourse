# -*- coding: utf-8 -*-
"""
Created on Sat May 30 09:50:17 2020

@author: Claudiney Martins
"""
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import numpy as np


x = [20, 27, 21, 37, 46, 53, 55, 47, 52, 32, 39, 41, 39, 48, 48]
y = [1000, 1200, 2900, 1850, 900, 950, 2000, 2100, 3000, 5900, 4100, 5100, 7000, 5000, 6500]
plt.scatter(x, y)

base = []
# criando a lista das listas de idade-rendimento
for i in range(0, len(x)):
    lista = []
    lista.insert(0, x[i])
    lista.insert(1, y[i])
    base.insert(i, lista)

base = np.array(base)

scaler = StandardScaler()
base = scaler.fit_transform(base)

dbscan = DBSCAN(eps=0.95, min_samples=2)
dbscan.fit(base)
previsoes = dbscan.labels_

cores = ('g.', 'r.', 'y.')

for i in range(len(base)):
    plt.plot(base[i][0], base[i][1], cores[previsoes[i]], markersize=15)
