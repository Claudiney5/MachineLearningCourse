# -*- coding: utf-8 -*-
"""
Created on Tue May 26 08:21:03 2020

@author: Claudiney Martins
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

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

scaler = StandardScaler()
base = scaler.fit_transform(base)

    
kmeans = KMeans(n_clusters=3)
kmeans.fit(base)

centroides = kmeans.cluster_centers_
rotulos = kmeans.labels_
cores = ['g.', 'r.', 'b.']

for i in range (len(x)):
    plt.plot(base[i][0], base[i][1], cores[rotulos[i]], markersize=15)

plt.scatter(centroides[:, 0], centroides[:, 1], marker='X')
    