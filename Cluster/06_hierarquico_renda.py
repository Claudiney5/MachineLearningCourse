# -*- coding: utf-8 -*-
"""
Created on Thu May 28 07:49:20 2020

@author: Claudiney Martins
"""

import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
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

dendrograma = dendrogram(linkage(base, method='ward')) # testar outros métodos!!
plt.title('Dendrograma')
plt.xlabel('Pessoas')
plt.ylabel('Dist. Euclidiana')

hc = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
previsoes = hc.fit_predict(base)

plt.scatter(base[previsoes==0, 0], base[previsoes==0, 1], s=100, c='red', label='Cluester 1')
plt.scatter(base[previsoes==1, 0], base[previsoes==1, 1], s=100, c='orange', label='Cluester 2')
plt.scatter(base[previsoes==2, 0], base[previsoes==2, 1], s=100, c='green', label='Cluester 3')
plt.xlabel('idade')
plt.ylabel('salario')
plt.legend()