# -*- coding: utf-8 -*-
"""
Created on Sat May 30 10:47:47 2020

@author: Claudiney Martins

"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn import datasets

# criando dados aleatórios
x, y = datasets.make_moons(n_samples=1500, noise=0.09)

plt.scatter(x[:, 0], x[:, 1], s=5)

cores = np.array(['red', 'blue'])

kmeans = KMeans(n_clusters=2)
previsoes = kmeans.fit_predict(x)
plt.scatter(x[:, 0], x[:, 1], s=5, color=cores[previsoes])

hc = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')
previsoes = hc.fit_predict(x)
plt.scatter(x[:, 0], x[:, 1], s=5, color=cores[previsoes])

dbscan = DBSCAN(eps=0.1)
previsoes = dbscan.fit_predict(x)
plt.scatter(x[:, 0], x[:, 1], s=5, color=cores[previsoes])
