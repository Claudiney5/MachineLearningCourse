# -*- coding: utf-8 -*-
"""
Created on Fri May 29 09:25:20 2020

@author: Claudiney Martins
"""

import matplotlib.pyplot as plt
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
import numpy as np

base = pd.read_csv('c_credito.csv', header=1)
# criando uma nova coluna
base['BILL_TOTAL'] = base['BILL_AMT1'] + base['BILL_AMT2'] + base['BILL_AMT3'] + base['BILL_AMT4'] + base['BILL_AMT5'] + base['BILL_AMT6']

X = base.iloc[:, [1,25]].values  #  1  'E'  25
scaler = StandardScaler()
X = scaler.fit_transform(X)

dendrograma = dendrogram(linkage(X, method='ward'))

# Hierarchy Cluster  (3 cluster como decido com o dendrogrma)
hc = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
previsoes = hc.fit_predict(X)

plt.scatter(X[previsoes==0,0], X[previsoes==0,1], s=10, c='red', label='Cluster 1')
plt.scatter(X[previsoes==1,0], X[previsoes==1,1], s=10, c='blue', label='Cluster 2')
plt.scatter(X[previsoes==2,0], X[previsoes==2,1], s=10, c='green', label='Cluster 3')
plt.title('Cartão de Crédito')
plt.xlabel('Limite')
plt.ylabel('Gastos')
plt.legend()