# -*- coding: utf-8 -*-
"""
Created on Sat May 30 10:27:01 2020

@author: Claudiney Martins
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


base = pd.read_csv('c_credito.csv', header=1)
# criando uma nova coluna
base['BILL_TOTAL'] = base['BILL_AMT1'] + base['BILL_AMT2'] + base['BILL_AMT3'] + base['BILL_AMT4'] + base['BILL_AMT5'] + base['BILL_AMT6']

X = base.iloc[:, [1,25]].values  #  1  'E'  25
scaler = StandardScaler()
X = scaler.fit_transform(X)

dbscan = DBSCAN(eps=0.37, min_samples=4)
previsoes = dbscan.fit_predict(X)
unicos, qtidade = np.unique(previsoes, return_counts=True)

plt.scatter(X[previsoes==0,0], X[previsoes==0,1], s=10, c='red', label='Cluster 1')
plt.scatter(X[previsoes==1,0], X[previsoes==1,1], s=10, c='blue', label='Cluster 2')
plt.scatter(X[previsoes==2,0], X[previsoes==2,1], s=10, c='green', label='Cluster 3')
plt.title('Cartão de Crédito')
plt.xlabel('Limite')
plt.ylabel('Gastos')
plt.legend()

lista_clientes = np.column_stack((base, previsoes))
lista_clientes = lista_clientes[lista_clientes[:,26].argsort()]