# -*- coding: utf-8 -*-
"""
Created on Wed May 27 08:42:38 2020

@author: Claudiney
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

base = pd.read_csv('c_credito.csv', header=1)

''' soamt´roio das atributos de dívidas mensais do cliente.'''
base['BILL_TOTAL'] = base['BILL_AMT1'] + base['BILL_AMT2'] + base['BILL_AMT3'] + base['BILL_AMT4'] + base['BILL_AMT5'] + base['BILL_AMT6']

X = base.iloc[:, [1,25]].values

scaler = StandardScaler()
X = scaler.fit_transform(X)


# DEFINIÇÃO DO NÚMERO DE CLUSTER
wcss = []   # Within Cluster Sum of Square

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11), wcss)
plt.title('gráfico dos KMeans')
plt.xlabel('Nº de Clusters')
plt.ylabel('WCSS')

# numero de clusters definido!

# GERAÇÃO DO GRÁFICO

kmeans = KMeans(n_clusters=4, random_state=0 )
previsoes = kmeans.fit_predict(X)

plt.scatter(X[previsoes == 0, 0], X[previsoes == 0, 1], s=100, c='red', label='Cluster 1')  
plt.scatter(X[previsoes == 1, 0], X[previsoes == 1, 1], s=100, c='orange', label='Cluster 2') 
plt.scatter(X[previsoes == 2, 0], X[previsoes == 2, 1], s=100, c='green', label='Cluster 3')  
plt.scatter(X[previsoes == 3, 0], X[previsoes == 3, 1], s=100, c='blue', label='Cluster 4')  

plt.xlabel('Limite') 
plt.ylabel('Gastos')
plt.title('Limite x Gastos')
plt.legend()

# VISUALIZANDO PESSOAS por grupos (última coluna)

lista_clientes = np.column_stack((base, previsoes))
lista_clientes = lista_clientes[lista_clientes[:,26].argsort()]

