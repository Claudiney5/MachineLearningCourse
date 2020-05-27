# -*- coding: utf-8 -*-
"""
Created on Wed May 27 08:42:38 2020

@author: Claudiney

ANÁLISE COM MAIS ATRIBUTOS
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

base = pd.read_csv('c_credito.csv', header=1)

''' soamt´roio das atributos de dívidas mensais do cliente.'''
base['BILL_TOTAL'] = base['BILL_AMT1'] + base['BILL_AMT2'] + base['BILL_AMT3'] + base['BILL_AMT4'] + base['BILL_AMT5'] + base['BILL_AMT6']

X = base.iloc[:, [1,2,3,4,5,25]].values

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

# numero de clusters definido! K = 4

# GERAÇÃO DO GRÁFICO

kmeans = KMeans(n_clusters=4, random_state=0 )
previsoes = kmeans.fit_predict(X)

# VISUALIZANDO PESSOAS por grupos (última coluna)

lista_clientes = np.column_stack((base, previsoes))
lista_clientes = lista_clientes[lista_clientes[:,26].argsort()]

