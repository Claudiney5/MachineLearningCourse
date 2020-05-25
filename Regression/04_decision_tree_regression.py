# -*- coding: utf-8 -*-
"""
Created on Fri May 22 10:27:50 2020

@author: Claudiney Martins
"""
import pandas as pd
import numpy as np

base = pd.read_csv('plano_saude2.csv')

X = base.iloc[:, 0:1].values
y = base.iloc[:, 1].values

from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor()
regressor.fit(X, y)

score = regressor.score(X, y)

import matplotlib.pyplot as plt
plt.scatter(X, y)
plt.plot(X, regressor.predict(X), color="red")
plt.title('Regressão com Árvores de Decisão')
plt.xlabel('Idade')
plt.ylabel('Custo')
# o gráfico gerado acima não reflete com correção um aárvore de decisão.

X_teste = np.arange(min(X), max(X), 0.1)
X_teste = X_teste.reshape(-1,1)
plt.scatter(X, y)
plt.plot(X_teste, regressor.predict(X_teste), color="red")
plt.title('Regressão com Árvores de Decisão')
plt.xlabel('Idade')
plt.ylabel('Custo')