# -*- coding: utf-8 -*-
"""
Created on Sat May 23 09:33:54 2020

@author: USUARIO-PC
"""
import pandas as pd

base = pd.read_csv('plano_saude2.csv')

X = base.iloc[:, 0:1].values
y = base.iloc[:, 1].values

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=10)
regressor.fit(X, y)

score = regressor.score(X, y)

import numpy as np
X_teste = np.arange(min(X), max(X), 0.1)
X_teste = X_teste.reshape(-1,1)

import matplotlib.pyplot as plt
plt.scatter(X, y)
plt.plot(X_teste, regressor.predict(X_teste), color='red')
plt.title('Regressão com Random Forest')
plt.xlabel('Idade')
plt.ylabel('Custo')

previsao = regressor.predict(np.array(40).reshape(-1,1))