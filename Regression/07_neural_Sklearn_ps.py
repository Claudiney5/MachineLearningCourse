# -*- coding: utf-8 -*-
"""
Created on Sun May 24 09:37:11 2020

@author: Claudiney Martins
"""
import pandas as pd
import numpy as np

base = pd.read_csv('plano_saude2.csv')

X = base.iloc[:, 0:1].values
y = base.iloc[:, 1:2].values

from sklearn.preprocessing import StandardScaler
scaler_x = StandardScaler()
X = scaler_x.fit_transform(X)
scaler_y = StandardScaler()
y = scaler_y.fit_transform(y)

from sklearn.neural_network import MLPRegressor
regressor = MLPRegressor(hidden_layer_sizes=150)
regressor.fit(X, y)

score1 = regressor.score(X, y)
    
import matplotlib.pyplot as plt
plt.scatter(X, y)
plt.plot(X, regressor.predict(X), color='red')
plt.title('regressão com redes neurais')
plt.xlabel('idade')
plt.ylabel('custo')

previsao = scaler_y.inverse_transform(regressor.predict(
    scaler_x.transform(np.array(40).reshape(-1,1))))
