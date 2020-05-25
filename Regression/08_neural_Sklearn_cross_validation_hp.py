# -*- coding: utf-8 -*-
"""
Created on Sun May 24 10:37:29 2020

@author:Claudiney Martins
"""
import pandas as pd

base = pd.read_csv('house_prices.csv')

X = base.iloc[:, 3:19].values
y = base.iloc[:, 2:3].values

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)
y = scaler.fit_transform(y)


from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPRegressor
regressor = MLPRegressor(hidden_layer_sizes=(9, 9, 9))

resultados = cross_val_score(regressor, X, y, cv=10)
media = resultados.mean()
desvio = resultados.std()
