# -*- coding: utf-8 -*-
"""
Created on Mon May 25 07:19:26 2020

@author: Claudiney Martins
"""
import pandas as pd
import numpy as np

base = pd.read_csv('house_prices.csv')

X = base.iloc[:, 3:19].values
y = base.iloc[:, 2:3].values

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)
y = scaler.fit_transform(y)


from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPRegressor
regressor = MLPRegressor(hidden_layer_sizes=(9, 9, 9))

X.shape     #  só para entender
X.shape[0]  # só para entender abaixo
b = np.zeros(shape=(X.shape[0], 1))

kfold = StratifiedKFold(n_splits=10, shuffle=True,random_state=0)
resultk = []


for i_train, i_test in kfold.split(X, np.zeros(shape=(X.shape[0], 1))):
    lista = []
    regressok = MLPRegressor(hidden_layer_sizes=(9, 9, 9))
    regressok.fit(X[i_train], y[i_train])
    previsoes = regressok.predict(X[i_test])
    lista.append(previsoes)
    precisao = regressok.score(y[i_test], lista)
    resultk.append(precisao)

resultk = np.asarray(resultk)
media = resultk.mean()
desvio = resultk.std