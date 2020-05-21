# -*- coding: utf-8 -*-
"""
Created on Thu May 21 07:35:46 2020

@author: Claudiney
"""
import pandas as pd

base = pd.read_csv('house_prices.csv')

X = base.iloc[:, 3:19].values  
y = base.iloc[:, 2].values

X = X.reshape(-1,1)
y = y.reshape(-1,1)

# BASES PARA TREINAMENTO E TESTE
from sklearn.model_selection import train_test_split
X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(X, y,
                                                test_size=0.3,
                                                random_state=0)

# CRIANDO O REGRESSOR E TREINANDO
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_treinamento, y_treinamento)

score = regressor.score(X_treinamento, y_treinamento)

previsoes = regressor.predict(X_teste)

from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_teste, previsoes)

regressor.score(X_teste, y_teste)

regressor.intercept_
regressor.coef_
len(regressor.coef_)
