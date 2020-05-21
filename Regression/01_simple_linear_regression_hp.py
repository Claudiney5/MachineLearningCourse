# -*- coding: utf-8 -*-
"""
Created on Wed May 20 07:30:57 2020

@author: Claudiney Martins

"""
import pandas as pd

base = pd.read_csv('house_prices.csv')

X = base.iloc[:, 5:6].values  # até 6 para não ter que fazer 'reshape'
y = base.iloc[:, 2].values

X = X.reshape(-1,1)
y = y.reshape(-1,1)

# BASES PARA TREINAMENTO E TESTE
from sklearn.model_selection import train_test_split
X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(X, y,
                                                test_size=0.3,
                                                random_state=0)

# CRIRANDO O REGRESSOR E TREINANDO
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_treinamento, y_treinamento)

# VERIFICANDO O ACERTO
score = regressor.score(X_treinamento, y_treinamento)

import matplotlib.pyplot as plt
plt.scatter(X_treinamento, y_treinamento)
plt.plot(X_treinamento, regressor.predict(X_treinamento), color='red')
# verificamos pelo gráfico qeu o modelo não se adaptou muito bem.

# fazendo previsões:
previsao = regressor.predict(X_teste)


# calculo de quanto ele esta errando
resultado = abs(y_teste - previsao)
resultado.mean()

from sklearn.metrics import mean_absolute_error, mean_squared_error
mae = mean_absolute_error(y_teste, previsao)
mse = mean_squared_error(y_teste, previsao)

# plotando com a base de teste
plt.scatter(X_teste, y_teste)
plt.plot(X_teste, regressor.predict(X_teste), color='green')
score_test = regressor.score(X_teste, y_teste)
