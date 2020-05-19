# -*- coding: utf-8 -*-
"""
Created on Tue May 19 07:42:27 2020

@author: Claudiney Martins
"""

import pandas as pd

base = pd.read_csv('plano_de_saude.csv')

X = base.iloc[:, 0].values
y = base.iloc[:, 1].values

import numpy as np
correlacao = np.corrcoef(X, y)

X = X.reshape(-1,1) # numpy function -> vetor para matriz. Scikit-learn 
                     # precisa que estejam no formato de matriz.
                     # -1, não altera as linhas, 1, coloca em coluna

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X, y)

# b0
regressor.intercept_

#b1
regressor.coef_

import matplotlib.pyplot as plt
plt.scatter(X, y)
plt.plot(X, regressor.predict(X), color='green')
plt.title('Regressão Linear Simples')
plt.xlabel('Idade')
plt.ylabel('Custo')

idd = 40

previsao1 = regressor.predict([[idd]])
previsao2 = regressor.intercept_ + regressor.coef_ * idd

score = regressor.score(X, y)  #  visualização de correlação (score) do modelo

#  resíduos (=/- = erro)
from yellowbrick.regressor import ResidualsPlot
visualizador = ResidualsPlot(regressor)
visualizador.fit(X,y)
visualizador.poof()



