# -*- coding: utf-8 -*-
"""
Created on Sat May 23 12:35:32 2020

@author: Claudiney Martins
"""
import pandas as pd
import numpy as np

base = pd.read_csv('plano_saude2.csv')

X = base.iloc[:, 0:1].values
y = base.iloc[:, 1:2].values


from sklearn.svm import SVR

# KERNEL Linear
regressor_linear = SVR(kernel='linear')
regressor_linear.fit(X, y)

import matplotlib.pyplot as plt
plt.scatter(X, y)
plt.plot(X, regressor_linear.predict(X), color='red')
score1 = regressor_linear.score(X, y)

#KERNEL Polynomial
regressor_poly = SVR(kernel='poly', degree=5)
regressor_poly.fit(X, y)

plt.scatter(X, y)
plt.plot(X, regressor_poly.predict(X), color='green')
score2 = regressor_poly.score(X, y)

# KERNEL rbf  (Regression Base Fucntion é mais indicado para SVR)
from sklearn.preprocessing import StandardScaler
scaler_x = StandardScaler()     # este escalonamento pode ser feitos nos outros Kernels
X = scaler_x.fit_transform(X)   # Para este caso ele melhoraria o Linear, mas pioroou o Poly
scaler_y = StandardScaler()
y = scaler_y.fit_transform(y)

regressor_rbf = SVR(kernel='rbf')
regressor_rbf.fit(X, y)

plt.scatter(X, y)
plt.plot(X, regressor_rbf.predict(X), color="yellow")
score3 = regressor_rbf.score(X, y)

previsao1 = regressor_linear.predict(np.array(40).reshape(-1,1))
previsão2 = regressor_poly.predict(np.array(40).reshape(-1,1))
previsao3 = scaler_y.inverse_transform(regressor_rbf.predict(scaler_x.transform(np.array(40).reshape(-1,1))))
