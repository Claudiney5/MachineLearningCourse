# -*- coding: utf-8 -*-
"""
Created on Thu May 21 08:12:53 2020

@author: Claudiney
"""
import pandas as pd
import numpy as np

base = pd.read_csv('plano_saude2.csv')

X = base.iloc[:, 0:1].values
y = base.iloc[:, 1].values

    
# linear --------------------------------------

from sklearn.linear_model import LinearRegression
regressor1 = LinearRegression()
regressor1.fit(X, y)
score1 = regressor1.score(X, y)

idd = 40

regressor1.predict(np.array(idd).reshape(-1,1))

# GRÁFICOS
import matplotlib.pyplot as plt
plt.scatter(X, y)
plt.plot(X, regressor1.predict(X), color='red')
plt.title('Regressão Linear')
plt.xlabel('Idade')
plt.ylabel('Custo')


# polinomial -----------------------------------

from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=4)
X_poly = poly.fit_transform(X)

regressor2 = LinearRegression()
regressor2.fit(X_poly,y)

score2 = regressor2.score(X_poly, y)

regressor2.predict(poly.transform(np.array(idd).reshape(-1,1)))

plt.scatter(X, y)
plt.plot(X, regressor2.predict(poly.fit_transform(X)), color='green')
plt.title('Regressão Polinomial')
plt.xlabel('Idade')
plt.ylabel('Custo')

