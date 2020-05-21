# -*- coding: utf-8 -*-
"""
Created on Thu May 21 08:12:53 2020

@author: Claudiney
"""
import pandas as pd

base = pd.read_csv('plano_saude2.csv')

X = base.iloc[:, 0:1].values
y = base.iloc[:, 1].values

from sklearn.linear_model import LinearRegression
regressor1 = LinearRegression()
regressor1.fit(X, y)
score1 = regressor1.score(X, y)

regressor1.predict(40)
