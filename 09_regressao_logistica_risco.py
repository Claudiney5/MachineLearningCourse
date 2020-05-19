# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 08:28:55 2020

@author: USUARIO-PC
"""

import pandas as pd

base = pd.read_csv('risco_credito2.csv')

predictors = base.iloc[:, 0:4].values
classe = base.iloc[:,4].values

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
predictors[:,0] = encoder.fit_transform(predictors[:,0])
predictors[:,1] = encoder.fit_transform(predictors[:,1])
predictors[:,2] = encoder.fit_transform(predictors[:,2])
predictors[:,3] = encoder.fit_transform(predictors[:,3])

from sklearn.linear_model import LogisticRegression
classificador = LogisticRegression()
classificador.fit(predictors, classe)
print(classificador.intercept_)
print(classificador.coef_)

# histórico, divida, garantias, tenda
#  1: bom, alta, nenhum, >35
#  2: ruim, alta, adequada, <15
result = classificador.predict([[0, 0, 1, 2], [3,0,0,0]]) # PREVISÃO 
result2 = classificador.predict_proba([[0, 0, 1, 2], [3,0,0,0]])
print(result)
print(result2)
