# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 18:44:59 2020

@author: Claudiney Martins

NAIVE BAYES

"""
import pandas as pd

base = pd.read_csv('risco_credito.csv')

predictors = base.iloc[:, 0:4].values
classe = base.iloc[:,4].values

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
predictors[:,0] = encoder.fit_transform(predictors[:,0])
predictors[:,1] = encoder.fit_transform(predictors[:,1])
predictors[:,2] = encoder.fit_transform(predictors[:,2])
predictors[:,3] = encoder.fit_transform(predictors[:,3])

# Geração da tabela de probabilidade
from sklearn.naive_bayes import GaussianNB
classificador =  GaussianNB()
classificador.fit(predictors, classe) # TREINO

# histórico, divida, garantias, tenda
#  1: bom, alta, nenhum, >35
#  2: ruim, alta, adequada, <15
result = classificador.predict([[0, 0, 1, 2], [3,0,0,0]]) # PREVISÃO

print(classificador.classes_)
print(classificador.class_count_)
print(classificador.class_prior_)
