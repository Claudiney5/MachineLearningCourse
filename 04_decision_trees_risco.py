# -*- coding: utf-8 -*-
"""
Created on Wed Apr 04 2020

@author: Claudiney Martins

ÁRVORES DE DECISÃO

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
from sklearn.tree import DecisionTreeClassifier, export
classificador = DecisionTreeClassifier(criterion='entropy')
classificador.fit(predictors, classe) # TREINO
print(classificador.feature_importances_)

export.export_graphviz(classificador,  # visto no Graphviz Online
                       out_file = 'arvore.dot',
                       feature_names = ['história', 'dívida', 'garantias', 'renda'],
                       class_names = ['alto', 'moderado', 'baixo'],
                       filled = True,
                       leaves_parallel = True)

# histórico, divida, garantias, tenda
#  1: bom, alta, nenhum, >35
#  2: ruim, alta, adequada, <15
result = classificador.predict([[0, 0, 1, 2], [3,0,0,0]]) # PREVISÃO

print(classificador.classes_)

