# -*- coding: utf-8 -*-

import pandas as pd

base = pd.read_csv('credit_data.csv')
# tratamento dos dados inválidos
base.loc[base.age < 0, 'age'] = 40.93

#divisão de previsores e classes
previsores = base.iloc[:, 1:4].values
classe = base.iloc[:, 4].values

# substituição dos valores faltantes pela média
from sklearn.impute import SimpleImputer
imputer = SimpleImputer( strategy='mean')
imputer = imputer.fit(previsores[:, 1:4])
previsores[:, 1:4] = imputer.transform(previsores[:, 1:4])

# ESCALONAMENTO
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)

# DIVISÃO DE BASES
from sklearn.model_selection import train_test_split
previsores_treino, previsores_teste, classe_treino, classe_teste = train_test_split(previsores, classe, test_size=0.25, random_state=0)

#
#
classificador.fit(previsores_treino, classe_treino)
previsoes = classificador.predict(previsores_teste)

from sklearn.metrics import confusion_matrix, accuracy_score
precisao = accuracy_score(classe_teste, previsoes)
matriz = confusion_matrix(classe_teste, previsoes)

import collections
collections.Counter(test_class)