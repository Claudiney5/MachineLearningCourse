# -*- coding: utf-8 -*-
"""
Spyder Editor

Este é um arquivo de script temporário.
"""
import pandas as pd
base = pd.read_csv('credit_data.csv')
base.describe()

# --  Tratando os valores da tabela  --

base.loc[base['age'] < 0]  # localizando

# apagando colunas
# base.drop('age', 1, inplace=True)

# apagando os registros com problemas
# base.drop(base[base.age < 0].index, inplace=True)

# preencher os valoes manualmente
#preencher os valores com a média é uma solução interesante
base.mean() # média com valores válidos e não válidos
base['age'].mean() # média com todos os valores apenas de 'age'
base['age'][base.age > 0].mean() # média somente com os valores válidos
base.loc[base.age < 0, 'age'] = 40.93

pd.isnull(base['age']) # Mostra quem tem e quem não tem Valor nulo
base.loc[pd.isnull(base['age'])]  # Mostra apenas as linhas que possuem nulo 
# em 'age'

predictors = base.iloc[:, 1:4].values
classifier = base.iloc[:, 4].values

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
imputer = imputer.fit(predictors[:, 0:3])
predictors[:, 0:3] = imputer.transform(predictors[:, 0:3])

# ESCALONAMENTO
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
predictors = scaler.fit_transform(predictors)




