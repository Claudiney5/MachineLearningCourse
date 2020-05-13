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

