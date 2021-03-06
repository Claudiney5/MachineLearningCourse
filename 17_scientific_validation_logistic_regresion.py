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

from sklearn.linear_model import LogisticRegression  #!!!

import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

resultados30 = []
for i in range(30):
    
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=i)
    resultados1 = []
    for indice_treinamento, indice_teste in kfold.split(previsores,
                                                        np.zeros(shape=(previsores.shape[0],1))):
        classificador = LogisticRegression(random_state=1)
        classificador.fit(previsores[indice_treinamento], classe[indice_treinamento])
        previsoes = classificador.predict(previsores[indice_teste])
        precisao = accuracy_score(classe[indice_teste], previsoes)
        resultados1.append(precisao)
        
    resultados1 = np.asarray(resultados1)
    media = resultados1.mean()
    resultados30.append(media)
    
resultados30 = np.asarray(resultados30)
for i in range(resultados30.size):
    print(str(resultados30[i]).replace('.', ','))
    


