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

from sklearn.neural_network import MLPClassifier   #!!!

import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

resultados30 = []
for i in range(30):
    
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=i)
    resultados1 = []
    for indice_treinamento, indice_teste in kfold.split(previsores,
                                                        np.zeros(shape=(previsores.shape[0],1))):
        classificador = MLPClassifier(hidden_layer_sizes=(100,),
                              activation="relu",  # default
                              solver='adam',  # default
                              batch_size='auto',  # default
                              learning_rate='constant',  # default
                              max_iter=1000, 
                              tol=1e-6,     
                              verbose=True,
                              momentum=0.9)
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
    


