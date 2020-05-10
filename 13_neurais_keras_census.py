# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 10:40:09 2020

@author: USUARIO-PC
"""

import pandas as pd
 
base = pd.read_csv('census.csv')
 
previsores = base.iloc[:, 0:14].values
classe = base.iloc[:, 14].values
                
from sklearn.preprocessing import LabelEncoder ,OneHotEncoder
from sklearn.compose import ColumnTransformer
 
onehotencorder = ColumnTransformer(transformers=[("OneHot", OneHotEncoder(), [1,3,5,6,7,8,9,13])],remainder='passthrough')
previsores = onehotencorder.fit_transform(previsores).toarray()
 
labelencoder_classe = LabelEncoder()
classe = labelencoder_classe.fit_transform(classe)

# ESCALONAMENTO 
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)
 
from sklearn.model_selection import train_test_split
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size=0.15, random_state=0)
 
# importação das bibliotecas
import keras
from keras.models import Sequential
from keras.layers import Dense
classificador = Sequential()

# criação das camadas ocultas
classificador.add(Dense(units=55,
                        activation='relu',
                        input_dim=108))
classificador.add(Dense(units=55,
                        activation='relu'))

# criação da camada(s) de saída
classificador.add(Dense(units=1,
                        activation='sigmoid'))

#  compilando a rede neural
classificador.compile(optimizer='adam', 
                      loss='binary_crossentropy', 
                      metrics=['accuracy'])

classificador.fit(previsores_treinamento, classe_treinamento,
                  batch_size=10,
                  epochs=100)
previsoes = classificador.predict(previsores_teste)
previsoes = (previsoes > 0.5)
 
from sklearn.metrics import confusion_matrix, accuracy_score
precisao = accuracy_score(classe_teste, previsoes)
matriz = confusion_matrix(classe_teste, previsoes)
 
