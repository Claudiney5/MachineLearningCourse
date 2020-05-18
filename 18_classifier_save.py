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

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

classificadorSVM = SVC(kernel='rbf', C = 2.0, probability=True)
classificadorSVM.fit(previsores, classe)

classificadorRandomForest = RandomForestClassifier(n_estimators=40, 
                                                       criterion='entropy')
classificadorRandomForest.fit(previsores, classe)

classificadorMLP = MLPClassifier(verbose=True, 
                                 max_iter=1000,
                                 tol = 0.000010, solver='adam',
                                 hidden_layer_sizes=(100),
                                 activation='relu',
                                 batch_size=200,
                                 learning_rate_init=0.001)
classificadorMLP.fit(previsores, classe)

import pickle
pickle.dump(classificadorSVM, open('svm_finalizado.sav', 'wb'))
pickle.dump(classificadorRandomForest, open('random_forest_finalizado.sav', 'wb'))
pickle.dump(classificadorMLP, open('mlp_finalizado.sav', 'wb'))
