# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 10:10:05 2020

@author: Master
"""
import pandas as pd

base = pd.read_csv('credit_data.csv')
base.loc[base.age < 0, 'age'] = 40.93

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

# DIVISÃO DE BASES
from sklearn.cross_validation import train_test_split
training_predictors, test_predictors, class_training, class_test = train_test_split(predictors, classifier, test_size=0.25, random_state=0)

# Geração da tabela de probabilidade (TREINANDO)
from sklearn.naive_bayes import GaussianNB
classificador = GaussianNB()
classificador.fit(training_predictors, class_training)

# Fazendo as estimativas
predicts = classificador.predict(test_predictors)

# comparação de previsão com o real
from sklearn.metrics import confusion_matrix, accuracy_score
precision = accuracy_score(class_test, predicts)
matrix = confusion_matrix(class_test, predicts)
