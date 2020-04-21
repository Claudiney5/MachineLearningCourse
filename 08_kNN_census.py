# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 10:40:09 2020

@author: USUARIO-PC
"""

import pandas as pd

basec = pd.read_csv('census.csv')

predictors = basec.iloc[:, 0:14].values
classifier = basec.iloc[:, 14].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_predictors = LabelEncoder()

predictors[:, 1] = labelencoder_predictors.fit_transform(predictors[:, 1])
predictors[:, 3] = labelencoder_predictors.fit_transform(predictors[:, 3])
predictors[:, 5] = labelencoder_predictors.fit_transform(predictors[:, 5])
predictors[:, 6] = labelencoder_predictors.fit_transform(predictors[:, 6])
predictors[:, 7] = labelencoder_predictors.fit_transform(predictors[:, 7])
predictors[:, 8] = labelencoder_predictors.fit_transform(predictors[:, 8])
predictors[:, 9] = labelencoder_predictors.fit_transform(predictors[:, 9])
predictors[:, 13] = labelencoder_predictors.fit_transform(predictors[:, 13])

# ONE HOT ENCODER

one_hot = OneHotEncoder(categories='auto')
predictors = one_hot.fit_transform(predictors).toarray()

labelencoder_class = LabelEncoder()
classifier = labelencoder_class.fit_transform(classifier)

# ESCALONAMENTO
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
predictors = scaler.fit_transform(predictors)

# DIVISÃO DE BASES
from sklearn.cross_validation import train_test_split
train_predictors, test_predictors, train_class, test_class = train_test_split(predictors, classifier, test_size=0.15, random_state=0)

# importação biblioteca
# criação do classificador

classificador.fit(train_predictors, train_class)
predicts = classificador.predict(test_predictors)

# comparação de previsão com o real
from sklearn.metrics import confusion_matrix, accuracy_score
precision = accuracy_score(test_class, predicts)
matrix = confusion_matrix(test_class, predicts)