# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 11:14:05 2020

@author: Claudiney Martins
"""

import pandas as pd

basec = pd.read_csv('census.csv')

education_weigths = {
    "Doctorate": 9,
    "Masters": 8,
    "Prof-school": 8,
    "Assoc-voc": 7,
    "Assoc-acdm": 7,
    "Bachelors": 6,
    "HS-grad": 5,
    "12th": 4,
    "11th": 4,
    "10th": 4,
    "9th": 4,
    "7th-8th": 3,
    "5th-6th": 2,
    "1st-4th": 2,
    "Preschool": 1,
    "Some-college": 0
}
 
basec["education"] = basec["education"].map(lambda x: education_weigths[x.strip()])

predictors = basec.iloc[:, 0:14].values
classifier = basec.iloc[:, 14].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_predictors = LabelEncoder()

predictors[:, 1] = labelencoder_predictors.fit_transform(predictors[:, 1])
#predictors[:, 3] = labelencoder_predictors.fit_transform(predictors[:, 3])
predictors[:, 5] = labelencoder_predictors.fit_transform(predictors[:, 5])
predictors[:, 6] = labelencoder_predictors.fit_transform(predictors[:, 6])
predictors[:, 7] = labelencoder_predictors.fit_transform(predictors[:, 7])
predictors[:, 8] = labelencoder_predictors.fit_transform(predictors[:, 8])
predictors[:, 9] = labelencoder_predictors.fit_transform(predictors[:, 9])
predictors[:, 13] = labelencoder_predictors.fit_transform(predictors[:, 13])

one_hot = OneHotEncoder(categorical_features=[1, 5, 6, 7, 8, 9, 13])
predictors = one_hot.fit_transform(predictors).toarray()

labelencoder_class = LabelEncoder()
classifier = labelencoder_class.fit_transform(classifier)

# ESCALONAMENTO
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
#predictors[:, 2] = scaler.fit_transform(predictors[:, 2])
predictors[:, 88:89] = scaler.fit_transform(predictors[:, 88:89])
predictors[:, 91:92] = scaler.fit_transform(predictors[:, 91:92])

# DIVISÃO DE BASES
from sklearn.cross_validation import train_test_split
train_predictors, test_predictors, train_class, test_class = train_test_split(predictors, classifier, test_size=0.15, random_state=0)

# Geração da tabela de probabilidade (TREINO)
from sklearn.naive_bayes import GaussianNB
classificador = GaussianNB()
classificador.fit(train_predictors, train_class)

# PREVISÕES COM O ARQUIVO TESTE
predicts = classificador.predict(test_predictors)

# comparação de previsão com o real
from sklearn.metrics import confusion_matrix, accuracy_score
precision = accuracy_score(test_class, predicts)
matrix = confusion_matrix(test_class, predicts)
