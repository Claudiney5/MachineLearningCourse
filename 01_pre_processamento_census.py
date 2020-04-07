# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 18:44:42 2020

@author: Master
"""

import pandas as pd

base = pd.read_csv('census.csv')

predictors = base.iloc[:, 0:14].values
classifier = base.iloc[:, 14].values

# substituindo dados strings por numéricos

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_predictors = LabelEncoder()
# labels = labelencoder_predictors.fit_transform(predictors[:, 1])
predictors[:, 1] = labelencoder_predictors.fit_transform(predictors[:, 1])
predictors[:, 3] = labelencoder_predictors.fit_transform(predictors[:, 3])
predictors[:, 5] = labelencoder_predictors.fit_transform(predictors[:, 5])
predictors[:, 6] = labelencoder_predictors.fit_transform(predictors[:, 6])
predictors[:, 7] = labelencoder_predictors.fit_transform(predictors[:, 7])
predictors[:, 8] = labelencoder_predictors.fit_transform(predictors[:, 8])
predictors[:, 9] = labelencoder_predictors.fit_transform(predictors[:, 9])
predictors[:, 13] = labelencoder_predictors.fit_transform(predictors[:, 13])

# Variáveis DUMMY
one_hot = OneHotEncoder(categorical_features=[1, 3, 5, 6, 7, 8, 9, 13])
predictors = one_hot.fit_transform(predictors).toarray()

labelencoder_class = LabelEncoder()
classifier = labelencoder_class.fit_transform(classifier)

# ESCALONAMENTO
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

predictors = scaler.fit_transform(predictors)


