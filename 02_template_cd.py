# -*- coding: utf-8 -*-

import pandas as pd

base = pd.read_csv('credit_data.csv')
base.loc[base.age < 0, 'age'] = 40.93

predictors = base.iloc[:, 1:4].values
classifier = base.iloc[:, 4].values

<<<<<<< HEAD
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
=======
from sklearn.impute import SimpleImputer
imputer = SimpleImputer( strategy='mean')
>>>>>>> 0fff2a319a5d24a70ca5a034d77b600fe7cb9af1
imputer = imputer.fit(predictors[:, 1:4])
predictors[:, 1:4] = imputer.transform(predictors[:, 1:4])

# ESCALONAMENTO
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
predictors = scaler.fit_transform(predictors)

# DIVISÃO DE BASES
<<<<<<< HEAD
from sklearn.cross_validation import train_test_split
training_predictors, test_predictors, training_class, test_class = train_test_split(predictors, classifier, test_size=0.25, random_state=0)

# importação biblioteca
# criação do classificador

classificador.fit(training_predictors, training_class)
predictors = classificador.predict(test_predictors)

from sklearn.metrics import confusion_matrix, accuracy_score
precisao = accuracy_score(test_class, predictors)
matriz = confusion_matrix(test_class, predictors)
=======
from sklearn.model_selection import train_test_split
training_predictors, test_predictors, training_class, test_class = train_test_split(predictors, classifier, test_size=0.25, random_state=0)

#importação da biblioteca
#criação do classificador
classificador.fit(training_predictors, training_class)
previsoes = classificador.predict(test_predictors)

from sklearn.metrics import confusion_matrix, accuracy_score
precisao = accuracy_score(test_class, previsoes)
matriz = confusion_matrix(test_class, previsoes)

import collections
collections.Counter(test_class)
>>>>>>> 0fff2a319a5d24a70ca5a034d77b600fe7cb9af1
