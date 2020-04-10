# -*- coding: utf-8 -*-


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


from sklearn.cross_validation import train_test_split
training_predictors, test_predictors, training_class, test_class = train_test_split(predictors, classifier, test_size=0.25, random_state=0)

from sklearn.tree import DecisionTreeClassifier
classificador = DecisionTreeClassifier(criterion='entropy', random_state=0)
classificador.fit(training_predictors, training_class)
previsores = classificador.predict(test_predictors)

from sklearn.metrics import confusion_matrix, accuracy_score
precisao = accuracy_score(test_class, previsores)
matriz = confusion_matrix(test_class, previsores)
