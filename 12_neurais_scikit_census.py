# -*- coding: utf-8 -*-
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
 
# importação biblioteca
from sklearn.neural_network import MLPClassifier
# criação do classificador
classificador = MLPClassifier(verbose=True,
                              max_iter=100,
                              tol=1e-4)

classificador.fit(previsores_treinamento, classe_treinamento)
previsoes = classificador.predict(previsores_teste)

# comparação de previsão com o real
from sklearn.metrics import confusion_matrix, accuracy_score
precisao = accuracy_score(classe_teste, previsoes)
matriz = confusion_matrix(classe_teste, previsoes)