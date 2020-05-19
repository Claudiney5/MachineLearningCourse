# -*- coding: utf-8 -*-
"""
Editor Spyder

by Claudiney 2020.04.19  10:41 hs
"""
import Orange

base = Orange.data.Table('credit_data.csv')
#    na base de dados c# = classificador, i# = ignorar
base.domain

# dividir base de dados em base treinamento(75%) e teste(25%)
base_dividida = Orange.evaluation.testing.sample(base, n=0.25)
base_treinamento = base_dividida[1]
base_teste = base_dividida[0]
len(base_treinamento)
len(base_teste)

#    algoritmo antigo para indução de regras
cn2_learner = Orange.classification.rules.CN2Learner()
#    Learner é a variável que gera as regras e as REGRAS são os classificadores
classificador = cn2_learner(base_treinamento)
#    vendo as regras criadas
for regras in classificador.rule_list:
    print(regras)
    
#    teste:
resultado = Orange.evaluation.testing.TestOnTestData(base_treinamento, base_teste, [classificador])
print(Orange.evaluation.CA(resultado))  # CA = classification accuracy
