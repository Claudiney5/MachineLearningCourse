# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 09:22:39 2020

@author: USUARIO-PC
"""
import Orange

base = Orange.data.Table('credit_data.csv')
base.domain

# dividir base de dados em base treinamento(75%) e teste(25%)
base_dividida = Orange.evaluation.testing.sample(base, n=0.25)
base_treinamento = base_dividida[1]
base_teste = base_dividida[0]
len(base_treinamento)
len(base_teste)

classificador = Orange.classification.MajorityLearner()
resultado = Orange.evaluation.testing.TestOnTestData(base_treinamento, base_teste, [classificador])
print(Orange.evaluation.CA(resultado))

from collections import Counter
print(Counter(str(d.get_class())for d in base_teste))

