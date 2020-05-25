# -*- coding: utf-8 -*-
"""
Created on Mon May 25 16:05:43 2020

@author: Claudiney Martins
"""
import pandas as pd

dados = pd.read_csv('mercado.csv', header=None)

transacoes = []

for i in range(0,10):
    transacoes.append([str(dados.values[i, j]) for j in range(0, 4)])
    
from apyori import apriori

regras = apriori(transacoes, min_support=0.3,
                             min_confidence=0.8,
                             min_lift=2.0,
                             min_legth=2)

resultados = list(regras)

resultados2 = [list(x) for x in resultados]
resultados2
res_formatado = []
for j in range(0,3):
    res_formatado.append([list(x) for x in resultados2[j][2]])
    
res_formatado
