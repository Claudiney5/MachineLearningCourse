# -*- coding: utf-8 -*-
"""
Created on Mon May 25 16:05:43 2020

@author: Claudiney Martins

mercado2.csv tras vendsa de um determinado mercado durante uma semana.

VALOR DO SUPORTE: Para encontrarmos regras de associação, digamos, para um produto vendido 4 vezes por dia:
    4 x 7 = 28
    28/7501 = 0,003732  (valor de suporte)



"""
import pandas as pd

dados = pd.read_csv('mercado2.csv', header=None)

transacoes = []

for i in range(0, 7501):
    transacoes.append([str(dados.values[i, j]) for j in range(0, 20)])
    
from apyori import apriori

regras = apriori(transacoes, min_support=0.0035,
                             min_confidence=0.3,
                             min_lift=4.0,
                             min_legth=2)

resultados = list(regras)

resultados2 = [list(x) for x in resultados]
resultados2
res_formatado = []
for j in range(0, 10):
    res_formatado.append([list(x) for x in resultados2[j][2]])
    
res_formatado
