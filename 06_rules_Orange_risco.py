# -*- coding: utf-8 -*-
"""
Editor Spyder

by Claudiney 2020.04.19  10:41 hs
"""
import Orange

base = Orange.data.Table('risco_credito.csv')
base.domain

#    algoritmo antigo para indução de regras
cn2_learner = Orange.classification.rules.CN2Learner()
#    Learner é a variável que gera as regras e as REGRAS são os classificadores
classificador = cn2_learner(base)
#    vendo as regras criadas
for regras in classificador.rule_list:
    print(regras)
    
#    testes:
#    história BOA, dívida ALTA, garantias NENHUMA, renda > 35 (como na base de dados)
#    história RUIM, dívida ALTA, garantias ADEQUADA, renda < 15

resultado = classificador([['boa', 'alta', 'nenhuma', 'acima_35'], ['ruim', 'alta', 'adequada', '0_15']])

for i in resultado:
    print(base.domain.class_var.values[i])

