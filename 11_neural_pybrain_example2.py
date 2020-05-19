# -*- coding: utf-8 -*-
"""
Created on Tue May  5 22:53:53 2020

@author: Claudiney Martins
"""
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules import SoftmaxLayer
from pybrain.structure.modules import  SigmoidLayer

'''Rede = buildNetwork(2, 3, 1, outclass = SoftmaxLayer,
                    hiddenclass = SigmoidLayer,
                    bias = False)
print(rede['in'])  # vendo a função da camada de entrada
print(rede['hidden0'])  # função utilizada da camada oculta
print(rede['out'])  # função utilizada na camada de saída 
print(rede['bias'])'''

rede = buildNetwork(2, 3, 1)
base = SupervisedDataSet(2, 1)
#  a base abaixo pode ser importada de um arquivo
base.addSample((0, 0), (0, ))
base.addSample((0, 1), (1, ))
base.addSample((1, 0), (1, ))
base.addSample((1, 1), (0, ))
print(base['input'])
print(base['target'])

# treinamento
treinamento = BackpropTrainer(rede, 
                              dataset = base, 
                              learningrate = 0.01, 
                              momentum = 0.06)

for i in range(1, 10000):
    erro = treinamento.train()
    if i % 1000 == 0:
        print("Erro: %s" % erro)
        
print(rede.activate([0, 0])) 
print(rede.activate([0, 1]))    
print(rede.activate([1, 0]))    
print(rede.activate([1, 1]))           