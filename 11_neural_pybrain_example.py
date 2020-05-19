<<<<<<< HEAD
# -*- coding: utf-8 -*-
"""
Created on Mon May  4 08:57:37 2020

@author: Claudiney
"""

from pybrain.structure import FeedForwardNetwork
from pybrain.structure import LinearLayer, SigmoidLayer, BiasUnit  # funções
from pybrain.structure import FullConnection  # ligação entre as camadas

rede = FeedForwardNetwork()

camadaEntrada = LinearLayer(2)  # valores de entrada não serão alterados (LinearLayer)
camadaOculta = SigmoidLayer(3)
camadaSaida = SigmoidLayer(1)
bias1 = BiasUnit()   # unidade de bias para a camada oculta
bias2 = BiasUnit()   # unidade de bias para a camada de saida

# definido as camadas, vamos acrescentá-las a rede
rede.addModule(camadaEntrada)
rede.addModule(camadaOculta)
rede.addModule(camadaSaida)
rede.addModule(bias1)
rede.addModule(bias2)

# fazendo as ligação das camadas
entradaOculta = FullConnection(camadaEntrada, camadaOculta)
saidaOculta = FullConnection(camadaOculta, camadaSaida)
biasOculta = FullConnection(bias1, camadaOculta)
biasSaida = FullConnection(bias2, camadaSaida)

rede.sortModules()

print(rede)
print(entradaOculta.params)
print(saidaOculta.params)
print(biasOculta.params)
print(biasSaida.params)
=======
# -*- coding: utf-8 -*-
"""
Created on Mon May  4 08:57:37 2020

@author: Claudiney
"""

from pybrain.structure import FeedForwardNetwork
from pybrain.structure import LinearLayer, SigmoidLayer, BiasUnit  # funções
from pybrain.structure import FullConnection  # ligação entre as camadas

rede = FeedForwardNetwork()

camadaEntrada = LinearLayer(2)  # valores de entrada não serão alterados (LinearLayer)
camadaOculta = SigmoidLayer(3)
camadaSaida = SigmoidLayer(1)
bias1 = BiasUnit()   # unidade de bias para a camada oculta
bias2 = BiasUnit()   # unidade de bias para a camada de saida

# definido as camadas, vamos acrescentá-las a rede
rede.addModule(camadaEntrada)
rede.addModule(camadaOculta)
rede.addModule(camadaSaida)
rede.addModule(bias1)
rede.addModule(bias2)

# fazendo as ligação das camadas
entradaOculta = FullConnection(camadaEntrada, camadaOculta)
saidaOculta = FullConnection(camadaOculta, camadaSaida)
biasOculta = FullConnection(bias1, camadaOculta)
biasSaida = FullConnection(bias2, camadaSaida)

rede.sortModules()

print(rede)
print(entradaOculta.params)
print(saidaOculta.params)
print(biasOculta.params)
print(biasSaida.params)
>>>>>>> 0fff2a319a5d24a70ca5a034d77b600fe7cb9af1
