# -*- coding: utf-8 -*-
"""
Created on Tue May 26 10:07:28 2020

@author: Claudiney Martins
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

base = pd.read_csv('c_credito.csv', header=1)

""" Para bases com muitos atributos é interessante iniciarmos as análises 
utilizando apenas 2 deles e depois ir gradativamente aumentando.

Para a base em questão e para a nálise que faremos podemos somar todas 
as BILLS para formar um único atributo """

base['']
