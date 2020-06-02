# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 07:40:52 2020

@author: Claudiney Martins
"""

import pandas as pd

base = pd.read_csv('credit_data.csv')
base = base.dropna()

import matplotlib.pyplot as plt

# outliers no atributo 'age'
plt.boxplot(base.iloc[:,  2], showfliers=True)
outliers_age = base[(base.age < 0)]

# outliers no atributo 'loan'
plt.boxplot(base.iloc[:,  3], showfliers=True)
outliers_loan = base[(base.loan > 13400)]