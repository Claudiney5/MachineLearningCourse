    # -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 08:09:45 2020

@author: Claudiney Martins
"""
import pandas as pd
import matplotlib.pyplot as plt

base = pd.read_csv('credit_data.csv')
base.dropna()
base.loc[base.age<0, 'age'] = 40.92 # média

# income x age
plt.scatter(base.iloc[:, 1], base.iloc[:, 2])
 
# income x loan
plt.scatter(base.iloc[:, 1], base.iloc[:, 3])

# age x loan
plt.scatter(base.iloc[:, 2], base.iloc[:, 3])

base_c = pd.read_csv('census.csv')

# age x final weight
plt.scatter(base_c.iloc[:, 0], base_c.iloc[:, 2])