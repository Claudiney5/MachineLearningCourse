# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 11:53:35 2020

@author: Claudiney Martins
"""

import pandas as pd
base_pl = pd.read_csv('planets.csv', sep=',')
base_pl.describe()

base_pl.drop([
           'loc_rowid', 
           'st_mass',
           'pl_massj',
           'pl_radj',
           'sy_snum'], 1, inplace=True)



base_pl.loc[pd.isnull(base_pl['pl_orbper'])]
base_pl.loc[pd.isnull(base_pl['pl_orbsmax'])]




