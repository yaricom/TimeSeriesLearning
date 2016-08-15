# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 21:28:07 2016

@author: yaric
"""

import pandas as pd

path_prefix = '../data/testData'#'data/training' 

# import data  
data = pd.read_csv(path_prefix + '.csv')

# slice data
small_data = data.loc[0 : 10000]
small_data.to_csv(path_prefix + '-small.csv',header=True,index=False)    
    