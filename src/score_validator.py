# -*- coding: utf-8 -*-
"""
Created on Sun Jul 24 17:15:12 2016

@author: yaric
"""
import pandas as pd
import numpy as np

# read predictions
y_pred_df = pd.read_csv('validation_predictions.csv', header=None)

# read validation
data_validation = pd.read_csv('data/training-validate.csv')#'data/training-small-validate.csv'
y_val_df = data_validation.loc[ :,'y1':'y3']
# replace nans with 0
y_val_df.fillna(0, inplace=True)
# get flags indicating if Y present in data
y_val_missing = np.array(data_validation.loc[:,'COVAR_y1_MISSING' : 'COVAR_y3_MISSING'])

# do scoring
y_pred = np.array(y_pred_df)
y_val = np.array(y_val_df)

assert len(y_pred) == len(y_val)

scores = np.abs(y_pred - y_val)

# the loops
n = len(scores)
means = np.zeros((n, 1))
for i in range(n): # simple loop
    means[i] = np.mean(scores[i][~y_val_missing[i]])
    
sum_r = np.sum(means)

score = 10 * (1 - sum_r/n)

print 'Score: %f, for %d rows' % (score, n)