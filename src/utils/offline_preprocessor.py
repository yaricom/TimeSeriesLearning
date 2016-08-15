# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 10:54:14 2016

@author: yaric
"""

import numpy as np
import pandas as pd

from sklearn import decomposition

import utils

# the input file prefix of data sets
input_file_prefix = '../../data/training-' # '../../data/training-small-'
output_file_prefix = '../../data/training-preprocessed-'

max_pca_components = 19

def createDataFrame(X, y, y_missing):
    """
    Creates pandas data frame from provided numpy arrays
    """
    data = np.concatenate((y, X), axis=1)
    columns = ['y1', 'y2', 'y3']
    for k in range(X.shape[1]):
        columns.append('X{}'.format(k))
        
    data_df = pd.DataFrame(data, columns=columns)
    ymiss_df = pd.DataFrame(y_missing, columns=['COVAR_y1_MISSING', 'COVAR_y2_MISSING', 'COVAR_y3_MISSING'])
    df = data_df.join(ymiss_df)
    return df


# import data  
train_df = pd.read_csv(input_file_prefix + 'train.csv')
validate_df = pd.read_csv(input_file_prefix + 'validate.csv')

# keep missing flags for both training and validation
ytr_missing = np.array(train_df.loc[ :,'COVAR_y1_MISSING':'COVAR_y3_MISSING'], dtype=bool)
yvl_missing = np.array(validate_df.loc[ :,'COVAR_y1_MISSING':'COVAR_y3_MISSING'], dtype=bool)

# read data
train_df['train_flag'] = True
validate_df['train_flag'] = False
data = pd.concat((train_df, validate_df))

# remove temporary data
del train_df
del validate_df

# basic formatting
Xtr, ytr, Xvl, yvl = utils.format_data(data, preprocessing=False)
del data

#
# do preprocessing
#
scaler = decomposition.RandomizedPCA()
#scaler = decomposition.SparsePCA(n_components=max_pca_components)
#scaler = decomposition.PCA(n_components='mle')
print 'PCA max features to keep: %d' % (max_pca_components)
Xtr = scaler.fit_transform(Xtr) # fit only for train data (http://cs231n.github.io/neural-networks-2/#datapre)
Xvl = scaler.transform(Xvl) 


#
# write result
#
train_df = createDataFrame(Xtr, ytr, ytr_missing)
validate_df = createDataFrame(Xvl, yvl, yvl_missing)

train_df.to_csv(output_file_prefix + 'train.csv', header=True, index=False)
validate_df.to_csv(output_file_prefix + 'validate.csv', header=True, index=False)

print '\n---------------------\nResult train:\n%s\n' % (train_df.describe())
print '\n---------------------\nResult validate:\n%s\n' % (validate_df.describe())



