# -*- coding: utf-8 -*-
"""
Created on Sun Jul 24 15:59:19 2016

@author: yaric
"""

from sklearn.cross_validation import train_test_split
import pandas as pd

path_prefix = '../../data/training-pca'#'../../data/training'#'../../data/training-small-pca'#

# import data  
df = pd.read_csv(path_prefix + '.csv')

print 'Input:\n%s\n\n' % (df.describe())

# read X, Y
y = df.loc[:, 'y1':'y3']
X = df.loc[:, 'STUDYID' : 'COVAR_y3_MISSING']
#X = df.loc[:, 'COVAR_y1_MISSING':'PC19']

# split
X_train, X_test, y_train, y_test = train_test_split(X, y)
X_train_df = pd.DataFrame(X_train, columns=X.columns)
print 'X train:\n%s\n' % (X_train_df.describe())
y_train_df = pd.DataFrame(y_train, columns=y.columns)
print '\n\nY train:\n%s\n' % (y_train_df.describe())

X_test_df = pd.DataFrame(X_test, columns=X.columns)
print '\n---------------------\nX test:\n%s\n' % (X_test_df.describe())
y_test_df = pd.DataFrame(y_test, columns=y.columns)
print '\n\nY test:\n%s\n---------------------\n' % (y_test_df.describe())

# combine and save
data_train_df = pd.concat([y_train_df, X_train_df], axis=1, join_axes=[y_train_df.index])
print '\n---------------------\nResult train:\n%s\n' % (data_train_df.describe())

data_train_df.to_csv(path_prefix + '-train.csv',header=True,index=False) 

data_test_df = pd.concat([y_test_df, X_test_df], axis=1, join_axes=[y_test_df.index])
print '\n---------------------\nResult test:\n%s\n' % (data_test_df.describe())

data_test_df.to_csv(path_prefix + '-validate.csv',header=True,index=False) 


