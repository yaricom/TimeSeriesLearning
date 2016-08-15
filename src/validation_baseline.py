# -*- coding: utf-8 -*-
'''
This is a baseline for the validation set.

We're making the simplest choices at each step, 
which can and should be improved on:

(1) The subject id variable is being ignored.
(2) The missing values are all being set to 0.
(3) There are three outputs and we are training three separate models.
(4) No feature selection or dimensionality reduction is being performed.
'''
import time
import datetime

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from utils import utils

def format_data(data):
    train_flag = np.array(data['train_flag'])    
    
    # outputs, nans included
    ytr1 = np.array(data['y1'])[train_flag]
    ytr2 = np.array(data['y2'])[train_flag]
    ytr3 = np.array(data['y3'])[train_flag]

    # dropping columns
    # subject id is not good for tree-based models
    data.drop(['y1','y2','y3','SUBJID','train_flag'], axis=1, inplace=True)

    # categorical encoding
    data = pd.get_dummies(data,columns=['STUDYID', u'SITEID', u'COUNTRY',
                                        'COVAR_NOMINAL_1','COVAR_NOMINAL_2',
                                        'COVAR_NOMINAL_3','COVAR_NOMINAL_4',
                                        'COVAR_NOMINAL_5','COVAR_NOMINAL_6',
                                        'COVAR_NOMINAL_7','COVAR_NOMINAL_8'])
        
    # replace nans with 0
    # the least sophisticated approach possible
    data.fillna(0,inplace=True)
    
    # split into training and test
    X = np.array(data).astype(np.float)
    Xtr = X[train_flag]
    Xvl = X[~train_flag]
    
    return Xtr, ytr1, ytr2, ytr3, Xvl
    
    
def format_data_preprocessed(data, dtype = np.float):
    columns_to_keep = ['COVAR_CONTINUOUS_1', 'COVAR_CONTINUOUS_10', 'COVAR_CONTINUOUS_11',
                       'COVAR_CONTINUOUS_12', 'COVAR_CONTINUOUS_13', 'COVAR_CONTINUOUS_14',
                       'COVAR_CONTINUOUS_15', 'COVAR_CONTINUOUS_16', 'COVAR_CONTINUOUS_17',
                       'COVAR_CONTINUOUS_18', 'COVAR_CONTINUOUS_2', 'COVAR_CONTINUOUS_20',
                       'COVAR_CONTINUOUS_21', 'COVAR_CONTINUOUS_22', 'COVAR_CONTINUOUS_23',
                       'COVAR_CONTINUOUS_23', 'COVAR_CONTINUOUS_24', 'COVAR_CONTINUOUS_25',
                       'COVAR_CONTINUOUS_26', 'COVAR_CONTINUOUS_27', 'COVAR_CONTINUOUS_28',
                       'COVAR_CONTINUOUS_29', 'COVAR_CONTINUOUS_3', 'COVAR_CONTINUOUS_30',
                       'COVAR_CONTINUOUS_4', 'COVAR_CONTINUOUS_5', 'COVAR_CONTINUOUS_6',
                       'COVAR_CONTINUOUS_7', 'COVAR_CONTINUOUS_8', 'COVAR_CONTINUOUS_9',
                       'COVAR_ORDINAL_1', 'COVAR_ORDINAL_2', 'COVAR_ORDINAL_3',
                       'COVAR_ORDINAL_4', 'TIMEVAR1', 'TIMEVAR2',
                       'COVAR_y1_MISSING', 'COVAR_y2_MISSING', 'COVAR_y3_MISSING']
    train_flag = np.array(data['train_flag'])

    print 'Formatting input data, size: %d' % (len(train_flag))

    # outputs, nans included
    ytr1 = np.array(data['y1'])[train_flag]
    ytr2 = np.array(data['y2'])[train_flag]
    ytr3 = np.array(data['y3'])[train_flag]

    print 'Train data outputs collected, size: %d' % (len(ytr1))

    # dropping columns
    features = data.loc[:, columns_to_keep]

    # do features construction
    """
    drop_columns = ['COVAR_CONTINUOUS_24', 'COVAR_CONTINUOUS_18', 'COVAR_ORDINAL_4',
                    'COVAR_CONTINUOUS_1', 'COVAR_ORDINAL_1', 'COVAR_CONTINUOUS_13']
    data.drop(drop_columns, axis=1, inplace=True)
    """

    # replace nans with 0
    # the least sophisticated approach possible
    features.fillna(0, inplace=True)
    
    print '\n\nData after encoding\n\n%s' % features.describe()
    
    # split into training and test
    X = np.array(features).astype(dtype)
    
    Xtr = X[train_flag]
    Xvl = X[~train_flag]

    #print 'Train data first: %s' % (Xtr[0])
    #print 'Evaluate data first: %s' % (Xvl[0])

    return Xtr, ytr1, ytr2, ytr3, Xvl

# the file prefix of debug data sets
debug_file_prefix = '../data/training-small-' # '../data/training-' # 
# debug mode switch
DEBUG = False # True #

# import data    
if DEBUG:
    data_train = pd.read_csv(debug_file_prefix + 'train.csv')
    data_validation = pd.read_csv(debug_file_prefix + 'validate.csv')
else:
    data_train = pd.read_csv('../data/training.csv')
    data_validation = pd.read_csv('../data/testData.csv')
    
data_train['train_flag'] = True
data_validation['train_flag'] = False
data_validation['y1'] = np.nan
data_validation['y2'] = np.nan
data_validation['y3'] = np.nan
data = pd.concat((data_train,data_validation))
del data_train
del data_validation

# basic formatting
Xtr, ytr1, ytr2, ytr3, Xvl = format_data_preprocessed(data) # format_data(data)
del data

print 'Start regressor'

start_time = datetime.datetime.fromtimestamp(time.time())

# random forest regressor
rfr = RandomForestRegressor(n_estimators=100)

# naive strategy: for each ytr, train where the output isn't missing
present_flag_1 = ~np.isnan(ytr1)
rfr.fit(Xtr[present_flag_1],ytr1[present_flag_1])
yvl1_est = rfr.predict(Xvl)

print 'yvl1_est estimated'

present_flag_2 = ~np.isnan(ytr2)
rfr.fit(Xtr[present_flag_2],ytr2[present_flag_2])
yvl2_est = rfr.predict(Xvl)

print 'yvl2_est estimated'

present_flag_3 = ~np.isnan(ytr3)
rfr.fit(Xtr[present_flag_3],ytr3[present_flag_3])
yvl3_est = rfr.predict(Xvl)

print 'yvl3_est estimated'

# The time spent
finish_date = datetime.datetime.fromtimestamp(time.time())
delta = finish_date - start_time
print '\n------------------------\nTrain/Test time: \n%s\n' % (delta)

# save results as csv
st = datetime.datetime.fromtimestamp(time.time()).strftime('%d_%m_%H_%M')
res_name = '../vp_tree_{}'.format(st)
yvl = pd.DataFrame({'yvl1_est':yvl1_est,'yvl2_est':yvl2_est,'yvl3_est':yvl3_est})
yvl.to_csv('{}.{}'.format(res_name, 'csv'), header=False, index=False)

# describe predictions
print '\n------------------------\nPredictions:\n%s' % yvl.describe()

# plot outputs
n = len(yvl1_est)
yvl_est = np.concatenate((np.reshape(yvl1_est, (n, 1)), np.reshape(yvl2_est, (n, 1)), np.reshape(yvl3_est, (n, 1))), axis=1)
utils.plotOutputs(yvl_est, res_name)