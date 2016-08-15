# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 22:42:08 2016

Utilities

@author: yaric
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn import decomposition

def plotResultsValidate(train_errors, train_scores, validation_errors, validation_scores):
    """
    Plots training results
    """
    nb_epochs = len(train_errors)
    epochs_range = np.arange(nb_epochs)
    
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.title("Train/Eval loss per epoch")
    plt.plot(epochs_range, train_errors * 100, 'b-', label='Train')
    plt.plot(epochs_range, validation_errors * 100, 'r-', label='Validate')
    plt.xlabel('epochs')
    plt.ylabel('error')
    plt.ylim(0., np.max(train_errors) * 100 + 5)
    plt.legend(loc="upper right")
    
    plt.subplot(2, 1, 2)
    plt.title("Train/Eval scores per epoch")
    plt.plot(epochs_range, train_scores, 'g-', label='Train')
    plt.plot(epochs_range, validation_scores, 'r-', label='Validate')
    plt.xlabel('epochs')
    plt.ylabel('score')
    plt.ylim(9., 10)
    plt.legend(loc="lower right")
    
    plt.subplots_adjust(0.1, 0.10, 0.98, 0.94, 0.2, 0.6)
    plt.show()
    
def plotResultsTest(train_errors, train_scores):
    """
    Plots test results
    """
    nb_epochs = len(train_errors)
    epochs_range = np.arange(nb_epochs)
    
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.title("Test loss per epoch")
    plt.plot(epochs_range, train_errors * 100, 'b-', label='Train')
    plt.xlabel('epochs')
    plt.ylabel('error')
    plt.ylim(0., np.maximum(0.5, np.max(train_errors)) * 100 + 5)
    
    plt.subplot(2, 1, 2)
    plt.title("Test scores per epoch")
    plt.plot(epochs_range, train_scores, 'g-', label='Train')
    plt.xlabel('epochs')
    plt.ylabel('score')
    plt.ylim(9., 10)
    
    plt.subplots_adjust(0.1, 0.10, 0.98, 0.94, 0.2, 0.6)
    plt.show()

def plotOutputs(y_pred, res_name):
    """
    Plot outputs
    """
    bins_count = 100
    # make histograms
    y1_hist, _ = np.histogram(y_pred[:,0], bins=bins_count)
    y2_hist, _ = np.histogram(y_pred[:,1], bins=bins_count)
    y3_hist, _ = np.histogram(y_pred[:,2], bins=bins_count)
    
    # draw scatter
    x = np.arange(bins_count)
    
    y1_plot = plt.scatter(x, np.log10(y1_hist), marker='o', color='b')
    y2_plot = plt.scatter(x, np.log10(y2_hist), marker='o', color='r')
    y3_plot = plt.scatter(x, np.log10(y3_hist), marker='o', color='g')
    
    plt.grid(color='black', linestyle='-')
    plt.title(res_name)
    plt.legend((y1_plot, y2_plot, y3_plot), ('y1','y2','y3'),
               scatterpoints=1, loc='upper right')
    # save figure
    plt.savefig('{}.{}'.format(res_name, 'png'), dpi=72)
    # show figure
    plt.show()

def format_data_preprocessed(data, dtype = np.float):
    """
    The input data preprocessing
    data the input data frame
    preprocessing whether to use features preprocessing (Default: False)
    dtype the data type for ndarray (Default: np.float)
    """
    train_flag = np.array(data['train_flag'])

    print 'Formatting input data, size: %d' % (len(train_flag))

    # outputs, nans excluded
    y = data.loc[ :,'y1':'y3']
    # replace nans with 0
    y.fillna(0, inplace=True)

    # collect only train data
    ytr = np.array(y)[train_flag]
    # collect only validation data
    yvl = np.array(y)[~train_flag]

    print 'Train data outputs collected, size: %d' % (len(ytr))
    print '\n\nData before encoding\n\n%s' % data.describe()


    # dropping target and synthetic columns
    data.drop(['y1','y2','y3','train_flag', 'COVAR_y1_MISSING', 'COVAR_y2_MISSING', 'COVAR_y3_MISSING'], axis=1, inplace=True)
    
    print '\n\nData after encoding\n\n%s' % data.describe()
    
    # split into training and test
    X = np.array(data).astype(dtype)
    
    Xtr = X[train_flag]
    Xvl = X[~train_flag]

    #print 'Train data first: %s' % (Xtr[0])
    #print 'Evaluate data first: %s' % (Xvl[0])

    return Xtr, ytr, Xvl, yvl

def format_data_features_selected(data, dtype = np.float):
    """
    The input data processign based on preselected relevant features
    """
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
                       'COVAR_ORDINAL_4', 'COVAR_ORDINAL_5', 'COVAR_ORDINAL_6',
                       'COVAR_ORDINAL_7', 'COVAR_ORDINAL_8',
                       'TIMEVAR1', 'TIMEVAR2',
                       'COVAR_y1_MISSING', 'COVAR_y2_MISSING', 'COVAR_y3_MISSING']
    train_flag = np.array(data['train_flag'])

    print 'Formatting input data, size: %d' % (len(train_flag))

    # outputs, nans excluded
    y = data.loc[ :,'y1':'y3']
    # replace nans with 0
    y.fillna(0, inplace=True)

    # collect only train data
    ytr = np.array(y)[train_flag]
    # collect only validation data
    yvl = np.array(y)[~train_flag]

    print 'Train data outputs collected, size: %d' % (len(ytr))
    print '\n\nData before encoding\n\n%s' % data.describe()


    # dropping columns
    features = data.loc[:, columns_to_keep]
    

    # do features construction
    drop_columns = ['COVAR_CONTINUOUS_24', 'COVAR_CONTINUOUS_18', 'COVAR_ORDINAL_4',
                    'COVAR_CONTINUOUS_1', 'COVAR_ORDINAL_1', 'COVAR_CONTINUOUS_13']
    data.drop(drop_columns, axis=1, inplace=True)
    """
    studyid = np.array(data.loc[:, 'STUDYID']).astype(dtype)
    subjid  = np.array(data.loc[:, 'SUBJID']).astype(dtype)
    del data
    
    userid = np.multiply(studyid, subjid)
    #userid = (userid - userid.mean()) / userid.std() # zero mean and standard deviation 1
    userid = np.log(userid) / np.sum(np.log(userid)) # 0 to 1
    
    userid_df = pd.DataFrame(userid, columns=['USERID'])
    features = features.join(userid_df)
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

    return Xtr, ytr, Xvl, yvl
  
def format_data(data, preprocessing=False, dtype = np.float):
    """
    The input data preprocessing
    data the input data frame
    preprocessing whether to use features preprocessing (Default: False)
    dtype the data type for ndarray (Default: np.float)
    """
    train_flag = np.array(data['train_flag'])

    print 'Formatting input data, size: %d' % (len(train_flag))

    # outputs, nans excluded
    y = data.loc[ :,'y1':'y3']
    # replace nans with 0
    y.fillna(0, inplace=True)

    # collect only train data
    ytr = np.array(y)[train_flag]
    # collect only validation data
    yvl = np.array(y)[~train_flag]

    print 'Train data outputs collected, size: %d' % (len(ytr))
    print '\n\nData before encoding\n\n%s' % data.describe()


    # dropping columns
    if preprocessing:
        data.drop(['y1','y2','y3','train_flag'], axis=1, inplace=True) # keep SUBJID
    else:
        data.drop(['y1','y2','y3','SUBJID','train_flag'], axis=1, inplace=True)

    # categorical encoding
    data = pd.get_dummies(data,columns=['STUDYID', u'SITEID', u'COUNTRY',
                                        'COVAR_NOMINAL_1','COVAR_NOMINAL_2',
                                        'COVAR_NOMINAL_3','COVAR_NOMINAL_4',
                                        'COVAR_NOMINAL_5','COVAR_NOMINAL_6',
                                        'COVAR_NOMINAL_7','COVAR_NOMINAL_8',
                                        'COVAR_y1_MISSING', 'COVAR_y2_MISSING',
                                        'COVAR_y3_MISSING'])

    # replace nans with 0
    # the least sophisticated approach possible
    data.fillna(0, inplace=True)
    
    print '\n\nData after encoding\n\n%s' % data.describe()
    
    # split into training and test
    X = np.array(data).astype(dtype)
    
    Xtr = X[train_flag]
    Xvl = X[~train_flag]

    #print 'Train data first: %s' % (Xtr[0])
    #print 'Evaluate data first: %s' % (Xvl[0])

    return Xtr, ytr, Xvl, yvl
    
# The data preprocessing
def preprocess(Xtr, Xvl, use_pca, max_pca_components=None):
    """
    The data preprocessing
    Xtr - the training data features
    Xvl - the test data features
    use_pca - whether to use PCA for feature space reduction
    max_pca_components - the maximal number of PCA components to extract
    return preprocessed features
    """
    if use_pca:
        if max_pca_components == None:
            raise "Please specify maximal number of PCA components to extract"
        #scaler = decomposition.RandomizedPCA(n_components=max_features)
        scaler = decomposition.SparsePCA(n_components=max_pca_components)
        print 'PCA max features to keep: %d' % (max_pca_components)
        Xtr = scaler.fit_transform(Xtr) # fit only for train data (http://cs231n.github.io/neural-networks-2/#datapre)
        Xvl = scaler.transform(Xvl) 
    else:
        scaler = StandardScaler(copy=False) 
        # scale only first column 'SUBJID'
        xtr_subj = Xtr[:,:1]
        xvl_subj = Xvl[:,:1]
        xtr_subj = scaler.fit_transform(xtr_subj) # fit only for train data (http://cs231n.github.io/neural-networks-2/#datapre)
        xvl_subj = scaler.transform(xvl_subj) 

    print 'Train data mean: %f, variance: %f' % (Xtr.mean(), Xtr.std())
    print 'Test data mean: %f, variance: %f' % (Xvl.mean(), Xvl.std())
    
    return Xtr, Xvl

def rescale(values, factor=1., dtype = np.float):
    
    factor = np.cast[dtype](factor)
    _,svs,_ = np.linalg.svd(values)
    #svs[0] is the largest singular value                      
    values = values / svs[0]
    return values