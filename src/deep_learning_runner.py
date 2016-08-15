# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 12:32:23 2016

The Deep Learning NN runner

@author: yaric
"""
import time
import datetime

import pandas as pd
import numpy as np

from deep.deep_learning_nn import DeepLearningNN

from utils import utils

# hyperparameters
n_neurons = [128, 32] # [64, 32]#  [256, 128] # size of hidden layers of neurons
n_epochs = 60 # the number of learning epochs

# for RMSProp it is good to have [1e-3, 1e-4], 
# for Adagrad [0.05], 
# for Adam [1e-4, 5e-5]
# for AdaMax [5e-4]
learning_rate = 5e-2 #2e-3 # 5e-4 # 
batch_step_size = 100#200
param_update_scheme = 'Adagrad' #'AdaMax' # 'RMSProp' #'Adam' #
activation_rule = 'ReLU'
relu_neg_slope = 0.001 # 0.01
sgd_shuffle = True

# The regularization parameters
use_dropout_regularization = False # True #
# The L2 regularization strength
reg_strenght = 1e-4
use_regularization = True

# Whether to preprocess input features (normalization, standardization, PCA, etc)
USE_PREPROCESSING = False #True#
# Whether to use single step (False) or batch step training (True)
USE_BATCH_TRAINING = True #False #
# Whether to check gradient
CHECK_GRADIENT = False #True 

# debug mode switch
DEBUG = False # True #
# Whether to save model when in debug mode (in production mode model will be saved anyway)
SAVE_MODEL_DEBUG = False #

# Whether to use existing trained model for predicition only
PREDICT_ONLY = False #True #

# the file prefix of debug data sets
debug_file_prefix = '../data/training-' # '../data/training-small-' # '../data/training-preprocessed-'

# whether data set in RAW form or already preprocessed
data_set_raw = True # False

def main():
    # import data
    if DEBUG:
        data_train = pd.read_csv(debug_file_prefix + 'train.csv')
        data_validation = pd.read_csv(debug_file_prefix + 'validate.csv')
    else:
        data_train = pd.read_csv('../data/training.csv')
        data_validation = pd.read_csv('../data/testData.csv')

    data_train['train_flag'] = True
    data_validation['train_flag'] = False
    data = pd.concat((data_train, data_validation))
    
    # keep missing flags for both training and validation
    ytr_missing = np.array(data_train.loc[ :,'COVAR_y1_MISSING':'COVAR_y3_MISSING'])
    yvl_missing = np.array(data_validation.loc[ :,'COVAR_y1_MISSING':'COVAR_y3_MISSING'])
    
    # remove temporary data
    del data_train
    del data_validation

    # basic formatting
    if data_set_raw:
        Xtr, ytr, Xvl, yvl = utils.format_data_features_selected(data)# utils.format_data(data, preprocessing=USE_PREPROCESSING)
    else:
        Xtr, ytr, Xvl, yvl = utils.format_data_preprocessed(data)
    del data
    
    # preprocess data
    if USE_PREPROCESSING:
        use_pca = False # apply PCA (True) or standard normalization (False)
        Xtr, Xvl = utils.preprocess(Xtr, Xvl, use_pca)
        
    # create RNN instance 
    n_features = len(Xtr[0])
    n_outputs = len(ytr[0])
    nn_solver = DeepLearningNN(n_features=n_features, n_outputs=n_outputs, 
                    n_neurons=n_neurons, param_update_scheme=param_update_scheme,
                    learning_rate = learning_rate, activation_rule = activation_rule,
                    use_dropout_regularization=use_dropout_regularization, 
                    reg_strenght=reg_strenght, use_regularization=use_regularization, 
                    relu_neg_slope=relu_neg_slope,
                    use_batch_step=USE_BATCH_TRAINING, batch_step_size=batch_step_size,
                    sgd_shuffle=sgd_shuffle)
                    
    if not PREDICT_ONLY:
        trainAndTest(nn_solver, Xtr, ytr, ytr_missing, Xvl, yvl, yvl_missing)
    else:
        predictByModel(nn_solver, Xvl, '../models/DeepNN/model_2016-08-03T15_39_15.mat')
        

def trainAndTest(nn_solver, Xtr, ytr, ytr_missing, Xvl, yvl, yvl_missing):
    """
    The train and test runner
    """
    if DEBUG:
        # train with validation
        train_errors, train_scores, validation_errors, validation_scores =  nn_solver.train(
                    Xtr = Xtr, ytr = ytr, ytr_missing = ytr_missing, 
                    n_epochs = n_epochs, Xvl = Xvl, yvl = yvl, yvl_missing = yvl_missing)
        # plot results
        utils.plotResultsValidate(train_errors, train_scores, validation_errors, validation_scores)
    else:
        # train without validation
        train_errors, train_scores =  nn_solver.train(
                    Xtr = Xtr, ytr = ytr, ytr_missing = ytr_missing, 
                    n_epochs = n_epochs)
        # plot results            
        utils.plotResultsTest(train_errors, train_scores)

    # and save model
    if DEBUG == False or (DEBUG and SAVE_MODEL_DEBUG):
        st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%dT%H_%M_%S')
        m_name = '../models/DeepNN/model_{}.mat'.format(st)
        nn_solver.saveModel(m_name)
    
    # test data predict
    predict(nn_solver, Xvl)


def predictByModel(nn_solver, Xvl, model_name):
    """    
    Method to make prediction on saved model
    """
    nn_solver.loadModel(model_name)
    
    predict(nn_solver, Xvl)


def predict(nn_solver, Xvl):
    """
    Do actual predicition 
    """
    yvl_est = nn_solver.predict(Xvl)
    
    # substitute negative with zeros (negative values mark absent Y)
    yvl_est = yvl_est.clip(min=0, max=1)
    
    assert len(yvl_est) == len(Xvl)
    
    # save predictions as csv
    if DEBUG:
        res_name = '../validation_predictions'
    else:
        st = datetime.datetime.fromtimestamp(time.time()).strftime('%d_%m_%H_%M')
        res_name = '../vp_{}'.format(st)
    yvl = pd.DataFrame({'yvl1_est':yvl_est[:,0],'yvl2_est':yvl_est[:,1],'yvl3_est':yvl_est[:,2]})
    yvl.to_csv('{}.{}'.format(res_name, 'csv'),header=False,index=False)
    
    # describe predictions
    print '\n------------------------\nPredictions:\n%s' % yvl.describe()
    
    # plot outputs
    utils.plotOutputs(yvl_est, res_name)


if __name__ == '__main__':
    main()