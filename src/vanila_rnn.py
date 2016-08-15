# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 18:13:32 2016

The plain vanila Recurrent NN with Tanhents/ReLU activation rules and
Adagrad/RMSProp parameters update schemes

@author: yaric
"""
import time
import datetime

import pandas as pd
import numpy as np

from rnn.simple_rnn import RNN

from utils import utils

# hyperparameters
hidden_size = 50 # size of hidden layer of neurons
n_epochs = 10 # 60 # 81#the number of learning epochs

# for RMSProp [0.0001](without regularization); 
# for Adagrad [0.05](without regularization);[1e-4](with dropout 0.8)
# for Adam[1e-3,1e-4] (with L2 regularization);
# for AdaMax [5e-4]
learning_rate = 5e-4#0.05#1e-4#
batch_step_size=100#200
param_update_scheme='Adam' #'Adagrad' #'RMSProp' # 'AdaMax' # 
activation_rule='Tanh' #'ReLU' #
relu_neg_slope=0.001 # 0.01
# whether to shuffle data samles in order to use Stochastic Gradient Descent like mechanics when batch processing
sgd_shuffle= True # False #

# The dropout regularization parameters
use_dropout_regularization=False#True# 
dropout_threshold=0.75
# The L2 regularization strength
reg_strenght=1e-3#
use_regularization=False # True #

# Whether to preprocess input features (normalization, standardization, PCA, etc)
USE_PREPROCESSING = False #True#
# Whether to use single step (False) or batch step training (True)
USE_BATCH_TRAINING = True #False #
# Whether to check gradient
CHECK_GRADIENT = False #True 

# debug mode switch
DEBUG = False #True #
# Whether to save model when in debug mode (in production mode model will be saved anyway)
SAVE_MODEL_DEBUG = False #True #

# Whether to use existing trained model for predicition only
PREDICT_ONLY = False #True #


def main():
    # import data
    if DEBUG:
        data_train = pd.read_csv('../data/training-train.csv')#pd.read_csv('../data/training-small-train.csv')#
        data_validation = pd.read_csv('../data/training-validate.csv')#pd.read_csv('../data/training-small-validate.csv')#
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
    Xtr, ytr, Xvl, yvl = utils.format_data(data, preprocessing=USE_PREPROCESSING)
    del data
    
    # preprocess data
    if USE_PREPROCESSING:
        use_pca = False # apply PCA (True) or standard normalization (False)
        Xtr, Xvl = utils.preprocess(Xtr, Xvl, use_pca)
        
    # create RNN instance 
    n_features = len(Xtr[0])
    n_outputs = len(ytr[0])
    nn_solver = RNN(n_features=n_features, n_outputs=n_outputs, 
                    n_neurons=hidden_size, param_update_scheme=param_update_scheme, 
                    learning_rate = learning_rate, activation_rule = activation_rule,
                    use_batch_step=USE_BATCH_TRAINING, batch_step_size=batch_step_size,
                    relu_neg_slope=relu_neg_slope,
                    use_dropout_regularization=use_dropout_regularization, dropout_threshold=dropout_threshold,
                    reg_strenght=reg_strenght, use_regularization=use_regularization,
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

