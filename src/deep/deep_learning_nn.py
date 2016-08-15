# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 10:30:44 2016

The cascade of connected NN forming deep learning 3-layered NN.

@author: yaric
"""
import time
import datetime
from math import sqrt

import numpy as np
import scipy.io as sio

DEBUG = True

class DeepLearningNN(object):
    
    def __init__(self, n_features, n_outputs, n_neurons=[50, 20], param_update_scheme='Adam', 
                 learning_rate=1e-1, activation_rule='ReLU', relu_neg_slope=0.01, 
                 use_dropout_regularization=True, input_dropout_threshold=0.75, 
                 hiden_dropout_threshold=0.5, reg_strenght=1e-3, use_regularization=True, 
                 use_batch_step=False, batch_step_size=25,
                 sgd_shuffle=True):
        """
        Initializes RNN
        n_features the number of features per data sample
        n_outputs the number of output values to find
        n_neurons the number of neurons per hidden layer (Default: [50, 20])
        param_update_scheme the algorithm used to update parameters after gradients update (Default: 'Adam')
        learning_rate - the start learning rate (Default: 1e-1)
        activation_rule - the single neuron non-linearity activation rule (Default: 'ReLU')
        relu_neg_slope the ReLU negative slope (Default: 0.01)
        use_dropout_regularization whether to use dropout regularization threshold (Default: True)
        input_dropout_threshold the input units dropout threshold (Default: 0.75)
        hiden_dropout_threshold the hidden units dropout threshold (Default: 0.5)
        reg_strenght the L2 regularization strength for training parameters (Default:1e-3)
        use_regularization the flag to turn on/off regularization (Default: True)
        use_batch_step the flag to indicate whether to use batch training (True), default - False
        batch_step_size the number of samples per batch (Default: 25)
        sgd_shuffle whether to shuffle data samples randomly after each epoch (Default: True)
        """
        self.hidden_size = n_neurons
        self.n_features = n_features
        self.n_outputs = n_outputs
        self.use_batch_step = use_batch_step
        self.batch_step_size = batch_step_size
        self.param_update_scheme = param_update_scheme
        self.learning_rate = learning_rate
        self.activation_rule = activation_rule
        self.relu_neg_slope = relu_neg_slope
        self.use_dropout_regularization = use_dropout_regularization
        self.input_dropout_threshold = input_dropout_threshold
        self.hiden_dropout_threshold = hiden_dropout_threshold
        self.reg_strenght = reg_strenght
        self.use_regularization = use_regularization
        self.sgd_shuffle = sgd_shuffle
        
    def train(self, Xtr, ytr, ytr_missing, n_epochs, Xvl=None, yvl=None, yvl_missing=None, check_gradient=False):
        """
        Trains neural network over specified epochs with optional validation if validation data provided
        Xtr - the train features tenzor with shape (num_samples, num_features)
        ytr - the train ground truth tenzor with shape (num_samples, num_outputs)
        ytr_missing - the boolean flags denoting missing train outputs with shape (num_samples, num_outputs)
        n_epochs - the number of epochs to use for training
        Xvl - the validation features tenzor with shape (num_samples, num_features) (Default: None)
        yvl - the validation ground truth tenzor with shape (num_samples, num_outputs) (Default: None)
        yvl_missing - the boolean flags denoting missing validation outputs with shape (num_samples, num_outputs) (Default: None)
        check_gradient - the boolean to indicate if gradient check should be done (Default: False)
        return trained model parameters as well as train/validation errors and scores per epoch
        """
        # parameters check
        assert len(Xtr[0]) == self.n_features
        assert len(ytr[0]) == self.n_outputs
        assert len(ytr_missing[0]) == self.n_outputs
        
        do_validation = (Xvl is not None)
        if do_validation and (yvl is None or yvl_missing is None):
            raise 'Validation outputs or missing falgs not specified when validation requested'
        elif do_validation:
            # check that validation parameters of correct size
            assert len(Xtr[0]) == len(Xvl[0])
            assert len(ytr[0]) == len(yvl[0])
            assert len(yvl[0]) == len(yvl_missing[0])
            
        # model parameters
        self.__initNNParameters()
        
        start_time = datetime.datetime.fromtimestamp(time.time())
        
        # do train
        mWxh, mWhh, mWhy = np.zeros_like(self.Wxh), np.zeros_like(self.Whh), np.zeros_like(self.Why)
        mbxh, mbhh, mbhy = np.zeros_like(self.bxh), np.zeros_like(self.bhh), np.zeros_like(self.bhy) # memory variables for Adagrad, RMSProp
        vWxh, vWhh, vWhy = np.zeros_like(self.Wxh), np.zeros_like(self.Whh), np.zeros_like(self.Why)
        vbxh, vbhh, vbhy = np.zeros_like(self.bxh), np.zeros_like(self.bhh), np.zeros_like(self.bhy) # memory variables for Adam
        train_errors = np.zeros((n_epochs, 1))
        train_scores = np.zeros_like(train_errors)
        if do_validation:
            validation_errors = np.zeros_like(train_errors)
            validation_scores = np.zeros_like(train_errors)
            
        n = 0
        step_f = self.__activationFunction()
        for epoch in range(n_epochs):
            # prepare for new epoch
            if self.use_batch_step:
                steps = len(Xtr) / self.batch_step_size
            else:
                steps = len(Xtr)
            epoch_error = np.zeros((steps, 1))
            epoch_score = np.zeros((steps, 1))
            
            # shuffle data for stohastic gradient descent before new epoch start
            if self.use_batch_step and self.sgd_shuffle:
                perm = np.arange(Xtr.shape[0])
                np.random.shuffle(perm)
                Xtr = Xtr[perm]
                ytr = ytr[perm]
            
            # proceed with mini-batches
            for j in range(steps): 
                if self.use_batch_step:
                    index = j * self.batch_step_size
                    inputs = Xtr[index : index + self.batch_step_size, :] # the slice of rows with batch_size length
                    targets = ytr[index : index + self.batch_step_size, :]
                    y_missing = ytr_missing[index : index + self.batch_step_size, :]
                    loss, score, dWxh, dWhh, dWhy, dbx, dbh, dby = step_f(inputs, targets, y_missing)
                else:
                    inputs = Xtr[j : j + 1, :] # just one row
                    targets = ytr[j : j + 1, :]
                    loss, score, dWxh, dWhh, dWhy, dbx, dbh, dby = step_f(inputs, targets, ytr_missing[j])
                
                epoch_error[j] = loss
                epoch_score[j] = score
    
                if j % 100 == 0: print '---iter %d, epoch: %d, step: %d from: %d, loss: %.5f' % (n, epoch, j, steps, loss) # print progress  
                
                n += 1 # total iteration counter
                
                if check_gradient:
                    self.__gradCheck(inputs, targets, ytr_missing[j])
            
                # perform parameter update 
                if self.param_update_scheme == 'Adagrad':
                    # with Adagrad
                    eps = 1e-8#1e-4#
                    for param, dparam, mem in zip([self.Wxh, self.Whh, self.Why, self.bxh, self.bhh, self.bhy], [dWxh, dWhh, dWhy, dbx, dbh, dby], [mWxh, mWhh, mWhy, mbxh, mbhh, mbhy]):
                        mem += dparam * dparam
                        param += -self.learning_rate * dparam / (np.sqrt(mem) + eps) # adagrad update
                elif self.param_update_scheme == 'RMSProp':
                    # with RMSProp
                    eps = 1e-8 # {1e−4, 1e−5, 1e−6}
                    decay_rate = 0.99# {0.9, 0.95}
                    for param, dparam, mem in zip([self.Wxh, self.Whh, self.Why, self.bxh, self.bhh, self.bhy], [dWxh, dWhh, dWhy, dbx, dbh, dby], [mWxh, mWhh, mWhy, mbxh, mbhh, mbhy]):
                        mem = decay_rate * mem + (1 - decay_rate) * (dparam * dparam) # cache = decay_rate * cache + (1 - decay_rate) * dx**2
                        param += -self.learning_rate * dparam / (np.sqrt(mem) + eps) # RMSProp update
                elif self.param_update_scheme == 'Adam':
                    # with Adam
                    eps = 1e-8
                    beta1 = 0.9
                    beta2 = 0.99 #0.95 #0.999# 
                    for param, dparam, m, v in zip([self.Wxh, self.Whh, self.Why, self.bxh, self.bhh, self.bhy], [dWxh, dWhh, dWhy, dbx, dbh, dby], [mWxh, mWhh, mWhy, mbxh, mbhh, mbhy], [vWxh, vWhh, vWhy, vbxh, vbhh, vbhy]):
                        m = beta1 * m + (1 - beta1) * dparam # Update biased first moment estimate
                        v = beta2 * v + (1 - beta2) * (dparam * dparam) # Update biased second raw moment estimate
                        #param += -self.learning_rate * m / (np.sqrt(v) + eps) # Adam update
                        # bias corrected estimates
                        mt = m / (1 - pow(beta1, j + 1)) #  N.B. j starts from 0
                        vt = v / (1 - pow(beta2, j + 1))
                        param += -self.learning_rate * mt / (np.sqrt(vt) + eps) # Adam update
                elif self.param_update_scheme == 'AdaMax':
                    # with AdaMax - a variant of Adam based on the infinity norm.
                    eps = 1e-8
                    beta1 = 0.9
                    beta2 = 0.99 #0.95 #0.999# 
                    step_size = self.learning_rate / (1 - pow(beta1, j + 1)) #bias correction
                    for param, dparam, m, v in zip([self.Wxh, self.Whh, self.Why, self.bxh, self.bhh, self.bhy], [dWxh, dWhh, dWhy, dbx, dbh, dby], [mWxh, mWhh, mWhy, mbxh, mbhh, mbhy], [vWxh, vWhh, vWhy, vbxh, vbhh, vbhy]):
                        m = beta1 * m + (1 - beta1) * dparam # Update biased first moment estimate
                        v = np.maximum(beta2 * v, np.abs(dparam) + eps) # Update the exponentially weighted infinity norm
                        param += - step_size * m / v 
                else:
                    raise "Uknown parameters update scheme: {}".format(self.param_update_scheme)
                
    
            # Annealing the learning rate but avoid dropping it too low
            if self.learning_rate >= 1e-6 and epoch != 0 and epoch % 20 == 0:  self.learning_rate *= 0.1
            
            train_scores[epoch] = self.__make_score(epoch_score) # the score per epoch
            train_errors[epoch] = np.average(epoch_error, axis=0) # the mean train error per epoch
            
            # calculate validation if appropriate
            if do_validation:
                y_predicted = self.predict(Xvl)
                validation_errors[epoch], validation_scores[epoch] = self.__validate(y_predicted, yvl, yvl_missing)
                
                print 'epoch: %d, train loss: %s, score: %s, learning rate: %s\nvalidation loss: %s, score: %s' % (epoch, train_errors[epoch], train_scores[epoch], self.learning_rate, validation_errors[epoch], validation_scores[epoch]) # print progress
            else:
                print 'epoch: %d, train loss: %s, score: %s, learning rate: %s' % (epoch, train_errors[epoch], train_scores[epoch], self.learning_rate) # print progress
    
        # The time spent
        finish_date = datetime.datetime.fromtimestamp(time.time())
        delta = finish_date - start_time
        print '\n------------------------\nTrain time: \n%s\nTrain error: \n%s\nscores:\n%s\n' % (delta, train_errors, train_scores)
        
        if do_validation:
            print '\n------------------------\nValidation error: \n%s\nscores:\n%s\n' % (validation_errors, validation_scores)
            return train_errors, train_scores, validation_errors, validation_scores
        else:
            return train_errors, train_scores
            
            
    def predict(self, Xvl, use_prev_state = False):
        """
        The method to predict outputs based on provided data samples
        Xvl the data samples with shape (num_samples, n_features)
        use_prev_state whether to use saved previous state of RNN or just reset its memory
        return predicitions per data sample with shape (num_samples, n_outputs)
        """
        # ensembled forward pass
        H1 = np.maximum(0, np.dot(Xvl, self.Wxh) + self.bxh)
        H2 = np.maximum(0, np.dot(H1, self.Whh) + self.bhh)
        out = np.dot(H2, self.Why) + self.bhy        
            
        return out 

    def saveModel(self, name):
        """
        Saves trained model using provided file name
        """
        vault = {'Wxh' : self.Wxh, 
                 'Whh' : self.Whh, 
                 'Why' : self.Why, 
                 'bxh' : self.bxh,
                 'bhh' : self.bhh, 
                 'byh' : self.bhy, 
                 'hidden_size' : self.hidden_size,
                 'n_features' : self.n_features,
                 'n_outputs' : self.n_outputs,
                 'use_batch_step' : self.use_batch_step,
                 'batch_step_size' : self.batch_step_size,
                 'param_update_scheme' : self.param_update_scheme,
                 'learning_rate' : self.learning_rate,
                 'use_dropout_regularization' : self.use_dropout_regularization,
                 'input_dropout_threshold' : self.input_dropout_threshold,
                 'hiden_dropout_threshold' : self.hiden_dropout_threshold,
                 'reg_strenght' : self.reg_strenght,
                 'use_regularization' : self.use_regularization,
                 'sgd_shuffle' : self.sgd_shuffle,
                 'activation_rule' : self.activation_rule}
                 
        sio.savemat(name, vault)
   
    def loadModel(self, name):
        """
        Loads model from spefied file
        name the path to the model file
        """
        mat_contents = sio.loadmat(name)
        self.Wxh = mat_contents['Wxh']
        self.Whh = mat_contents['Whh']
        self.Why = mat_contents['Why']
        self.bxh = mat_contents['bxh']
        self.bhh = mat_contents['bhh']
        self.bhy = mat_contents['byh']
        self.hidden_size = mat_contents['hidden_size']
        self.n_features = mat_contents['n_features']
        self.n_outputs = mat_contents['n_outputs']
        self.use_batch_step = mat_contents['use_batch_step']
        self.batch_step_size = mat_contents['batch_step_size']
        self.param_update_scheme = mat_contents['param_update_scheme']
        self.learning_rate = mat_contents['learning_rate']
        self.use_dropout_regularization = mat_contents['use_dropout_regularization']
        self.input_dropout_threshold = mat_contents['input_dropout_threshold']
        self.hiden_dropout_threshold = mat_contents['hiden_dropout_threshold']
        self.reg_strenght = mat_contents['reg_strenght']
        self.use_regularization = mat_contents['use_regularization']
        self.sgd_shuffle = mat_contents['sgd_shuffle']
        self.activation_rule = mat_contents['activation_rule']
        
    def __step_relu(self, inputs, targets, ytr_missing):
        """
        The one step in NN computations using ReLU function as non-linear activation function
        inputs, targets are both arrays of real numbers with shapes (input_size, 1) and (target_size, 1) respectively.
        hprev is array of initial hidden state with shape (hidden_size, 1)
        Wxh, Whh, Why - the neurons input/output weights
        bh, by - the hidden/output layer bias
        returns the loss, score_mean, gradients on model parameters, and last hidden state
        """
        #
        # forward pass
        #
        xs = inputs
        hidden_1 = np.maximum(0, np.dot(xs, self.Wxh) + self.bxh) # input-to-hidden, ReLU activation
        if self.use_regularization and self.use_dropout_regularization:
            U1 = (np.random.rand(*hidden_1.shape) < self.input_dropout_threshold ) / self.input_dropout_threshold  # first dropout mask
            hidden_1 *= U1 # drop! and scale the activations by p at test time. (see: http://cs231n.github.io/neural-networks-2/#reg - Inverted Dropout)
            
        hidden_2 = np.maximum(0, np.dot(hidden_1, self.Whh)  + self.bhh) # hidden-to-hidden, ReLU activation
        if self.use_regularization and self.use_dropout_regularization:
            U2 = (np.random.rand(*hidden_2.shape) < self.hiden_dropout_threshold) / self.hiden_dropout_threshold # second dropout mask
            hidden_2 *= U2 # drop! and scale the activations by p at test time.
        
        ys = np.dot(hidden_2, self.Why)  + self.bhy # hidden-to-output, ReLU activation
        ps = ys - targets # error
        loss = np.sum(np.abs(ps), axis=1) # L1 norm
        
        #
        # backward pass: compute gradients going backwards
        #
        dy = np.sign(ps) # the gradient for y only inherits the sign of the difference for L1 norm (http://cs231n.github.io/neural-networks-2/#reg)
        # first backprop into parameters Why and bhy
        dWhy = np.dot(hidden_2.T, dy)
        dby = np.sum(dy, axis=0, keepdims=True)
        # next backprop into hidden layer
        dhidden_2 = np.dot(dy, self.Why.T)
        
        # backprop the ReLU non-linearity
        dhidden_2[hidden_2 <= 0] = 0
        # backprop into Whh, bhh
        dWhh = np.dot(hidden_1.T, dhidden_2)
        dbh = np.sum(dhidden_2, axis=0, keepdims=True)  
        # next backprop into hidden layer
        dhidden_1 = np.dot(dhidden_2, self.Whh.T)
        
        # backprop the ReLU non-linearity
        dhidden_1[hidden_1 <= 0] = 0
        # backprop into Wxh, bxh
        dWxh = np.dot(xs.T, dhidden_1)
        dbx = np.sum(dhidden_1, axis=0, keepdims=True) 
        
        # add L2 regularization gradient contribution if not dropout
        if self.use_regularization and not self.use_dropout_regularization:
            dWhy += self.reg_strenght * self.Why
            dWhh += self.reg_strenght * self.Whh
            dWxh += self.reg_strenght * self.Wxh
          
        # calculate score
        score = np.zeros((inputs.shape[0], 1))
        for t in range(inputs.shape[0]):
            score[t] = self.__score_mean(np.abs(ps[t, :]), ytr_missing[t, :]) # IMPORTANT: use COVAR_y_MISSING flags for mean calculation without missed Y
        return np.average(loss), np.average(score), dWxh, dWhh, dWhy, dbx, dbh, dby
            
        
    def __score_mean(self, abs_diff, y_missing):
        """
        Calculates score mean on based absolute differences between Y predicted and target
        abs_diff = |Ypred - Yeval|
        y_missing the array with COVAR_y_MISSING flags with shape (target_size, 1)
        """  
        scores = abs_diff.flat[~y_missing]
        return np.mean(scores)
        
    def __make_score(self, mean_scores):
        """
        Calculates final score from provided array of mean scores
        mean_scores the array of mean scores
        return score value
        """
        n = len(mean_scores)
        sum_r = np.sum(mean_scores)
        score = 10 * (1 - sum_r/n)
        return score
    
    def __validate(self, y, y_target, y_missing):
        """
        The method to validate calculated validation outputs against ground truth
        y the calculated predictions with shape (num_samples, output_size)
        y_target the ground trouth with shape (num_samples, output_size)
        y_missing the array of flags denoting missed ground trouth value for predicition with shape (num_samples, output_size)
        return calculated score and error values over provided data set
        """
        ps = np.abs(y - y_target)
        errors = np.sum(ps, axis=1) # L1 norm     
        
        scores = np.zeros((y.shape[0], 1))
        for t in range(y.shape[0]):
            # find score per sample
            scores[t] = self.__score_mean(ps[t], y_missing[t])
            
        # find total score and error
        score = self.__make_score(scores)
        error = np.average(errors, axis=0)
        return error, score
        
    def __activationFunction(self):
        """
        Finds appropriate activation function depending on configuration
        """
        step_f = None
        if self.activation_rule == 'ReLU':
            step_f = self.__step_relu
        
        if step_f == None:
            raise 'Unsupported activation function specified: {}'.format(self.activation_rule)
            
        return step_f
        
    def __initNNParameters(self):
        """
        Do NN parameters initialization according to provided data samples
        input_size the input layer size
        output_size the output layer size
        """
        if self.activation_rule == 'ReLU':
            self.Wxh = np.random.randn(self.n_features, self.hidden_size[0]) * sqrt(2.0/self.n_features) # input to hidden
            self.Whh = np.random.randn(self.hidden_size[0], self.hidden_size[1]) * sqrt(2.0/self.hidden_size[0]) # hidden to hidden
            self.Why = np.random.randn(self.hidden_size[1], self.n_outputs) * sqrt(2.0/self.hidden_size[1]) # hidden to output
        else:
            self.Wxh = np.random.randn(self.n_features, self.hidden_size[0]) * 0.01 # input to hidden
            self.Whh = np.random.randn(self.hidden_size[0], self.hidden_size[1]) * 0.01 # hidden to hidden
            self.Why = np.random.randn(self.hidden_size[1], self.n_outputs) * 0.01 # hidden to output
            
        self.bxh = np.zeros((1, self.hidden_size[0])) # input-to-hidden bias
        self.bhh = np.zeros((1, self.hidden_size[1])) # hidden-to-hidden bias
        self.bhy = np.zeros((1, self.n_outputs)) # hidden-to-output bias
        
        if DEBUG:
            print 'Wxh: %s, Whh: %s, Why: %s\nbxh: %s, bhh: %s, bhy: %s' % (np.shape(self.Wxh), np.shape(self.Whh), np.shape(self.Why), np.shape(self.bxh), np.shape(self.bhh), np.shape(self.bhy))