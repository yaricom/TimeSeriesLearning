# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 10:55:34 2016

The plain vanila implementation of Recurrent Neural Network

@author: yaric
"""
import time
import datetime
from random import uniform

import numpy as np
import scipy.io as sio

class RNN(object):
    
    def __init__(self, n_features, n_outputs, n_neurons=100, param_update_scheme='Adagrad', 
                 learning_rate=1e-1, activation_rule='Tanh', 
                 use_batch_step=False, batch_step_size=25, relu_neg_slope=0.01, 
                 use_dropout_regularization=True, dropout_threshold=0.8, 
                 reg_strenght=0.5, use_regularization=True, 
                 sgd_shuffle=True):
        """
        Initializes RNN
        n_features the number of features per data sample
        n_outputs the number of output values to find
        n_neurons the number of neurons in hidden layer (Default: 100)
        param_update_scheme the algorithm used to update parameters after gradients update (Default: 'Adagrad')
        learning_rate - the start learning rate (Default: 1e-1)
        activation_rule - the single neuron non-linearity activation rule (Default: 'Tanh')
        use_batch_step the flag to indicate whether to use batch training (True), default - False
        batch_step_size the number of samples per batch (Default: 25)
        relu_neg_slope the ReLU negative slope (Default: 0.01)
        use_dropout_regularization whether to use dropout regularization threshold (Default: True)
        dropout_threshold the dropout threshold (Default: 0.8)
        reg_strenght the L2 regularization strength for training parameters (Default:0.001)
        use_regularization the flag to turn on/off regularization (Default: True)
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
        self.dropout_threshold = dropout_threshold
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
        mbh, mby = np.zeros_like(self.bh), np.zeros_like(self.by) # memory variables for Adagrad, RMSProp
        vWxh, vWhh, vWhy = np.zeros_like(self.Wxh), np.zeros_like(self.Whh), np.zeros_like(self.Why)
        vbh, vby = np.zeros_like(self.bh), np.zeros_like(self.by) # memory variables for Adam
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
            self.hprev = np.zeros((self.hidden_size, 1)) # reset RNN memory at start of new epoch
            
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
                    loss, score, dWxh, dWhh, dWhy, dbh, dby, self.hprev = step_f(inputs, targets, y_missing)
                else:
                    inputs = Xtr[j : j + 1, :] # just one row
                    targets = ytr[j : j + 1, :]
                    loss, score, dWxh, dWhh, dWhy, dbh, dby, self.hprev = step_f(inputs, targets, ytr_missing[j])
                
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
                    for param, dparam, mem in zip([self.Wxh, self.Whh, self.Why, self.bh, self.by], [dWxh, dWhh, dWhy, dbh, dby], [mWxh, mWhh, mWhy, mbh, mby]):
                        mem += dparam * dparam
                        param += -self.learning_rate * dparam / (np.sqrt(mem) + eps) # adagrad update
                elif self.param_update_scheme == 'RMSProp':
                    # with RMSProp
                    eps = 1e-8 # {1e−4, 1e−5, 1e−6}
                    decay_rate = 0.95# {0.9, 0.95}
                    for param, dparam, mem in zip([self.Wxh, self.Whh, self.Why, self.bh, self.by], [dWxh, dWhh, dWhy, dbh, dby], [mWxh, mWhh, mWhy, mbh, mby]):
                        mem = decay_rate * mem + (1 - decay_rate) * dparam * dparam # cache = decay_rate * cache + (1 - decay_rate) * dx**2
                        param += -self.learning_rate * dparam / (np.sqrt(mem) + eps) # RMSProp update
                elif self.param_update_scheme == 'Adam':
                    # with Adam
                    eps = 1e-8
                    beta1 = 0.9
                    beta2 = 0.999#0.99
                    for param, dparam, m, v in zip([self.Wxh, self.Whh, self.Why, self.bh, self.by], [dWxh, dWhh, dWhy, dbh, dby], [mWxh, mWhh, mWhy, mbh, mby], [vWxh, vWhh, vWhy, vbh, vby]):
                        m = beta1 * m + (1 - beta1) * dparam
                        v = beta2 * v + (1 - beta2) * (dparam * dparam)
                        #param += -self.learning_rate * m / (np.sqrt(v) + eps) # Adam update
                        # bias corrected
                        mt = m / (1 - pow(beta1, j + 1)) # N.B. j starts from 0
                        vt = v / (1 - pow(beta2, j + 1))
                        param += -self.learning_rate * mt / (np.sqrt(vt) + eps) # Adam update
                elif self.param_update_scheme == 'AdaMax':
                    # with AdaMax - a variant of Adam based on the infinity norm.
                    eps = 1e-8
                    beta1 = 0.9
                    beta2 = 0.99 #0.999# 0.95 #
                    step_size = self.learning_rate / (1 - pow(beta1, j + 1)) #bias correction
                    for param, dparam, m, v in zip([self.Wxh, self.Whh, self.Why, self.bh, self.by], [dWxh, dWhh, dWhy, dbh, dby], [mWxh, mWhh, mWhy, mbh, mby], [vWxh, vWhh, vWhy, vbh, vby]):
                        m = beta1 * m + (1 - beta1) * dparam # Update biased first moment estimate
                        v = np.maximum(beta2 * v, np.abs(dparam) + eps) # Update the exponentially weighted infinity norm
                        param += - step_size * m / v 
                else:
                    raise "Uknown parameters update scheme: {}".format(self.param_update_scheme)
                
    
            # Annealing the learning rate but avoid dropping it too low
            if self.learning_rate > 1e-6 and epoch != 0 and epoch % 20 == 0:  self.learning_rate *= 0.1
            
            train_scores[epoch] = self.__make_score(epoch_score) # the score per epoch
            train_errors[epoch] = np.average(epoch_error, axis=0) # the mean train error per epoch
            
            # calculate validation if appropriate
            if do_validation:
                y_predicted = self.__predict(Xvl, np.zeros_like(self.hprev))
                validation_errors[epoch], validation_scores[epoch] = self.__validate(y_predicted, yvl, yvl_missing)
                
                print 'epoch: %d, learning rate: %s, train loss: %s, score: %s\nvalidation loss: %s, score: %s' % (epoch, self.learning_rate, train_errors[epoch], train_scores[epoch], validation_errors[epoch], validation_scores[epoch]) # print progress
            else:
                print 'epoch: %d, learning rate: %s, train loss: %s, score: %s' % (epoch, self.learning_rate, train_errors[epoch], train_scores[epoch]) # print progress
    
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
        hprev = self.hprev if use_prev_state else np.zeros_like(self.hprev)
        return self.__predict(Xvl, hprev)

    def saveModel(self, name):
        """
        Saves trained model using provided file name
        """
        vault = {'Wxh' : self.Wxh, 
                 'Whh' : self.Whh, 
                 'Why': self.Why, 
                 'bh' : self.bh, 
                 'by' : self.by, 
                 'hprev' : self.hprev,
                 'hidden_size' : self.hidden_size,
                 'n_features' : self.n_features,
                 'n_outputs' : self.n_outputs,
                 'use_batch_step' : self.use_batch_step,
                 'batch_step_size' : self.batch_step_size,
                 'param_update_scheme' : self.param_update_scheme,
                 'learning_rate' : self.learning_rate,
                 'activation_rule' : self.activation_rule,
                 'relu_neg_slope' : self.relu_neg_slope,
                 'use_dropout_regularization' : self.use_dropout_regularization,
                 'dropout_threshold' : self.dropout_threshold,
                 'reg_strenght' : self.reg_strenght,
                 'use_regularization' : self.use_regularization }
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
        self.bh = mat_contents['bh']
        self.by = mat_contents['by']
        self.hprev = mat_contents['hprev']
        self.hidden_size = mat_contents['hidden_size']
        self.n_features = mat_contents['n_features']
        self.n_outputs = mat_contents['n_outputs']
        self.use_batch_step = mat_contents['use_batch_step']
        self.batch_step_size = mat_contents['batch_step_size']
        self.param_update_scheme = mat_contents['param_update_scheme']
        self.learning_rate = mat_contents['learning_rate']
        self.activation_rule = mat_contents['activation_rule']
        self.relu_neg_slope = mat_contents['relu_neg_slope']
        self.use_dropout_regularization = mat_contents['use_dropout_regularization']
        self.dropout_threshold = mat_contents['dropout_threshold']
        self.reg_strenght = mat_contents['reg_strenght']
        self.use_regularization = mat_contents['use_regularization']

    def __step_tanh(self, inputs, targets, ytr_missing):
        """
        The one step in RNN computations using Tanhents function as non-linear activation function
        inputs, targets are both arrays of real numbers with shapes (input_size, 1) and (target_size, 1) respectively.
        hprev is array of initial hidden state with shape (hidden_size, 1)
        Wxh, Whh, Why - the neurons input/output weights
        bh, by - the hidden/output layer bias
        returns the loss, score_mean, gradients on model parameters, and last hidden state
        """
        #
        # forward pass
        #
        xs = inputs.T
        hs = np.tanh(np.dot(self.Wxh, xs) + np.dot(self.Whh, self.hprev) + self.bh) # hidden state
        if self.use_regularization and self.use_dropout_regularization:
            U1 = (np.random.rand(*hs.shape) < self.dropout_threshold) / self.dropout_threshold # dropout mask
            hs *= U1 # drop!
        ys = np.dot(self.Why, hs) + self.by # unnormalized next outputs
        ps = ys - targets.T
        loss = np.sum(np.abs(ps)) # L1 norm
        
        #
        # backward pass: compute gradients going backwards
        #
        dy = np.sign(ps) # the gradient for y only inherits the sign of the difference for L1 norm (http://cs231n.github.io/neural-networks-2/#reg)
        dWhy = np.dot(dy, hs.T)
        dby = dy
        dh = np.dot(self.Why.T, dy) # backprop into h
        dhraw = (1 - hs * hs) * dh # backprop through tanh nonlinearity
        dbh = dhraw
        dWxh = np.dot(dhraw, inputs)
        dWhh = np.dot(dhraw, self.hprev.T)
        
        # add L2 regularization gradient contribution if not dropout
        if self.use_regularization and not self.use_dropout_regularization:
            dWhy += self.reg_strenght * self.Why
            dWhh += self.reg_strenght * self.Whh
            dWxh += self.reg_strenght * self.Wxh
            
        for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
            np.clip(dparam, -5, 5, out=dparam) # clip to mitigate exploding gradients
            
        score = self.__score_mean(np.abs(ps), ytr_missing) # IMPORTANT: use COVAR_y_MISSING flags for mean calculation without missed Y
        return loss, score, dWxh, dWhh, dWhy, dbh, dby, hs
        
    def __batch_step_tanh(self, inputs, targets, ytr_missing):
        """
        The one step in RNN computations over min batch of input features using Tanhents function as non-linear activation function
        inputs,targets are both list of real numbers.
        hprev is Hx1 array of initial hidden state
        returns the loss, gradients on model parameters, and last hidden state
        """
        input_size = len(inputs[0])
        target_size = len(targets[0])
        xs, hs, ys, ps = {}, {}, {}, {}
        hs[-1] = np.copy(self.hprev)
        loss = np.zeros((len(inputs), 1))
        score = np.zeros((len(inputs), 1))
        # forward pass
        for t in range(len(inputs)):
            xs[t] = np.reshape(inputs[t], (input_size, 1))
            hs[t] = np.tanh(np.dot(self.Wxh, xs[t]) + np.dot(self.Whh, hs[t-1]) + self.bh) # hidden state
            if self.use_regularization and self.use_dropout_regularization:
                U1 = (np.random.rand(*hs[t].shape) < self.dropout_threshold) / self.dropout_threshold # dropout mask
                hs[t] *= U1 # drop!
            ys[t] = np.dot(self.Why, hs[t]) + self.by
            ps[t] = ys[t] - np.reshape(targets[t], (target_size, 1))
            loss[t] = np.sum(np.abs(ps[t])) # L1 norm
            score[t] = self.__score_mean(np.abs(ps[t]), ytr_missing[t]) 
            
        # backward pass: compute gradients going backwards
        dWxh, dWhh, dWhy = np.zeros_like(self.Wxh), np.zeros_like(self.Whh), np.zeros_like(self.Why)
        dbh, dby = np.zeros_like(self.bh), np.zeros_like(self.by)
        dhnext = np.zeros_like(hs[0])
        for t in reversed(range(len(inputs))):
            dy = np.sign(ps[t]) # the gradient for y only inherits the sign of the difference for L1 norm (http://cs231n.github.io/neural-networks-2/#losses)
            dWhy += np.dot(dy, hs[t].T)
            dby += dy
            dh = np.dot(self.Why.T, dy) + dhnext # backprop into h
            dhraw = (1 - hs[t] * hs[t]) * dh # backprop through tanh nonlinearity
            dbh += dhraw
            dWxh += np.dot(dhraw, xs[t].T)
            dWhh += np.dot(dhraw, hs[t-1].T)
            dhnext = np.dot(self.Whh.T, dhraw)
          
        # add L2 regularization gradient contribution if not dropout
        if self.use_regularization and not self.use_dropout_regularization:
            dWhy += self.reg_strenght * self.Why
            dWhh += self.reg_strenght * self.Whh
            dWxh += self.reg_strenght * self.Wxh          
          
        for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
            np.clip(dparam, -5, 5, out=dparam) # clip to mitigate exploding gradients
            
        return np.average(loss), np.average(score), dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs)-1]
      
    def __step_relu(self, inputs, targets, ytr_missing):
        """
        The one step in RNN computations using ReLU function as non-linear activation function
        inputs, targets are both arrays of real numbers with shapes (input_size, 1) and (target_size, 1) respectively.
        hprev is array of initial hidden state with shape (hidden_size, 1)
        Wxh, Whh, Why - the neurons input/output weights
        bh, by - the hidden/output layer bias
        returns the loss, score_mean, gradients on model parameters, and last hidden state
        """
        #
        # forward pass
        #
        xs = inputs.T
        #hs = np.maximum(0, np.dot(self.Wxh, xs) + np.dot(self.Whh, self.hprev) + self.bh) # hidden state, ReLU activation
        hs = np.dot(self.Wxh, xs) + np.dot(self.Whh, self.hprev) + self.bh
        hs[hs<0] *= self.relu_neg_slope
        if self.use_regularization and self.use_dropout_regularization:
            U1 = (np.random.rand(*hs.shape) < self.reg_strenght) / self.reg_strenght # dropout mask
            hs *= U1 # drop!
        ys = np.dot(self.Why, hs) + self.by # unnormalized next outputs
        ps = ys - targets.T
        loss = np.sum(np.abs(ps)) # L1 norm
        
        #
        # backward pass: compute gradients going backwards
        #
        dy = np.sign(ps) # the gradient for y only inherits the sign of the difference for L1 norm (http://cs231n.github.io/neural-networks-2/#reg)
        dWhy = np.dot(dy, hs.T)
        dby = dy
        dh = np.dot(self.Why.T, dy) # backprop into h
        dh[hs < 0] = 0 # backprop through ReLU non-linearity
        dbh = dh
        dWxh = np.dot(dh, inputs)
        dWhh = np.dot(dh, self.hprev.T)
         
        # add L2 regularization gradient contribution if not dropout
        if self.use_regularization and not self.use_dropout_regularization:
            dWhy += self.reg_strenght * self.Why
            dWhh += self.reg_strenght * self.Whh
            dWxh += self.reg_strenght * self.Wxh  
            
        #for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
        #    np.clip(dparam, -5, 5, out=dparam) # clip to mitigate exploding gradients
            
        score = self.__score_mean(np.abs(ps), ytr_missing) # IMPORTANT: use COVAR_y_MISSING flags for mean calculation without missed Y
        return loss, score, dWxh, dWhh, dWhy, dbh, dby, hs
        
    def __batch_step_relu(self, inputs, targets, ytr_missing):
        """
        The one step in RNN computations over min batch of input features using ReLU function as non-linear activation function
        inputs,targets are both list of real numbers.
        hprev is Hx1 array of initial hidden state
        returns the loss, gradients on model parameters, and last hidden state
        """
        input_size = len(inputs[0])
        target_size = len(targets[0])
        xs, hs, ys, ps = {}, {}, {}, {}
        hs[-1] = np.copy(self.hprev)
        loss = np.zeros((len(inputs), 1))
        score = np.zeros((len(inputs), 1))
        # forward pass
        for t in range(len(inputs)):
            xs[t] = np.reshape(inputs[t], (input_size, 1))
            #hs[t] = np.maximum(0, np.dot(self.Wxh, xs[t]) + np.dot(self.Whh, hs[t-1]) + self.bh) # hidden state, ReLU Activation
            hs[t] = np.dot(self.Wxh, xs[t]) + np.dot(self.Whh, hs[t-1]) + self.bh
            hs[t][hs<0] *= self.relu_neg_slope            
            if self.use_regularization and self.use_dropout_regularization:
                U1 = (np.random.rand(*hs[t].shape) < self.reg_strenght) / self.reg_strenght # dropout mask
                hs[t] *= U1 # drop!
            ys[t] = np.dot(self.Why, hs[t]) + self.by
            ps[t] = ys[t] - np.reshape(targets[t], (target_size, 1))
            loss[t] = np.sum(np.abs(ps[t])) # L1 norm
            score[t] = self.__score_mean(np.abs(ps[t]), ytr_missing[t]) 
            
        # backward pass: compute gradients going backwards
        dWxh, dWhh, dWhy = np.zeros_like(self.Wxh), np.zeros_like(self.Whh), np.zeros_like(self.Why)
        dbh, dby = np.zeros_like(self.bh), np.zeros_like(self.by)
        dhnext = np.zeros_like(hs[0])
        for t in reversed(range(len(inputs))):
            dy = np.sign(ps[t]) # the gradient for y only inherits the sign of the difference for L1 norm (http://cs231n.github.io/neural-networks-2/#losses)
            dWhy += np.dot(dy, hs[t].T)
            dby += dy
            dh = np.dot(self.Why.T, dy) + dhnext # backprop into h
            dh[hs[t] < 0] = 0 # backprop through ReLU non-linearity
            dbh += dh
            dWxh += np.dot(dh, xs[t].T)
            dWhh += np.dot(dh, hs[t-1].T)
            dhnext = np.dot(self.Whh.T, dh)
           
        # add L2 regularization gradient contribution if not dropout
        if self.use_regularization and not self.use_dropout_regularization:
            dWhy += self.reg_strenght * self.Why
            dWhh += self.reg_strenght * self.Whh
            dWxh += self.reg_strenght * self.Wxh 
            
        #for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
        #    np.clip(dparam, -5, 5, out=dparam) # clip to mitigate exploding gradients
            
        return np.average(loss), np.average(score), dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs)-1]      
      
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
        num_samples = len(y)
        scores = np.zeros((num_samples, 1))
        errors = np.zeros_like(scores)
        for t in range(num_samples):
            # find error per sample
            ps = y[t] - y_target[t]
            errors[t] = np.sum(np.abs(ps)) # L1 norm
            # find score per sample
            scores[t] = self.__score_mean(np.abs(ps), y_missing[t])
            
        # find total score and error
        score = self.__make_score(scores)
        error = np.average(errors, axis=0)
        return error, score
        
    def __predict(self, Xvl, hprev):
        """
        The RNN predict method
        Xvl - the test data features
        """
        n = len(Xvl)
        input_size = len(Xvl[0])
        y_est = np.zeros((n, self.n_outputs))
        for t in range(n):
            x = np.reshape(Xvl[t], (input_size, 1))
            hprev = np.tanh(np.dot(self.Wxh, x) + np.dot(self.Whh, hprev) + self.bh)
            y = np.dot(self.Why, hprev) + self.by
            y_est[t] = y.T
            
        return y_est  
        
    def __initNNParameters(self):
        """
        Do NN parameters initialization according to provided data samples
        input_size the input layer size
        output_size the output layer size
        """
        self.Wxh = np.random.randn(self.hidden_size, self.n_features) * 0.01 # input to hidden
        self.Whh = np.random.randn(self.hidden_size, self.hidden_size) * 0.01 # hidden to hidden
        self.Why = np.random.randn(self.n_outputs, self.hidden_size) * 0.01 # hidden to output
        self.bh = np.zeros((self.hidden_size, 1)) # hidden bias
        self.by = np.zeros((self.n_outputs, 1)) # output bias
        self.hprev = np.zeros((self.hidden_size,1))
        
    def __activationFunction(self):
        """
        Finds appropriate activation function depending on configuration
        """
        step_f = None
        if self.use_batch_step:
            if self.activation_rule == 'Tanh':
                step_f = self.__batch_step_tanh
            elif self.activation_rule == 'ReLU':
                step_f = self.__batch_step_relu
        else:
            if self.activation_rule == 'Tanh':
                step_f = self.__step_tanh
            elif self.activation_rule == 'ReLU':
                step_f = self.__step_relu
        
        if step_f == None:
            raise 'Unsupported activation function specified: {}'.format(self.activation_rule)
            
        return step_f
        
    # gradient checking
    def __gradCheck(self, inputs, targets, ytr_missing):
        """
        The gradient check to test if analytic and numerical gradients converge
        returns found gradient errors per paarameter as map
        """
        num_checks, delta = 10, 1e-5
        step_f = self.__activationFunction()
            
        _, dWxh, dWhh, dWhy, dbh, dby, _ = step_f(inputs, targets, ytr_missing)
        
        gradient_rel_errors = {}
        for param,dparam,name in zip([self.Wxh, self.Whh, self.Why, self.bh, self.by], [dWxh, dWhh, dWhy, dbh, dby], ['Wxh', 'Whh', 'Why', 'bh', 'by']):
            s0 = dparam.shape
            s1 = param.shape
            assert s0 == s1, 'Error dims dont match: %s and %s.' % (`s0`, `s1`)
            print name
            errors = np.zeros((num_checks, 1))
            for i in xrange(num_checks):
                ri = int(uniform(0, param.size))
                # evaluate cost at [x + delta] and [x - delta]
                old_val = param.flat[ri]
                param.flat[ri] = old_val + delta
                cg0, _, _, _, _, _, _ = step_f(inputs, targets, ytr_missing)
                param.flat[ri] = old_val - delta
                cg1, _, _, _, _, _, _ = step_f(inputs, targets, ytr_missing)
                param.flat[ri] = old_val # reset old value for this parameter
                # fetch both numerical and analytic gradient
                grad_analytic = dparam.flat[ri]
                grad_numerical = (cg0 - cg1) / ( 2 * delta )
                if grad_numerical + grad_analytic != 0:
                    rel_error = abs(grad_analytic - grad_numerical) / abs(grad_numerical + grad_analytic)
                    print '%f, %f => %e ' % (grad_numerical, grad_analytic, rel_error)
                    # rel_error should be on order of 1e-7 or less
                    errors[i] = rel_error
                else:
                    errors[i] = 0
                    
            # store relative gradient error average per parameter
            gradient_rel_errors[name] = np.average(errors)
            
        return gradient_rel_errors