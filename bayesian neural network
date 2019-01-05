#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 14:44:56 2018

@author: unalunsal
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import theano.tensor as tt
import pandas as pd
import numpy as np 
import theano
import pymc3 as pm
import time

# let`s get the data first
X, y = load_iris(True)

# Dependent variable y is a numpy array with 3 distinct values [0,1,2]
# I will turn it into a np ndarray so it can be used in the network.  
y0, y1, y2 = [],[],[]
for i in range(0, y.shape[0]):
    if y[i] == 0:
        y0.append(1)
        y1.append(0)
        y2.append(0)
    elif y[i] == 1:
        y0.append(0)
        y1.append(1)
        y2.append(0)        
    elif y[i] == 2:
        y0.append(0)
        y1.append(0)
        y2.append(1) 
         
Y = np.concatenate([np.array(y0).reshape(y.shape[0],1), np.array(y1).reshape(y.shape[0],1), np.array(y2).reshape(y.shape[0],1)], axis = 1)

# Define training and test sets. 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.5)

# Let`s define the function for the network:
def make_nn(ann_input, ann_output, n_hidden):

    init_1 = np.random.randn(X.shape[1], n_hidden)
    init_2 = np.random.randn(n_hidden, n_hidden)
    init_out = np.random.randn(n_hidden, Y.shape[1])
    
    with pm.Model() as nn_model:
        # Define weights
        w_1 = pm.Normal('w_1', mu=0, sd=1, shape=(X.shape[1], n_hidden), testval=init_1)
        w_2 = pm.Normal('w_2', mu=0, sd=1, shape=(n_hidden, n_hidden), testval=init_2)
        w_out = pm.Normal('w_out', mu=0, sd=1, shape=(n_hidden, Y.shape[1]), testval=init_out)

        # Define activations
        acts_1 = pm.Deterministic('activations_1', tt.tanh(tt.dot(ann_input, w_1)))
        acts_2 = pm.Deterministic('activations_2', tt.tanh(tt.dot(acts_1, w_2)))
        acts_out = pm.Deterministic('activations_out', tt.nnet.softmax(tt.dot(acts_2, w_out)))
        
        # Define likelihood
        out = pm.Multinomial('likelihood', n=1, p=acts_out, 
                              observed=ann_output)
        
    return nn_model


# I will define the ann_input and ann_output as theano shared variables. 
# This will make it possible to update them with the X_test and Y_test values later. 
# Otherwise, I will not be able to get the out of sample forecast accuray ratio. 

ann_input = theano.shared(X_train)
ann_output = theano.shared(Y_train)


n_hidden = 4  # define the number of hidden units. 

model = make_nn(X_train, Y_train, n_hidden=n_hidden) # define the network 

with model:
    s = theano.shared(pm.floatX(1.1))
    inference = pm.ADVI(cost_part_grad_scale=s)
    approx = pm.fit(100000, 
                    callbacks=[pm.callbacks.CheckParametersConvergence(tolerance=1e-1)])

with model:    
    trace = approx.sample(10000)

# update the theano shared variables with the Testing sets 
ann_input.set_value(X_test)
ann_output.set_value(Y_test)

# get the out of sample predictions
with model:
    ppc = pm.sample_posterior_predictive(trace, samples=10000, progressbar=True)
    
    
# convert the ppc into numpy nd array. It will be used below to get the accuracy ratio. 
preds_proba = ppc['likelihood'].mean(axis=0)
preds = (preds_proba == np.max(preds_proba, axis=1, keepdims=True)) * 1


accu = np.abs(preds - Y_test)


print("Out of sample acuracy ratio: ",
      np.unique(np.sum(accu, axis =1), return_counts = True)[1][0] / np.sum(np.unique(np.sum(accu, axis =1), return_counts = True)[1]))

