
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 5 10:16:58 2018

@author: unalunsal
"""

import pylab as pl
from pymc3 import Model, glm
import pymc3 as pm 
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from scipy.stats import mode
from sklearn.model_selection import train_test_split
import theano.tensor as tt
import pandas as pd
import numpy as np 
import theano, time
import pymc3 as pm
import seaborn as sns


# let`s get the data first
X, y = load_iris(True)
         

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5)

# define the Xt and yt as theano shared variables. Otherwise, I will not be able to update them with the test sets. 
Xt = theano.shared(X_train)
yt = theano.shared(y_train)


now = time.time()
with pm.Model() as model:  
    # Coefficients for the dependent variables 
    β = pm.Normal('β', 0, sd=1e2, shape=(4,1)) 
    a = pm.Flat('a', shape=(3,))
    p = tt.nnet.softmax(Xt.dot(β) + a)
    #p = tt.nnet.softmax(Xt.dot(β))
    observed = pm.Categorical('obs', p=p, observed=yt)
    
    trace = pm.sample(init = 'auto',cores = 1,  draws = 1000, tune = 1000, nuts_kwargs = {'target_accept':0.999}) 

print("It takes ",round((time.time() - now)/60,1), "minutes to run.")


# update the Xt and yt with the test sets
Xt.set_value(X_test)
yt.set_value(np.zeros(y_test.shape[0]).tolist())

# by using the model, sample from the posterior. 
with model:
    post_pred = pm.sample_ppc(trace, samples=1000)

# get the predictions 
pred = post_pred['obs'].mean(axis=0)
unique2, counts2 = np.unique(y_test, return_counts=True)
print(np.asarray((unique2, counts2)).T)

