# BayesianAnalysis
Bayesian Approach with Python

# BayesianLogisticRegression.py
I am calculating the out of sample prediction accuracy by using Bayesian Logistic Regression. 
There are not many examples for the multinomial categorical outcome models by using pymc3. I thought this will be a good resource for anyone interested in using Bayesian approach with Logistic Regression. 

Thanks. 

# BayesianNeuralNetwork.py
This code calculates the out of sample prediction accuracy by using Bayesian Neural Network. 
Most of the examples on pymc3 documentation focus on the models for data with binary outcomes. 
With this code, I am predicting a dependent variable with multinomial output by using Bayesian Neural Network. 
There are two main differences between this code and the code in the pymc3 documentation. The first one is the likelihood function which is multinomial rather than binomial which gives an option to focus on multinomial outcome. Second difference is the shape of the dependent variable in training and test sets (y_train and y_test). They need to be defined as numpy nd array rather than a numpy vector. 
Data that I am using is the Iris data set and its independent variable has 3 distinct values (0,1,2). 
If you want to expand this code for data with more than 3 outputs then you will need to update the Y numpy nd array. 
Currently, Y has 3 binary columns since it has 3 distinct outputs. I will automate this part in the future. 

Thanks. 
