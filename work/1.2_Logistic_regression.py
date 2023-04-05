# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 04:24:37 2023

@author: Vikki
"""
import numpy as np

class LogisticRegression:
    
    def __init__(self, learning_rate=0.1, num_iterations=1000, verbose=False):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.verbose = verbose
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def cost(self, y, y_pred):
        m = y.shape[0]
        J = (-1/m) * np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
        return J
    
    def fit(self, X, y):
        m, n = X.shape
        
        # add bias column to X
        X = np.hstack((np.ones((m, 1)), X))
        
        # initialize parameters
        self.theta = np.zeros(n+1)
        
        # gradient descent
        for i in range(self.num_iterations):
            z = np.dot(X, self.theta)
            y_pred = self.sigmoid(z)
            J = self.cost(y, y_pred)
            
            # update parameters
            gradient = (1/m) * np.dot(X.T, (y_pred - y))
            self.theta = self.theta - self.learning_rate * gradient
            
            # print cost every 100 iterations (optional)
            if self.verbose and i % 100 == 0:
                print(f"iteration {i}: cost = {J}")
    
    def predict(self, X):
        m = X.shape[0]
        # add bias column to X
        X = np.hstack((np.ones((m, 1)), X))
        y_pred = self.sigmoid(np.dot(X, self.theta))
        return np.round(y_pred).astype(int)



# create example data
X = np.array([[2.5, 3.0], [1.5, 2.0], [3.5, 4.0], [2.0, 2.5], [3.0, 2.0], [1.0, 1.5]])
y = np.array([1, 0, 1, 0, 1, 0])

# create logistic regression model
lr = LogisticRegression(learning_rate=0.1, num_iterations=1000, verbose=True)

# train model
lr.fit(X, y)

# predict classes for new data
X_test = np.array([[3.0, 3.5], [1.0, 2.5]])
y_pred = lr.predict(X_test)

# print predictions
print(y_pred)
