# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 04:24:37 2023

@author: Vikki
"""
#in case of lg, gradient  is calculated as (1/m) * np.dot(X.T, (y_pred - y)),
# we obtain the gradient of the cost function with respect to the parameters.
#This gradient points in the direction of steepest ascent of
#the cost function, and we subtract it from the current parameters to update them in 
#the opposite direction (gradient descent). This iterative parameter update process helps
#minimize the cost function and find the optimal values for the parameters in logistic regression


## in linear regression
#y_pred = X.dot(w) + b
#error = y_pred - y
#gradient_w = (2 / m) * X.T.dot(error)
#gradient_b = (2 / m) * np.sum(error)
#w = w - alpha * gradient_w
#b = b - alpha * gradient_b





import numpy as np

class LogisticRegression:
    
    def __init__(self, learning_rate=0.1, num_iterations=1000, verbose=False):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.verbose = verbose
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    #binary cross entropy for weight update in the lg
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
            #weighted sum = w1 * x1 + w2 * x2
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
X = np.array([[25, 3.0], [1.5, 2.0], [35, 4.0], [2.0, 2.5], [30, 2.0], [1.0, 1.5]])
y = np.array([1, 0, 1, 0, 1, 0])

# create logistic regression model
lr = LogisticRegression(learning_rate=0.1, num_iterations=1000, verbose=True)

# train model
#lr.fit(X, y)

# predict classes for new data
#X_test = np.array([[30, 3.5], [1.0, 2.5]])
#y_pred = lr.predict(X_test)

# print predictions
#print(y_pred)


#======================



# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 04:24:37 2023

@author: Vikki
"""
#in case of lg, gradient  is calculated as (1/m) * np.dot(X.T, (y_pred - y)),
# we obtain the gradient of the cost function with respect to the parameters.
#This gradient points in the direction of steepest ascent of
#the cost function, and we subtract it from the current parameters to update them in 
#the opposite direction (gradient descent). This iterative parameter update process helps
#minimize the cost function and find the optimal values for the parameters in logistic regression


## in linear regression
#y_pred = X.dot(w) + b
#error = y_pred - y
#gradient_w = (2 / m) * X.T.dot(error)
#gradient_b = (2 / m) * np.sum(error)
#w = w - alpha * gradient_w
#b = b - alpha * gradient_b





import numpy as np

import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def _initialize_parameters(self, num_features):
        self.weights = np.zeros(num_features)
        #print(self.weights)
        self.bias = 0

    def fit(self, X, y):
        num_samples, num_features = X.shape
        self._initialize_parameters(num_features)

        for i in range(self.num_iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = self._sigmoid(linear_model)

            dw = (1 / num_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / num_samples) * np.sum(y_pred - y)

            self.weights = self.weights - self.learning_rate * dw
            self.bias =  self.bias - self.learning_rate * db

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_pred = self._sigmoid(linear_model)
        y_pred_class = np.where(y_pred > 0.5, 1, 0)
        return y_pred_class



# Sample input
X = np.array([[1.5, 2.0], [2.0, 2.5], [2.5, 3.0], [3.0, 3.5], [4.0, 4.5]])
y = np.array([0, 0, 0, 1, 1])

# Create logistic regression model
model = LogisticRegression(learning_rate=0.1, num_iterations=1000)

# Train the model
model.fit(X, y)

# Predict classes for new data
X_test = np.array([[30, 3.5], [1.0, 2.5]])
y_pred = model.predict(X_test)

# Print predictions
print("Predicted classes:", y_pred)

