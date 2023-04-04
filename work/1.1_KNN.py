# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 08:56:43 2023

@author: Vikki
"""
import numpy as np

# Define a function to calculate the Euclidean distance between two points
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))

# Define the KNN classifier
class KNN:
    def __init__(self, k):
        self.k = k
        
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        
    def predict(self, X):
        y_pred = []  # Create an empty list to store the predicted labels
        for i in range(len(X)):  # Loop through each input point
            distances = [round(euclidean_distance(X[i], x_train)) for x_train in self.X_train] 
            # Calculate the Euclidean distance between the input point and each training point
            print(distances)
            print("---------------------")
            print(np.argsort(distances))
            sorted_indices = np.argsort(distances)[:self.k]  
            print("---------------------")
            print(np.argsort(distances)[:self.k]  )
            # Find the indices of the k nearest neighbors
            k_nearest_labels = [self.y_train[j] for j in sorted_indices] 
            print("k_nearest_labels---------------------")
            print(k_nearest_labels)
            # Get the class labels of the k nearest neighbors
            y_pred.append(max(set(k_nearest_labels), key = k_nearest_labels.count)) 
            print("---------------------")
            print(y_pred)
            print("-------------------------------------------------------------")
            # Choose the most frequent class label among the k nearest neighbors and append it to the list of predicted labels
        return np.array(y_pred) 


    
    
# Create a small dataset with 10 data points and 2 features
X = np.array([[11, 2], [12, 1], [13, 4], [14, 3], [25, 6], [26, 5], [27, 8], [28, 7], [39, 10], [38, 9]])
y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2])

# Create a KNN classifier with k=5
clf = KNN(k=3)

# Train the classifier on the dataset
clf.fit(X, y)

# Predict the labels of a set of input points
X_test = np.array([[26, 1]])
y_pred = clf.predict(X_test)

# Print the predicted labels
print(y_pred)