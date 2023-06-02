# -*- coding: utf-8 -*-
"""
Created on Wed May 31 00:18:24 2023

@author: Vikki
"""

#    Standardize the data.
#    Obtain the covariance matrix.
#    Compute the eigenvalues and eigenvectors of the covariance matrix.
#    Sort eigenvalues and their corresponding eigenvectors.
#    Select a subset from the rearranged eigenvalue matrix.
#    Transform the data.
    
import numpy as np

# Define the PCA function
def pca(X, num_components):
    # Step 1: Standardize the data
    X_std = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

    # Step 2: Obtain the covariance matrix
    covariance_matrix = np.cov(X_std.T)

    # Step 3: Compute the eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

    # Step 4: Sort the eigenvalues in descending order
    idx = np.argsort(eigenvalues)[::-1]
    
    # Rearrange the eigenvalues and eigenvectors
    eigenvectors = eigenvectors[:, idx]
    
    # Step 5: Select the first 'num_components' eigenvectors
    eigenvectors = eigenvectors[:, :num_components]
    
    # Transform the data using the selected eigenvectors
    transformed = X_std.dot(eigenvectors)
    
    return transformed

# Create a random numpy array for testing
X = np.array([[332, 3, 4], [5, 7, 9], [8, 6, 9]])

# Call the PCA function and transform the data to 2D
transformed_data = pca(X, 2)

print(transformed_data.shape)


#====================

import numpy as np

# Define the PCA function
def pca(X, num_components):
    xs = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    
    con = np.cov(xs.T)
    
    eva, eve =  np.linalg.eig(con)
    
    idx = np.argsort(eva)[::-1]
    
    matrix = eve[:,idx][:, :num_components]
    
    return xs.dot(matrix)


# Create a random numpy array for testing
X = np.array([[1,2,3], [4,3,2], [6, 1,7]])

# Call the PCA function and transform the data to 2D
transformed_data = pca(X, 2)

print(transformed_data)