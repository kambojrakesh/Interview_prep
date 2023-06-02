# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 09:36:51 2023

@author: Vikki
"""
import numpy as np

def lda(X, y, n_components):
    # Compute class-wise means
    class_means = []
    for class_label in np.unique(y):
        class_means.append(np.mean(X[y == class_label], axis=0))
    class_means = np.array(class_means)

    # Compute within-class scatter matrix
    within_class_scatter = np.zeros((X.shape[1], X.shape[1]))
    for class_label in np.unique(y):
        class_data = X[y == class_label]
        centered_data = class_data - class_means[class_label]
        within_class_scatter += np.dot(centered_data.T, centered_data)

    # Compute between-class scatter matrix
    total_mean = np.mean(X, axis=0)
    between_class_scatter = np.zeros((X.shape[1], X.shape[1]))
    for class_label in np.unique(y):
        n_samples = X[y == class_label].shape[0]
        class_mean_diff = class_means[class_label] - total_mean
        between_class_scatter += n_samples * np.outer(class_mean_diff, class_mean_diff)

    # Compute eigenvalues and eigenvectors of (within_class_scatter)^-1 * between_class_scatter
    eigenvalues, eigenvectors = np.linalg.eig(np.linalg.inv(within_class_scatter).dot(between_class_scatter))
    
    # Sort eigenvalues in descending order and select top 'n_components'
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]
    selected_eigenvectors = sorted_eigenvectors[:, :n_components]

    # Transform data to the selected eigenvectors (discriminant components)
    transformed_data = np.dot(X, selected_eigenvectors)

    return transformed_data

# Example usage:
X = np.array([[2, 3], [4, 5], [6, 7], [8, 9], [1, 2], [3, 4]])
y = np.array([0, 0, 0, 1, 1, 1])
n_components = 1

transformed_data = lda(X, y, n_components)
print(transformed_data)
