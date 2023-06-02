import numpy as np

# Sample data for demonstration
X = np.array([[1, 2],
              [3, 4],
              [5, 6]])
y = np.array([3, 6, 9])

# Hyperparameters
learning_rate = 0.01
epochs = 2

# Initialize the weights and bias
weights = np.zeros(X.shape[1])
bias = 0

# Stochastic gradient descent
for epoch in range(epochs):
    
    # Shuffle the data
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]

    # Iterate over each sample
    for i in range(X.shape[0]):
        # Compute prediction for the current sample
        y_pred = np.dot(X[i], weights) + bias

        # Compute gradients for the current sample
        error = y_pred - y[i]
        gradient_weights = np.dot(X[i].T, error)#X[i] * error
        gradient_bias = np.sum(error)

        # Update weights and bias
        weights = weights - learning_rate * gradient_weights
        bias = bias - learning_rate * gradient_bias

        # Compute and store the cost
        cost = np.mean(np.square(error)) / 2
        #costs.append(cost)

# Print the final weights, bias, and costs
print("Weights:", weights)
print("Bias:", bias)
print("Costs:", cost)
