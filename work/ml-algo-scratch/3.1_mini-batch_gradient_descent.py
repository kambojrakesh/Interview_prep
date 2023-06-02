import numpy as np

# Sample data for demonstration
X = np.array([[1, 2],
              [3, 4],
              [5, 6]])
y = np.array([3, 6, 9])

# Hyperparameters
learning_rate = 0.01
batch_size = 2
epochs = 100

# Initialize the weights and bias
weights = np.zeros(X.shape[1])
bias = 0

# Lists to store cost values
costs = []

# Mini-batch gradient descent
for epoch in range(epochs):
    # Shuffle the data
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]

    # Perform mini-batch updates
    for i in range(0, X.shape[0], batch_size):
        # Select mini-batch
        X_batch = X[i:i+batch_size]
        y_batch = y[i:i+batch_size]

        # Compute predictions
        y_pred = np.dot(X_batch, weights) + bias

        # Compute gradients
        error = y_pred - y_batch
        gradient_weights = np.dot(X_batch.T, error) / batch_size
        gradient_bias = np.sum(error) / batch_size

        # Update weights and bias
        weights = weights - learning_rate * gradient_weights
        bias = bias - learning_rate * gradient_bias

        # Compute and store the cost
        cost = np.mean(np.square(error)) / 2
        costs.append(cost)

# Print the final weights, bias, and costs
print("Weights:", weights)
print("Bias:", bias)
print("Costs:", costs)
