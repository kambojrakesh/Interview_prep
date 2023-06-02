import numpy as np

#GD is used to find optimal param model by updating weight and noise
# loss for linear regression is mse and derivate  of loss is calculate by np.dot(X.T, error) / len(X)
# Sample data for demonstration
X = np.array([[1, 2],
              [3, 4],
              [5, 6]])
y = np.array([3, 6, 9])

# Hyperparameters
learning_rate = 0.01
epochs = 3

# Initialize the weights and bias
weights = np.zeros(X.shape[1])
bias = 0

# Lists to store cost values
costs = []

# Gradient descent
for epoch in range(epochs):
    # Compute predictions
    y_pred = np.dot(X, weights) + bias

    # Compute gradients
    error = y_pred - y
    gradient_weights = np.dot(X.T, error) / len(X)
    gradient_bias = np.sum(error) / len(X)

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
