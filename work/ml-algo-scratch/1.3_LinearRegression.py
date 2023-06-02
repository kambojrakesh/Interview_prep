import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score, mean_squared_error

class LinearRegression:
    def fit(self, x, y):
        self.x = x
        self.y = y
        self.coefficients = self._get_coefficients()

    def _get_coefficients(self):
        x_mean = np.mean(self.x)
        y_mean = np.mean(self.y)
        covariance = np.sum((self.x - x_mean) * (self.y - y_mean))
        variance = np.sum((self.x - x_mean) ** 2)
        slope = covariance / variance
        intercept = y_mean - slope * x_mean
        return slope, intercept

    def predict(self, x_test):
        slope, intercept = self.coefficients
        return slope * x_test + intercept

    def plot_regression_line(self):
        plt.scatter(self.x, self.y)
        x_min, x_max = np.min(self.x), np.max(self.x)
        x_range = np.linspace(x_min, x_max, 100)
        y_pred = self.predict(x_range)
        plt.plot(x_range, y_pred, color='red')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Linear Regression')
        plt.show()
        
    def evaluate(self, y_pred):
        r2 = r2_score(self.y, y_pred)
        mse = mean_squared_error(self.y, y_pred)
        print("The R2 score of the model is:", r2)
        print("The MSE score of the model is:", mse)
        
# Sample input
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 6, 8, 10])

# Instantiate and fit the LinearRegression model
model = LinearRegression()
model.fit(x, y)

# Plot the regression line
model.plot_regression_line()


# Call predict on new input values
x_new = np.array([6, 7, 8, 9 ,2])
y_pred = model.predict(x_new)

print("Predicted values:", y_pred)

model.evaluate(y_pred)