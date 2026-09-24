import numpy as np


# Each row is one training example
X = np.array([
    [1, 2],
    [2, 3],
    [3, 4],
    [4, 5]
])


# Correct answers
y = np.array([5, 8, 11, 14])


# Two inputs → therefore two weights
weights = np.random.randn(2)

bias = 0.0

learning_rate = 0.01


for epoch in range(1000):

    for i in range(len(X)):

        # Prediction
        prediction = np.dot(X[i], weights) + bias

        # Error
        error = prediction - y[i]

        # Gradients
        weight_gradient = 2 * error * X[i]
        bias_gradient = 2 * error

        # Update
        weights = weights - learning_rate * weight_gradient
        bias = bias - learning_rate * bias_gradient


print("Weights:", weights)
print("Bias:", bias)


# Test
input_value = np.array([5, 6])

prediction = np.dot(input_value, weights) + bias

print("Input:", input_value)
print("Prediction:", prediction)