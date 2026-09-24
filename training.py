import numpy as np


# Input words represented as vectors
X = np.array([
    [1, 0],
    [1, 0],
    [1, 0],
    [0, 1]
])


# Expected answers
y = np.array([
    1,
    1,
    1,
    0
])


# Start with random weights
weights = np.random.randn(2)

# Start with zero bias
bias = 0.0

learning_rate = 0.1


for epoch in range(100):

    for i in range(len(X)):

        # Forward pass
        prediction = np.dot(X[i], weights) + bias

        # Error
        error = y[i] - prediction

        # Update weights
        weights = weights + learning_rate * error * X[i]

        # Update bias
        bias = bias + learning_rate * error


print("Weights:", weights)
print("Bias:", bias)