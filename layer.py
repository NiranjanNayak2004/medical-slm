import numpy as np


X = np.array([
    [1, 2],
    [2, 3],
    [3, 4]
])


weights = np.random.randn(2, 3)

bias = np.zeros(3)


# Linear calculation
output = np.dot(X, weights) + bias


# ReLU activation
relu_output = np.maximum(0, output)


print("Before ReLU:")
print(output)

print("\nAfter ReLU:")
print(relu_output)