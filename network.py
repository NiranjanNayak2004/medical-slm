import numpy as np


# Input data
X = np.array([
    [1, 2],
    [2, 3],
    [3, 4]
])


# Layer 1
weights1 = np.random.randn(2, 3)
bias1 = np.zeros(3)

output1 = np.dot(X, weights1) + bias1

# ReLU
output1 = np.maximum(0, output1)


# Layer 2
weights2 = np.random.randn(3, 1)
bias2 = np.zeros(1)

output2 = np.dot(output1, weights2) + bias2


print("Layer 1 output:")
print(output1)

print("\nFinal output:")
print(output2)
