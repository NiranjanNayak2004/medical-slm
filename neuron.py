import numpy as np


# Training data
X = np.array([1, 2, 3, 4])

# Correct answers
y = np.array([2, 4, 6, 8])


# Start with random weight
weight = np.random.randn()

# Start with zero bias
bias = 0.0

learning_rate = 0.01


for epoch in range(1000):

    for i in range(len(X)):

        # 1. Prediction
        prediction = X[i] * weight + bias

        # 2. Calculate error
        error = prediction - y[i]

        # 3. Calculate gradients
        weight_gradient = 2 * error * X[i]
        bias_gradient = 2 * error

        # 4. Update weight and bias
        weight = weight - learning_rate * weight_gradient
        bias = bias - learning_rate * bias_gradient


print("Weight:", weight)
print("Bias:", bias)


# Test the trained model
input_value = 5

prediction = input_value * weight + bias

print("Input:", input_value)
print("Prediction:", prediction)