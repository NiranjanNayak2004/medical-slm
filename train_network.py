import numpy as np


# Training data
X = np.array([
    [1, 2],
    [2, 3],
    [3, 4],
    [4, 5]
])

# Correct answers
y = np.array([
    [5],
    [8],
    [11],
    [14]
])


# Layer 1
weights1 = np.random.randn(2, 3)
bias1 = np.zeros((1, 3))


# Layer 2
weights2 = np.random.randn(3, 1)
bias2 = np.zeros((1, 1))


learning_rate = 0.01


for epoch in range(1000):

    # ----------------
    # Forward pass
    # ----------------

    layer1 = np.dot(X, weights1) + bias1

    relu = np.maximum(0, layer1)

    output = np.dot(relu, weights2) + bias2


    # ----------------
    # Calculate loss
    # ----------------

    error = output - y

    loss = np.mean(error ** 2)


    # ----------------
    # Backward pass
    # ----------------

    output_gradient = 2 * error / len(X)

    weights2_gradient = np.dot(relu.T, output_gradient)

    bias2_gradient = np.sum(output_gradient, axis=0, keepdims=True)


    relu_gradient = np.dot(output_gradient, weights2.T)

    layer1_gradient = relu_gradient * (layer1 > 0)

    weights1_gradient = np.dot(X.T, layer1_gradient)

    bias1_gradient = np.sum(
        layer1_gradient,
        axis=0,
        keepdims=True
    )


    # ----------------
    # Update parameters
    # ----------------

    weights2 -= learning_rate * weights2_gradient
    bias2 -= learning_rate * bias2_gradient

    weights1 -= learning_rate * weights1_gradient
    bias1 -= learning_rate * bias1_gradient


    if epoch % 100 == 0:
        print("Epoch:", epoch, "Loss:", loss)


# Test the trained network

test_input = np.array([[5, 6]])

layer1 = np.dot(test_input, weights1) + bias1

relu = np.maximum(0, layer1)

prediction = np.dot(relu, weights2) + bias2

print("\nPrediction:", prediction)