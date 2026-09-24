import numpy as np

# -------------------------
# 1. Model setup
# -------------------------

vocab_size = 20
embedding_size = 8

embedding_matrix = np.random.randn(
    vocab_size,
    embedding_size
)

output_weights = np.random.randn(
    embedding_size,
    vocab_size
)

output_bias = np.zeros(vocab_size)

learning_rate = 0.01


# -------------------------
# 2. Training example
# -------------------------

input_ids = np.array([0, 1, 2])

# Correct next token
target_id = 3


# -------------------------
# 3. Training
# -------------------------

for epoch in range(1000):

    # Get embeddings
    embeddings = embedding_matrix[input_ids]

    # Combine context
    context_vector = np.mean(
        embeddings,
        axis=0
    )

    # Calculate scores
    logits = np.dot(
        context_vector,
        output_weights
    ) + output_bias

    # Softmax
    exp_logits = np.exp(logits - np.max(logits))
    probabilities = exp_logits / np.sum(exp_logits)

    # Loss
    loss = -np.log(probabilities[target_id])

    # -------------------------
    # Backpropagation
    # -------------------------

    output_gradient = probabilities.copy()

    output_gradient[target_id] -= 1

    weights_gradient = np.outer(
        context_vector,
        output_gradient
    )

    bias_gradient = output_gradient

    context_gradient = np.dot(
        output_weights,
        output_gradient
    )

    embedding_gradient = (
        context_gradient / len(input_ids)
    )

    # -------------------------
    # Update
    # -------------------------

    output_weights -= (
        learning_rate * weights_gradient
    )

    output_bias -= (
        learning_rate * bias_gradient
    )

    for token_id in input_ids:
        embedding_matrix[token_id] -= (
            learning_rate * embedding_gradient
        )

    if epoch % 100 == 0:
        print(
            "Epoch:",
            epoch,
            "Loss:",
            loss
        )


# -------------------------
# 4. Test
# -------------------------

embeddings = embedding_matrix[input_ids]

context_vector = np.mean(
    embeddings,
    axis=0
)

logits = np.dot(
    context_vector,
    output_weights
) + output_bias

prediction = np.argmax(logits)

print("\nCorrect token:", target_id)
print("Predicted token:", prediction)