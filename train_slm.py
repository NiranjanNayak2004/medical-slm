import numpy as np
from tokenizer import tokenize


# -------------------------
# 1. Load medical text
# -------------------------

with open("data/medical_text.txt", "r") as file:
    text = file.read()


# -------------------------
# 2. Tokenize
# -------------------------

tokens = tokenize(text)


# -------------------------
# 3. Create vocabulary
# -------------------------

vocabulary = {}

for token in tokens:
    if token not in vocabulary:
        vocabulary[token] = len(vocabulary)


# -------------------------
# 4. Convert tokens to IDs
# -------------------------

token_ids = []

for token in tokens:
    token_ids.append(vocabulary[token])


# -------------------------
# 5. Create training data
# -------------------------

context_size = 3

X = []
y = []

for i in range(len(token_ids) - context_size):

    input_sequence = token_ids[i:i + context_size]

    target = token_ids[i + context_size]

    X.append(input_sequence)
    y.append(target)

X = np.array(X)
y = np.array(y)


print("Number of training examples:", len(X))
print("Vocabulary size:", len(vocabulary))


# -------------------------
# 6. Model
# -------------------------

vocab_size = len(vocabulary)
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
# 7. Training
# -------------------------

for epoch in range(1000):

    total_loss = 0

    for i in range(len(X)):

        input_ids = X[i]
        target_id = y[i]

        # Get embeddings
        embeddings = embedding_matrix[input_ids]

        # Combine context
        context_vector = np.mean(
            embeddings,
            axis=0
        )

        # Calculate logits
        logits = np.dot(
            context_vector,
            output_weights
        ) + output_bias

        # Softmax
        exp_logits = np.exp(
            logits - np.max(logits)
        )

        probabilities = (
            exp_logits / np.sum(exp_logits)
        )

        # Loss
        loss = -np.log(
            probabilities[target_id]
        )

        total_loss += loss

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
        # Update parameters
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

    average_loss = total_loss / len(X)

    if epoch % 100 == 0:

        print(
            "Epoch:",
            epoch,
            "Loss:",
            average_loss
        )
# -------------------------
# 8. Prediction function
# -------------------------

id_to_word = {}

for word, token_id in vocabulary.items():
    id_to_word[token_id] = word


def predict_next_word(input_words):

    input_ids = []

    for word in input_words:
        input_ids.append(vocabulary[word])

    embeddings = embedding_matrix[input_ids]

    context_vector = np.mean(
        embeddings,
        axis=0
    )

    logits = np.dot(
        context_vector,
        output_weights
    ) + output_bias

    predicted_id = np.argmax(logits)

    predicted_word = id_to_word[predicted_id]

    return predicted_word


# Test
input_words = ["fever", "is", "a"]

prediction = predict_next_word(input_words)

print("\nInput:", input_words)
print("Predicted next word:", prediction)
# -------------------------
# 9. Save trained model
# -------------------------

np.save("embedding_matrix.npy", embedding_matrix)
np.save("output_weights.npy", output_weights)
np.save("output_bias.npy", output_bias)

np.save("vocabulary.npy", vocabulary)

print("\nModel saved successfully!")