import numpy as np


# -------------------------
# 1. Load trained model
# -------------------------

embedding_matrix = np.load(
    "embedding_matrix.npy"
)

output_weights = np.load(
    "output_weights.npy"
)

output_bias = np.load(
    "output_bias.npy"
)

vocabulary = np.load(
    "vocabulary.npy",
    allow_pickle=True
).item()


# -------------------------
# 2. Prediction function
# -------------------------

def predict_next_word(
    input_words,
    vocabulary,
    embedding_matrix,
    output_weights,
    output_bias
):

    # Convert words → token IDs
    input_ids = []

    for word in input_words:
        input_ids.append(vocabulary[word])

    # Get embeddings
    embeddings = embedding_matrix[input_ids]

    # Combine the context embeddings
    context_vector = np.mean(
        embeddings,
        axis=0
    )

    # Calculate scores for every word
    logits = np.dot(
        context_vector,
        output_weights
    ) + output_bias

    # Select the word with highest score
    predicted_id = np.argmax(logits)

    # Convert token ID → word
    id_to_word = {
        token_id: word
        for word, token_id in vocabulary.items()
    }

    predicted_word = id_to_word[predicted_id]

    return predicted_word


# -------------------------
# 3. Test the model
# -------------------------

input_words = [
    "fever",
    "is",
    "a"
]

prediction = predict_next_word(
    input_words,
    vocabulary,
    embedding_matrix,
    output_weights,
    output_bias
)

print("Input:", input_words)
print("Predicted next word:", prediction)