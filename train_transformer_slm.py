import numpy as np

from data_loader import medical_data
from tokenizer import tokenize
from slm_model import MedicalSLM


# ==========================================
# 1. Load medical training text
# ==========================================

with open(
    "data/medical_text.txt",
    "r"
) as file:

    text = file.read()


# ==========================================
# 2. Tokenize
# ==========================================

tokens = tokenize(text)

print("Number of tokens:")
print(len(tokens))


# ==========================================
# 3. Build vocabulary
# ==========================================

vocabulary = {}

for token in tokens:

    if token not in vocabulary:

        vocabulary[token] = len(vocabulary)


id_to_word = {
    value: key
    for key, value in vocabulary.items()
}


print("Vocabulary size:")
print(len(vocabulary))


# ==========================================
# 4. Convert words → token IDs
# ==========================================

token_ids = np.array([
    vocabulary[token]
    for token in tokens
])


# ==========================================
# 5. Create training examples
# ==========================================

context_size = 3

training_data = []

for i in range(
    len(token_ids) - context_size
):

    input_ids = token_ids[
        i:i + context_size
    ]

    target_id = token_ids[
        i + context_size
    ]

    training_data.append(
        (
            input_ids,
            target_id
        )
    )


print("Training examples:")
print(len(training_data))


# ==========================================
# 6. Create model
# ==========================================

embedding_size = 16

model = MedicalSLM(
    vocabulary_size=len(vocabulary),
    embedding_size=embedding_size
)


# ==========================================
# 7. Training settings
# ==========================================

learning_rate = 0.01

epochs = 100


# ==========================================
# 8. Training loop
# ==========================================

for epoch in range(epochs):

    total_loss = 0

    for input_ids, target_id in training_data:

        # -----------------------------
        # Forward pass
        # -----------------------------

        logits, probabilities = model.forward(
            input_ids
        )


        # -----------------------------
        # Cross entropy loss
        # -----------------------------

        loss = -np.log(
            probabilities[target_id] + 1e-9
        )

        total_loss += loss


        # -----------------------------
        # Output gradient
        # -----------------------------

        grad_logits = probabilities.copy()

        grad_logits[target_id] -= 1


        # -----------------------------
        # Output layer gradients
        # -----------------------------

        transformer_output = model.transformer.forward(
            model.embedding_matrix[input_ids]
        )

        last_token = transformer_output[-1]


        grad_output_weights = np.outer(
            last_token,
            grad_logits
        )

        grad_output_bias = grad_logits


        # Gradient flowing back into last token
        grad_last_token = np.dot(
            model.output_weights,
            grad_logits
        )


        # -----------------------------
        # Transformer backward
        # -----------------------------

        token_embeddings = model.embedding_matrix[
            input_ids
        ]

        transformer_output, cache = (
            model.transformer.forward(
                token_embeddings,
                return_cache=True
            )
        )


        grad_transformer_output = np.zeros_like(
            transformer_output
        )

        grad_transformer_output[-1] = (
            grad_last_token
        )


        grad_embeddings, gradients = (
            model.transformer.backward(
                grad_transformer_output,
                cache
            )
        )


        # -----------------------------
        # Update output layer
        # -----------------------------

        model.output_weights -= (
            learning_rate
            * grad_output_weights
        )

        model.output_bias -= (
            learning_rate
            * grad_output_bias
        )


        # -----------------------------
        # Update Transformer
        # -----------------------------

        model.transformer.W_Q -= (
            learning_rate
            * gradients["W_Q"]
        )

        model.transformer.W_K -= (
            learning_rate
            * gradients["W_K"]
        )

        model.transformer.W_V -= (
            learning_rate
            * gradients["W_V"]
        )

        model.transformer.W1 -= (
            learning_rate
            * gradients["W1"]
        )

        model.transformer.b1 -= (
            learning_rate
            * gradients["b1"]
        )

        model.transformer.W2 -= (
            learning_rate
            * gradients["W2"]
        )

        model.transformer.b2 -= (
            learning_rate
            * gradients["b2"]
        )

        model.transformer.gamma1 -= (
            learning_rate
            * gradients["gamma1"]
        )

        model.transformer.beta1 -= (
            learning_rate
            * gradients["beta1"]
        )

        model.transformer.gamma2 -= (
            learning_rate
            * gradients["gamma2"]
        )

        model.transformer.beta2 -= (
            learning_rate
            * gradients["beta2"]
        )


        # -----------------------------
        # Update embeddings
        # -----------------------------

        for position, token_id in enumerate(
            input_ids
        ):

            model.embedding_matrix[token_id] -= (
                learning_rate
                * grad_embeddings[position]
            )


    # ======================================
    # Epoch result
    # ======================================

    average_loss = (
        total_loss
        / len(training_data)
    )

    if epoch % 10 == 0:

        print(
            f"Epoch {epoch} | Loss: {average_loss:.4f}"
        )


print("\nTraining complete!")


# ==========================================
# Test prediction
# ==========================================

test_words = [
    "fever",
    "is",
    "a"
]


test_ids = np.array([
    vocabulary[word]
    for word in test_words
])


predicted_word, probabilities = (
    model.predict_next_word(
        test_ids,
        id_to_word
    )
)


print("\nInput:")
print(test_words)

print("\nPredicted next word:")
print(predicted_word)
# ==========================================
# Save trained model
# ==========================================

np.save(
    "transformer_embedding_matrix.npy",
    model.embedding_matrix
)

np.save(
    "transformer_output_weights.npy",
    model.output_weights
)

np.save(
    "transformer_output_bias.npy",
    model.output_bias
)

np.save(
    "transformer_vocabulary.npy",
    vocabulary
)


# Save Transformer parameters

np.savez(
    "transformer_parameters.npz",

    W_Q=model.transformer.W_Q,
    W_K=model.transformer.W_K,
    W_V=model.transformer.W_V,

    W1=model.transformer.W1,
    b1=model.transformer.b1,

    W2=model.transformer.W2,
    b2=model.transformer.b2,

    gamma1=model.transformer.gamma1,
    beta1=model.transformer.beta1,

    gamma2=model.transformer.gamma2,
    beta2=model.transformer.beta2
)


print("\nModel saved successfully!")