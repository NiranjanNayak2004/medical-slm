import numpy as np

from tokenizer import tokenize
from slm_model import MedicalSLM


# ==========================================
# 1. Load saved model data
# ==========================================

embedding_matrix = np.load(
    "transformer_embedding_matrix.npy"
)

output_weights = np.load(
    "transformer_output_weights.npy"
)

output_bias = np.load(
    "transformer_output_bias.npy"
)

vocabulary = np.load(
    "transformer_vocabulary.npy",
    allow_pickle=True
).item()

parameters = np.load(
    "transformer_parameters.npz"
)


# ==========================================
# 2. Create model
# ==========================================

model = MedicalSLM(
    vocabulary_size=len(vocabulary),
    embedding_size=embedding_matrix.shape[1]
)


# ==========================================
# 3. Load embeddings and output layer
# ==========================================

model.embedding_matrix = embedding_matrix

model.output_weights = output_weights

model.output_bias = output_bias


# ==========================================
# 4. Load Transformer parameters
# ==========================================

model.transformer.W_Q = parameters["W_Q"]

model.transformer.W_K = parameters["W_K"]

model.transformer.W_V = parameters["W_V"]

model.transformer.W1 = parameters["W1"]

model.transformer.b1 = parameters["b1"]

model.transformer.W2 = parameters["W2"]

model.transformer.b2 = parameters["b2"]

model.transformer.gamma1 = parameters["gamma1"]

model.transformer.beta1 = parameters["beta1"]

model.transformer.gamma2 = parameters["gamma2"]

model.transformer.beta2 = parameters["beta2"]


# ==========================================
# 5. Reverse vocabulary
# ==========================================

id_to_word = {
    value: key
    for key, value in vocabulary.items()
}


# ==========================================
# 6. Prediction function
# ==========================================

def predict(text):

    words = tokenize(text)

    input_ids = []

    for word in words:

        if word not in vocabulary:

            print(
                f"Unknown word: {word}"
            )

            return

        input_ids.append(
            vocabulary[word]
        )

    input_ids = np.array(
        input_ids
    )

    predicted_word, probabilities = (
        model.predict_next_word(
            input_ids,
            id_to_word
        )
    )

    return predicted_word


# ==========================================
# 7. Test
# ==========================================

text = "fever is a"

prediction = predict(text)

print("Input:")
print(text)

print("\nPredicted next word:")
print(prediction)